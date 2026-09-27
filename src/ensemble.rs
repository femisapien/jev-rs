//! Stacked meta-calibrator over several backends' probability vectors.
//!
//! Each member contributes one `jev eval --rows` file. Rows are aligned on
//! `(case_index, id)`; a question is used only if every member answered it.
//! A multinomial logistic model is fit on a case-level train split (no case
//! leaks across the split) and reported on the held-out test split next to
//! each member and the uniform probability average, all through
//! [`eval::metrics`] so the numbers line up with `jev eval`.
//!
//! Features per question: each member's temperature-calibrated probability
//! vector padded to `max_classes`, a one-hot question kind, and `n` scaled.
//! Logits beyond the question's option count are masked, so the ensemble's
//! output is again a distribution over the declared options.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::backend::{BackendError, FullBackend, Scorer};
use crate::eval::{self, Metrics, Row};
use crate::full_eval;
use crate::judge::{Judge, RawQuestion};
use crate::protocol::{parse_questions, Request};
use crate::score::{softmax, Calibration};

/// Options are single-letter labels in the scorer path, so at most 26.
pub const MAX_CLASSES: usize = 26;

const KINDS: [&str; 3] = ["noul", "choice", "score"];

/// A fitted stacker. `members` order defines the feature layout and must
/// match the member order when applying it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stacker {
    pub members: Vec<String>,
    pub max_classes: usize,
    pub dim: usize,
    /// `weights[class][feature]`, class < max_classes.
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub train_loss: f64,
    pub iterations: usize,
    pub lambda: f64,
    /// Per-member temperature calibration fit on the train split; the same
    /// transform must be applied to member log-probs at apply/live time so
    /// features match training. Empty in models saved before this field —
    /// callers fall back to temperature 1.
    #[serde(default)]
    pub temps: Vec<Calibration>,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub members: Vec<String>,
    pub aligned_questions: usize,
    /// Questions dropped because a member failed or rows disagreed.
    pub dropped_questions: usize,
    pub train_questions: usize,
    pub test_questions: usize,
    pub train_cases: usize,
    pub test_cases: usize,
    pub max_classes: usize,
    pub feature_dim: usize,
    pub train_loss: f64,
    /// Test-split metrics per member (temperature fit on the train split).
    pub member_test: BTreeMap<String, Metrics>,
    /// Uniform average of the members' probability vectors, test split.
    pub mean_test: Metrics,
    pub stacker_test: Metrics,
    pub stacker_train: Metrics,
    pub model: Option<String>,
    pub latency_note: String,
}

/// One question aligned across every member.
struct Aligned<'a> {
    case_index: usize,
    kind: String,
    n: usize,
    gold: usize,
    rows: Vec<&'a Row>,
}

pub fn load_rows(path: &Path) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|e| format!("{}: line {}: {e}", path.display(), i + 1))
        })
        .collect()
}

/// Deterministic case-level split: ~`test_frac` of cases go to test.
fn is_test_case(case_index: usize, test_frac: f64) -> bool {
    let mut x = case_index as u64 ^ 0x9E37_79B9_7F4A_7C15;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let bucket = x % 1000;
    bucket < (test_frac.clamp(0.0, 1.0) * 1000.0) as u64
}

fn align(members: &[Vec<Row>]) -> (Vec<Aligned<'_>>, usize) {
    let maps: Vec<BTreeMap<(usize, String), &Row>> = members
        .iter()
        .map(|rs| {
            rs.iter()
                .map(|r| ((r.case_index, r.id.clone()), r))
                .collect()
        })
        .collect();
    let mut dropped = 0usize;
    let mut out = Vec::new();
    for (key, r0) in &maps[0] {
        let rows: Vec<&Row> = maps.iter().filter_map(|m| m.get(key).copied()).collect();
        if rows.len() != members.len()
            || rows
                .iter()
                .any(|r| r.kind != r0.kind || r.n != r0.n || r.gold != r0.gold)
        {
            dropped += 1;
            continue;
        }
        out.push(Aligned {
            case_index: key.0,
            kind: r0.kind.clone(),
            n: r0.n,
            gold: r0.gold,
            rows,
        });
    }
    (out, dropped)
}

fn fit_member_temps(rows: &[&Row]) -> Calibration {
    let owned: Vec<Row> = rows.iter().map(|r| (*r).clone()).collect();
    eval::fit(&owned)
}

fn features(probs: &[Vec<f64>], kind: &str, n: usize, max_classes: usize) -> Vec<f64> {
    let mut x = Vec::with_capacity(probs.len() * max_classes + KINDS.len() + 1);
    for p in probs {
        x.extend_from_slice(&p[..p.len().min(max_classes)]);
        x.extend(std::iter::repeat(0.0).take(max_classes - p.len().min(max_classes)));
    }
    for k in KINDS {
        x.push(if kind == k { 1.0 } else { 0.0 });
    }
    x.push(n as f64 / max_classes as f64);
    x
}

impl Stacker {
    pub fn dim_for(&self, max_classes: usize) -> usize {
        self.members.len() * max_classes + KINDS.len() + 1
    }

    /// Probabilities over the question's `n` options (`n <= max_classes`).
    pub fn predict(&self, x: &[f64], n: usize) -> Vec<f64> {
        let n = n.min(self.max_classes);
        let logits: Vec<f64> = (0..n)
            .map(|c| {
                self.weights[c]
                    .iter()
                    .zip(x)
                    .map(|(w, f)| w * f)
                    .sum::<f64>()
                    + self.bias[c]
            })
            .collect();
        softmax(&logits, 1.0)
    }

    /// Temperature for member `m` on this question (1.0 if the model predates
    /// saved temps or the bucket was never seen in training).
    pub fn temp(&self, member: usize, kind: &str, n: usize) -> f64 {
        self.temps
            .get(member)
            .map(|c| c.temperature(kind, n))
            .unwrap_or(1.0)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap())
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Fit the stacker on the train split, report on train and test.
pub fn fit_report(
    members: &[(String, Vec<Row>)],
    test_frac: f64,
    iterations: usize,
    lambda: f64,
) -> Result<(Report, Stacker), String> {
    if members.len() < 2 {
        return Err("ensemble needs at least two member row files".into());
    }
    let names: Vec<String> = members.iter().map(|(n, _)| n.clone()).collect();
    let rowsets: Vec<Vec<Row>> = members.iter().map(|(_, r)| r.clone()).collect();
    let (aligned, dropped) = align(&rowsets);
    if aligned.is_empty() {
        return Err("no questions are present in every member's rows".into());
    }
    let max_classes = aligned
        .iter()
        .map(|a| a.n)
        .max()
        .unwrap_or(2)
        .min(MAX_CLASSES);
    let train: Vec<&Aligned> = aligned
        .iter()
        .filter(|a| !is_test_case(a.case_index, test_frac))
        .collect();
    let test: Vec<&Aligned> = aligned
        .iter()
        .filter(|a| is_test_case(a.case_index, test_frac))
        .collect();
    if train.is_empty() || test.is_empty() {
        return Err(format!(
            "split produced {} train / {} test questions (need both; adjust --test)",
            train.len(),
            test.len()
        ));
    }

    // Per-member temperature fitted on the train split only.
    let temps: Vec<Calibration> = (0..members.len())
        .map(|m| fit_member_temps(&train.iter().map(|a| a.rows[m]).collect::<Vec<_>>()))
        .collect();
    let probs_of = |a: &Aligned| -> Vec<Vec<f64>> {
        (0..a.rows.len())
            .map(|m| {
                let t = temps[m].temperature(&a.kind, a.n);
                softmax(&a.rows[m].raw_logprobs, t)
            })
            .collect()
    };

    let dim = members.len() * max_classes + KINDS.len() + 1;
    let train_x: Vec<Vec<f64>> = train
        .iter()
        .map(|a| features(&probs_of(a), &a.kind, a.n, max_classes))
        .collect();
    let train_golds: Vec<usize> = train.iter().map(|a| a.gold).collect();
    let train_ns: Vec<usize> = train.iter().map(|a| a.n).collect();
    let (weights, bias, loss) = fit_logreg(
        &train_x,
        &train_golds,
        &train_ns,
        max_classes,
        iterations,
        lambda,
    );
    let stacker = Stacker {
        members: names.clone(),
        max_classes,
        dim,
        weights,
        bias,
        train_loss: loss,
        iterations,
        lambda,
        temps: temps.clone(),
    };

    let stacker_rows = |set: &[&Aligned]| -> Vec<Row> {
        set.iter()
            .map(|a| {
                let p = stacker.predict(&features(&probs_of(a), &a.kind, a.n, max_classes), a.n);
                meta_row(a, &p, a.rows.iter().map(|r| r.latency_ms).sum())
            })
            .collect()
    };
    let mean_rows = |set: &[&Aligned]| -> Vec<Row> {
        set.iter()
            .map(|a| {
                let ps = probs_of(a);
                let avg: Vec<f64> = (0..a.n)
                    .map(|c| ps.iter().map(|p| p[c]).sum::<f64>() / ps.len() as f64)
                    .collect();
                meta_row(a, &avg, a.rows.iter().map(|r| r.latency_ms).sum())
            })
            .collect()
    };
    let member_test: BTreeMap<String, Metrics> = (0..members.len())
        .map(|m| {
            let rows: Vec<Row> = test.iter().map(|a| a.rows[m].clone()).collect();
            (names[m].clone(), eval::metrics(&rows, 0, &temps[m]))
        })
        .collect();

    let mut train_cases: Vec<usize> = train.iter().map(|a| a.case_index).collect();
    train_cases.sort_unstable();
    train_cases.dedup();
    let mut test_cases: Vec<usize> = test.iter().map(|a| a.case_index).collect();
    test_cases.sort_unstable();
    test_cases.dedup();

    let report = Report {
        members: names,
        aligned_questions: aligned.len(),
        dropped_questions: dropped,
        train_questions: train.len(),
        test_questions: test.len(),
        train_cases: train_cases.len(),
        test_cases: test_cases.len(),
        max_classes,
        feature_dim: dim,
        train_loss: loss,
        member_test,
        mean_test: eval::metrics(&mean_rows(&test), 0, &Calibration::default()),
        stacker_test: eval::metrics(&stacker_rows(&test), 0, &Calibration::default()),
        stacker_train: eval::metrics(&stacker_rows(&train), 0, &Calibration::default()),
        model: None,
        latency_note: "meta rows report the sum of member latencies (all members run)".into(),
    };
    Ok((report, stacker))
}

/// Apply a saved stacker to new rows; metrics are raw (no re-fit).
pub fn apply_report(
    model_path: &Path,
    members: &[(String, Vec<Row>)],
) -> Result<serde_json::Value, String> {
    let stacker = Stacker::load(model_path)?;
    let names: Vec<String> = members.iter().map(|(n, _)| n.clone()).collect();
    if names != stacker.members {
        return Err(format!(
            "model expects members {:?}, got {:?}",
            stacker.members, names
        ));
    }
    let rowsets: Vec<Vec<Row>> = members.iter().map(|(_, r)| r.clone()).collect();
    let (aligned, dropped) = align(&rowsets);
    if aligned.is_empty() {
        return Err("no questions are present in every member's rows".into());
    }
    let probs_of = |a: &Aligned| -> Vec<Vec<f64>> {
        (0..a.rows.len())
            .map(|m| {
                let t = stacker.temp(m, &a.kind, a.n);
                softmax(&a.rows[m].raw_logprobs, t)
            })
            .collect()
    };
    let stacker_rows: Vec<Row> = aligned
        .iter()
        .map(|a| {
            let p = stacker.predict(
                &features(&probs_of(a), &a.kind, a.n, stacker.max_classes),
                a.n,
            );
            meta_row(a, &p, a.rows.iter().map(|r| r.latency_ms).sum())
        })
        .collect();
    let member: BTreeMap<String, Metrics> = (0..members.len())
        .map(|m| {
            let rows: Vec<Row> = aligned.iter().map(|a| a.rows[m].clone()).collect();
            (
                names[m].clone(),
                eval::metrics(&rows, 0, &Calibration::default()),
            )
        })
        .collect();
    Ok(serde_json::json!({
        "mode": "apply",
        "model": model_path.display().to_string(),
        "members": names,
        "aligned_questions": aligned.len(),
        "dropped_questions": dropped,
        "member": member,
        "stacker": eval::metrics(&stacker_rows, 0, &Calibration::default()),
        "latency_note": "meta rows report the sum of member latencies (all members run)",
    }))
}

/// A live member backend: answers a request with raw, uncalibrated option
/// log-probs per question — the same intermediate `eval --rows` stores.
pub trait Member: Send + Sync {
    fn name(&self) -> &str;
    fn raw(&self, req: &Request) -> Result<Vec<RawQuestion>, BackendError>;
}

/// A [`Scorer`] judge as a live member.
pub struct JudgeMember {
    pub name: String,
    pub judge: Judge<Box<dyn Scorer>>,
}

impl Member for JudgeMember {
    fn name(&self) -> &str {
        &self.name
    }
    fn raw(&self, req: &Request) -> Result<Vec<RawQuestion>, BackendError> {
        self.judge.raw(req)
    }
}

/// A whole-request [`FullBackend`] (Laya, AgentJev, hosted Jev) as a live
/// member; answers are converted to the same log-prob intermediate.
pub struct FullMember {
    pub name: String,
    pub backend: Box<dyn FullBackend>,
}

impl Member for FullMember {
    fn name(&self) -> &str {
        &self.name
    }
    fn raw(&self, req: &Request) -> Result<Vec<RawQuestion>, BackendError> {
        let (answers, ms) = self.backend.evaluate(req)?;
        full_eval::raw_from_answers(req, &answers, ms)
    }
}

/// Ask-time stacker: fans a request out to every member (in parallel),
/// rebuilds the trained features, and answers from the meta-model. Satisfies
/// [`FullBackend`], so `jev serve` / `jev ask` / `jev eval` treat it like any
/// other whole-request backend.
pub struct LiveStacker {
    pub stacker: Stacker,
    pub members: Vec<Box<dyn Member>>,
}

impl LiveStacker {
    pub fn new(stacker: Stacker, members: Vec<Box<dyn Member>>) -> Result<Self, String> {
        if members.len() != stacker.members.len() {
            return Err(format!(
                "model expects {} members {:?}, got {}",
                stacker.members.len(),
                stacker.members,
                members.len()
            ));
        }
        for (m, want) in members.iter().zip(&stacker.members) {
            if m.name() != want {
                return Err(format!(
                    "member order mismatch: model position holds `{want}`, got `{}`",
                    m.name()
                ));
            }
        }
        Ok(Self { stacker, members })
    }
}

impl FullBackend for LiveStacker {
    fn evaluate(
        &self,
        req: &Request,
    ) -> Result<(serde_json::Map<String, serde_json::Value>, f64), BackendError> {
        let t0 = std::time::Instant::now();
        // Fan out in parallel; members are Send + Sync.
        let raws: Vec<Vec<RawQuestion>> = std::thread::scope(|s| {
            let handles: Vec<_> = self
                .members
                .iter()
                .map(|m| s.spawn(move || m.raw(req)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(BackendError::Http("member thread panicked".into()))
                    })
                })
                .collect::<Result<_, _>>()
        })?;

        let parsed = parse_questions(&req.questions).map_err(BackendError::Rejected)?;
        if raws.iter().any(|r| r.len() != parsed.len()) {
            return Err(BackendError::Malformed(
                "a member returned a different number of answers".into(),
            ));
        }
        let mut answers = serde_json::Map::new();
        for (qi, (id, _q)) in parsed.iter().enumerate() {
            let r0 = &raws[0][qi];
            for r in &raws {
                if r[qi].id != r0.id || r[qi].kind != r0.kind || r[qi].keys != r0.keys {
                    return Err(BackendError::Malformed(format!(
                        "members disagree on question `{id}` (order or options differ)"
                    )));
                }
            }
            let kind = r0.kind;
            let n = r0.keys.len();
            if n > self.stacker.max_classes {
                return Err(BackendError::Rejected(format!(
                    "question `{id}`: {n} options; this ensemble was fit with max_classes={}",
                    self.stacker.max_classes
                )));
            }
            let probs: Vec<Vec<f64>> = raws
                .iter()
                .enumerate()
                .map(|(m, rs)| softmax(&rs[qi].logprobs, self.stacker.temp(m, kind, n)))
                .collect();
            let p = self
                .stacker
                .predict(&features(&probs, kind, n, self.stacker.max_classes), n);
            let answer = crate::judge::make_answer(kind, &r0.keys, &p, req, id);
            answers.insert(id.clone(), serde_json::to_value(answer).unwrap());
        }
        Ok((answers, t0.elapsed().as_secs_f64() * 1e3))
    }

    fn model_name(&self) -> String {
        format!("ensemble({})", self.stacker.members.join("+"))
    }
}

fn meta_row(a: &Aligned, probs: &[f64], latency_ms: f64) -> Row {
    Row {
        case_index: a.case_index,
        id: String::new(),
        kind: a.kind.clone(),
        n: a.n,
        raw_logprobs: probs.iter().map(|p| p.max(1e-12).ln()).collect(),
        gold: a.gold,
        latency_ms,
        prompt_evaluated: a.rows.iter().map(|r| r.prompt_evaluated).sum(),
        prompt_cached: a.rows.iter().map(|r| r.prompt_cached).sum(),
    }
}

/// Multinomial logistic regression by full-batch gradient descent with
/// momentum; logits beyond each question's `n` are masked out of the softmax.
fn fit_logreg(
    x: &[Vec<f64>],
    gold: &[usize],
    ns: &[usize],
    max_classes: usize,
    iterations: usize,
    lambda: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
    let dim = x[0].len();
    let m = x.len() as f64;
    let mut w = vec![vec![0.0; dim]; max_classes];
    let mut b = vec![0.0; max_classes];
    let mut vw = vec![vec![0.0; dim]; max_classes];
    let mut vb = vec![0.0; max_classes];
    let lr = 0.5;
    let momentum = 0.9;
    let mut loss = 0.0;
    for _ in 0..iterations {
        let mut gw = vec![vec![0.0; dim]; max_classes];
        let mut gb = vec![0.0; max_classes];
        loss = 0.0;
        for ((xi, &g), &n) in x.iter().zip(gold).zip(ns) {
            let n = n.min(max_classes);
            let logits: Vec<f64> = (0..n)
                .map(|c| w[c].iter().zip(xi).map(|(wi, xi)| wi * xi).sum::<f64>() + b[c])
                .collect();
            let p = softmax(&logits, 1.0);
            loss -= p[g].max(1e-12).ln();
            for c in 0..n {
                let d = p[c] - if c == g { 1.0 } else { 0.0 };
                for (wc, &xc) in gw[c].iter_mut().zip(xi) {
                    *wc += d * xc;
                }
                gb[c] += d;
            }
        }
        loss = loss / m + 0.5 * lambda * w.iter().flatten().map(|v| v * v).sum::<f64>();
        for c in 0..max_classes {
            for f in 0..dim {
                gw[c][f] = gw[c][f] / m + lambda * w[c][f];
                vw[c][f] = momentum * vw[c][f] + lr * gw[c][f];
                w[c][f] -= vw[c][f];
            }
            gb[c] /= m;
            vb[c] = momentum * vb[c] + lr * gb[c];
            b[c] -= vb[c];
        }
    }
    (w, b, loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Answer;

    fn row(case: usize, gold: usize, p_yes: f64, lat: f64) -> Row {
        let p = p_yes.clamp(1e-6, 1.0 - 1e-6);
        Row {
            case_index: case,
            id: "q".into(),
            kind: "noul".into(),
            n: 2,
            raw_logprobs: vec![p.ln(), (1.0 - p).ln()],
            gold,
            latency_ms: lat,
            prompt_evaluated: 0,
            prompt_cached: 0,
        }
    }

    /// Member A is confidently right on i%4 in {0,1}, uniform otherwise;
    /// member B is the complement. Each reaches ~75% alone; the stacker
    /// should learn to trust whoever is confident and beat both.
    #[test]
    fn stacker_beats_members_on_synthetic() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..400 {
            let gold = i % 2;
            let pa = if i % 4 <= 1 {
                if gold == 0 {
                    0.9
                } else {
                    0.1
                }
            } else {
                0.5
            };
            let pb = if i % 4 >= 2 {
                if gold == 0 {
                    0.9
                } else {
                    0.1
                }
            } else {
                0.5
            };
            a.push(row(i, gold, pa, 10.0));
            b.push(row(i, gold, pb, 20.0));
        }
        let members = vec![("a".to_string(), a), ("b".to_string(), b)];
        let (report, _stacker) = fit_report(&members, 0.3, 600, 1e-3).unwrap();
        let a_acc = report.member_test["a"].accuracy;
        let b_acc = report.member_test["b"].accuracy;
        let e_acc = report.stacker_test.accuracy;
        assert!(
            e_acc > a_acc.max(b_acc) + 0.1,
            "stacker {e_acc} should beat members ({a_acc}, {b_acc})"
        );
        assert!(e_acc > 0.9, "stacker should be near-perfect, got {e_acc}");
    }

    #[test]
    fn roundtrip_save_load() {
        let s = Stacker {
            members: vec!["x".into()],
            max_classes: 2,
            dim: 6,
            weights: vec![vec![0.0; 6], vec![0.0; 6]],
            bias: vec![0.0, 0.0],
            train_loss: 0.1,
            iterations: 10,
            lambda: 0.001,
            temps: vec![],
        };
        let p = std::env::temp_dir().join("jev_stacker_test.json");
        s.save(&p).unwrap();
        let l = Stacker::load(&p).unwrap();
        assert_eq!(l.members, s.members);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn split_is_deterministic_and_partitioned() {
        for i in 0..1000 {
            assert_eq!(is_test_case(i, 0.3), is_test_case(i, 0.3));
        }
        let tests = (0..1000).filter(|&i| is_test_case(i, 0.3)).count();
        assert!(
            (250..350).contains(&tests),
            "test fraction ~0.3, got {tests}"
        );
    }

    /// A member that always answers the same fixed distribution.
    struct StubMember {
        name: String,
        p_yes: f64,
    }

    impl Member for StubMember {
        fn name(&self) -> &str {
            &self.name
        }
        fn raw(&self, _req: &Request) -> Result<Vec<RawQuestion>, BackendError> {
            let p = self.p_yes.clamp(1e-6, 1.0 - 1e-6);
            Ok(vec![RawQuestion {
                id: "churn".into(),
                kind: "noul",
                keys: vec!["yes".into(), "no".into()],
                logprobs: vec![p.ln(), (1.0 - p).ln()],
                prompt_evaluated: 0,
                prompt_cached: 0,
                latency_ms: 1.0,
            }])
        }
    }

    fn one_noul_request() -> Request {
        let mut questions = serde_json::Map::new();
        questions.insert(
            "churn".into(),
            serde_json::json!({"type": "noul", "instructions": "leaves?"}),
        );
        Request {
            model: None,
            state: serde_json::json!("customer threatens to cancel"),
            questions,
        }
    }

    #[test]
    fn live_stacker_answers_through_fullbackend() {
        // Train a stacker whose member "hot" is right when it says yes.
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        for i in 0..200 {
            let gold = i % 2;
            hot.push(row(i, gold, if gold == 0 { 0.9 } else { 0.1 }, 5.0));
            cold.push(row(i, gold, 0.5, 5.0));
        }
        let members = vec![("hot".to_string(), hot), ("cold".to_string(), cold)];
        let (_report, stacker) = fit_report(&members, 0.3, 400, 1e-3).unwrap();

        let live = LiveStacker::new(
            stacker,
            vec![
                Box::new(StubMember {
                    name: "hot".into(),
                    p_yes: 0.9,
                }),
                Box::new(StubMember {
                    name: "cold".into(),
                    p_yes: 0.5,
                }),
            ],
        )
        .unwrap();
        let req = one_noul_request();
        let (answers, ms) = live.evaluate(&req).unwrap();
        assert!(ms >= 0.0);
        let a: Answer = serde_json::from_value(answers["churn"].clone()).unwrap();
        match a {
            Answer::Noul { noul } => assert!(noul > 0.5, "stacker should side with hot: {noul}"),
            other => panic!("expected noul, got {other:?}"),
        }
        assert_eq!(live.model_name(), "ensemble(hot+cold)");
    }

    #[test]
    fn live_stacker_rejects_member_order_mismatch() {
        let s = Stacker {
            members: vec!["a".into(), "b".into()],
            max_classes: 2,
            dim: 6,
            weights: vec![vec![0.0; 6], vec![0.0; 6]],
            bias: vec![0.0, 0.0],
            train_loss: 0.0,
            iterations: 0,
            lambda: 0.0,
            temps: vec![],
        };
        let err = match LiveStacker::new(
            s,
            vec![
                Box::new(StubMember {
                    name: "b".into(),
                    p_yes: 0.5,
                }),
                Box::new(StubMember {
                    name: "a".into(),
                    p_yes: 0.5,
                }),
            ],
        ) {
            Ok(_) => panic!("expected an order mismatch error"),
            Err(e) => e,
        };
        assert!(err.contains("order mismatch"), "{err}");
    }
}
