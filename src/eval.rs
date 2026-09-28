//! Labelled evaluation: accuracy, Brier, ECE, latency, and temperature fitting.
//!
//! Case file: JSON Lines, one object per line:
//! `{"state": ..., "questions": {...}, "gold": {"<id>": "<key or level index>"}}`

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::backend::{BackendError, Scorer};
use crate::judge::{Judge, RawQuestion};
use crate::protocol::Request;
use crate::score::{argmax, confidence, softmax, Calibration};

#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    pub questions: Map<String, Value>,
    pub gold: Map<String, Value>,
}

pub fn load_cases(path: &Path) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .enumerate()
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect()
}

/// One scored question with its gold index, kept for fitting/reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub case_index: usize,
    pub id: String,
    pub kind: String,
    pub n: usize,
    pub raw_logprobs: Vec<f64>,
    pub gold: usize,
    pub latency_ms: f64,
    pub prompt_evaluated: u64,
    pub prompt_cached: u64,
}

/// Map a case's `gold` value onto an option index in `keys`. Accepts the key
/// itself, the index as a number or numeric string, booleans, and the strings
/// `"true"`/`"false"` (noul). Returns an error when gold is not one of the
/// options — an out-of-range index would later panic in `Calibration::fit` and
/// `fit_logreg`, so it is rejected here instead.
pub fn parse_gold(g: &Value, keys: &[String]) -> Result<usize, String> {
    let idx = match g {
        Value::Number(n) => n.as_u64().map(|x| x as usize),
        Value::String(s) => keys
            .iter()
            .position(|k| k == s)
            .or_else(|| s.parse().ok())
            .or(match s.as_str() {
                "true" => Some(0),
                "false" => Some(1),
                _ => None,
            }),
        Value::Bool(b) => Some(if *b { 0 } else { 1 }),
        _ => None,
    };
    match idx {
        Some(i) if i < keys.len() => Ok(i),
        _ => Err(format!("gold {g} is not an option")),
    }
}

/// Scored rows plus the number of cases whose backend call failed. A failed
/// case contributes no rows; it is reported next to accuracy, as the public
/// Jev benchmarks do. A rejected request (malformed case) still aborts.
pub fn run<S: Scorer>(judge: &Judge<S>, cases: &[Case]) -> Result<(Vec<Row>, usize), BackendError> {
    let mut rows = Vec::new();
    let mut failed = 0usize;
    for (ci, c) in cases.iter().enumerate() {
        let req = Request {
            model: c.model.clone(),
            state: c.state.clone(),
            questions: c.questions.clone(),
        };
        let raws: Vec<RawQuestion> = match judge.raw(&req) {
            Ok(r) => r,
            Err(BackendError::Rejected(m)) => return Err(BackendError::Rejected(m)),
            Err(e) => {
                eprintln!("case {ci}: {e} (counted as failed)");
                failed += 1;
                continue;
            }
        };
        for r in raws {
            let Some(g) = c.gold.get(&r.id) else { continue };
            let gold = parse_gold(g, &r.keys).map_err(|m| {
                BackendError::Rejected(format!("case {ci} question `{}`: {m}", r.id))
            })?;
            rows.push(Row {
                case_index: ci,
                id: r.id.clone(),
                kind: r.kind.to_string(),
                n: r.keys.len(),
                raw_logprobs: r.logprobs.clone(),
                gold,
                latency_ms: r.latency_ms,
                prompt_evaluated: r.prompt_evaluated,
                prompt_cached: r.prompt_cached,
            });
        }
    }
    Ok((rows, failed))
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub questions: usize,
    /// Cases whose backend call failed (no rows; not counted in accuracy).
    pub failed_cases: usize,
    pub accuracy: f64,
    pub brier: f64,
    /// Top-label expected calibration error, 10 equal-width bins.
    pub ece: f64,
    pub mean_confidence: f64,
    /// Coverage at <=5% empirical error when gating on confidence.
    pub coverage_at_5pct_error: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub prompt_tokens_evaluated: u64,
    pub prompt_tokens_cached: u64,
    pub by_kind: BTreeMap<String, f64>,
}

pub fn metrics(rows: &[Row], failed_cases: usize, cal: &Calibration) -> Metrics {
    let mut correct = 0usize;
    let mut brier = 0.0;
    let mut conf_sum = 0.0;
    let mut bins = vec![(0usize, 0usize, 0.0f64); 10]; // (n, correct, conf_sum)
    let mut gated: Vec<(f64, bool)> = Vec::new();
    let mut kind_tot: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut lat: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for r in rows {
        let t = cal.temperature(&r.kind, r.n);
        let p = softmax(&r.raw_logprobs, t);
        let pred = argmax(&p);
        let ok = pred == r.gold;
        correct += ok as usize;
        let pmax = p[pred];
        conf_sum += confidence(&p);
        brier += p
            .iter()
            .enumerate()
            .map(|(i, &pi)| {
                let y = if i == r.gold { 1.0 } else { 0.0 };
                (pi - y).powi(2)
            })
            .sum::<f64>();
        let b = ((pmax * 10.0).floor() as usize).min(9);
        bins[b].0 += 1;
        bins[b].1 += ok as usize;
        bins[b].2 += pmax;
        gated.push((confidence(&p), ok));
        let e = kind_tot.entry(r.kind.clone()).or_default();
        e.0 += 1;
        e.1 += ok as usize;
    }
    let n = rows.len().max(1) as f64;
    let ece = bins
        .iter()
        .filter(|(c, _, _)| *c > 0)
        .map(|(c, k, s)| (*c as f64 / n) * ((*k as f64 / *c as f64) - (s / *c as f64)).abs())
        .sum();
    // coverage: sort by confidence desc, take the longest prefix with error <= 5%
    gated.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut best = 0usize;
    let mut wrong = 0usize;
    for (i, (_, ok)) in gated.iter().enumerate() {
        wrong += (!ok) as usize;
        if wrong as f64 / (i + 1) as f64 <= 0.05 {
            best = i + 1;
        }
    }
    let pct = |q: f64| -> f64 {
        if lat.is_empty() {
            0.0
        } else {
            lat[((lat.len() - 1) as f64 * q).round() as usize]
        }
    };
    Metrics {
        questions: rows.len(),
        failed_cases,
        accuracy: correct as f64 / n,
        brier: brier / n,
        ece,
        mean_confidence: conf_sum / n,
        coverage_at_5pct_error: best as f64 / n,
        latency_p50_ms: pct(0.5),
        latency_p95_ms: pct(0.95),
        prompt_tokens_evaluated: rows.iter().map(|r| r.prompt_evaluated).sum(),
        prompt_tokens_cached: rows.iter().map(|r| r.prompt_cached).sum(),
        by_kind: kind_tot
            .into_iter()
            .map(|(k, (t, c))| (k, c as f64 / t.max(1) as f64))
            .collect(),
    }
}

pub fn fit(rows: &[Row]) -> Calibration {
    let samples: Vec<(String, Vec<f64>, usize)> = rows
        .iter()
        .map(|r| {
            (
                Calibration::bucket(&r.kind, r.n),
                r.raw_logprobs.clone(),
                r.gold,
            )
        })
        .collect();
    Calibration::fit(&samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn keys() -> Vec<String> {
        vec!["yes".into(), "no".into()]
    }

    #[test]
    fn parse_gold_accepts_documented_forms() {
        assert_eq!(parse_gold(&json!("yes"), &keys()), Ok(0));
        assert_eq!(parse_gold(&json!("1"), &keys()), Ok(1));
        assert_eq!(parse_gold(&json!(0), &keys()), Ok(0));
        assert_eq!(parse_gold(&json!(true), &keys()), Ok(0));
        assert_eq!(parse_gold(&json!("true"), &keys()), Ok(0));
        assert_eq!(parse_gold(&json!("false"), &keys()), Ok(1));
    }

    #[test]
    fn parse_gold_rejects_out_of_range() {
        // Regression: this used to build a row whose gold exceeded n and later
        // panicked in Calibration::fit (index out of bounds).
        assert!(parse_gold(&json!(9), &keys()).is_err());
        assert!(parse_gold(&json!("9"), &keys()).is_err());
        assert!(parse_gold(&json!(-1), &keys()).is_err());
        assert!(parse_gold(&json!("maybe"), &keys()).is_err());
        assert!(parse_gold(&json!(null), &keys()).is_err());
    }

    #[test]
    fn fit_on_empty_rows_is_not_a_panic() {
        let m = metrics(&[], 3, &Calibration::default());
        assert_eq!(m.questions, 0);
        assert_eq!(m.failed_cases, 3);
        let _ = fit(&[]);
    }
}
