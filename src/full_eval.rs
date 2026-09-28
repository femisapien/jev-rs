//! Eval path for whole-request backends (Laya, AgentJev): one HTTP call per
//! case answers every question; probabilities are converted to log-probs so
//! [`metrics`] / [`fit`] score them identically to a [`Scorer`] run.

use serde_json::Value;

use crate::backend::{BackendError, FullBackend};
use crate::eval::{self, Case, Row};
use crate::judge::RawQuestion;
use crate::protocol::{parse_questions, Question, Request};
use crate::score::Calibration;

/// (kind, keys) for one parsed question, in request order.
pub(crate) fn kind_keys(q: &Question) -> (&'static str, Vec<String>) {
    match q {
        Question::Noul { .. } => ("noul", vec!["yes".to_string(), "no".to_string()]),
        Question::Choice { criteria, .. } => {
            ("choice", criteria.keys().cloned().collect::<Vec<_>>())
        }
        Question::Score { criteria, .. } => (
            "score",
            (0..criteria.len()).map(|i| i.to_string()).collect(),
        ),
    }
}

/// Convert a whole-request `answers` map into per-question raw log-probs —
/// the same intermediate the Scorer path produces, so live ensemble members
/// of both kinds feed identical features to the stacker.
pub fn raw_from_answers(
    req: &Request,
    answers: &serde_json::Map<String, Value>,
    latency_ms: f64,
) -> Result<Vec<RawQuestion>, BackendError> {
    let parsed = parse_questions(&req.questions).map_err(BackendError::Rejected)?;
    let mut out = Vec::with_capacity(parsed.len());
    for (id, q) in &parsed {
        let (kind, keys) = kind_keys(q);
        let logprobs = answer_logprobs(kind, &keys, answers.get(id).unwrap_or(&Value::Null))?;
        out.push(RawQuestion {
            id: id.clone(),
            kind,
            keys,
            logprobs,
            prompt_evaluated: 0,
            prompt_cached: 0,
            latency_ms,
        });
    }
    Ok(out)
}

/// Run cases through a full-request backend. Latency recorded on each row is
/// the whole-case wall time (how long a Jev-style multi-question call took).
pub fn run(backend: &dyn FullBackend, cases: &[Case]) -> Result<(Vec<Row>, usize), BackendError> {
    let mut rows = Vec::new();
    let mut failed = 0usize;
    for (ci, c) in cases.iter().enumerate() {
        let req = Request {
            model: c.model.clone(),
            state: c.state.clone(),
            questions: c.questions.clone(),
        };
        let (answers, ms) = match backend.evaluate(&req) {
            Ok(x) => x,
            Err(BackendError::Rejected(m)) => return Err(BackendError::Rejected(m)),
            Err(e) => {
                eprintln!("case {ci}: {e} (counted as failed)");
                failed += 1;
                continue;
            }
        };
        let parsed = parse_questions(&req.questions).map_err(BackendError::Rejected)?;
        let mut case_rows = Vec::with_capacity(parsed.len());
        let mut case_failed = false;
        for (id, q) in &parsed {
            let (kind, keys) = kind_keys(q);
            if let Some(g) = c.gold.get(id) {
                let gold = eval::parse_gold(g, &keys).map_err(|m| {
                    BackendError::Rejected(format!("case {ci} question `{id}`: {m}"))
                })?;
                // A missing or malformed answer is the backend's fault, not the
                // case file's: count the case as failed instead of aborting the
                // whole run (the Scorer path treats backend errors the same way).
                let logprobs =
                    match answer_logprobs(kind, &keys, answers.get(id).unwrap_or(&Value::Null)) {
                        Ok(lp) => lp,
                        Err(e) => {
                            eprintln!("case {ci} question `{id}`: {e} (counted as failed)");
                            case_failed = true;
                            break;
                        }
                    };
                case_rows.push(Row {
                    case_index: ci,
                    id: id.clone(),
                    kind: kind.to_string(),
                    n: keys.len(),
                    raw_logprobs: logprobs,
                    gold,
                    latency_ms: ms,
                    prompt_evaluated: 0,
                    prompt_cached: 0,
                });
            }
        }
        if case_failed {
            // A failed case contributes no rows (same contract as the Scorer
            // path): `case_rows` is dropped rather than half-reported.
            failed += 1;
        } else {
            rows.extend(case_rows);
        }
    }
    Ok((rows, failed))
}

/// Convert a Jev answer object into ln(prob) over `keys` (same order the
/// Scorer path uses). Temperature stays 1 here; `metrics`/`fit` apply it.
fn answer_logprobs(kind: &str, keys: &[String], ans: &Value) -> Result<Vec<f64>, BackendError> {
    let bad = |m: &str| BackendError::Malformed(m.to_string());
    match kind {
        "noul" => {
            let p = ans
                .get("noul")
                .and_then(Value::as_f64)
                .ok_or_else(|| bad("noul answer missing `noul`"))?;
            let p = p.clamp(1e-12, 1.0 - 1e-12);
            Ok(vec![p.ln(), (1.0 - p).ln()])
        }
        "choice" => {
            let probs = ans
                .get("probabilities")
                .and_then(Value::as_object)
                .ok_or_else(|| bad("choice answer missing probabilities"))?;
            keys.iter()
                .map(|k| {
                    let p = probs
                        .get(k)
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        .clamp(1e-12, 1.0);
                    Ok(p.ln())
                })
                .collect()
        }
        "score" => {
            let probs = ans
                .get("probabilities")
                .and_then(Value::as_object)
                .ok_or_else(|| bad("score answer missing probabilities"))?;
            keys.iter()
                .map(|k| {
                    let p = probs
                        .get(k)
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        .clamp(1e-12, 1.0);
                    Ok(p.ln())
                })
                .collect()
        }
        other => Err(BackendError::Malformed(format!("unknown kind `{other}`"))),
    }
}

/// Convenience: metrics with the default (identity) calibration, then the
/// same after fitting temperatures on this run — both matter for comparing
/// against Scorer backends that report raw and calibrated views.
pub fn metrics_pair(rows: &[Row], failed: usize) -> (eval::Metrics, eval::Metrics, Calibration) {
    let raw = eval::metrics(rows, failed, &Calibration::default());
    let cal = eval::fit(rows);
    let fitted = eval::metrics(rows, failed, &cal);
    (raw, fitted, cal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendError;
    use serde_json::json;

    /// Answers only the first question of every request; omits the rest.
    struct HalfAnswering;

    impl FullBackend for HalfAnswering {
        fn evaluate(
            &self,
            req: &Request,
        ) -> Result<(serde_json::Map<String, Value>, f64), BackendError> {
            let mut answers = serde_json::Map::new();
            if let Some((id, _)) = req.questions.iter().next() {
                answers.insert(id.clone(), json!({"type": "noul", "noul": 0.7}));
            }
            Ok((answers, 3.0))
        }
        fn model_name(&self) -> String {
            "half".into()
        }
    }

    fn case() -> Case {
        let mut questions = serde_json::Map::new();
        questions.insert("a".into(), json!({"type": "noul", "instructions": "one"}));
        questions.insert("b".into(), json!({"type": "noul", "instructions": "two"}));
        let mut gold = serde_json::Map::new();
        gold.insert("a".into(), json!("yes"));
        gold.insert("b".into(), json!("yes"));
        Case {
            model: None,
            state: json!("s"),
            questions,
            gold,
        }
    }

    #[test]
    fn malformed_answer_counts_failed_case_not_run_abort() {
        // Regression: one unanswered question used to abort the whole eval via
        // `?` on answer_logprobs; now the case is counted as failed and the run
        // continues (a failed case contributes no rows, per `eval::run`).
        let (rows, failed) = run(&HalfAnswering, &[case()]).unwrap();
        assert_eq!(failed, 1);
        assert!(rows.is_empty(), "failed case contributes no rows");

        // A second, fully-answered case still runs: the first failure did not
        // abort the eval.
        let mut c2 = case();
        c2.questions.remove("b");
        c2.gold.remove("b");
        let (rows, failed) = run(&HalfAnswering, &[case(), c2]).unwrap();
        assert_eq!(failed, 1);
        assert_eq!(rows.len(), 1, "second case contributes its row");
    }

    #[test]
    fn out_of_range_gold_is_rejected_not_panicked() {
        let mut c = case();
        c.gold.insert("a".into(), json!(9));
        let err = run(&HalfAnswering, &[c]).unwrap_err();
        assert!(matches!(err, BackendError::Rejected(_)), "{err:?}");
    }
}
