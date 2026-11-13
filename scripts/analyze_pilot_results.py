"""Offline statistical analysis of the cached 96-question pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_store import load_cache  # noqa: E402
from scripts.evaluate_split_cache import (  # noqa: E402
    CACHE_PATH,
    DATASET_PATH,
    MANIFEST_PATH,
    RESULTS_PATH as EVALUATION_PATH,
    TIERS,
    _latency_seconds,
    _load_pilot,
    evaluate_pilot,
)

RESULTS_PATH = PROJECT_ROOT / "results" / "pilot_statistical_analysis.json"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "PILOT_STATISTICAL_ANALYSIS.md"
TASK_POPULATION = {"math": 100, "mcq": 100, "extract": 51, "reformat": 33, "classify": 16}
Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> dict[str, float] | None:
    """Return a two-sided Wilson score interval, or None for no observations."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials")
    if trials == 0:
        return None
    proportion = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (proportion + z2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / trials + z2 / (4 * trials * trials)
    ) / denominator
    return {"lower": max(0.0, center - half_width), "upper": min(1.0, center + half_width)}


def exact_mcnemar(left_only: int, right_only: int) -> dict[str, int | float]:
    """Two-sided exact McNemar test from the discordant paired outcomes."""
    if left_only < 0 or right_only < 0:
        raise ValueError("Discordant counts cannot be negative")
    discordant = left_only + right_only
    tail = min(left_only, right_only)
    p_value = 1.0 if discordant == 0 else min(
        1.0,
        2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant),
    )
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p_value": p_value,
    }


def weighted_task_accuracy(
    task_results: dict[str, dict[str, Any]], population: dict[str, int] = TASK_POPULATION
) -> dict[str, Any]:
    """Estimate accuracy by reweighting pilot task accuracies to a stated mix."""
    total_population = sum(population.values())
    contributions = {}
    weighted_correct = 0.0
    for task, size in population.items():
        result = task_results.get(task)
        accuracy = result.get("accuracy") if result else None
        if accuracy is None:
            contributions[task] = {"population_questions": size, "pilot_accuracy": None, "weighted_correct_estimate": None}
            continue
        estimate = size * accuracy
        weighted_correct += estimate
        contributions[task] = {
            "population_questions": size,
            "pilot_accuracy": accuracy,
            "weighted_correct_estimate": estimate,
        }
    return {
        "label": "exploratory task-proportion-weighted estimate; not measured full-test accuracy",
        "population_task_counts": population,
        "population_total": total_population,
        "estimated_correct": weighted_correct,
        "estimated_accuracy": weighted_correct / total_population,
        "task_contributions": contributions,
    }


def _latency_stats(values: list[float]) -> dict[str, Any]:
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else None
    return {
        "n": len(values),
        "mean_seconds": statistics.mean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "q1_seconds": quartiles[0] if quartiles else (values[0] if values else None),
        "q3_seconds": quartiles[2] if quartiles else (values[0] if values else None),
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
    }


def analyze_pilot(
    manifest_path: Path = MANIFEST_PATH,
    dataset_path: Path = DATASET_PATH,
    cache_path: Path = CACHE_PATH,
    evaluation_path: Path = EVALUATION_PATH,
) -> dict[str, Any]:
    """Recompute pilot correctness from raw cached answers and analyze it."""
    from checker import check_answer

    manifest, questions = _load_pilot(manifest_path, dataset_path)
    cache = load_cache(cache_path, "test")
    saved_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    fresh_evaluation = evaluate_pilot(manifest_path, dataset_path, cache_path)
    if saved_evaluation != fresh_evaluation:
        raise ValueError("Saved pilot evaluation does not match current manifest, data, and cache")

    outcomes: dict[str, dict[str, bool | None]] = {tier: {} for tier in TIERS}
    latencies: dict[str, dict[str, float]] = {tier: {} for tier in TIERS}
    for question in questions:
        item_id = question["id"]
        entry = cache["entries"].get(item_id)
        if not isinstance(entry, dict) or entry.get("question") != question["question"]:
            raise ValueError(f"Cache entry missing or mismatched for {item_id}")
        answer_records = entry.get("answers")
        if not isinstance(answer_records, dict):
            raise ValueError(f"Cache answers malformed for {item_id}")
        for tier in TIERS:
            record = answer_records.get(tier)
            if not isinstance(record, dict) or not isinstance(record.get("answer"), str):
                outcomes[tier][item_id] = None
                continue
            outcomes[tier][item_id] = bool(
                check_answer(record["answer"], question["answer"], question["checker"])
            )
            latency = _latency_seconds(record)
            if latency is not None:
                latencies[tier][item_id] = latency

    task_by_id = {row["id"]: row["task_type"] for row in questions}
    overall: dict[str, Any] = {}
    per_task: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        included = [value for value in outcomes[tier].values() if value is not None]
        successes = sum(value is True for value in included)
        overall[tier] = {
            "correct": successes,
            "evaluated": len(included),
            "missing_or_invalid": len(questions) - len(included),
            "accuracy": successes / len(included) if included else None,
            "confidence_interval_95_wilson": wilson_interval(successes, len(included)),
        }
        per_task[tier] = {}
        for task in sorted(set(task_by_id.values())):
            values = [outcomes[tier][item_id] for item_id, current_task in task_by_id.items() if current_task == task]
            valid = [value for value in values if value is not None]
            correct = sum(value is True for value in valid)
            per_task[tier][task] = {
                "correct": correct,
                "evaluated": len(valid),
                "accuracy": correct / len(valid) if valid else None,
                "confidence_interval_95_wilson": wilson_interval(correct, len(valid)),
                "small_sample_note": f"n={len(valid)}; pilot task sample is small",
            }

    mcnemar = {}
    for left, right in (("local", "cheap"), ("local", "strong"), ("cheap", "strong")):
        common = [item_id for item_id in task_by_id if outcomes[left][item_id] is not None and outcomes[right][item_id] is not None]
        left_only = sum(outcomes[left][item_id] is True and outcomes[right][item_id] is False for item_id in common)
        right_only = sum(outcomes[left][item_id] is False and outcomes[right][item_id] is True for item_id in common)
        mcnemar[f"{left}_vs_{right}"] = {
            "paired_n": len(common),
            **exact_mcnemar(left_only, right_only),
        }

    ids = {
        "cheap_correct_strong_incorrect": [],
        "strong_correct_cheap_incorrect": [],
        "local_correct_both_hosted_incorrect": [],
        "all_three_not_unanimous": [],
    }
    for item_id in task_by_id:
        local, cheap, strong = (outcomes[tier][item_id] for tier in TIERS)
        if cheap is True and strong is False:
            ids["cheap_correct_strong_incorrect"].append(item_id)
        if strong is True and cheap is False:
            ids["strong_correct_cheap_incorrect"].append(item_id)
        if local is True and cheap is False and strong is False:
            ids["local_correct_both_hosted_incorrect"].append(item_id)
        if None not in (local, cheap, strong) and len({local, cheap, strong}) > 1:
            ids["all_three_not_unanimous"].append(item_id)

    paired_latency_ids = sorted(set(latencies["cheap"]) & set(latencies["strong"]))
    differences = [latencies["cheap"][item_id] - latencies["strong"][item_id] for item_id in paired_latency_ids]
    latency_report = {
        "method": "descriptive only; no latency significance test performed",
        "tiers": {tier: _latency_stats(list(latencies[tier].values())) for tier in ("cheap", "strong")},
        "missing_valid_latency_records": {tier: len(questions) - len(latencies[tier]) for tier in ("cheap", "strong")},
        "paired_cheap_minus_strong": {
            "paired_n": len(paired_latency_ids),
            "cheap_minus_strong_seconds": _latency_stats(differences),
            "cheap_faster_count": sum(value < 0 for value in differences),
            "strong_faster_count": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "question_ids": paired_latency_ids,
        },
    }
    weighted = {
        tier: weighted_task_accuracy(fresh_evaluation["tier_results"][tier]["per_task"])
        for tier in TIERS
    }
    return {
        "schema_version": 1,
        "analysis_type": "offline descriptive and exact paired analysis",
        "source": {
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)) if manifest_path.is_relative_to(PROJECT_ROOT) else str(manifest_path),
            "source_test_sha256": manifest["source_test_sha256"],
            "pilot_questions": len(questions),
            "evaluation_summary": "docs/PILOT_CACHE_EVALUATION.md",
        },
        "overall_accuracy": overall,
        "per_task_accuracy": per_task,
        "exact_mcnemar_tests": mcnemar,
        "correctness_pattern_question_ids": ids,
        "correctness_pattern_definitions": {
            "all_three_not_unanimous": "At least one tier's correctness outcome differs from another; with binary outcomes, this means the three are not all identical.",
        },
        "latency": latency_report,
        "exploratory_weighted_estimates": weighted,
        "research_interpretation": {
            "supports": "On this fixed 96-question pilot, cached hosted answers were more often checker-correct than local answers; cheap and strong were close in observed accuracy.",
            "uncertain": "Small task strata and one pilot sample limit precision and generalization. Latency summaries are descriptive only. The task-weighted estimate is exploratory, not measured full-test accuracy.",
            "classifier": "The existing question-only classifier has not demonstrated robust, calibrated, prospective routing performance across the observed distribution mismatch; it is not suitable for live routing.",
            "larger_experiment": "A larger preregistered, task- and source-stratified paired evaluation, with provenance-aligned training and validation and a routing policy fixed before evaluation, would provide stronger evidence.",
        },
        "dollar_cost": {"reported": False, "reason": "No pricing was inferred or applied."},
    }


def _pct_interval(interval: dict[str, float] | None) -> str:
    if interval is None:
        return "unavailable"
    return f"{interval['lower']:.1%}–{interval['upper']:.1%}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pilot Statistical Analysis", "",
        "Offline analysis of the completed 96-question pilot. Correctness was recomputed with each question's dataset checker and the raw cached answer.",
        "", f"- Pilot n: {report['source']['pilot_questions']}",
        f"- Frozen test SHA-256: `{report['source']['source_test_sha256']}`",
        "- Provider calls: none; dollar cost: not calculated.", "",
        "## Overall accuracy and 95% Wilson intervals", "",
        "| Tier | Correct / evaluated | Accuracy | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for tier, data in report["overall_accuracy"].items():
        accuracy = "—" if data["accuracy"] is None else f"{data['accuracy']:.1%}"
        lines.append(f"| {tier} | {data['correct']}/{data['evaluated']} | {accuracy} | {_pct_interval(data['confidence_interval_95_wilson'])} |")
    lines.extend(["", "## Per-task accuracy", "", "Intervals use the Wilson score method. Task samples are small (n=16 or 20); interpret them cautiously.", "", "| Tier | Task | Correct / n | Accuracy | 95% CI |", "|---|---|---:|---:|---:|"])
    for tier, tasks in report["per_task_accuracy"].items():
        for task, data in tasks.items():
            accuracy = "—" if data["accuracy"] is None else f"{data['accuracy']:.1%}"
            lines.append(f"| {tier} | {task} | {data['correct']}/{data['evaluated']} | {accuracy} | {_pct_interval(data['confidence_interval_95_wilson'])} |")
    lines.extend(["", "## Exact two-sided McNemar tests", "", "These exact conditional binomial tests use paired correctness discordances. P-values are nominal and are not adjusted for the three comparisons.", "", "| Pair | Paired n | Left only correct | Right only correct | Exact p |", "|---|---:|---:|---:|---:|"])
    for name, data in report["exact_mcnemar_tests"].items():
        lines.append(f"| {name.replace('_vs_', ' vs ')} | {data['paired_n']} | {data['left_only_correct']} | {data['right_only_correct']} | {data['two_sided_exact_p_value']:.6g} |")
    lines.extend(["", "## Correctness pattern IDs", ""])
    lines.append("The three-model pattern lists questions where at least one tier's binary correctness outcome differs; it does not imply all three outcomes are pairwise distinct.")
    for label, question_ids in report["correctness_pattern_question_ids"].items():
        lines.append(f"- **{label}:** {', '.join(question_ids) if question_ids else 'none'}")
    lines.extend(["", "## Latency", "", "Descriptive only; no latency significance test was performed. Values are included only when cache latency metadata is numeric and valid.", "", "| Tier | n | Mean (s) | Median (s) | Q1–Q3 (s) | Min (s) | Max (s) | Missing |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for tier, data in report["latency"]["tiers"].items():
        missing = report["latency"]["missing_valid_latency_records"][tier]
        fields = [data[key] for key in ("mean_seconds", "median_seconds", "min_seconds", "max_seconds")]
        formatted = ["—" if value is None else f"{value:.4f}" for value in fields]
        q1 = "—" if data["q1_seconds"] is None else f"{data['q1_seconds']:.4f}"
        q3 = "—" if data["q3_seconds"] is None else f"{data['q3_seconds']:.4f}"
        lines.append(f"| {tier} | {data['n']} | {formatted[0]} | {formatted[1]} | {q1}–{q3} | {formatted[2]} | {formatted[3]} | {missing} |")
    paired = report["latency"]["paired_cheap_minus_strong"]
    diff = paired["cheap_minus_strong_seconds"]
    lines.extend(["", f"Paired cheap-minus-strong latencies: n={paired['paired_n']}; mean {diff['mean_seconds']:.4f}s, median {diff['median_seconds']:.4f}s; cheap faster on {paired['cheap_faster_count']}, strong faster on {paired['strong_faster_count']}, ties {paired['ties']}. Negative differences mean cheap was faster for the same question. No significance test was performed.", ""])
    lines.extend(["## Exploratory task-proportion-weighted estimates", "", "These reweight pilot task accuracies to the supplied full-test task counts. They are exploratory estimates, not measured full-test accuracy.", "", "| Tier | Estimated accuracy | Estimated correct / 300 |", "|---|---:|---:|"])
    for tier, data in report["exploratory_weighted_estimates"].items():
        lines.append(f"| {tier} | {data['estimated_accuracy']:.1%} | {data['estimated_correct']:.2f}/300 |")
    lines.extend(["", "## Research interpretation", ""])
    for key, description in report["research_interpretation"].items():
        lines.append(f"- **{key.replace('_', ' ').capitalize()}:** {description}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--evaluation", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    report = analyze_pilot(args.manifest, args.dataset, args.cache, args.evaluation)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.results} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
