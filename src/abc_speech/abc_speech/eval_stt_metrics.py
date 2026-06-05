import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SUCCESS_LABELS = {
    "SUCCESS",
    "CORRECTION_SUCCESS",
}

LOGIC_EXCLUDED_LABELS = {
    "UNKNOWN_VALUE",
    "STT_FAIL_SEVERE",
}

KNOWN_LABELS = {
    "SUCCESS",
    "CORRECTION_SUCCESS",
    "CORRECTION_FAIL",
    "PARSING_FAIL",
    "STT_FAIL_SEVERE",
    "FALSE_POSITIVE",
    "UNKNOWN_VALUE",
}

FIELDNAMES = [
    "case_id",
    "attempt",
    "intent_order",
    "stt_raw",
    "final_order",
    "label",
    "note",
]

DEFAULT_EVAL_PATH = Path(
    "/home/ssu/team_abc_ws/src/abc_speech/abc_speech/stt_eval.csv"
)


def _percent(numerator, denominator):
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.1f}%"


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("case_id", str(line_no))
            row.setdefault("attempt", "1")
            rows.append(row)
    return rows


def read_rows(path):
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    else:
        rows = _read_csv(path)

    normalized = []
    for index, row in enumerate(rows, start=1):
        label = str(row.get("label", "")).strip().upper()
        case_id = str(row.get("case_id", "")).strip() or str(index)
        attempt = str(row.get("attempt", "")).strip() or "1"

        normalized.append({
            **row,
            "case_id": case_id,
            "attempt": attempt,
            "label": label,
        })

    return normalized


def validate_rows(rows):
    errors = []
    for index, row in enumerate(rows, start=1):
        label = row["label"]
        if label not in KNOWN_LABELS:
            errors.append(
                f"row {index}: unknown label '{label}'"
            )

        try:
            int(row["attempt"])
        except ValueError:
            errors.append(
                f"row {index}: attempt must be an integer"
            )

    return errors


def summarize_attempts(rows):
    total = len(rows)
    labels = Counter(row["label"] for row in rows)
    success = sum(labels[label] for label in SUCCESS_LABELS)

    logic_rows = [
        row for row in rows
        if row["label"] not in LOGIC_EXCLUDED_LABELS
    ]
    logic_success = sum(
        1 for row in logic_rows
        if row["label"] in SUCCESS_LABELS
    )

    return {
        "total": total,
        "labels": labels,
        "success": success,
        "logic_total": len(logic_rows),
        "logic_success": logic_success,
    }


def summarize_orders(rows):
    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)

    total_cases = len(cases)
    one_shot_success = 0
    final_success = 0
    success_attempts = []

    for case_rows in cases.values():
        case_rows.sort(key=lambda row: int(row["attempt"]))
        first_success_attempt = None

        for row in case_rows:
            if row["label"] in SUCCESS_LABELS:
                first_success_attempt = int(row["attempt"])
                break

        if first_success_attempt is None:
            continue

        final_success += 1
        success_attempts.append(first_success_attempt)

        if first_success_attempt == 1:
            one_shot_success += 1

    avg_attempt = "N/A"
    if success_attempts:
        avg_attempt = f"{sum(success_attempts) / len(success_attempts):.2f}"

    return {
        "total_cases": total_cases,
        "one_shot_success": one_shot_success,
        "final_success": final_success,
        "final_failure": total_cases - final_success,
        "avg_success_attempt": avg_attempt,
    }


def print_summary(rows):
    attempt_summary = summarize_attempts(rows)
    order_summary = summarize_orders(rows)

    print("=== Attempt metrics ===")
    print(f"Total attempts: {attempt_summary['total']}")

    for label in sorted(KNOWN_LABELS):
        count = attempt_summary["labels"][label]
        print(
            f"{label}: {count} "
            f"({_percent(count, attempt_summary['total'])})"
        )

    print(
        "Overall attempt success rate: "
        f"{attempt_summary['success']} / {attempt_summary['total']} "
        f"= {_percent(attempt_summary['success'], attempt_summary['total'])}"
    )
    print(
        "Parsing/correction success rate: "
        f"{attempt_summary['logic_success']} / "
        f"{attempt_summary['logic_total']} "
        f"= {_percent(attempt_summary['logic_success'], attempt_summary['logic_total'])}"
    )

    print()
    print("=== Order metrics ===")
    print(f"Total test cases: {order_summary['total_cases']}")
    print(
        "1-attempt success rate: "
        f"{order_summary['one_shot_success']} / "
        f"{order_summary['total_cases']} "
        f"= {_percent(order_summary['one_shot_success'], order_summary['total_cases'])}"
    )
    print(
        "Retry-included success rate: "
        f"{order_summary['final_success']} / "
        f"{order_summary['total_cases']} "
        f"= {_percent(order_summary['final_success'], order_summary['total_cases'])}"
    )
    print(f"Final failures: {order_summary['final_failure']}")
    print(f"Average success attempt: {order_summary['avg_success_attempt']}")


def print_template():
    print(",".join(FIELDNAMES))
    print(
        "1,1,펩시 3개,앱 실행해 주세요,,STT_FAIL_SEVERE,"
        "STT raw text had no useful product clue"
    )
    print(
        "1,2,펩시 3개,펩씨 세 개 주세요,펩시 3개,"
        "CORRECTION_SUCCESS,"
        "corrected product name"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize STT/order evaluation metrics."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Evaluation CSV or JSONL path. "
            f"Default: {DEFAULT_EVAL_PATH}"
        ),
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Print a CSV template and exit.",
    )

    args = parser.parse_args()

    if args.template:
        print_template()
        return

    path = Path(args.path) if args.path else DEFAULT_EVAL_PATH

    if not path.exists():
        print(f"Evaluation file not found: {path}")
        print("Create the file with this header:")
        print(",".join(FIELDNAMES))
        raise SystemExit(1)

    rows = read_rows(path)
    errors = validate_rows(rows)

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print_summary(rows)


if __name__ == "__main__":
    main()
