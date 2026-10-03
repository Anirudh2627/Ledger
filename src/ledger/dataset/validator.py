"""Dataset-level validation beyond per-case schema checks.

Schema validation (pydantic) guarantees *shape*; this module guarantees
*dataset health*: unique ids, category coverage, critical-case sanity, field
length limits and signals that a case is unevaluatable in practice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ledger.dataset.schema import KNOWN_CATEGORIES, TestCase


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    case_id: str | None
    message: str

    def __str__(self) -> str:
        where = f"[{self.case_id}] " if self.case_id else ""
        return f"{self.severity.upper()}: {where}{self.message}"


@dataclass
class ValidationReport:
    n_cases: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    category_counts: Counter[str] = field(default_factory=Counter)
    critical_count: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_cases(
    cases: list[TestCase],
    *,
    max_question_chars: int = 20_000,
    max_reference_chars: int = 20_000,
    warn_unknown_category: bool = True,
    min_cases: int = 1,
) -> ValidationReport:
    """Run dataset-level health checks over already-parsed cases."""
    report = ValidationReport(n_cases=len(cases))

    if len(cases) < min_cases:
        report.issues.append(
            ValidationIssue("error", None, f"dataset has fewer than {min_cases} cases")
        )

    ids = Counter(case.id for case in cases)
    for case_id, count in ids.items():
        if count > 1:
            report.issues.append(
                ValidationIssue("error", case_id, f"duplicate id appears {count} times")
            )

    report.category_counts = Counter(case.category for case in cases)
    report.critical_count = sum(1 for case in cases if case.critical)

    if warn_unknown_category:
        for category in sorted(set(report.category_counts) - set(KNOWN_CATEGORIES)):
            report.issues.append(
                ValidationIssue(
                    "warning",
                    None,
                    f"category '{category}' is not in the known taxonomy {KNOWN_CATEGORIES}",
                )
            )

    for case in cases:
        if len(case.question) > max_question_chars:
            report.issues.append(
                ValidationIssue("error", case.id, f"question exceeds {max_question_chars} chars")
            )
        if case.reference_answer and len(case.reference_answer) > max_reference_chars:
            report.issues.append(
                ValidationIssue(
                    "error", case.id, f"reference_answer exceeds {max_reference_chars} chars"
                )
            )
        if case.critical and not (case.reference_answer or case.expected_behavior):
            report.issues.append(
                ValidationIssue(
                    "error",
                    case.id,
                    "critical case must define reference_answer and/or expected_behavior",
                )
            )
        if case.must_contain and not case.reference_answer and not case.expected_behavior:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    case.id,
                    "must_contain without reference_answer/expected_behavior may over-constrain",
                )
            )
        if case.category == "retrieval_correctness" and not case.expected_sources:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    case.id,
                    "retrieval_correctness case without expected_sources "
                    "cannot score retrieval_hit",
                )
            )
        output_format = case.metadata.get("output_format")
        if output_format is not None and output_format not in {"json", "csv", "markdown_list"}:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    case.id,
                    f"unknown metadata.output_format '{output_format}' (metrics will ignore it)",
                )
            )

    if report.critical_count == 0 and cases:
        report.issues.append(ValidationIssue("warning", None, "dataset contains no critical cases"))

    return report
