"""Golden dataset handling: schema, loading, validation."""

from ledger.dataset.loader import DatasetError, load_dataset
from ledger.dataset.schema import KNOWN_CATEGORIES, RUBRIC_DIMENSIONS, TestCase
from ledger.dataset.validator import ValidationIssue, ValidationReport, validate_cases

__all__ = [
    "KNOWN_CATEGORIES",
    "RUBRIC_DIMENSIONS",
    "DatasetError",
    "TestCase",
    "ValidationIssue",
    "ValidationReport",
    "load_dataset",
    "validate_cases",
]
