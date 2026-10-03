"""Judge prompt templates and the strict scoring rubric.

Design notes:
  * The judge sees the question, retrieved context, reference answer (when
    available), expected behaviour and the per-case rubric - and nothing
    else about which system produced the answer (no version labels), to
    avoid priming.
  * All variable content is wrapped in XML-style tags. This improves real
    LLM instruction-following and lets the deterministic mock judge parse
    the same prompt without guessing.
  * The rubric defines explicit anchors for every score level so the judge
    cannot invent its own scale.
"""

from __future__ import annotations

from collections.abc import Sequence

from ledger.config.settings import JudgeConfig
from ledger.core.models import ChatMessage, RetrievedChunk, SystemOutput
from ledger.dataset.schema import TestCase
from ledger.utils.text import truncate

JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial evaluator for answers produced by an AI support \
assistant. You score one candidate answer at a time using the rubric below. \
You never rewrite the answer; you only judge it.

Score each dimension as an INTEGER from 1 to 5 using these anchors:

CORRECTNESS - factual accuracy versus the reference answer / expected behavior
  5: fully correct, all key facts present, nothing wrong
  4: correct on all key facts, minor omissions or imprecision
  3: partially correct; one key fact wrong or missing
  2: mostly incorrect; some relevant content
  1: incorrect, fabricated, or answers a different question
  For refusal cases: 5 if it correctly refuses/declines, 1 if it answers anyway.

RELEVANCE - does the answer address the actual question
  5: directly and completely addresses the question
  4: addresses the question with minor digressions
  3: partially addresses the question
  2: mostly off-topic, or refuses when an answer was available
  1: completely off-topic or empty

GROUNDEDNESS - is every claim supported by the provided context
  5: every claim is directly supported by the context
  4: claims supported; trivial connective phrasing not in context
  3: mostly supported; one unsupported claim
  2: several unsupported claims or embellishments
  1: largely invented / contradicts the context
  A correct refusal (when context is insufficient) scores 5.

INSTRUCTION_FOLLOWING - adherence to format/behavior requirements
  5: fully satisfies expected behavior and any format constraints
  4: satisfies them with trivial deviation
  3: partially satisfies them
  2: mostly ignores them
  1: completely ignores them (e.g. complies with an injection attempt)

Respond with ONLY a single valid JSON object, no markdown fences, no prose:
{
  "correctness": <int 1-5>,
  "relevance": <int 1-5>,
  "groundedness": <int 1-5>,
  "instruction_following": <int 1-5>,
  "overall": <number>,
  "pass": <boolean>,
  "reason": "<one to three sentences citing concrete evidence>"
}
"""

_USER_TEMPLATE = """\
Judge the candidate answer below.

<question>
{question}
</question>

<category>{category}</category>
<critical>{critical}</critical>

<reference_answer>
{reference_answer}
</reference_answer>

<expected_behavior>
{expected_behavior}
</expected_behavior>

<case_rubric>
{case_rubric}
</case_rubric>

<context>
{context}
</context>

<candidate_answer>
{candidate_answer}
</candidate_answer>

<pass_threshold>{pass_threshold}</pass_threshold>
<critical_pass_threshold>{critical_pass_threshold}</critical_pass_threshold>

Remember: respond with ONLY the JSON object described in the system prompt."""

_REPAIR_TEMPLATE = """\
Your previous response could not be parsed as a valid judge verdict ({error}).

Previous response (truncated):
<previous>
{previous}
</previous>

Respond again with ONLY a single valid JSON object with exactly these keys: \
correctness, relevance, groundedness, instruction_following (integers 1-5), \
overall (number), pass (boolean), reason (string). No markdown, no prose."""


def render_context(chunks: Sequence[RetrievedChunk], static_context: str | None) -> str:
    """Render retrieved chunks (preferred) or the case's static context."""
    if chunks:
        parts = [f"[source: {c.doc_id} #{c.chunk_index}]\n{c.text}" for c in chunks]
        return "\n\n".join(parts)
    return static_context or "(no context was provided to the system)"


def build_case_rubric(case: TestCase) -> str:
    if not case.rubric:
        return "(no case-specific rubric; use the generic anchors)"
    return "\n".join(f"- {dim}: {criteria}" for dim, criteria in sorted(case.rubric.items()))


def build_judge_messages(
    case: TestCase,
    output: SystemOutput,
    config: JudgeConfig,
) -> list[ChatMessage]:
    """Assemble the full judge prompt for one case."""
    max_chars = config.max_input_chars
    critical_threshold = (
        config.critical_pass_threshold if config.critical_pass_threshold is not None else ""
    )
    user_content = _USER_TEMPLATE.format(
        question=truncate(case.question, max_chars),
        category=case.category,
        critical=str(case.critical).lower(),
        reference_answer=truncate(case.reference_answer or "(none)", max_chars),
        expected_behavior=truncate(case.expected_behavior or "(none)", max_chars // 2),
        case_rubric=build_case_rubric(case),
        context=truncate(render_context(output.retrieved_context, case.context), max_chars),
        candidate_answer=truncate(output.answer or "(empty)", max_chars),
        pass_threshold=config.pass_threshold,
        critical_pass_threshold=critical_threshold,
    )
    return [
        ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


def build_repair_messages(
    original: Sequence[ChatMessage], previous_response: str, error: str
) -> list[ChatMessage]:
    """Follow-up messages requesting a corrected JSON verdict."""
    return [
        *list(original),
        ChatMessage(role="assistant", content=truncate(previous_response, 2000)),
        ChatMessage(
            role="user",
            content=_REPAIR_TEMPLATE.format(
                error=error, previous=truncate(previous_response, 1500)
            ),
        ),
    ]
