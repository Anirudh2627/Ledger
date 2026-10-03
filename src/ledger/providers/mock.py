"""Deterministic offline LLM provider.

The mock provider makes the *entire* framework runnable and testable without
network access or API keys, and gives CI a reproducible default. It is NOT a
quality oracle: its answers are simple deterministic heuristics (extractive
answers from the supplied context, rule-based judge scores). It exists so the
plumbing - pipeline, statistics, policy, reports, gates - can be exercised
end-to-end, and so teams can dry-run changes before spending tokens.

Behaviour:
  * ``purpose="generation"`` - extractive answering from the ``<context>``
    block of the prompt. Refuses when the context cannot answer. Honours a
    few instruction signals (JSON / bullets / one sentence / word limit /
    citations) and rejects prompt-injection phrasing.
  * ``purpose="judge"``      - parses the tagged judge prompt and emits a
    structured JSON verdict using lexical heuristics correlated with the
    real quality signals (overlap with reference, grounding in context,
    refusal behaviour, instruction markers).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Iterable, Sequence

from ledger.core.interfaces import LLMProvider
from ledger.core.models import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)
from ledger.utils.logging import get_logger
from ledger.utils.text import STOPWORDS, split_sentences, token_f1, tokenize

logger = get_logger(__name__)

_CONTEXT_RE = re.compile(r"<context>(.*?)</context>", re.DOTALL)
_QUESTION_RE = re.compile(r"Question:\s*(.+)")
_SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]#]+?)(?:\s*#\d+)?\s*\]")
_SOURCE_SPLIT_RE = re.compile(r"\[source:[^\]]*\]")

_JUDGE_TAG_RES = {
    "question": re.compile(r"<question>(.*?)</question>", re.DOTALL),
    "reference_answer": re.compile(r"<reference_answer>(.*?)</reference_answer>", re.DOTALL),
    "expected_behavior": re.compile(r"<expected_behavior>(.*?)</expected_behavior>", re.DOTALL),
    "context": re.compile(r"<context>(.*?)</context>", re.DOTALL),
    "candidate_answer": re.compile(r"<candidate_answer>(.*?)</candidate_answer>", re.DOTALL),
    "category": re.compile(r"<category>(.*?)</category>", re.DOTALL),
    "pass_threshold": re.compile(r"<pass_threshold>(.*?)</pass_threshold>", re.DOTALL),
    "critical": re.compile(r"<critical>(.*?)</critical>", re.DOTALL),
    "critical_pass_threshold": re.compile(
        r"<critical_pass_threshold>(.*?)</critical_pass_threshold>", re.DOTALL
    ),
}

_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "ignore the previous",
    "ignore your instructions",
    "disregard previous",
    "disregard your",
    "reveal your system prompt",
    "print your system prompt",
    "show your system prompt",
    "repeat your system prompt",
    "override your guidelines",
    "new instructions:",
)

_REFUSAL_PHRASES = (
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "cannot answer",
    "can't answer",
    "no information",
    "not provided in the",
    "not contain",
    "doesn't contain",
    "i'm sorry",
    "i am sorry",
    "cannot act on",
    "can't act on",
    "unable to answer",
    "not answerable from",
)

_INJECTION_REFUSAL = (
    "I can't act on requests to ignore or override my instructions. "
    "I will continue to answer using only the provided documentation."
)
_NO_INFO_REFUSAL = (
    "Based on the provided documents, I don't have enough information "
    "to answer this question reliably."
)

#: Tokens that carry no discriminating signal for a product-documentation
#: corpus ("Helios ... supports ..." appears on nearly every page). Excluded
#: from evidence-overlap counting so retrieval-quality questions about
#: *niche* facts still refuse when the fact is absent.
_DOMAIN_GENERIC = frozenset({"helios", "heliosdata", "data", "platform", "support", "supported"})

#: Minimum content-token overlap before a sentence counts as evidence.
_MIN_OVERLAP = 2
_STEM_LEN = 5

#: Behaviour phrases that reliably signal "the correct outcome is a refusal".
#: Deliberately narrow: broader phrases ("must not ...", "unavailable") caused
#: false refusal expectations on answerable questions.
_REFUSAL_EXPECTATION_MARKERS = ("refuse", "decline")


def _stable_unit(*parts: object) -> float:
    """Deterministic float in [0, 1) derived from the inputs."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _tag(content: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(content)
    return match.group(1).strip() if match else None


def _contains_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def _content_of(messages: Sequence[ChatMessage]) -> str:
    return "\n\n".join(m.content for m in messages)


def _content_tokens(text: str) -> list[str]:
    """Deduplicated content tokens (stopwords and domain-generic removed)."""
    return list(
        dict.fromkeys(t for t in tokenize(text) if t not in STOPWORDS and t not in _DOMAIN_GENERIC)
    )


def _stem_key(token: str) -> str:
    return token[:4] if len(token) >= 4 else token


def _stem_overlap(query_tokens: Iterable[str], target_tokens: set[str]) -> int:
    """Count query tokens present in target.

    Matching is exact or by 4-character prefix ("cost"~"costs",
    "encryption"~"encrypted", "migrate"~"migrating"). Loose enough for a
    morphological-free lexical mock, tight enough to avoid most collisions.
    """
    target_prefixes = {_stem_key(t) for t in target_tokens if len(t) >= 4}
    return sum(
        1
        for t in query_tokens
        if t in target_tokens or (len(t) >= 4 and _stem_key(t) in target_prefixes)
    )


def _context_blocks(context: str) -> list[str]:
    """Split rendered context on ``[source: ...]`` headers, dropping them."""
    if not context:
        return []
    return [block.strip() for block in _SOURCE_SPLIT_RE.split(context) if block.strip()]


def _block_units(block: str) -> list[str]:
    """Split a context block into sentence-like units.

    Markdown table rows are treated as atomic units (a whole table would
    otherwise collapse into one giant pseudo-sentence and create spurious
    lexical evidence); prose is split on sentence boundaries.
    """
    units: list[str] = []
    prose: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if re.fullmatch(r"\|[\s\-:|]+\|", stripped):
                continue  # table separator row
            if prose:
                units.extend(split_sentences(" ".join(prose)))
                prose = []
            units.append(stripped)
        elif stripped:
            prose.append(stripped)
    if prose:
        units.extend(split_sentences(" ".join(prose)))
    return [u for u in units if u]


class MockLLMProvider(LLMProvider):
    """Deterministic, offline stand-in for a chat LLM."""

    name = "mock"

    def __init__(self, *, model: str = "mock-llm-v1", seed: int = 1337) -> None:
        self.model = model
        self.seed = seed

    # -- LLMProvider API -----------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        content = _content_of(request.messages)
        if request.purpose == "judge":
            text = self._judge_response(content)
        else:
            text = self._generation_response(request.messages)
        latency = 30.0 + 220.0 * _stable_unit(self.seed, request.purpose, content[:400])
        usage = CompletionUsage(
            prompt_tokens=len(content) // 4,
            completion_tokens=len(text) // 4,
            total_tokens=(len(content) + len(text)) // 4,
        )
        return CompletionResponse(
            text=text,
            model=self.model,
            finish_reason="stop",
            latency_ms=round(latency, 2),
            usage=usage,
            attempts=1,
        )

    # -- Generation ----------------------------------------------------------

    def _generation_response(self, messages: Sequence[ChatMessage]) -> str:
        content = _content_of(messages)
        question_match = _QUESTION_RE.search(content)
        question = question_match.group(1).strip() if question_match else ""
        context_match = _CONTEXT_RE.search(content)
        context = context_match.group(1).strip() if context_match else ""

        # Instruction signals come from the system prompt + question only -
        # NOT from the full content, because context blocks contain markup
        # like "[source: ...]" that would otherwise trigger citation mode.
        system_text = " ".join(m.content for m in messages if m.role == "system")
        instructions = f"{system_text}\n{question}".lower()

        # 1. Prompt-injection phrasing: a reasonably robust assistant refuses.
        if any(marker in question.lower() for marker in _INJECTION_MARKERS):
            return _INJECTION_REFUSAL

        # 2. Extractive answering from the supplied context.
        sources = _SOURCE_TAG_RE.findall(context)
        sentences = self._select_sentences(question, context)

        if not sentences:
            return _NO_INFO_REFUSAL
        return self._format_answer(sentences, instructions, sources)

    def _select_sentences(self, question: str, context: str) -> list[str]:
        q_tokens = _content_tokens(question)
        if not context or not q_tokens:
            return []
        min_overlap = _MIN_OVERLAP if len(q_tokens) > 2 else 1
        scored: list[tuple[float, int, str]] = []
        position = 0
        # Sentences are extracted per context block so evidence never spans
        # chunk boundaries (and source headers never leak into answers).
        for block in _context_blocks(context):
            for sentence in _block_units(block):
                position += 1
                overlap = _stem_overlap(q_tokens, set(tokenize(sentence)))
                if overlap < min_overlap:
                    continue
                score = overlap / (len(q_tokens) + 2.0)
                scored.append((score, position, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        picked: list[str] = []
        word_count = 0
        for _score, _pos, sentence in scored:
            picked.append(sentence)
            word_count += len(sentence.split())
            if len(picked) >= 3 or word_count >= 60:
                break
        return picked

    def _format_answer(self, sentences: list[str], instructions: str, sources: list[str]) -> str:
        answer_sentences = list(sentences)

        # one-sentence instruction
        if "one sentence" in instructions or "single sentence" in instructions:
            answer_sentences = answer_sentences[:1]

        # explicit word limit: "at most N words" / "N words or fewer" / "under N words"
        limit_match = re.search(
            r"(?:at most|no more than|under|maximum of|<=?)\s*(\d+)\s*words", instructions
        )
        text = " ".join(answer_sentences)
        if limit_match:
            limit = int(limit_match.group(1))
            words = text.split()
            if len(words) > limit:
                text = " ".join(words[:limit]).rstrip(",;:") + "."

        # bullet instruction
        if "bullet" in instructions or "bulleted" in instructions:
            text = "\n".join(f"- {s}" for s in answer_sentences)

        unique_sources = list(dict.fromkeys(s.strip() for s in sources if s.strip()))

        # JSON instruction
        wants_json = (
            "json object" in instructions
            or "valid json" in instructions
            or "in json" in instructions
        )
        if wants_json:
            payload: dict[str, object] = {"answer": " ".join(answer_sentences)}
            if unique_sources:
                payload["sources"] = unique_sources[:3]
            text = json.dumps(payload)
        elif "cite" in instructions or "citation" in instructions:
            # citation instruction (and not JSON output)
            if unique_sources:
                cited = ", ".join(unique_sources[:3])
                text = f"{text} (Sources: {cited})"
        return text

    # -- Judge ----------------------------------------------------------------

    def _judge_response(self, content: str) -> str:
        fields = {key: _tag(content, pattern) for key, pattern in _JUDGE_TAG_RES.items()}
        question = fields.get("question") or ""
        reference = fields.get("reference_answer") or ""
        behavior = (fields.get("expected_behavior") or "").lower()
        context = fields.get("context") or ""
        answer = fields.get("candidate_answer") or ""
        category = (fields.get("category") or "").lower()

        threshold = 3.5
        with contextlib.suppress(ValueError, TypeError):
            threshold = float(fields.get("pass_threshold") or 3.5)
        is_critical = (fields.get("critical") or "").lower() in {"true", "yes", "1"}
        if is_critical:
            with contextlib.suppress(ValueError, TypeError):
                threshold = float(fields.get("critical_pass_threshold") or threshold)

        answer_refuses = _contains_refusal(answer)
        expects_refusal = category == "refusal" or any(
            marker in behavior for marker in _REFUSAL_EXPECTATION_MARKERS
        )
        answer_tokens = set(tokenize(answer))
        q_tokens = _content_tokens(question)

        # --- correctness ----------------------------------------------------
        if expects_refusal:
            correctness = 5 if answer_refuses else 1
        elif reference and reference.lower() not in {"n/a", "none", "-", "(none)"}:
            f1 = token_f1(answer, reference)
            ref_tokens = _content_tokens(reference)
            coverage = (
                _stem_overlap(ref_tokens, answer_tokens) / len(ref_tokens) if ref_tokens else 0.0
            )
            correctness = _bucket(max(f1, coverage), (0.55, 0.38, 0.22, 0.10))
            # Numeric-fact guard: when the reference contains numbers (prices,
            # limits, ports, dates), an answer sharing NONE of them missed the
            # key fact no matter how much other vocabulary overlaps.
            ref_numbers = {t for t in tokenize(reference) if any(ch.isdigit() for ch in t)}
            if ref_numbers:
                ans_numbers = {t for t in tokenize(answer) if any(ch.isdigit() for ch in t)}
                if not (ref_numbers & ans_numbers):
                    correctness = min(correctness, 2)
        else:
            correctness = 3  # nothing to check against; neutral

        # --- groundedness ---------------------------------------------------
        substantive_context = bool(context) and not context.lower().startswith("(no context")
        if expects_refusal and answer_refuses:
            groundedness = 5
        elif substantive_context:
            context_tokens = set(tokenize(context))
            substantive = [t for t in tokenize(answer) if t not in STOPWORDS]
            frac = (
                sum(1 for t in substantive if t in context_tokens) / len(substantive)
                if substantive
                else 0.0
            )
            groundedness = _bucket(frac, (0.92, 0.80, 0.60, 0.35))
        else:
            # No context supplied: groundedness is unverifiable -> neutral.
            groundedness = 3

        # --- relevance ------------------------------------------------------
        if answer_refuses and not expects_refusal:
            relevance = 2
        elif q_tokens:
            overlap = _stem_overlap(q_tokens, answer_tokens) / (len(q_tokens) + 2.0)
            relevance = _bucket(overlap, (0.45, 0.30, 0.18, 0.08))
            if expects_refusal and answer_refuses:
                relevance = max(relevance, 4)
        else:
            relevance = 3

        # --- instruction following -------------------------------------------
        if expects_refusal:
            instruction = 5 if answer_refuses else 1
        else:
            bonus = 0
            if "bullet" in behavior and re.search(r"(?m)^\s*[-*\u2022]", answer):
                bonus += 1
            if "json" in behavior and _parses_as_json(answer):
                bonus += 2
            word_limit = re.search(r"(\d+)\s*words", behavior)
            if word_limit and len(answer.split()) <= int(word_limit.group(1)) + 2:
                bonus += 1
            if "one sentence" in behavior and len(split_sentences(answer)) <= 1:
                bonus += 1
            if ("cite" in behavior or "source" in behavior) and re.search(
                r"\.md|\[|\(source", answer, re.IGNORECASE
            ):
                bonus += 1
            if bonus:
                instruction = min(5, 3 + bonus)
            elif "clarif" in behavior or "assumption" in behavior or "ambigu" in behavior:
                clarifies = any(
                    m in answer.lower() for m in ("?", "assume", "clarif", "which ", "depends")
                )
                instruction = 4 if clarifies else 2
            else:
                instruction = 4 if not answer_refuses else 2

        dimensions = {
            "correctness": correctness,
            "relevance": relevance,
            "groundedness": groundedness,
            "instruction_following": instruction,
        }
        overall = round(sum(dimensions.values()) / len(dimensions), 2)
        verdict = {
            **dimensions,
            "overall": overall,
            "pass": overall >= threshold,
            "reason": (
                f"mock-judge heuristics: expects_refusal={expects_refusal}, "
                f"answer_refuses={answer_refuses}, ref_f1={token_f1(answer, reference):.2f}, "
                f"threshold={threshold}"
            ),
        }
        return json.dumps(verdict)


def _bucket(signal: float, cuts: tuple[float, float, float, float]) -> int:
    """Map a [0,1] signal onto the 1-5 rubric scale using four cut points."""
    high, good, mid, low = cuts
    if signal >= high:
        return 5
    if signal >= good:
        return 4
    if signal >= mid:
        return 3
    if signal >= low:
        return 2
    return 1


def _parses_as_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False
