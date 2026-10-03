"""Versioned prompt templates for the sample RAG system.

Prompt versions are first-class: every evaluation result records which
prompt produced it, so a prompt change is testable exactly like a model or
retrieval change. This module is the single source of truth for template
text; adding ``v3`` means adding an entry here and referencing it in config.
"""

from __future__ import annotations

from dataclasses import dataclass

from ledger.core.models import ChatMessage, RetrievedChunk

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are the Helios Support Assistant. You answer questions about the Helios "
    "Data Platform using ONLY the provided documentation context. Be precise and "
    "factual. Cite the source document names you used. If the context does not "
    "contain enough information, say so explicitly instead of guessing."
)


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system_template: str
    user_template: str
    #: Marker text identifying the block the context is injected into. The
    #: mock provider parses the same tags, so they must stay stable.
    context_open: str = "<context>"
    context_close: str = "</context>"

    def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        system_instruction: str | None = None,
        max_context_chars: int = 12_000,
    ) -> list[ChatMessage]:
        system_text = (system_instruction or self.system_template).strip()
        context_block = self._render_context(chunks, max_context_chars)
        user_text = self.user_template.format(question=question.strip(), context=context_block)
        return [
            ChatMessage(role="system", content=system_text),
            ChatMessage(role="user", content=user_text),
        ]

    def _render_context(self, chunks: list[RetrievedChunk], max_context_chars: int) -> str:
        parts: list[str] = []
        used = 0
        for chunk in chunks:
            header = f"[source: {chunk.doc_id} #{chunk.chunk_index}]"
            block = f"{header}\n{chunk.text}"
            if used + len(block) > max_context_chars:
                remaining = max_context_chars - used
                if remaining > 80:
                    parts.append(block[:remaining] + " [...]")
                break
            parts.append(block)
            used += len(block) + 2
        return ("\n\n".join(parts)).strip()


_PROMPT_VERSIONS: dict[str, PromptTemplate] = {
    # Baseline prompt: grounding + citations, no length constraints.
    "v1_grounding": PromptTemplate(
        version="v1_grounding",
        system_template=DEFAULT_SYSTEM_INSTRUCTION,
        user_template=(
            "Answer the question using only the documentation context below.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    ),
    # Candidate prompt: same grounding rules, adds concision + inline citation
    # requirements. A realistic "prompt tweak" a team would ship and test.
    "v2_concise_cited": PromptTemplate(
        version="v2_concise_cited",
        system_template=(
            "You are the Helios Support Assistant. Answer strictly from the provided "
            "documentation context. Keep answers concise: at most three sentences. "
            "End every answer with an inline citation list of the document names you "
            "used, e.g. (Sources: pricing.md). If the context is insufficient, state "
            "clearly that you cannot answer from the provided documents."
        ),
        user_template=(
            "<context>\n{context}\n</context>\n\n"
            "Question: {question}\n\n"
            "Provide a concise, grounded answer with source citations."
        ),
    ),
    # Intentionally weak prompt used by the degraded demo config: no grounding
    # instruction, no citations. Exists to exercise the regression gate.
    "v0_sloppy": PromptTemplate(
        version="v0_sloppy",
        system_template=(
            "You are a helpful assistant. Answer the user's question as best you can."
        ),
        user_template=(
            "Context (maybe useful):\n{context}\n\nQuestion: {question}\nAnswer briefly:"
        ),
    ),
}


def get_prompt_template(version: str) -> PromptTemplate:
    try:
        return _PROMPT_VERSIONS[version]
    except KeyError:
        raise KeyError(
            f"unknown prompt_version '{version}'; available: {sorted(_PROMPT_VERSIONS)}"
        ) from None


def available_prompt_versions() -> list[str]:
    return sorted(_PROMPT_VERSIONS)
