from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

_PROMPTS_FILE = Path(__file__).parent.parent / "subagent_prompts.md"


@lru_cache(maxsize=None)
def _load_prompts_doc() -> str:
    if not _PROMPTS_FILE.exists():
        return ""
    return _PROMPTS_FILE.read_text(encoding="utf-8")


def _extract_section(doc: str, keyword: str) -> str:
    """Extract content under the first heading containing keyword."""
    pattern = re.compile(
        r"^(#{1,4})\s+.*?" + re.escape(keyword) + r".*?$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = pattern.search(doc)
    if not m:
        return ""
    level = len(m.group(1))
    start = m.end()
    next_heading = re.compile(r"^#{1," + str(level) + r"}\s+", re.MULTILINE)
    end_m = next_heading.search(doc, start)
    end = end_m.start() if end_m else len(doc)
    return doc[start:end].strip()


def _extract_subsection_code(section: str, subheading_keyword: str) -> str:
    """Extract first code block from after a subheading containing keyword."""
    sub_pat = re.compile(
        r"^#{1,6}\s+.*?" + re.escape(subheading_keyword) + r".*?$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = sub_pat.search(section)
    if not m:
        return ""
    remainder = section[m.end():]
    next_sub = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    end_m = next_sub.search(remainder)
    subsection = remainder[:end_m.start()].strip() if end_m else remainder.strip()
    return _extract_code_block(subsection) or subsection


def _extract_code_block(section: str, label: str = "") -> str:
    if label:
        pattern = re.compile(
            r"```" + re.escape(label) + r"\s*\n(.*?)\n```",
            re.DOTALL | re.IGNORECASE,
        )
    else:
        pattern = re.compile(r"```\w*\s*\n(.*?)\n```", re.DOTALL)
    m = pattern.search(section)
    return m.group(1).strip() if m else section.strip()


def _fmt(template: str, **kwargs) -> str:
    """Format template, silently leaving unrecognised {vars} as empty string."""
    return template.format_map(defaultdict(str, **kwargs))


class PromptBuilder:
    """Parse prompt templates from subagent_prompts.md on demand."""

    @classmethod
    def _sys(cls, section_keyword: str) -> str:
        doc = _load_prompts_doc()
        section = _extract_section(doc, section_keyword)
        return _extract_subsection_code(section, "system:") or section

    @classmethod
    def _usr(cls, section_keyword: str) -> str:
        doc = _load_prompts_doc()
        section = _extract_section(doc, section_keyword)
        return _extract_subsection_code(section, "user:")

    # ------------------------------------------------------------------ MPSR
    @classmethod
    def mpsr_panel_system(cls, jurisdiction: str, contract_type: str,
                          jurisdiction_language: str) -> str:
        return _fmt(cls._sys("mpsr_panel_member"),
                    jurisdiction=jurisdiction,
                    contract_type=contract_type,
                    jurisdiction_language=jurisdiction_language)

    @classmethod
    def mpsr_panel_user(cls, contract_type: str, track: str, jurisdiction: str,
                        engagement_context_block: str = "") -> str:
        return _fmt(cls._usr("mpsr_panel_member"),
                    contract_type=contract_type,
                    track=track,
                    jurisdiction=jurisdiction,
                    engagement_context_block=engagement_context_block)

    # --------------------------------------------------------------- Framework
    @classmethod
    def framework_system(cls, jurisdiction: str, jurisdiction_language: str,
                         contract_type: str = "", mode_verb: str = "reviewed",
                         charter_json: str = "") -> str:
        return _fmt(cls._sys("framework"),
                    jurisdiction=jurisdiction,
                    jurisdiction_language=jurisdiction_language,
                    contract_type=contract_type,
                    mode_verb=mode_verb,
                    charter_json=charter_json)

    @classmethod
    def framework_user(cls, mode: str, contract_text: str,
                       rag_statutory_context: str = "",
                       rag_template_context: str = "") -> str:
        return _fmt(cls._usr("framework"),
                    mode=mode,
                    contract_text=contract_text,
                    rag_statutory_context=rag_statutory_context or "[not available]",
                    rag_template_context=rag_template_context or "[not available]")

    # ---------------------------------------------------------------- RedTeam
    @classmethod
    def red_team_system(cls, jurisdiction: str, jurisdiction_language: str,
                        contract_type: str = "", target_label: str = "contract",
                        charter_json: str = "") -> str:
        return _fmt(cls._sys("red_team"),
                    jurisdiction=jurisdiction,
                    jurisdiction_language=jurisdiction_language,
                    contract_type=contract_type,
                    target_label=target_label,
                    charter_json=charter_json)

    @classmethod
    def red_team_user(cls, mode: str, contract_text: str,
                      target_label: str = "contract",
                      rag_statutory_context: str = "",
                      rag_treatise_context: str = "",
                      jurisdiction_language: str = "") -> str:
        return _fmt(cls._usr("red_team"),
                    mode=mode,
                    target_label=target_label,
                    contract_text=contract_text,
                    rag_statutory_context=rag_statutory_context or "[not available]",
                    rag_treatise_context=rag_treatise_context or "[not available]",
                    jurisdiction_language=jurisdiction_language)

    # --------------------------------------------------------- CharterDelta
    @classmethod
    def charter_delta_few_shot_examples(cls, jurisdiction: str) -> str:
        if jurisdiction.lower() != "japan":
            return "[No few-shot examples configured for this jurisdiction]"
        doc = _load_prompts_doc()
        section = _extract_section(doc, "charter_delta_few_shot_examples")
        return _extract_code_block(section) or section

    @classmethod
    def charter_delta_residual_system(cls, jurisdiction: str,
                                      jurisdiction_language: str = "",
                                      charter_json: str = "",
                                      precomputed_statute_deltas_json: str = "[]") -> str:
        examples = cls.charter_delta_few_shot_examples(jurisdiction)
        return _fmt(cls._sys("charter_delta_residual"),
                    jurisdiction=jurisdiction,
                    jurisdiction_language=jurisdiction_language,
                    charter_json=charter_json,
                    charter_delta_few_shot_examples=examples,
                    precomputed_statute_deltas_json=precomputed_statute_deltas_json)

    @classmethod
    def charter_delta_residual_user(cls, red_team_findings_json: str) -> str:
        return _fmt(cls._usr("charter_delta_residual"),
                    red_team_findings_json=red_team_findings_json)

    # ----------------------------------------------------------------- Intake
    @classmethod
    def intake_system(cls, jurisdiction: str, charter_json: str = "") -> str:
        return _fmt(cls._sys("intake"),
                    jurisdiction=jurisdiction,
                    charter_json=charter_json)

    @classmethod
    def intake_user(cls, termsheet_text: str, termsheet_format: str = "plain_text") -> str:
        return _fmt(cls._usr("intake"),
                    termsheet_text=termsheet_text,
                    termsheet_format=termsheet_format)

    # ---------------------------------------------------------------- Drafter
    @classmethod
    def drafter_system(cls, jurisdiction: str, jurisdiction_language: str,
                       contract_type: str = "", report_language: str = "English",
                       charter_json: str = "") -> str:
        return _fmt(cls._sys("drafter"),
                    jurisdiction=jurisdiction,
                    jurisdiction_language=jurisdiction_language,
                    contract_type=contract_type,
                    report_language=report_language,
                    charter_json=charter_json)

    @classmethod
    def drafter_user_b2(cls, requirements_spec_json: str, skeleton_clauses_json: str,
                        jurisdiction_language: str = "") -> str:
        return _fmt(cls._usr("drafter"),
                    mode="b2_draft_pass",
                    requirements_spec_json=requirements_spec_json,
                    skeleton_clauses_json=skeleton_clauses_json,
                    jurisdiction_language=jurisdiction_language,
                    previous_draft="",
                    accepted_hardenings_json="[]",
                    charter_delta_findings_json="[]",
                    incremental_rag_context="")

    @classmethod
    def drafter_user_b4(cls, requirements_spec_json: str, skeleton_clauses_json: str,
                        previous_draft: str, accepted_hardenings_json: str,
                        charter_delta_findings_json: str = "[]",
                        incremental_rag_context: str = "",
                        jurisdiction_language: str = "",
                        report_language: str = "English") -> str:
        return _fmt(cls._usr("drafter"),
                    mode="b4_synthesis",
                    requirements_spec_json=requirements_spec_json,
                    skeleton_clauses_json=skeleton_clauses_json,
                    jurisdiction_language=jurisdiction_language,
                    report_language=report_language,
                    previous_draft=previous_draft,
                    accepted_hardenings_json=accepted_hardenings_json,
                    charter_delta_findings_json=charter_delta_findings_json,
                    incremental_rag_context=incremental_rag_context)

    # ----------------------------------------------------------------- Verdict
    @classmethod
    def verdict_system(cls, jurisdiction: str, jurisdiction_language: str,
                       report_language: str = "English",
                       charter_json: str = "") -> str:
        return _fmt(cls._sys("verdict"),
                    jurisdiction=jurisdiction,
                    jurisdiction_language=jurisdiction_language,
                    report_language=report_language,
                    charter_json=charter_json)

    @classmethod
    def verdict_user(cls, framework_clauses_json: str,
                     framework_missing_clauses_proposed_language_json: str,
                     red_team_findings_json: str,
                     charter_delta_findings_json: str = "[]",
                     rag_treatise_context: str = "",
                     rag_internal_context: str = "",
                     contract_text: str = "",
                     report_language: str = "English",
                     jurisdiction_language: str = "") -> str:
        return _fmt(cls._usr("verdict"),
                    framework_clauses_json=framework_clauses_json,
                    framework_missing_clauses_proposed_language_json=framework_missing_clauses_proposed_language_json,
                    red_team_findings_json=red_team_findings_json,
                    charter_delta_findings_json=charter_delta_findings_json,
                    rag_treatise_context=rag_treatise_context or "[not available]",
                    rag_internal_context=rag_internal_context or "[not available]",
                    contract_text=contract_text,
                    report_language=report_language,
                    jurisdiction_language=jurisdiction_language)

    # ----------------------------------------------------------------- Auditor
    @classmethod
    def auditor_system(cls, jurisdiction: str, jurisdiction_language: str = "",
                       report_language: str = "English") -> str:
        return _fmt(cls._sys("auditor_phase1"),
                    jurisdiction=jurisdiction,
                    jurisdiction_language=jurisdiction_language,
                    report_language=report_language)

    @classmethod
    def auditor_user(cls, citation_string: str, document_claim_excerpt: str,
                     retrieved_source_text: str = "[retrieval not available]") -> str:
        return _fmt(cls._usr("auditor_phase1"),
                    citation_string=citation_string,
                    document_claim_excerpt=document_claim_excerpt,
                    retrieved_source_text=retrieved_source_text)

    @classmethod
    def get_raw(cls, keyword: str) -> str:
        doc = _load_prompts_doc()
        return _extract_section(doc, keyword)


_VALIDATION_STRINGS = {
    "charter_delta_residual": ["NOT_NEW"],
    "verdict": ["[OMISSION:", "[/OMISSION]"],
    "red_team": ["MUST NOT BE SKIPPED"],
}


def validate_prompts() -> None:
    doc = _load_prompts_doc()
    if not doc:
        print("WARNING: subagent_prompts.md not found — skipping prompt validation.")
        return
    for section_name, required_strings in _VALIDATION_STRINGS.items():
        section = _extract_section(doc, section_name)
        for s in required_strings:
            if s not in section:
                raise ValueError(
                    f"Prompt validation failed: '{s}' not found in section '{section_name}'"
                )
    print("Prompt validation passed.")
