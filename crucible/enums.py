from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConsensusLevel(str, Enum):
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    MINORITY = "minority"


class ClauseStatus(str, Enum):
    PRESENT = "PRESENT"
    PRESENT_BUT_DEFICIENT = "PRESENT_BUT_DEFICIENT"
    MISSING = "MISSING"


class AuditClassification(str, Enum):
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    DISPUTED = "DISPUTED"
    OMITTED_FROM_VERDICT = "OMITTED_FROM_VERDICT"
    OMITTED_WITH_EXPLANATION = "OMITTED_WITH_EXPLANATION"


class Decision(str, Enum):
    APPROVE = "approve"
    APPROVE_WITH_EDITS = "approve_with_edits"
    REJECT = "reject"
    EXTEND_BUDGET = "extend_budget"
    ABORT = "abort"
    MANUAL_PROMPT_REWRITE = "manual_prompt_rewrite"
    CONTINUE_UNCACHED = "continue_uncached"
    RETRY = "retry"
    PROCEED_WITH_DEGRADED_PANEL = "proceed_with_degraded_panel"


class CheckpointId(str, Enum):
    SETUP_CONFIRM = "setup_confirm"
    SCOPE_CONFIRM = "scope_confirm"
    DELTA_CONFIRM = "delta_confirm"
    POST_INTAKE = "post_intake"
    POST_RED = "post_red"
    POST_SELF_RED = "post_self_red"
    POST_VERDICT = "post_verdict"
    POST_FINAL_DRAFT = "post_final_draft"
    POST_AUDIT = "post_audit"
    PRE_NEGOTIATION_MEMO = "pre_negotiation_memo"
    GUARDRAIL_TRIP = "guardrail_trip"
    CONTENT_POLICY_HALT = "content_policy_halt"
    CACHE_REBUILD_OVERDRAFT_POST = "cache_rebuild_overdraft_post"
    CACHE_REBUILD_OVERDRAFT_AUTHORIZE = "cache_rebuild_overdraft_authorize"
    PANEL_DEGRADED = "panel_degraded"


class ModelTier(str, Enum):
    FLAGSHIP = "flagship"
    FAST = "fast"


class RetryClass(str, Enum):
    RETRY = "retry"
    HALT_AUTH = "halt_auth"
    HALT_POLICY = "halt_policy"
    RATE_LIMIT_RETRY = "rate_limit_retry"
    HALT_OTHER = "halt_other"


class _TokenType(Enum):
    ARTICLE_REF = "article_ref"
    STATUTE_NAME = "statute_name"
    RANGE_FROM = "range_from"
    RANGE_TO = "range_to"
    LIST_JOINER = "list_joiner"
    APPLY_VERB = "apply_verb"
    EXCLUDE_PREP = "exclude_prep"
    RELATIVE_REF = "relative_ref"
    SAME_STATUTE = "same_statute"
    SENTENCE_END = "sentence_end"
    COMMA = "comma"
    OTHER = "other"
