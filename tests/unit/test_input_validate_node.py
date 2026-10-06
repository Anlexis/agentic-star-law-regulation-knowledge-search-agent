# GOV-C2-005 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state), so the framework's own input gate runs ahead of
# execute(). Payloads are lowercase and free of personal data so the platform's
# redaction pass leaves them untouched and these assertions stay about parsing.
#
# Mirrors docs/03_test_spec.md Sec.2.2 (VAL-01..VAL-09).
# Deterministic — no model call, no network.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json, to_json

_NO_OPTIONS = {"category": None, "top_k": None, "score_threshold": None}


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_query(self):
        result = InputValidateNode()(_make_state("retention period for administrative documents"))
        assert result["search_query"] == "retention period for administrative documents"
        filters = from_json(result["query_filters"])
        assert filters == _NO_OPTIONS

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  retention   period\n for documents "))
        assert result["search_query"] == "retention period for documents"

    def test_query_filters_is_json_string(self):
        # Structured state fields travel as JSON strings, never bare dicts —
        # state is checkpointed and bare containers do not survive that.
        result = InputValidateNode()(_make_state("retention period"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestJsonEnvelopeParsing:
    def test_val_03_envelope_query_category_top_k(self):
        payload = json.dumps({"query": "disclosure request processing deadline", "category": "act", "top_k": 2})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "disclosure request processing deadline"
        filters = from_json(result["query_filters"])
        assert filters == {"category": "act", "top_k": 2, "score_threshold": None}

    def test_question_alias_accepted(self):
        payload = json.dumps({"question": "what is the standard processing period for an application?"})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "what is the standard processing period for an application?"

    def test_category_is_normalised(self):
        payload = json.dumps({"query": "retention checks", "category": "  Cabinet_Order "})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["category"] == "cabinet_order"

    def test_val_04_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == payload
        notes = from_json(result.get("intake_notes"), [])
        assert any("did not parse" in n for n in notes)


class TestEnvelopeOptionsFailClosed:
    """VAL-05..07: an envelope option that fails its bounds rejects the request.

    Rejecting rather than substituting is the point. A clamped or dropped value
    returns an answer the caller did not ask for and cannot distinguish from the
    one it did — here the caller is told the request was not run.
    """

    @pytest.mark.parametrize(
        "value",
        [99, -5, 0, "many", "NaN", "Infinity", "-Infinity", float("nan"), float("inf"), True, 2.5],
    )
    def test_val_05_out_of_bounds_top_k_is_rejected(self, value):
        payload = json.dumps({"query": "retention period", "top_k": value})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert "search_query" not in result
        assert any("top_k" in str(e) for e in result["error_log"])

    @pytest.mark.parametrize("value", ["NaN", "Infinity", float("nan"), float("-inf"), 5.0, -0.5, True, "high"])
    def test_val_06_out_of_bounds_score_threshold_is_rejected(self, value):
        payload = json.dumps({"query": "retention period", "score_threshold": value})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("score_threshold" in str(e) for e in result["error_log"])

    @pytest.mark.parametrize("value", ["bogus", "ACT; DROP TABLE", "", "a" * 40, 7])
    def test_val_07_unrecognised_category_is_rejected(self, value):
        payload = json.dumps({"query": "retention period", "category": value})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("category" in str(e) for e in result["error_log"])

    def test_rejection_never_echoes_the_rejected_value(self):
        secret = "9876543210987654"
        payload = json.dumps({"query": "retention period", "top_k": secret})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert secret not in str(result["error_log"])

    def test_in_bounds_score_threshold_is_carried(self):
        payload = json.dumps({"query": "retention period", "score_threshold": 0.5})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["score_threshold"] == 0.5


class TestContextChannelOptions:
    """Options validated upstream arrive through the context bridge."""

    def test_bridged_options_are_applied(self):
        result = InputValidateNode()(
            _make_state(
                "retention period",
                caller_options=to_json({"category": "cabinet_order", "top_k": 2}),
            )
        )
        filters = from_json(result["query_filters"])
        assert filters["category"] == "cabinet_order"
        assert filters["top_k"] == 2

    def test_context_channel_wins_over_the_envelope(self):
        payload = json.dumps({"query": "retention period", "category": "act"})
        result = InputValidateNode()(_make_state(payload, caller_options=to_json({"category": "cabinet_order"})))
        assert from_json(result["query_filters"])["category"] == "cabinet_order"

    def test_non_option_context_keys_are_not_carried_into_filters(self):
        # `channel` is validated for storage but is not a retrieval option.
        result = InputValidateNode()(_make_state("retention period", caller_options=to_json({"channel": "portal"})))
        assert from_json(result["query_filters"]) == _NO_OPTIONS


class TestSizeAndEmptyGuards:
    def test_val_08_oversize_query_is_truncated(self):
        payload = "regulatory " * 300  # ~3300 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["search_query"]) == 2000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_09_empty_request_yields_note_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)
