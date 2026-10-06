# GOV-C2-005 — Unit Tests: manifest / config consistency
#
# The repo carries two config files with different jobs, and both are live:
#
#   config/agent.yaml   The registry entry. Flat — every key at root level.
#                       Identity, entry point, trust level, and the compile-time
#                       `requires` declarations. No runtime tuning.
#   config/config.yaml  Runtime parameters, handed to the graph constructor:
#                       max_retry, timeout_s, and the retrieval tuning that
#                       RegulationKnowledgeGraphNode forwards inward.
#
# These tests pin each file to the code that reads it, so a config drift fails
# in CI rather than in a deployment. The pairing matters most in one direction:
# tuning read from the wrong file does not raise, it silently returns nothing
# and the pipeline runs on module defaults.
#
# Mirrors docs/03_test_spec.md Sec.2.8 (CFG-01..CFG-07).
# Deterministic — no model call, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import GovernmentRegulationKnowledgeAgent, RegulationKnowledgeGraphNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.runtime_config import load_runtime_config

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_template_id_is_consistent(self):
        assert _MANIFEST["id"] == "GOV-C2-005"
        assert _MANIFEST["namespace"] == "gov"

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # The manifest's entry point is a single dotted path, and it must
        # resolve to the class the server imports.
        assert _MANIFEST["class"] == ("src.graph.graph.GovernmentRegulationKnowledgeAgent")
        assert _MANIFEST["class"].rsplit(".", 1)[-1] == (GovernmentRegulationKnowledgeAgent.__name__)
        assert _MANIFEST["name"] == GovernmentRegulationKnowledgeAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "GOV"
        assert _MANIFEST["base_type"] == "RAGAgent"
        assert _MANIFEST["enabled"] is True

    def test_manifest_declares_no_secrets_or_extras(self):
        # Both are code-derived: this template calls no secrets and constructs
        # no optional client. Declaring either would fail compilation in an
        # environment where it is not provisioned.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []

    def test_manifest_generation_mode_matches_the_pipeline(self):
        # Every node assembles its output by rule; nothing calls a model.
        assert _MANIFEST["generation_mode"] == "deterministic"


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared


class TestRuntimeConfigFile:
    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # framework retry ceiling

    def test_runtime_config_loader_reads_the_file(self):
        assert load_runtime_config()["max_retry"] == _RUNTIME["max_retry"]
        assert load_runtime_config()["timeout_s"] == _RUNTIME["timeout_s"]

    def test_runtime_values_reach_the_constructed_agent(self):
        # The entry point hands this mapping to the constructor. Constructing
        # with no config would leave every declared value inert.
        agent = GovernmentRegulationKnowledgeAgent(config=load_runtime_config())
        assert agent.config["max_retry"] == _RUNTIME["max_retry"]
        assert agent.config["timeout_s"] == _RUNTIME["timeout_s"]

    def test_hitl_is_not_enabled(self):
        # This template declares no human-in-the-loop step, so no checkpointer
        # is required and no interrupt path is live.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False


class TestRetrievalBlock:
    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror the shipped tuning — a drift would
        # silently change retrieval behaviour wherever the file is unreadable.
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults

        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_parent_config_forwards_the_runtime_retrieval_block(self):
        cfg = RegulationKnowledgeGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty retrieval block"

    def test_the_manifest_carries_no_tuning_to_read_by_mistake(self):
        # Guards the failure this file exists for: if tuning ever reappears in
        # the registry entry, two files would disagree and the reader that lost
        # would degrade in silence.
        for key in ("retrieval", "config", "llm", "agent"):
            assert key not in _MANIFEST


class TestSeededKnowledgeBase:
    def _entries(self):
        return json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))

    def test_kb_is_a_well_formed_entry_list(self):
        entries = self._entries()
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded corpus must carry a usable body of text"
        for entry in entries:
            assert set(entry.keys()) == {
                "id",
                "title",
                "category",
                "source",
                "tags",
                "content",
            }
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        ids = [e["id"] for e in self._entries()]
        assert len(ids) == len(set(ids))

    def test_kb_categories_match_the_accepted_vocabulary(self):
        # Every seeded entry's category must be one a caller is allowed to ask
        # for, or that category filter could never match anything.
        from src.caller_contract import CATEGORY_VALUES

        assert {e["category"] for e in self._entries()} <= set(CATEGORY_VALUES)
