# Test Specification — GOV-C2-005

**Template ID:** GOV-C2-005
**Template Name:** GovernmentRegulationKnowledgeAgent
**Category:** Cat 2 (nested retrieval pipeline)

This document is the contract the test suite implements. Every case below maps
to a test that ships in `tests/`.

## 1. Scope and invocation conventions

- Per-node unit tests for the five inner domain nodes and the two outer
  boundary nodes.
- The caller-data contract: bounds, refusal, and fail-closed behaviour.
- The output boundary: credential redaction and the grounding + disclaimer
  contract.
- Config consistency (`config/agent.yaml` and `config/config.yaml` against the
  code that reads each) and seeded-corpus integrity.
- Retrieval quality — golden queries over `config/kb/gov_regulations_kb.json`,
  one per corpus entry.
- Inner-graph and outer-graph composition.
- Boundary proofs: import isolation, state serialisation safety, invoke order,
  human-in-the-loop propagation (auto-waived), server boot, and end-to-end
  behaviour through the real HTTP entry point.

**Invocation canon.** Per-node tests invoke the node via `node(state)`, which
runs the framework's trust check, its redaction pass, `execute()`, and its
output check in order — never a bare `node.execute(state)`, which would step
around the trust check. The state builder sets `caller_trust_level` to
`VERIFIED_EXTERNAL` for the two outer slots (the manifest's declared caller
level) and `ANONYMOUS` for the five inner domain nodes.

**One deliberate exception.** The refusal tests in
`test_pre_process_node.py::TestTemplateOwnedRefusal` call `execute()`
**directly**, with no framework wrapper in front. Their whole claim is that the
TEMPLATE refuses; run through `node(state)` they could pass on a platform gate's
behaviour and say nothing about this template. They assert behaviour — an error
status and nothing carried forward — never a gate's wording, and they pin both
directions: attack forms refused, ordinary questions using the same vocabulary
unaffected.

**No `config` parameter.** `execute(self, state)` takes no `config` argument
anywhere in this template. Config knobs are seeded on the state dict
(`retrieval_config`, `caller_options` — the same fields
`DomainWorkflowGraph._extra_initial_state()` seeds at runtime), never as a
second positional argument.

**Redaction expectations, and a documented interaction with this payload.** The
framework's input gate masks `user_input` / `validated_input` (e-mail addresses,
grouped digit runs, Title-Case name-shaped runs) to `[MASKED]` before
`execute()` runs. Happy-path tests use lowercase phrasing so the mask leaves
them untouched; the deliberate-identifier tests assert the raw identifier is
gone and the marker is present (`[MASKED]` from the framework, `[REDACTED]` from
the node's own screen).

The canonical payload for this template — *"…under the Public Records Management
Act?"* — contains a four-word Title-Case run that the framework masks, because a
statute name and a person's name are the same shape. This does not break the
success path: the surviving tokens still clear the relevance floor. It does mean
the echoed question shows `[MASKED]` where the act name would be, while the
Sources list and inline citations still carry the real act name, because those
are built from the corpus entries rather than from the query. Pinned at
`test_pre_process_node.py::TestPlatformMaskingOfTitleCaseStatuteNames` and
`test_pb_invoke_order.py::TestRedactionObservedBehaviour`.

**Audit muting.** `shared.*` is never stubbed out of `sys.modules` (the
framework imports `shared.security` at load time). The domain audit emitter is
muted by an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; the
audit assertion test re-patches the same attribute with a spy and asserts on
`call.args[1]`, the event payload.

## 2. Unit test cases

### 2.1 PreProcessNode (outer pre_process slot) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query | lowercase regulatory question | success; `validated_input` set; `enriched_context` carries channel and source |
| PRE-02 | Empty input | `""` / whitespace | error; `error_log` non-empty; no `validated_input` |
| PRE-03 | Missing / non-string input | `user_input` absent; dict payload | error |
| PRE-04 | Identifier screen | account-shaped code and long 4-4-6 reference number → node screen; e-mail and 4-4-4 digit group → framework | raw identifier absent from `validated_input`; `[REDACTED]` (node) / `[MASKED]` (framework) present |
| — | Redaction interaction | canonical payload (`"…under the Public Records Management Act?"`) | `[MASKED]` replaces the act name; surrounding text untouched |
| PRE-08 | Audit | valid query | `pre_process_complete` emitted; payload carries `input_chars` |
| REF-01 | Attack payloads refused | instruction override, prompt disclosure, role reassignment, chat-role markers — via **direct `execute()`** | error; no `validated_input`, no `caller_options` |
| REF-02 | Ordinary questions unaffected | domain questions reusing the same words ("system", "instructs", "above", "act as") | success; `validated_input` set |
| REF-03 | Context channel screened alike | attack payload moved into an option field | error; nothing carried forward |
| REF-04 | Refusal describes nothing | any attack payload | the message names neither the matched rule nor any part of the payload |
| OPT-01 | Non-finite / out-of-bounds `top_k` | `"NaN"`, `"Infinity"`, `"-Infinity"`, raw `nan`, raw `inf`, `True`, `0`, `21`, `2.5`, `"two"` | error naming `top_k`; request not run |
| OPT-02 | Non-finite / out-of-bounds `score_threshold` | same matrix plus `1.5`, `-0.1` | error naming `score_threshold` |
| OPT-03 | Unrecognised `category` | `"bogus"`, `"ACT; DROP TABLE"`, `""`, 40 chars, an int | error naming `category` |
| OPT-04 | `channel` must be inert | markup, spaces, hyphen, 33 chars, an int, empty | error naming `channel` |
| OPT-05 | Case and padding normalised | `"Portal"`, `"  ACT "` | accepted; stored lowercase and trimmed |
| OPT-06 | Rejected value never echoed | a digit run as `top_k` | the value is absent from `error_log` |
| OPT-07 | Oversized context mapping | 40 keys | error naming `input_context` |
| OPT-08 | Valid options carried | category + top_k + score_threshold | `caller_options` is a JSON string with exactly those values |
| OPT-09 | Unknown keys ignored | an unrecognised envelope key | success — the platform may add keys of its own |
| OPT-10 | Absent options | no context | `caller_options` is empty; configured tuning stands |

### 2.2 InputValidateNode (inner node 1) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text query | whole string becomes `search_query`; no options set |
| VAL-02 | Whitespace | ragged spacing / newlines | collapsed to single spaces |
| VAL-03 | Request envelope | `{"query","category","top_k"}` | all parsed; `question` alias accepted; category lower-cased and trimmed |
| VAL-04 | Malformed JSON | `{`-prefixed non-JSON | treated as a plain-text query, plus a parse note |
| VAL-05 | Out-of-bounds `top_k` | full non-finite matrix plus `99`, `-5`, `0`, `2.5`, `True` | error naming `top_k`; no `search_query` produced |
| VAL-06 | Out-of-bounds `score_threshold` | full non-finite matrix plus `5.0`, `-0.5`, `True` | error naming `score_threshold` |
| VAL-07 | Unrecognised `category` | `"bogus"`, injection-shaped, empty, oversize, an int | error naming `category` |
| VAL-08 | Oversize query | > 2000 chars | truncated to 2000, plus a note |
| VAL-09 | Empty request | `""` | `search_query=""` plus an "empty request" note (non-fatal) |
| VAL-10 | Envelope bounds equal the context bounds | same values on both channels | the same rejection either way — neither channel is the lenient one |
| VAL-11 | Context channel wins | category on both channels | the context value applies |
| VAL-12 | Non-option context keys | `channel` | not carried into `query_filters` |
| — | Serialisation | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 RetrieveNode (inner node 2) — `test_retrieve_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Happy path | retention / documents query | top candidate is `gov-001` |
| RET-02 | Ordering | retention / documents query | scores strictly descending; all above zero |
| RET-03 | Entry shape | any hit | keys `{id,title,category,source,score,excerpt}`; excerpt ≤ 400 chars |
| RET-04 | Category filter | `category="administrative_guidance"` | only that category; top candidate `gov-009` |
| RET-05 | Empty query | `""` | no candidates |
| RET-06 | Corpus unreadable | bogus `kb_path` in `retrieval_config` | empty result plus a "not readable" note |
| RET-07 | `top_k` from tuning | `retrieval_config.top_k=1` | still returns ranked candidates; top candidate `gov-001` |
| RET-08 | Notes accumulate | prior `intake_notes` | appended, never clobbered |

### 2.4 RerankFilterNode (inner node 3) — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | Relevance floor | scores 0.9 / 0.1 | 0.1 dropped (0.25 floor) |
| RRF-02 | Floor from tuning | `score_threshold=0.5` | 0.3 dropped |
| RRF-03 | Cap from tuning | `top_k=1` | one survivor, highest score |
| RRF-04 | Category boost | matching category | +0.1, re-ranked ahead |
| RRF-05 | Boost cap | 0.95 plus boost | capped at 1.0 |
| RRF-06 | Caller `top_k` | stricter wins; looser does not widen | enforced |
| RRF-07 | Garbage entries | non-dict; uncoercible score | skipped, or coerced to 0.0 and dropped |
| RRF-08 | Tie-break | equal scores | deterministic id-ascending order |

### 2.5 GenerateAnswerNode (inner node 4) — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Citation markers | 2 ranked passages | `[1]` / `[2]` markers with titles |
| GEN-02 | Lead sentence | query present | the question is quoted in the lead |
| GEN-03 | Citation list | ranked passages | refs 1..n mirror ranked order; id / title / source carried |
| GEN-04 | Groundedness | single passage | the answer body traces to ranked passages only |
| GEN-05 | No coverage | empty or missing `ranked_documents` | escalation answer; no citations |

### 2.6 OutputFormatNode (inner node 5, terminal) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body plus citations | header, body, `## Sources` rows, disclaimer; success |
| FMT-02 | Source suffix | citation with a source | rendered once, no empty `()` |
| FMT-03 | Disclaimer | every input | the disclaimer rides with every answer |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing body | no `grounded_answer` | fallback text; success |
| GND-01 | Marker with no citation | `[7]` in the body, one citation | answer withheld; `ungrounded_citation_marker` |
| GND-02 | Source never retrieved | citation id absent from `ranked_documents` | answer withheld; `citation_source_was_not_retrieved` |
| GND-03 | Unattributed citation | blank `source` | answer withheld; `citation_without_source` |
| GND-04 | Untitled citation | blank `title` | answer withheld; `citation_without_title` |
| GND-05 | Citations with nothing retrieved | citations present, `ranked_documents` empty | answer withheld |
| GND-06 | Withheld answer is non-empty | any grounding violation | `formatted_answer` holds the withheld placeholder and `citations` is cleared — refusing by omitting the field would leave it falsy, and a falsy answer reads downstream as "look elsewhere" rather than "refused" |
| GND-07 | Refusal names the rule only | violation over a body carrying a quotable sentence | the returned delta reproduces no part of the refused answer |

### 2.7 PostProcessNode (outer post_process slot) — `test_post_process_node.py`

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | a complete rendered answer | `formatted_output=result`; success |
| POST-02 | Empty result | `""` | forwarded as-is; success (non-fatal) |
| POST-03..06 | Credential leak | `sk-` key, `password=` assignment, token (built at runtime), Bearer token | error; EVERY output-bearing field replaced by the blocked placeholder and `citations` cleared; the secret absent from the whole delta and from the resolved envelope |
| POST-07 | Identifiers byte-identical | act number, article reference, corpus id, year, citation marker | output equals input exactly — this boundary refuses, it never rewrites |
| POST-08 | Missing disclaimer | disclaimer stripped | answer withheld; `missing_legal_disclaimer` |
| POST-09 | Forged citation marker | marker with no citation record | answer withheld; `ungrounded_citation_marker` |
| POST-10 | Unattributed citation | blank `source` | answer withheld; `citation_without_source` |
| POST-11 | Layers independent | credential AND invariant breach together | the credential layer wins — it is the one that rewrites |
| POST-12 | Withheld answer is non-empty | any invariant breach | every output-bearing field holds the withheld placeholder and `citations` is cleared. Asserted against the RESOLVED envelope (`formatted_output` OR `result`), because blanking one field while a populated sibling remains releases the refused answer through the fallback |
| POST-13 | Breach does not reach the caller | breaching sentence in the answer | the sentence appears nowhere in the resolved envelope, nor anywhere in the returned delta |
| POST-14 | Already-refused runs bypass this gate | error status on the incoming state | `execute()` does not run — containment for those runs belongs at the envelope, and this pins the framework behaviour that makes a local branch unnecessary |
| — | Recursive scan | credential nested in a dict or list | detected at any depth — see `test_trust_and_output_gates.py::TestOutputGateNestedScan` |

### 2.8 Config consistency — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Identity | `id` = `GOV-C2-005`; `namespace` = `gov` |
| CFG-02 | Entry point | `class` = `src.graph.graph.GovernmentRegulationKnowledgeAgent`, resolving to the graph class the server imports |
| CFG-03 | Classification | Cat 2 / GOV / RAGAgent / enabled |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` equals both outer nodes' `required_trust_level` |
| CFG-05 | `max_retry` | int, `0 ≤ v < 10` (framework ceiling); human-in-the-loop not enabled |
| CFG-06 | Retrieval tuning | `top_k` / `score_threshold` mirror the node module defaults; `kb_path` exists |
| CFG-07 | `_parent_config()` | forwards the `retrieval` block from `config/config.yaml`; never empty, even when the file is unreadable |
| CFG-08 | Runtime values reach the agent | the constructed agent carries `max_retry` and `timeout_s` |
| CFG-09 | No tuning in the registry entry | `retrieval`, `config`, `llm`, `agent` are all absent from `config/agent.yaml` |
| CFG-10 | Declarations are code-derived | `requires.secrets` and `requires.extras` are empty; `generation_mode` is `deterministic` |
| — | Corpus integrity | JSON list of ≥ 5 entries; unique ids; required keys per entry; categories within the accepted set |

### 2.9 Retrieval quality (golden queries) — `test_retrieval_quality.py`

| ID | Case | Expected |
|----|------|----------|
| QUAL-01 | 12 golden queries, one per corpus entry | the expected entry ranks first |
| QUAL-02 | Relevance floor | every survivor at or above 0.25 |
| QUAL-03 | Citation integrity | every survivor id exists in the corpus |
| QUAL-04 | Precision | the national-security exemption query keeps only `gov-004` |
| QUAL-05 | Category filter | `cabinet_order` keeps only `gov-008` |
| QUAL-06 | No coverage | an out-of-domain query yields zero survivors |
| QUAL-07 | Escalation answer | no coverage yields explicit escalation text and no citations |

## 3. Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the five domain nodes; no initialize or finalize |
| INT-02 | State seeding | `_extra_initial_state()` seeds `retrieval_config` **and** `caller_options`, both as JSON strings |
| INT-02b | Bridge pickup | options published on the bridge arrive in inner state; with no publish, the mapping is empty and defaults apply |
| INT-03 | Output shape | `get_output()` emits the merge contract; `route()` returns END on error |
| INT-04 | Inner end-to-end | a full inner `invoke()` succeeds: formatted answer, disclaimer, `gov-001` citation, inner `node_history` in linear order |

### 3.2 Outer graph — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` directly; `Graph` alias present; `add_edges()` not overridden |
| INT-06 | Backbone slots | `compile()` fills all five; pre / main / post are the expected classes |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded tuning |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` maps to both `regulation_answer` and `result`; citations and status mapped; changed keys only |
| INT-10 | Tuning source | `_parent_config()` matches `config/config.yaml`; with the file unreadable it degrades to the shipped tuning, never to an empty block |
| INT-11 | End-to-end happy path | a VERIFIED_EXTERNAL invoke succeeds; `output` is the gated answer; PostProcessNode traversed |
| INT-12 | End-to-end trust denial | an ANONYMOUS invoke errors; empty `output`; PostProcessNode not traversed; `citations` absent from the envelope |
| — | Structured citations | surfaced only on success, absent on error |
| CNT-01 | `merge_output()` withholds | inner run did not succeed | every outer output field gets the withheld placeholder and `citations` is cleared, rather than mapping an absent answer through as `None` |
| CNT-02 | Non-success envelope | answer text left in `result` under an error status | the envelope surfaces the placeholder, never the text — a refusal that ships the answer is decoration |
| CNT-03 | Placeholder is preserved | a boundary already wrote its own placeholder | the more specific wording survives the envelope check |
| CNT-04 | Nothing produced stays nothing | error status, no output fields written | the envelope reports no output; it does not invent one |
| CNT-05 | Citation-gate violations | credential-shaped citation field / malformed citation on an otherwise successful run | status flips to error, `citations` withheld, AND the rendered answer withheld — the answer and its citations are one product |
| CNT-06 | Clean control | a well-formed successful run | the answer and the citation list are surfaced unchanged; the containment checks cost a good answer nothing |
| — | State helpers | `to_json` / `from_json` round-trip; None and malformed handling |

## 4. Boundary proofs

| ID | File | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no direct platform-SDK import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no model-class or context annotations |
| PB-6 | `test_pb_invoke_order.py` | a full `Graph().invoke()` at `VERIFIED_EXTERNAL` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json` succeeds, with outer `node_history` exactly `[InitializeNode, PreProcessNode, RegulationKnowledgeGraphNode, PostProcessNode, FinalizeNode]`. Also pins the observed redaction behaviour for that payload (see §1) |
| PB-5 | `test_state_safety.py` | **Auto-waived — checkpointing disabled** (`config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`); the conditional gate and the non-lossy traversal helper ship with the stub |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **auto-waived** — this template declares no human-in-the-loop step, so the conditional skip stub is retained and its skip must not block the gate |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; the module-level agent is this template's class, compiled; a fresh constructor plus `compile()` fills the five slots; `/health` reports the agent |
| PB-E2E | `test_invoke_e2e.py` | the real ASGI app with Bearer auth: a grounded cited answer, the no-coverage success path, the disclaimer on every answer, caller options reaching the inner graph (category narrows, `top_k` caps, a stricter floor narrows, a looser floor cannot widen), the request envelope still working, every out-of-bounds option rejected including raw `NaN` / `Infinity` JSON literals, oversized context rejected at the adapter, hostile payloads refused with nothing published, identifiers byte-identical, and caller text unable to forge a citation marker or a Sources section |
| PB-CONTAIN | `test_invoke_e2e.py::TestOutputContainmentEndToEnd` | a refused answer does not reach the caller, measured at the envelope rather than at the node. Both output-boundary layers are reached through the real request path from the seeded corpus — the citation invariant via a `source` string carrying a bracketed number (rendered only into the Sources block, so the inner check passes it and the outer one catches it), the credential layer via a credential-shaped passage. In both cases the envelope carries a non-empty withheld placeholder, no answer text, no traceback and no source paths, and no citation list. A clean control over the same corpus still produces a real cited answer |

> **Gate checklist:** PB-IMPORT, PB-STATE, PB-6, PB-BOOT and PB-E2E are
> mandatory. PB-7 applies only to templates with a human-in-the-loop step; this
> one has none.

## 5. Test execution summary

- Runner: the real framework wheel (`agenticstar-agentcore[anthropic]==1.0.1`),
  the same version CI installs.
- `tests/` (full tree): **327 passed, 1 skipped** (PB-7, auto-waived).
- `tests/proof_of_boundary/` (boundary subset): **65 passed, 1 skipped**.
- Determinism: no model call, no network. Retrieval and answer assembly are
  rule-based, so a given question always produces the same passages in the same
  order.

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `GovernmentRegulationKnowledgeAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.
