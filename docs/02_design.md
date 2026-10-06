# Template Design Specification

## Position in the framework architecture

- **Agent class**: `GovernmentRegulationKnowledgeAgent`

| Role | Class |
|---|---|
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph | `DomainWorkflowGraph` (`BaseGraph`, fully custom topology) |
| `main`-slot wrapper | `RegulationKnowledgeGraphNode` (`GraphNode`) |

This is the **nested** shape: a `GraphNode` occupies the outer `main` slot and
wraps an inner `BaseGraph` that runs the five-node retrieval pipeline. Domain
complexity lives entirely inside the inner graph; the outer backbone stays a
thin, uniform shell and is never modified.

**Three-layer separation**

- **State** — flat `TypedDict` composition, never a model class: graph
  checkpoints are serialised with msgpack, and model objects corrupt silently
  there. Structured fields (dict / list) are stored as JSON **strings**
  (`to_json()` / `from_json()`), never as bare containers in a checkpointed
  field.
- **Node** — framework inheritance with a single override:
  `execute(self, state) -> dict`. There is **no** `config` parameter, so
  configuration reaches the inner domain nodes exclusively through state fields
  seeded by `DomainWorkflowGraph._extra_initial_state()`.
- **Graph** — composition. The outer `register_nodes()` fills the fixed backbone
  slots; the inner graph composes the five domain nodes into a linear topology.

## Configuration: two files, two jobs

| File | Job | Read by |
|---|---|---|
| `config/agent.yaml` | Registry entry. Flat — every key at root level. Identity, entry point, trust level, and the compile-time `requires` declarations. **No runtime tuning.** | the platform registry |
| `config/config.yaml` | Runtime parameters: `max_retry`, `timeout_s`, and the `retrieval` tuning block. | `src/runtime_config.py`; each entry point calls it and passes the result as `Graph(config=...)` |

The split matters in one direction in particular. Tuning read from the wrong
file does not raise — it returns nothing, and the pipeline runs on its module
defaults while the deployment believes it configured something. So:
`RegulationKnowledgeGraphNode._parent_config()` receives the runtime config from
the outer graph and nothing else — the outer graph reads `self.config`, which the
entry point loaded from `config/config.yaml`, and threads it into the node at
`register_nodes()` time. `src/api/server.py` hands the same mapping to the graph
constructor (constructing with no config would leave `max_retry` inert), and
`tests/unit/test_config_manifest.py` asserts that the registry entry carries no
tuning for a future reader to pick up by mistake.

## Architecture overview

### Outer backbone (fixed five-node pipeline — never modified)

| Node | Responsibility | Input state | Output state | Trust level | Class |
|------|---------------|-------------|--------------|-------------|-------|
| initialize | schema_version, session_id, trust level | — | — | — | `InitializeNode` (framework default) |
| pre_process | the caller contract: trust, refusal, redaction, option bounds | `user_input`, `input_context` | `validated_input`, `caller_options`, `enriched_context` | **VERIFIED_EXTERNAL** | `PreProcessNode` |
| main | delegate to the inner workflow | `validated_input`, `caller_options` | `regulation_answer`, `result`, `citations`, `status` | n/a (`GraphNode`) | `RegulationKnowledgeGraphNode` |
| post_process | the output boundary: credential redaction, then the grounding and disclaimer contract | `result`, `citations` | `formatted_output`, `status` — on a refusal also `result`, `regulation_answer` and `citations`, so no output-bearing field is left holding the refused answer | **VERIFIED_EXTERNAL** | `PostProcessNode` |
| finalize | response metadata, total time | — | — | — | `FinalizeNode` (framework default) |

### Inner workflow (`DomainWorkflowGraph`)

| Node | Responsibility | Input state | Output state | Trust level |
|------|---------------|-------------|--------------|-------------|
| input_validate | settle the query: plain text or a `{query, category, top_k, score_threshold}` envelope, reconciled with the caller options that arrived through the context bridge; normalise whitespace; cap length | `validated_input`, `caller_options` | `search_query`, `query_filters`, `intake_notes` | **ANONYMOUS** |
| retrieve | keyword search over the seeded corpus (`config/kb/gov_regulations_kb.json`) | `search_query`, `query_filters`, `retrieval_config` | `retrieved_documents`, `intake_notes` | **ANONYMOUS** |
| rerank_filter | category-match boost, relevance floor, passage cap | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` | **ANONYMOUS** |
| generate_answer | rule-based grounded assembly, one numbered `[n]` citation per surviving passage | `ranked_documents`, `search_query` | `grounded_answer`, `citations` | **ANONYMOUS** |
| output_format | compose the answer: body, Sources list, legal-advice disclaimer — after checking the citations against the retrieved set | `grounded_answer`, `citations`, `ranked_documents` | `formatted_answer`, `status` | **ANONYMOUS** |

All five inner nodes declare `required_trust_level = TrustLevel.ANONYMOUS`: the
external boundary lives on the outer `pre_process` / `post_process` slots, and
duplicating it inside would only make the real boundary harder to find.

### Data flow

```
START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
                                        |  (retry)
                                        -> pre_process
```

The `main` slot delegates to the inner workflow:

```
START -> input_validate -> retrieve -> rerank_filter -> generate_answer -> output_format -> END
```

### The caller-context bridge

The framework calls `subgraph.invoke(user_input, session_id=..., ctx=...)`,
which carries the question text but **not** the caller's `input_context`
mapping. Left alone, the inner graph starts with an empty context and every
caller option falls back to its default — a caller asking for cabinet orders
only, or for a single most-relevant passage, would silently receive the default
unfiltered answer with no way to tell the difference.

`src/graph/context_bridge.py` closes the gap with a `ContextVar`:
`RegulationKnowledgeGraphNode.extract_input()` publishes the already-validated
options, and `DomainWorkflowGraph._extra_initial_state()` reads them back. Both
run on the same call stack inside one `GraphNode` invocation, so concurrent
invocations each see their own value. Only validated values travel — the bridge
is a transport, never a second validator.

This is proven end to end, not at node level, in
`tests/proof_of_boundary/test_invoke_e2e.py`.

## The caller-data contract

`src/caller_contract.py` is the one place where caller input becomes something
the pipeline may act on. `PreProcessNode` owns it; nothing else re-derives these
rules.

| Field | Channel | Bounds |
|---|---|---|
| `category` | context mapping or request envelope | one of `act`, `cabinet_order`, `ministerial_ordinance`, `administrative_guidance` |
| `top_k` | context mapping or request envelope | finite integer, 1–20 |
| `score_threshold` | context mapping or request envelope | finite number, 0.0–1.0 |
| `channel` | context mapping | inert identifier, `[a-z0-9_]{1,32}` |

Three rules hold for all of them.

- **Finite and bounded.** Numbers are checked for finiteness *before* the range
  check. `float("nan")` and `float("inf")` parse cleanly, and they arrive
  verbatim through JSON request bodies as the bare literals `NaN` and
  `Infinity`. Every comparison against NaN evaluates False, so a NaN relevance
  floor would keep every passage rather than raise — a silent failure on exactly
  the decision this template exists to make. Booleans are rejected explicitly,
  since `True` is an `int` in Python and would otherwise pass as 1.
- **Inert.** Strings that survive into state are restricted to a short lowercase
  identifier. No free text from the context channel reaches the rendered answer.
- **Fail closed.** A present-but-out-of-bounds value **rejects the request**,
  naming the field. It is not clamped and not dropped: a substituted value
  returns an answer the caller did not ask for and cannot distinguish from the
  one it did. The rejected value itself is never echoed into an error, a log
  line, or an audit event.

**Completion is not the same as answering.** A run that ends with
`AgentStatus.SUCCESS` reports that the request was handled safely to a defined
end, not that an answer was produced. A value the caller can correct (an
out-of-contract parameter, an empty or over-long request) ends this way so the
caller receives the reason and can send a corrected request on the same
conversation; terminating instead would end the calling surface's turn and
surface only an exception type, leaving the reason reachable solely from the
audit trail. The reason travels as `error_code` in State, every later domain
node passes through without doing work once it is set, the structured output
fields are withheld, and `PostProcessNode` renders the reason as a static
caller-facing sentence.

Two classes keep terminating, and must not be folded into the above: content
the agent refuses outright (an instruction-override payload — re-sending a
reworded variant is not a correction), and a breach of a contract the caller
cannot influence.

A caller override applies only in the **stricter** direction — a smaller passage
cap, or a higher relevance floor. A caller may narrow what it receives; it
cannot talk the template into citing passages the deployment's own relevance
floor rejected.

### Refusal is the template's own

`PreProcessNode` refuses instruction-override, prompt-disclosure,
role-reassignment and chat-role-marker payloads itself, in both the question and
the context mapping. A template whose only defence is a platform gate returns a
normal successful answer wherever that gate is absent or configured differently.
The refusal tests therefore call `execute()` **directly**, with no framework
wrapper in front, and assert behaviour — an error status and nothing carried
forward — never a gate's wording.

The patterns are phrases, not keywords, and both directions are pinned: *"what
are the record management **system** requirements?"* and *"which ordinance
**instructs** organs on disposal?"* are ordinary questions in this domain and
must pass through untouched.

## The output boundary

### What is enforced here, and what is not

Some domains enforce a numeric rounding grid at their output boundary. **This
template has no such grid, because it renders no monetary aggregates** — its
output is statutory text, act titles and article references. Nothing rewrites
numbers in the answer, and that is deliberate: act numbers (`Act No. 66 of
2009`), article references (`Article 5`), corpus identifiers (`gov-001`) and
years reach the caller byte-for-byte. A boundary that rewrote numbers here would
corrupt the very citations the template exists to deliver.

What replaces the grid is this template's own stated promise: **every provision
is traceable to a passage retrieval actually returned, and every answer carries
the legal-advice disclaimer.** `src/output_invariants.py` defines those checks,
and they are enforced at two independent boundaries:

| Boundary | Sees | Catches |
|---|---|---|
| `OutputFormatNode` (inner, terminal) | the citations **and** the retrieved passage set | a citation naming a source retrieval never returned |
| `PostProcessNode` + `get_output()` (outer) | the rendered answer and the structured citation list | a citation marker with nothing behind it, an unattributed citation, a missing disclaimer |

Neither boundary rewrites the answer; each accepts or refuses it whole. That
keeps them order-independent with respect to the credential redaction layer,
which *does* rewrite — a check that mutated text could otherwise destroy the
very pattern the credential scan is looking for. A violation withholds the
answer entirely rather than shipping a partly-trustworthy document, which in
this domain is the worse outcome.

### What withholding actually requires

Returning an error status is not withholding. The caller-visible answer is
resolved as `formatted_output` **or**, when that is falsy, `result` — an OR
across two state fields. A boundary that refuses by blanking only the first
field has refused nothing: the blank is falsy, the OR falls through, and the
answer that was just refused is what the caller receives, wrapped in a status
that says otherwise.

So every refusal in this template writes a **non-empty** placeholder over *every*
field that can carry released text, and drops the structured citation list in
the same delta — half an answer released under an error status is still a
release. `src/output_invariants.py` owns both the placeholders and the field
sets, so the boundaries cannot drift apart one field at a time. The envelope
carries a final check of the same rule: a non-success run never surfaces answer
text, whatever an earlier pass may have left in those fields.

A refusal message names the boundary and the rule that was broken, never the
content that broke it. That restraint is structural: the framework's output gate
scans every value of the delta a node returns, so a message quoting a
credential-shaped match makes that gate raise on the refusal itself — and a node
that raises has its delta discarded, clearing included, leaving the un-cleared
answer in state for the fallback to ship. A refusal that quotes what it caught
destroys itself.

### Caller text in the rendered answer

The question is echoed into the answer's lead sentence, so caller text becomes
report content. `sanitize_echo()` collapses all whitespace to single spaces, so
a multi-line payload cannot forge a heading or a list item, and removes markdown
structure characters and bracketed numbers, so it cannot forge a citation marker
that no retrieved passage backs. Bracketed **non**-numbers are left alone — the
redaction marker the platform writes over detected personal data is one of them,
and it should stay legible.

### Credential redaction

`PostProcessNode.execute()` calls a module-level scan directly rather than
defining a `_extra_security_gate_*` instance method (the framework auto-wraps
those hooks). The scan is **recursive**: it walks dict values and list, tuple and
set elements at any depth, checking every string leaf, because a scan that only
looked at a top-level string would miss a violation nested inside a structured
payload — which is exactly the shape the citation list has. The same function is
reused by `GovernmentRegulationKnowledgeAgent.get_output()` as a second,
fail-closed pass over the structured `citations` field before it is surfaced.

## State definition

| Field | Type | Purpose |
|-------|------|---------|
| `validated_input` | `str` | redacted question (from `PreProcessNode`) |
| `caller_options` | `Optional[str]` (JSON) | validated caller options, carried inward by the bridge |
| `regulation_answer` | `str` | final answer, mapped from the inner graph by `merge_output()` |
| `search_query` | `str` | normalised question |
| `query_filters` | `Optional[str]` (JSON) | `{"category", "top_k", "score_threshold"}`, each present only if supplied |
| `retrieval_config` | `Optional[str]` (JSON) | retrieval tuning forwarded from `config/config.yaml` |
| `retrieved_documents` | `Optional[str]` (JSON) | scored candidates |
| `ranked_documents` | `Optional[str]` (JSON) | reranked, floor-filtered passages, capped at `top_k` |
| `grounded_answer` | `str` | answer body with `[n]` citation markers |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, title, source}]` — the structured product |
| `formatted_answer` | `str` | final rendered answer (inner terminal output) |
| `intake_notes` | `Optional[str]` (JSON) | parse notes accumulated during intake |

**State constraints**

- Flat `TypedDict` only — primitives and JSON-serialisable types.
- No credentials of any kind in state (it is checkpointed).
- No model classes, dataclasses or arbitrary Python objects (msgpack cannot
  serialise them).
- Structured fields travel as JSON **strings**, never bare containers.

## Framework utilisation

- **`InvocationContext`** — session, correlation and trust, wired by
  `src/api/server.py`.
- **Input gate** — the framework's default runs before every `execute()`. The
  template's own refusal and redaction run *inside* `execute()` as
  defence in depth, not as a replacement.
- **Output gate** — the framework's default runs after every `execute()`; the
  template's boundary layers run inside `PostProcessNode.execute()` and
  `get_output()`.
- **`emit_trace_event()`** — one domain event per node (`pre_process_complete`,
  `pre_process_refused`, `input_validate_complete`, `retrieve_complete`,
  `rerank_filter_complete`, `generate_answer_complete`,
  `output_format_complete`, `output_format_grounding_violation`,
  `post_process_complete`, `post_process_invariant_violation`). The framework
  already emits node start / complete / error, so node code never duplicates
  those.
- **Retry / timeout** — `max_retry` and `timeout_s` from `config/config.yaml`,
  passed to the graph constructor.

Neither `_extra_security_gate_input()` nor `_extra_security_gate_output()` is
defined as an instance method anywhere in this template.

### Composition pattern

- **Pattern**: `GraphNode` (subgraph).
- **Target**: `DomainWorkflowGraph`, instantiated by
  `RegulationKnowledgeGraphNode.get_subgraph()` with the runtime config.
- **Error propagation**: `propagate` — an inner-graph exception is re-raised as
  `SubgraphError`. No partial or best-effort answer is ever surfaced.

## Import isolation

- The template imports `framework/` and `shared/` only; it never imports the
  underlying platform SDK directly. Enforced by
  `tests/proof_of_boundary/test_import_isolation.py`.

## Deterministic answer assembly

`GenerateAnswerNode` assembles its answer by rule from `ranked_documents` — a
lead sentence plus one numbered citation per surviving passage. No model client
is constructed or called anywhere in this template, and `config/agent.yaml`
declares `generation_mode: deterministic` to say so. The same question always
returns the same passages in the same order.

Grounding therefore holds *by construction*: the node only ever emits a `[n]`
marker for a passage that survived retrieval and reranking, and returns a fixed
"insufficient coverage" message — never a fabricated citation — when nothing
clears the relevance floor. The output boundary re-checks it anyway, because
"grounded by construction" holds only while the construction is unchanged, and
the next person to edit that node will not have read this paragraph.

`config/prompts/answer_synthesis_prompt.md` writes down the contract a
model-based replacement would have to satisfy: the same `ranked_documents`
input, the same `grounded_answer` + `citations` output, the same grounding and
no-coverage rules. Swapping the assembly touches the inside of that one node and
nothing else.

## Design decision record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | `AgentBaseGraph` | `AutonomousBaseGraph` | **`AgentBaseGraph`** | A fixed multi-step retrieval pipeline, not an autonomous think / act / observe loop |
| Composition | standalone | `GraphNode` (subgraph) | **`GraphNode` (subgraph)** | The five-node pipeline is fully encapsulated in `DomainWorkflowGraph`; the outer backbone is never overridden |
| `get_output()` | inherit default | extend `super()` | **extend `super()`** | The citations are the caller-facing product, not a side note. Surfaced only on success, and only after the boundary checks pass — fail closed |
| Answer assembly | live model call | deterministic rule-based | **deterministic rule-based** | Grounding is verifiable and the output reproducible; the model seam is documented rather than built |
| Out-of-bounds caller option | clamp to the nearest valid value | reject the request | **reject the request** | A clamped value returns an answer the caller did not ask for and cannot distinguish from the one it did |
| Output invariant | numeric rounding grid | grounding + disclaimer contract | **grounding + disclaimer** | Nothing monetary is rendered; the promise this template makes is traceability, so that is what the boundary enforces — and numbers stay byte-identical |
| Corpus | live statutory-database integration | seeded representative corpus | **seeded corpus** (`config/kb/gov_regulations_kb.json`) | Corpus ingestion is an outstanding dependency. The shipped set is representative and non-authoritative, covering all four categories, so the pipeline shape, the boundaries and the state contract are all fully exercised |

## Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.
