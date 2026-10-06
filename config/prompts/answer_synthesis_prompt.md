# Answer Synthesis Prompt — GOV-C2-005

> **Nothing reads this file at runtime.** `GenerateAnswerNode` assembles its
> answer by rule from `ranked_documents`; no model is called. This document is
> the contract a model-based replacement would have to satisfy, written down so
> that swapping the assembly changes only the inside of
> `GenerateAnswerNode.execute()` and nothing else. The seam is described in
> `docs/02_design.md`.

## Contract (a model-based GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) and `search_query` the v1 node reads.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers) and `citations` (JSON list of
  `{ref, id, title, source}`).
- **Grounding rule:** every statutory provision in the answer must be
  traceable to one of the supplied passages via a `[n]` marker; content not
  present in the passages must not be asserted. This is not advisory — the
  output boundary enforces it (`src/output_invariants.py`) and withholds an
  answer that cites a marker or a source it cannot trace.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend refining the query or escalating to the legal/compliance team —
  never answer from parametric knowledge.
- **Tone:** neutral, precise, no individualized legal recommendations (the
  mandatory legal-advice disclaimer is appended downstream by
  `OutputFormatNode`).

## Prompt template

```
You answer government law and regulation questions strictly from the
knowledge-base passages provided below.

Question:
{search_query}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   knowledge base has insufficient coverage and stop.
2. Mark every statutory statement with the [n] reference of its passage.
3. Do not give individualized legal advice or a formal regulatory determination.
4. Keep the answer under 300 words.
```

## Manifest coupling

The `llm` block in `config/agent.yaml` (`temperature`, `max_tokens`) is already
forwarded to the inner graph via
`RegulationKnowledgeGraphNode._parent_config()` under
`config["configurable"]["llm"]`; the v2 node reads it from there.
