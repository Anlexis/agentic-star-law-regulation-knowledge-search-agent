# Law & Regulation Knowledge Search Agent

AI agent for searching law and regulation knowledge bases, built with Agentic Star.

> **Category**: Cat 2 (domain-specific retrieval pipeline)
> **Industry**: Government
> **Template ID**: GOV-C2-005

## Overview

Answers questions about government law and regulation from a knowledge base of acts, cabinet
orders, ministerial ordinances and administrative guidance, and cites the provisions it used.

Ask it something like *"what retention period applies to administrative documents before
disposal?"* and it returns the relevant provisions with their exact statutory citation — act name,
act number, article — plus a standing notice that the answer is not legal advice. The alternative
it replaces is keyword search across a statutory corpus, read by hand: slow, dependent on legal
literacy, and easy to get subtly wrong.

Two properties are enforced rather than assumed, because in this domain a confidently-worded
answer nobody can trace is worse than no answer:

- **Every provision is traceable.** The answer is assembled only from passages retrieval actually
  returned, and the output boundary re-checks that — a citation marker with nothing behind it, or
  a citation naming a source that was never retrieved, withholds the answer instead of shipping
  it.
- **Nothing rewrites the text.** Act numbers, article references and years reach the caller
  byte-for-byte.

Retrieval is deterministic: no model call and no network request, so the same question always
returns the same passages. The bundled corpus is a small representative seed set meant to be
replaced with your own.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >= 3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the framework version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/graph/          outer agent, inner workflow graph, caller-context bridge
src/nodes/          the five pipeline steps
src/caller_contract.py    bounds every value a caller can supply
src/output_invariants.py  what every answer must satisfy before it ships
src/schemas/state.py      the state passed between nodes
src/api/server.py         standalone HTTP entry point
config/agent.yaml         registry entry (identity, entry point, trust level)
config/config.yaml        runtime parameters and retrieval tuning
config/kb/                the seeded corpus
tests/                    unit, boundary and end-to-end tests
docs/                     design and test specification
```

See [`docs/02_design.md`](docs/02_design.md) for the architecture and the caller contract, and
[`docs/03_test_spec.md`](docs/03_test_spec.md) for the test specification.

## Customising

1. Replace `config/kb/gov_regulations_kb.json` with your own corpus. Each entry needs an `id`,
   `title`, `category`, `source`, `tags` and `content`; `category` must be one of the values in
   `src/caller_contract.py`.
2. Tune retrieval in `config/config.yaml` (`top_k`, `score_threshold`).
3. Adjust the disclaimer and the answer layout in `src/output_invariants.py` and
   `src/nodes/output_format_node.py`.
4. Re-run the test suite — the boundary tests will tell you if a change broke the grounding
   contract.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
