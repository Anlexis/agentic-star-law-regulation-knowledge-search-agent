"""AgentCore Platform v1.0"""

# Service layer: domain queries, external API wrappers, data aggregation.
# Must NOT contain business logic, routing, or credentials.
# Nodes call this; this calls the shared service layer for external integrations.
#
# Nothing in the shipped build calls it. Retrieval grounds on the seeded corpus
# file that RetrieveNode reads (src/nodes/retrieve_node.py), so there is no
# external statutory-database call to wrap. The module is kept because it is the
# seam a deployment replaces: implementing fetch() here lets RetrieveNode source
# its passages from a live corpus behind the same node contract, with no change
# to the pipeline, the state contract, or the output boundary.

from __future__ import annotations

from typing import Any


class Service:
    """Domain service — the seam described in the module note above."""

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch domain data for the given query.

        Deliberately unimplemented in the shipped build. Raising keeps the
        unimplemented contract explicit rather than silently returning empty
        data, which would look like a corpus with no coverage.
        """
        raise NotImplementedError("Service.fetch() is a deliberate stub in the shipped build")
