# Copyright (C) 2026  Edin Jelacic — AGPL-3.0-or-later
"""Neo4j graph database client for industrial knowledge graphs.

Manages connections, entity upserts, relationship creation, and graph queries
for the GraphRAG pipeline. All operations are idempotent — re-ingesting
documents won't duplicate nodes or edges.

Usage
-----
    from neo4j_client import Neo4jClient, init_graph_db

    client = init_graph_db()

    # Upsert entities and relationships from extraction results
    client.merge_entities(entities)       # list[ExtractedEntity]
    client.merge_relationships(relationships)  # list[ExtractedRelationship]

    # Query the graph
    nodes = client.get_nodes_by_type("machine")
    paths = client.multi_hop_traversal(start="Coolant Pump CP-101", depth=3)

    client.close()
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from neo4j import Driver, GraphDatabase

logger = logging.getLogger("industryorch-neo4j")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "industryorch2026")

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

_driver: Driver | None = None


def init_graph_db(
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
) -> Neo4jClient:
    """Initialize the Neo4j connection and ensure schema exists.

    Creates indexes for fast lookups on entity names and types.
    Idempotent — safe to call multiple times.
    """
    global _driver
    _driver = GraphDatabase.driver(uri, auth=(user, password))

    with _driver.session() as session:
        result = session.run("RETURN 1 AS test")
        assert result.single()["test"] == 1, "Neo4j connection failed"

    _ensure_schema()

    logger.info("Neo4j connected at %s", uri)
    return Neo4jClient(_driver)


def get_graph_db() -> Neo4jClient:
    """Get the global Neo4j client instance."""
    if _driver is None:
        raise RuntimeError(
            "Neo4j client not initialized. Call init_graph_db() first."
        )
    return Neo4jClient(_driver)


def close_graph_db() -> None:
    """Close the Neo4j connection."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j connection closed.")


def _ensure_schema() -> None:
    """Create indexes and constraints for optimal graph operations."""
    with _driver.session() as session:  # type: ignore[union-attr]
        session.run("""
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """)
        session.run("""
            CREATE INDEX entity_type_index IF NOT EXISTS
            FOR (e:Entity) ON (e.type)
        """)
        session.run("""
            CREATE INDEX entity_location_index IF NOT EXISTS
            FOR (e:Location) ON (e.name)
        """)


# ---------------------------------------------------------------------------
# Neo4jClient — all graph operations live here
# ---------------------------------------------------------------------------


class Neo4jClient:
    """Thin wrapper around neo4j.Driver for knowledge graph operations."""

    def __init__(self, driver: Driver):
        self._driver = driver

    def close(self) -> None:
        self._driver.close()

    # -----------------------------------------------------------------------
    # Merge operations (idempotent upserts)
    # -----------------------------------------------------------------------

    def merge_entity(
        self, name: str, etype: str, attributes: dict | None = None
    ) -> None:
        """Create or update a single entity. Uses MERGE for idempotency."""
        attrs = attributes or {}
        attrs_json = json.dumps(attrs, default=str) if attrs else "{}"
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.attributes = $attributes,
                    e.updated_at = datetime()
                """,
                name=name,
                type=etype,
                attributes=attrs_json,
            )

    def merge_entities(self, entities: list[Any]) -> int:
        """Bulk upsert entities. Returns count of entities processed."""
        if not entities:
            return 0

        items = []
        for ent in entities:
            d = (
                asdict(ent) if hasattr(ent, "__dataclass_fields__") else dict(ent)
            )
            items.append({
                "name": str(d["name"]).strip(),
                "type": str(d["type"]).strip().lower(),
                "attributes": json.dumps(
                    d.get("attributes", {}) or {}, default=str
                ),
            })

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $items AS item
                MERGE (e:Entity {name: item.name})
                SET e.type = item.type,
                    e.attributes = item.attributes,
                    e.updated_at = datetime()
                """,
                items=items,
            )

        logger.info("Upserted %d entities into Neo4j", len(items))
        type_counts = {}
        for item in items:
            type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
        logger.info("  Entity types stored: %s", type_counts)
        return len(items)

    def merge_relationship(
        self,
        source: str,
        target: str,
        rtype: str,
        attributes: dict | None = None,
    ) -> None:
        """Create or update a relationship between two entities.

        Creates the entity nodes if they don't exist yet (auto-merge).
        Idempotent — won't duplicate relationships of the same type.
        """
        attrs = attributes or {}
        attrs_json = json.dumps(attrs, default=str) if attrs else "{}"
        safe_type = rtype.replace("'", "").replace('"', "")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", safe_type):
            logger.warning("Invalid relationship type '%s', skipping", rtype)
            return

        with self._driver.session() as session:
            session.run(
                f"""
                MERGE (src:Entity {{name: $source}})
                MERGE (tgt:Entity {{name: $target}})
                MERGE (src)-[r:{safe_type}]->(tgt)
                SET r.attributes = $attributes,
                    r.updated_at = datetime()
                """,
                source=source,
                target=target,
                attributes=attrs_json,
            )

    def merge_relationships(self, relationships: list[Any]) -> int:
        """Bulk upsert relationships. Returns count processed.

        Groups items by relationship type and uses UNWIND for each group so
        the total number of round-trips equals the number of distinct types
        rather than the number of relationships.
        """
        if not relationships:
            return 0

        # Bucket items by type — Cypher requires literal relationship types
        by_type: dict[str, list[dict]] = defaultdict(list)
        for rel in relationships:
            d = (
                asdict(rel) if hasattr(rel, "__dataclass_fields__") else dict(rel)
            )
            safe_type = (
                str(d["type"]).strip().lower()
                .replace("'", "").replace('"', "")
            )
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", safe_type):
                logger.debug(
                    "Invalid relationship type '%s', skipping", d["type"]
                )
                continue
            by_type[safe_type].append({
                "source": str(d["source"]).strip(),
                "target": str(d["target"]).strip(),
                "attributes": json.dumps(
                    d.get("attributes", {}) or {}, default=str
                ),
            })

        total = sum(len(v) for v in by_type.values())
        with self._driver.session() as session:
            for rel_type, batch in by_type.items():
                session.run(
                    f"""
                    UNWIND $items AS item
                    MERGE (src:Entity {{name: item.source}})
                    MERGE (tgt:Entity {{name: item.target}})
                    MERGE (src)-[r:{rel_type}]->(tgt)
                    SET r.attributes = item.attributes,
                        r.updated_at = datetime()
                    """,
                    items=batch,
                )

        logger.info(
            "Upserted %d relationships (%d type(s)) into Neo4j",
            total,
            len(by_type),
        )
        return total

    # -----------------------------------------------------------------------
    # Query operations
    # -----------------------------------------------------------------------

    def get_nodes_by_type(self, etype: str, limit: int = 100) -> list[dict]:
        """Get all entities of a given type."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {type: $type})
                RETURN e.name AS name, e.type AS type,
                       e.attributes AS attributes
                ORDER BY e.name
                LIMIT $limit
                """,
                type=etype,
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_entity(self, name: str) -> dict | None:
        """Get a single entity by name."""
        with self._driver.session() as session:
            record = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e",
                name=name,
            ).single()
            if record is None:
                return None
            e = record["e"]
            raw = e.get("attributes", "{}")
            attrs = json.loads(raw) if raw else {}
            return {
                "name": e.get("name"),
                "type": e.get("type"),
                "attributes": attrs,
            }

    def get_entity_connections(
        self, name: str, depth: int = 2, max_results: int = 50
    ) -> dict:
        """Get all entities connected to a given entity (multi-hop).

        Returns a structured graph for visualization:
        {"center": {...}, "neighbors": [...], "edges": [...]}
        """
        with self._driver.session() as session:
            query = f"""
                MATCH path = (center:Entity {{name: $name}})
                             -[r*1..{depth}]-(neighbor)
                RETURN path
                LIMIT $max_results
            """
            result = session.run(query, name=name, max_results=max_results)

            center = None
            neighbors: dict = {}
            edges: list = []

            for record in result:
                path = record["path"]
                nodes = list(path.nodes)
                rels = list(path.relationships)

                c = nodes[0]
                if center is None:
                    raw = c.get("attributes", "{}")
                    center = {
                        "name": c.get("name"),
                        "type": c.get("type"),
                        "attributes": json.loads(raw) if raw else {},
                    }

                for i, rel in enumerate(rels):
                    other = nodes[i + 1] if i + 1 < len(nodes) else None
                    if other is None:
                        continue
                    n_name = other.get("name")
                    n_type = other.get("type")
                    if n_name not in neighbors:
                        raw = other.get("attributes", "{}")
                        neighbors[n_name] = {
                            "name": n_name,
                            "type": n_type,
                            "attributes": json.loads(raw) if raw else {},
                        }
                    raw_rel = rel.get("attributes", "{}")
                    edges.append({
                        "source": nodes[i].get("name"),
                        "target": n_name,
                        "type": rel.get("type", ""),
                        "attributes": json.loads(raw_rel) if raw_rel else {},
                    })

            return {
                "center": center,
                "neighbors": list(neighbors.values()),
                "edges": edges,
            }

    def multi_hop_traversal(
        self, start: str, rel_type: str | None = None, depth: int = 3
    ) -> list[dict]:
        """Find all paths from a starting entity with optional rel filter.

        Useful for questions like "what does Coolant Pump CP-101 feed?" or
        "what is connected to valve PRV-5?"
        """
        if rel_type:
            safe_type = rel_type.replace("'", "").replace('"', "")
            rel_clause = f"-[r:{safe_type}*1..{depth}]-"
        else:
            rel_clause = f"-[r*1..{depth}]-"

        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (start:Entity {{name: $start}})
                             {rel_clause}(end)
                RETURN path
                ORDER BY length(path)
                LIMIT 50
                """,
                start=start,
            )

            paths = []
            for record in result:
                path = record["path"]
                nodes = [n.get("name") for n in path.nodes]
                rels = [r.get("type", "") for r in path.relationships]
                paths.append({
                    "nodes": nodes,
                    "relationships": rels,
                    "length": len(nodes) - 1,
                })

            return paths

    def community_detection(self) -> list[dict]:
        """Run Louvain community detection on the knowledge graph.

        Returns communities (clusters of related entities). Useful for
        answering global questions across the document collection.

        Requires Neo4j Graph Data Science library.
        """
        with self._driver.session() as session:
            try:
                session.run("CALL gds.version()")
            except Exception:
                logger.warning(
                    "GDS library not available — skipping community detection"
                )
                return []

            result = session.run("""
                CALL gds.louvain.write({
                    nodeProjection: 'Entity',
                    relationshipProjection: {
                        ALL: {
                            type: '*',
                            orientation: 'UNDIRECTED'
                        }
                    },
                    writeProperty: 'community'
                })
            """)

            stats = result.single()
            return [{
                "communities": int(stats.get("communityCount", 0)),
                "modularity": float(stats.get("modularity", 0)),
            }]

    def get_relationship_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        with self._driver.session() as session:
            total_nodes = session.run(
                "MATCH (e:Entity) RETURN count(e) AS cnt"
            ).single()["cnt"]
            total_rels = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS cnt"
            ).single()["cnt"]

            by_type = session.run("""
                MATCH (e:Entity)
                RETURN e.type AS type, count(e) AS count
                ORDER BY count DESC
            """)
            type_stats = {r["type"]: r["count"] for r in by_type}

            rel_types = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
            """)
            rel_type_stats = {r["type"]: r["count"] for r in rel_types}

            return {
                "total_entities": total_nodes,
                "total_relationships": total_rels,
                "by_entity_type": type_stats,
                "by_relationship_type": rel_type_stats,
            }

    def clear_graph(self) -> None:
        """Delete all nodes and relationships. Use with caution."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared — all nodes and relationships deleted.")
