import urllib
import json
import time
import hashlib
import logging
import signal
import sys
from datetime import datetime
from typing import TypedDict

import pandas as pd
from sqlalchemy import create_engine, text
from neo4j import GraphDatabase
import google.generativeai as genai
from langgraph.graph import StateGraph, END

# ===============================
# CHANGE ONLY THIS
# ===============================
DATABASE_NAME   = "DATABASE_NAME"
SQL_PASSWORD    = "YOUR_PASSWORD"
NEO4J_PASSWORD  = "YOUR_PASSWORD"
GEMINI_API_KEY  = "YOUR_API_KEY"

POLL_INTERVAL_SECONDS = 60   # how often to check for changes (change as needed)

# ===============================
# Logging Setup
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sync_pipeline.log")
    ]
)
log = logging.getLogger(__name__)

# ===============================
# Connections
# ===============================
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER=localhost,1433;"
    f"DATABASE={DATABASE_NAME};"
    f"UID=sa;"
    f"PWD={SQL_PASSWORD};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

neo4j_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", NEO4J_PASSWORD)
)

log.info("Connections established successfully.")

# ===============================
# State Definition
# ===============================
class AgentState(TypedDict):
    metadata: dict
    semantic_plan: dict
    execution_summary: dict
    validation_report: dict

# ===============================
# Hash Store  (in-memory snapshot)
# { table_name: { primary_key_value: row_hash } }
# ===============================
_snapshot: dict[str, dict] = {}
_semantic_plan_cache: dict = {}   # cache so we don't re-ask Gemini every poll

# ===============================
# Helpers
# ===============================
def get_connection():
    """Always return a fresh connection to avoid stale connections."""
    return engine.connect()


def hash_row(row: dict) -> str:
    stable = json.dumps(row, sort_keys=True, default=str)
    return hashlib.md5(stable.encode()).hexdigest()


def get_primary_keys(table_name: str, conn) -> list[str]:
    """Fetch primary key columns for a table from SQL Server."""
    query = text("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE OBJECTPROPERTY(
            OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
        AND TABLE_NAME = :table
    """)
    result = conn.execute(query, {"table": table_name})
    pks = [row[0] for row in result]
    return pks


def make_row_key(row: dict, pk_cols: list[str]) -> str:
    """Create a stable string key from primary key column values."""
    return "|".join(str(row[col]) for col in pk_cols)


def detect_changes(table_name: str, df: pd.DataFrame, pk_cols: list[str]) -> dict:
    """
    Compare current table data against the in-memory snapshot.
    Returns dict with: inserted, updated, deleted rows.
    """
    current_hashes = {}
    current_rows = {}

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        key = make_row_key(row_dict, pk_cols)
        current_hashes[key] = hash_row(row_dict)
        current_rows[key] = row_dict

    old_hashes = _snapshot.get(table_name, {})

    inserted = [current_rows[k] for k in current_hashes if k not in old_hashes]
    updated  = [current_rows[k] for k in current_hashes
                if k in old_hashes and current_hashes[k] != old_hashes[k]]
    deleted_keys = [k for k in old_hashes if k not in current_hashes]

    return {
        "inserted": inserted,
        "updated": updated,
        "deleted_keys": deleted_keys,
        "current_hashes": current_hashes,
        "current_rows": current_rows,
        "pk_cols": pk_cols
    }

# ===============================
# LangGraph Agent Nodes
# ===============================
def planner(state: AgentState):
    # Preserve the cached semantic plan so incremental_sync can access it
    return {
        "metadata": {},
        "semantic_plan": _semantic_plan_cache,
        "execution_summary": {},
        "validation_report": {}
    }


def metadata_extractor(state: AgentState):
    with get_connection() as conn:
        tables = pd.read_sql("""
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
        """, conn)

        columns = pd.read_sql("""
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
        """, conn)

        fks = pd.read_sql("""
            SELECT
                FK.TABLE_SCHEMA AS from_schema, FK.TABLE_NAME AS from_table,
                CU.COLUMN_NAME AS from_column,
                PK.TABLE_SCHEMA AS to_schema, PK.TABLE_NAME AS to_table,
                PT.COLUMN_NAME AS to_column
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS C
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS FK
                ON C.CONSTRAINT_NAME = FK.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS PK
                ON C.UNIQUE_CONSTRAINT_NAME = PK.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE CU
                ON C.CONSTRAINT_NAME = CU.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE PT
                ON C.UNIQUE_CONSTRAINT_NAME = PT.CONSTRAINT_NAME
        """, conn)

    return {
        "metadata": {
            "tables": tables.to_dict(),
            "columns": columns.to_dict(),
            "foreign_keys": fks.to_dict()
        }
    }


def semantic_engine(state: AgentState):
    global _semantic_plan_cache

    # Use cached plan if schema hasn't changed
    if _semantic_plan_cache:
        log.info("Using cached semantic plan (schema unchanged).")
        return {"semantic_plan": _semantic_plan_cache}

    log.info("Asking Gemini to design the knowledge graph plan...")
    prompt = f"""
You are a knowledge graph designer. Given this SQL metadata:
{json.dumps(state['metadata'], indent=2)}

Design a knowledge graph plan. Output JSON:
{{
    "nodes": [
        {{"label": "...", "source_table": "..."}}
    ],
    "relationships": [
        {{
            "from": "...",
            "to": "...",
            "type": "...",
            "from_column": "...",
            "to_column": "..."
        }}
    ]
}}
"""
    response = llm.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )
    plan = json.loads(response.text)
    _semantic_plan_cache = plan
    return {"semantic_plan": plan}


def initial_graph_builder(state: AgentState):
    """Full initial load — only runs on first startup."""
    plan = state["semantic_plan"]
    log.info("Building full graph from scratch (initial load)...")

    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for node in plan["nodes"]:
            table  = node["source_table"]
            label  = node["label"]
            log.info(f"  Loading table: {table} -> :{label}")
            with get_connection() as conn:
                df = pd.read_sql(f"SELECT * FROM {table}", conn)
                pk_cols = get_primary_keys(table, conn)

            current_hashes = {}
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                key = make_row_key(row_dict, pk_cols) if pk_cols else hash_row(row_dict)
                current_hashes[key] = hash_row(row_dict)
                props = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row_dict.items()}
                session.run(f"CREATE (n:{label} $props)", props=props)

            # Seed the snapshot
            _snapshot[table] = current_hashes

        for rel in plan["relationships"]:
            session.run(f"""
                MATCH (a:{rel['from']})
                MATCH (b:{rel['to']})
                WHERE a.{rel['from_column']} = b.{rel['to_column']}
                MERGE (a)-[:{rel['type']}]->(b)
            """)

    log.info("Initial graph build complete.")
    return {"execution_summary": {"status": "initial_load_complete"}}


def incremental_sync(state: AgentState):
    """Check each table for changes and apply only deltas to Neo4j."""
    # The planner resets state to {} on every cycle, so always use the cache
    plan = state.get("semantic_plan") or _semantic_plan_cache

    if not plan or "nodes" not in plan:
        log.error("No semantic plan available in state or cache. Skipping sync cycle.")
        return {"execution_summary": {"status": "error", "reason": "no_plan_available"}}

    total_inserted = total_updated = total_deleted = 0
    changed_tables = []

    for node in plan["nodes"]:
        table = node["source_table"]
        label = node["label"]

        with get_connection() as conn:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            pk_cols = get_primary_keys(table, conn)

        if not pk_cols:
            log.warning(f"Table '{table}' has no PK — skipping incremental sync.")
            continue

        changes = detect_changes(table, df, pk_cols)

        inserted     = changes["inserted"]
        updated      = changes["updated"]
        deleted_keys = changes["deleted_keys"]
        current_rows = changes["current_rows"]

        if not inserted and not updated and not deleted_keys:
            continue  # No change for this table

        changed_tables.append(table)
        log.info(f"Changes in '{table}': +{len(inserted)} inserted, "
                 f"~{len(updated)} updated, -{len(deleted_keys)} deleted")

        with neo4j_driver.session() as session:
            # INSERT new rows
            for row in inserted:
                props = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}
                session.run(f"CREATE (n:{label} $props)", props=props)

            # UPDATE changed rows
            for row in updated:
                pk_filter = " AND ".join(
                    [f"n.{col} = $pk_{col}" for col in pk_cols]
                )
                pk_params = {f"pk_{col}": row[col] for col in pk_cols}
                props = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}
                session.run(
                    f"MATCH (n:{label}) WHERE {pk_filter} SET n = $props",
                    {**pk_params, "props": props}
                )

            # DELETE removed rows
            for key in deleted_keys:
                row_vals = dict(zip(pk_cols, key.split("|")))
                pk_filter = " AND ".join([f"n.{col} = $pk_{col}" for col in pk_cols])
                pk_params = {f"pk_{col}": v for col, v in row_vals.items()}
                session.run(
                    f"MATCH (n:{label}) WHERE {pk_filter} DETACH DELETE n",
                    pk_params
                )

            # Re-sync relationships for changed tables
            for rel in plan["relationships"]:
                if rel["from"] == label or rel["to"] == label:
                    session.run(f"""
                        MATCH (a:{rel['from']})
                        MATCH (b:{rel['to']})
                        WHERE a.{rel['from_column']} = b.{rel['to_column']}
                        MERGE (a)-[:{rel['type']}]->(b)
                    """)

        # Update snapshot
        _snapshot[table] = changes["current_hashes"]

        total_inserted += len(inserted)
        total_updated  += len(updated)
        total_deleted  += len(deleted_keys)

    summary = {
        "status": "synced" if changed_tables else "no_changes",
        "changed_tables": changed_tables,
        "inserted": total_inserted,
        "updated": total_updated,
        "deleted": total_deleted,
        "timestamp": datetime.now().isoformat()
    }
    return {"execution_summary": summary}


def validator(state: AgentState):
    with neo4j_driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return {
        "validation_report": {
            "total_nodes": node_count,
            "total_relationships": rel_count,
            "checked_at": datetime.now().isoformat()
        }
    }

# ===============================
# Build Workflows
# ===============================
def build_initial_workflow():
    wf = StateGraph(AgentState)
    wf.add_node("planner",            planner)
    wf.add_node("metadata_extractor", metadata_extractor)
    wf.add_node("semantic_engine",    semantic_engine)
    wf.add_node("graph_builder",      initial_graph_builder)
    wf.add_node("validator",          validator)

    wf.set_entry_point("planner")
    wf.add_edge("planner",            "metadata_extractor")
    wf.add_edge("metadata_extractor", "semantic_engine")
    wf.add_edge("semantic_engine",    "graph_builder")
    wf.add_edge("graph_builder",      "validator")
    wf.add_edge("validator",          END)
    return wf.compile()


def build_sync_workflow():
    wf = StateGraph(AgentState)
    wf.add_node("planner",         planner)
    wf.add_node("incremental_sync", incremental_sync)
    wf.add_node("validator",       validator)

    wf.set_entry_point("planner")
    wf.add_edge("planner",          "incremental_sync")
    wf.add_edge("incremental_sync", "validator")
    wf.add_edge("validator",        END)
    return wf.compile()

# ===============================
# Graceful Shutdown
# ===============================
_running = True

def handle_shutdown(sig, frame):
    global _running
    log.info("Shutdown signal received. Stopping after current cycle...")
    _running = False

signal.signal(signal.SIGINT,  handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# ===============================
# Main Loop
# ===============================
def run_pipeline():
    log.info("=" * 60)
    log.info("  Continuous SQL -> Neo4j Sync Pipeline Starting")
    log.info(f"  Poll interval : {POLL_INTERVAL_SECONDS}s")
    log.info(f"  Database      : {DATABASE_NAME}")
    log.info("=" * 60)

    # --- Step 1: Full initial load ---
    log.info("Running initial full load...")
    initial_app = build_initial_workflow()
    result = initial_app.invoke({})
    log.info(f"Initial load done. {result['validation_report']}")

    # --- Step 2: Continuous sync loop ---
    sync_app = build_sync_workflow()

    log.info(f"\nWatching for changes every {POLL_INTERVAL_SECONDS} seconds. "
             f"Press Ctrl+C to stop.\n")

    while _running:
        time.sleep(POLL_INTERVAL_SECONDS)

        if not _running:
            break

        log.info(f"--- Polling SQL Server for changes ---")
        try:
            result = sync_app.invoke({})
            summary = result["execution_summary"]

            if summary["status"] == "no_changes":
                log.info("No changes detected.")
            else:
                log.info(
                    f"Sync complete | Tables changed: {summary['changed_tables']} | "
                    f"+{summary['inserted']} inserted, "
                    f"~{summary['updated']} updated, "
                    f"-{summary['deleted']} deleted"
                )
            log.info(f"Graph state: {result['validation_report']}")

        except Exception as e:
            log.error(f"Error during sync cycle: {e}", exc_info=True)
            log.info("Will retry on next poll cycle...")

    log.info("Pipeline stopped gracefully.")
    neo4j_driver.close()
    engine.dispose()


if __name__ == "__main__":

    run_pipeline()
