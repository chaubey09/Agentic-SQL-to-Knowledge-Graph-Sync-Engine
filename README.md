**Agentic SQL-to-Knowledge Graph Sync Engine**

An LLM-driven autonomous pipeline that converts a relational SQL database into a Neo4j knowledge graph and keeps it continuously synchronized using incremental change detection.

1. Overview

This project builds a self-orchestrating agent pipeline that:

Extracts relational database metadata

Uses a Large Language Model (Gemini) to design a semantic knowledge graph schema

Performs an initial full graph build

Continuously detects row-level changes (insert, update, delete)

Applies incremental updates to Neo4j

Validates graph integrity after each sync cycle

The system is built using LangGraph for agent orchestration and follows a modular agent design.

2. Architecture
Agent Workflow (LangGraph)

Planner
→ Metadata Extractor
→ Semantic Engine (LLM)
→ Initial Graph Builder
→ Incremental Sync
→ Validator

3. Tech Stack

Python

LangGraph (Agent orchestration)

Google Gemini (gemini-2.5-flash) – LLM for schema design

Microsoft SQL Server

Neo4j (Graph Database)

SQLAlchemy + PyODBC

Pandas

Neo4j Python Driver

Hash-based snapshot change detection (MD5)

4. How It Works
1️⃣ Initial Load

Extracts SQL schema metadata

Sends metadata to Gemini

LLM generates a knowledge graph plan (nodes + relationships)

Builds full Neo4j graph

Stores row hashes in an in-memory snapshot

2️⃣ Continuous Sync

Polls SQL Server at fixed intervals

Detects changes using row-level hashing

Applies:

CREATE for new rows

UPDATE for modified rows

DELETE for removed rows

Re-syncs relationships if needed

Validates graph state

5. Setup Instructions
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Ensure System Requirements

SQL Server running on localhost:1433

Neo4j running on bolt://localhost:7687

ODBC Driver 18 for SQL Server installed

3️⃣ Configure Credentials

Update these variables in the script:

DATABASE_NAME   = "Chinook"
SQL_PASSWORD    = "your_password"
NEO4J_PASSWORD  = "your_password"
GEMINI_API_KEY  = "your_api_key"
4️⃣ Run the Pipeline
python main.py

Press Ctrl + C to stop gracefully.

6. Key Features

✅ LLM-driven semantic graph generation
✅ Modular agent-based architecture
✅ Incremental row-level synchronization
✅ Automatic relationship re-linking
✅ Snapshot-based change detection
✅ Continuous polling system
✅ Graph validation after each cycle

7. Use Cases

Building Knowledge Graphs from relational databases

AI-ready data infrastructure

RAG + Graph-based retrieval systems

Enterprise data modernization

Semantic layer automation

8. Future Improvements

Persistent snapshot storage (Redis / file-based)

Schema-change detection & re-planning

Event-driven sync (CDC instead of polling)

Dockerized deployment

Monitoring dashboard

**Author**

Anmol Chaubey
M.Tech – AI & ML
Focused on Agentic AI, LLM Systems, Knowledge Graphs, and Distributed Data Systems
