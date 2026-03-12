# 🔐 The Encrypted Vault

A multi-agent AI game where 4 LLM-powered agents compete to find a hidden 4-digit Master Key stored in a dynamic RAG system.

## Overview

**The Encrypted Vault** is a turn-based simulation built with LangGraph, ChromaDB, and Streamlit. Four distinct AI agents — each with unique strategies and tool access — race to crack a secret code hidden across 10 vault fragments. Agents can search the vault, corrupt clues to mislead rivals, send public broadcasts, and exchange private messages.

### The Agents

| Agent | Emoji | Strategy | Unique Tool |
|-------|-------|----------|-------------|
| **Infiltrator** | 🕵️ | Aggressive vault searching; secret alliances | — |
| **Saboteur** | 💣 | Corrupts vault fragments; spreads disinformation | `obfuscate_clue` |
| **Scholar** | 🎓 | Deductive reasoning; cross-references all sources | `submit_guess` |
| **Enforcer** | 👊 | Social manipulation via private DMs | `submit_guess` |

### Win Conditions

| Condition | Trigger |
|-----------|---------|
| 🏆 Agent Win | First agent to call `submit_guess` with the correct 4-digit code |
| 💀 System Win | 20 turns elapsed OR all agents exceed token budget |

---

## Architecture

The project follows a strict **5-layer architecture** with one-way dependencies:

```
UI (Streamlit) → Orchestration (LangGraph) → Agents → Services → DB (ChromaDB)
```

### Layer 1 — Database (`db/`)
- [`AbstractVaultRepository`](src/encrypted_vault/db/base_repository.py) — ABC interface
- [`ChromaVaultRepository`](src/encrypted_vault/db/chroma_repository.py) — production ChromaDB backend
- [`InMemoryVaultRepository`](src/encrypted_vault/db/in_memory_repository.py) — test/CI backend

### Layer 2 — Services (`services/`)
- [`VaultService`](src/encrypted_vault/services/vault_service.py) — vault query, obfuscate, health
- [`ChatService`](src/encrypted_vault/services/chat_service.py) — public broadcast + private DMs
- [`GameService`](src/encrypted_vault/services/game_service.py) — seeding, win checking, reset
- [`ServiceContainer`](src/encrypted_vault/services/container.py) — dependency injection

### Layer 3 — Agents (`agents/`)
- [`BaseAgent`](src/encrypted_vault/agents/base_agent.py) — abstract base with `run_turn()` loop
- [`Infiltrator`](src/encrypted_vault/agents/infiltrator.py), [`Saboteur`](src/encrypted_vault/agents/saboteur.py), [`Scholar`](src/encrypted_vault/agents/scholar.py), [`Enforcer`](src/encrypted_vault/agents/enforcer.py)

### Layer 4 — Orchestration (`graph/`)
- [`GameGraphBuilder`](src/encrypted_vault/graph/builder.py) — LangGraph StateGraph construction
- [`GameRunner`](src/encrypted_vault/graph/runner.py) — game lifecycle + streaming + reset

### Layer 5 — UI (`ui/`)
- [`app.py`](src/encrypted_vault/ui/app.py) — Streamlit real-time dashboard

---

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd TheEncryptedVault

# Install dependencies with UV
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running the Game

```bash
# Launch the Streamlit dashboard
uv run streamlit run src/encrypted_vault/ui/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

1. Click **▶ Start Game** to begin
2. Watch agents reason, search, and deceive in real-time
3. Use the **Speed slider** to control turn pace (0–3 seconds)
4. Click **🔄 Restart** or **🔄 Play Again** to start a new game

### Running Tests

```bash
# Run all tests (no API key needed — uses InMemoryVaultRepository)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_db.py -v
uv run pytest tests/test_services.py -v
uv run pytest tests/test_state.py -v
```

---

## Configuration

All settings are in `.env`:

```env
# Required
OPENAI_API_KEY=sk-...

# Optional (defaults shown)
LLM_MODEL=gpt-4o-mini
MAX_TURNS=20
TOKEN_BUDGET_PER_AGENT=8000
CHROMA_PERSIST_DIR=./chroma_db
```

---

## Swapping LLM Providers

The [`LLMFactory`](src/encrypted_vault/llm_factory.py) supports OpenAI, Anthropic, and Ollama:

```python
from encrypted_vault.llm_factory import LLMFactory, LLMProvider

# OpenAI (default)
llm = LLMFactory.create(LLMProvider.OPENAI, model="gpt-4o-mini")

# Anthropic
llm = LLMFactory.create(LLMProvider.ANTHROPIC, model="claude-3-5-haiku-20241022")

# Ollama (local)
llm = LLMFactory.create(LLMProvider.OLLAMA, model="llama3.2")
```

---

## Swapping the Vector Store

To add a new vector store backend (e.g. Pinecone):

1. Create `src/encrypted_vault/db/pinecone_repository.py`
2. Implement `PineconeVaultRepository(AbstractVaultRepository)` — 5 methods
3. Update `ServiceContainer.create_production()` to use it

Zero changes required in any other layer.

---

## UI Features

| Panel | Description |
|-------|-------------|
| **Public Chat** | All agent broadcasts + 🔒 private DMs (spectator sees all) |
| **Agent Progress** | Suspected key, confirmed digits, closeness bar (0–4/4) |
| **Real Master Key** | Spectator-only reveal — never shown to agents |
| **Thought Traces** | Internal agent reasoning — hidden from other agents |
| **Vault Status** | All 10 fragments: KEY ✅ / CORRUPTED ⚠️ / NOISE |

---

## Project Structure

```
TheEncryptedVault/
├── pyproject.toml
├── .env.example
├── README.md
├── src/
│   └── encrypted_vault/
│       ├── config.py
│       ├── llm_factory.py
│       ├── state/          # Pydantic models (no logic)
│       ├── db/             # Layer 1: Repository pattern
│       ├── services/       # Layer 2: Business logic
│       ├── agents/         # Layer 3: LLM agents
│       ├── graph/          # Layer 4: LangGraph orchestration
│       └── ui/             # Layer 5: Streamlit dashboard
├── tests/
└── plans/
    └── design.md
```
