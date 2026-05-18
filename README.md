# REVEAL: Retrieval and Evidence-based Validated Interpretation Analysis for gene Lists

A stateful, LLM-powered pipeline for comprehensive gene function interpretation and analysis. Combines local database queries, PubMed literature mining, and AI-driven synthesis to produce detailed gene interpretation reports.

## Features

- **Natural Language Queries** — Analyse genes using plain English ("What is the functional role of MED12, EOMES, PEG3, ZIM2, PCDHA6, PCDHGA3, F8A2, MIMT1, ADPRHL1, RGPD1, and F8A3 in neurodevelopmental disorders?")
- **Comprehensive Data Integration** — Gene info, GO terms, expression data, protein interactions
- **Adaptive Literature Mining** — Tiered PubMed queries with BM25 pre-ranking and LLM-powered relevance ranking
- **LLM-Powered Interpretation** — Generate biological insights and cross-gene synthesis
- **State Persistence** — Resume interrupted runs and reuse expensive computations
- **Resume & Checkpointing** — Pick up from any node if a run is interrupted
- **Web Interface** — Streamlit app for interactive exploration of results
- **Optional Observability** — Phoenix tracing integration for debugging LLM calls

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/brettrj03/REVEAL.git
cd REVEAL

# 2. Create virtual environment (Python 3.11+ required)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your API keys (see below)

# 5. Build the database (downloads ~500MB, creates ~1.3GB database)
python scripts/setup_database.py

# 6. Verify setup
python scripts/verify_database.py

# 7. Launch the web interface
streamlit run streamlit_app.py
# Or run via command line
python run_stateful_pipeline.py "What is the functional role of MED12, EOMES, PEG3, ZIM2, PCDHA6, PCDHGA3, F8A2, MIMT1, ADPRHL1, RGPD1, and F8A3 in neurodevelopmental disorders?"
```

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11 | 3.11 or 3.12 |
| RAM | 8 GB | 16 GB |
| Disk Space | 10 GB | 20 GB |
| OS | macOS 11+, Ubuntu 20.04+, Windows 10+ (WSL2 recommended) |

### Required API Keys

| Key | Required | Purpose | Get it at |
|-----|----------|---------|-----------|
| `OPENAI_API_KEY` | Yes | Gene interpretation, literature ranking | [platform.openai.com](https://platform.openai.com/api-keys) |
| `NCBI_EMAIL` | Yes | PubMed API identification | Any valid email |
| `NCBI_API_KEY` | No | Higher PubMed rate limits (10 req/s vs 3) | [ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/account/) |

---

## Installation

### Step 1: Clone and Set Up Environment

```bash
git clone https://github.com/brettrj03/REVEAL.git
cd REVEAL

python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Verify core imports work:
```bash
python -c "import pydantic_graph; import openai; import streamlit; print('All imports OK')"
```

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
NCBI_EMAIL=your.email@example.com
# NCBI_API_KEY=your-ncbi-key-here  # Optional — delete this line if you don't have one
```

> **Important:** If you don't have an NCBI API key, remove or comment out the `NCBI_API_KEY` line entirely. Leaving it blank causes 400 errors.

### Step 4: Build the Database

```bash
python scripts/setup_database.py
```

This downloads and processes:
- GENCODE v49 gene annotations (~45 MB)
- Gene Ontology terms and associations (~53 MB)
- GTEx v8 tissue expression data (~13 MB)
- STRING v12 protein interactions (~450 MB)
- NCBI gene descriptions (~8 MB)

Takes approximately 10–15 minutes. Final database size: ~1.3 GB at `src/database/gene_database.sqlite`.

### Step 5: Verify Installation

```bash
python scripts/verify_database.py
```

This checks database integrity, table counts, and API configuration.

---

## Usage

### Web Interface

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser. Load a saved state file from the sidebar to explore previous results interactively.

### Command Line

```bash
# Full analysis with LLM interpretation (default)
python run_stateful_pipeline.py "What is the functional role of MED12, EOMES, PEG3, ZIM2, PCDHA6, PCDHGA3, F8A2, MIMT1, ADPRHL1, RGPD1, and F8A3 in neurodevelopmental disorders?"

# Fast mode — database facts only, no LLM calls, free
python run_stateful_pipeline.py "What is the functional role of GATA4, PTGFR, BNC1, PRAG1, IQGAP1, TXNDC5, ENAM, ZNF619, TMEM71, ZNF717, and NETO1-DT in congenital heart disease and cardiomyocyte function?" --factual-only

# Resume an interrupted run
python run_stateful_pipeline.py "What is the functional role of SETBP1, COL2A1, IGFBP5, VIM, DNAJC6, KYAT3, ESRP2, JARID2, NXPH4, CDK19, and DLL4 in neurodevelopmental disorders and neural progenitor cell differentiation?" --resume

# Resume from a specific node
python run_stateful_pipeline.py "What is the functional role of MED12, EOMES, PEG3, ZIM2, PCDHA6, PCDHGA3, F8A2, MIMT1, ADPRHL1, RGPD1, and F8A3 in neurodevelopmental disorders?" --resume-from InterpretAllGenes

# Check the status of a previous run
python run_stateful_pipeline.py --status results/stateful_pipeline/run_20260429_123456_query_abc123

# Run without saving state to disk
python run_stateful_pipeline.py "What is the functional role of GATA4, PTGFR, BNC1, PRAG1, IQGAP1, TXNDC5, ENAM, ZNF619, TMEM71, ZNF717, and NETO1-DT in congenital heart disease and cardiomyocyte function?" --no-persist
```

### Analysis Modes

| Mode | Speed | LLM Calls | Cost | Use Case |
|------|-------|-----------|------|----------|
| Default (interpreted) | ~20s/gene | Yes | ~$0.01–0.03/gene | Full analysis with biological insights |
| `--factual-only` | ~2s/gene | No | Free | Quick data retrieval, debugging |

---

## Project Structure

```
gene-annotation/
├── run_stateful_pipeline.py     # CLI entry point
├── streamlit_app.py             # Web interface
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package configuration
├── .env.example                 # Environment variable template
├── src/
│   ├── agents/                  # LLM agents (8 agents)
│   ├── nodes/                   # Pipeline nodes (21 nodes)
│   ├── graph/                   # Graph definition & state
│   ├── models/                  # Pydantic data models
│   ├── reports/                 # Report generation
│   ├── utils/                   # Utilities (BM25, persistence, tracing)
│   ├── integrations/            # External API clients (PubMed, etc.)
│   ├── database/                # SQLite database layer
│   ├── validation/              # Output validation logic
│   └── validation_config/       # Validation configuration
├── scripts/
│   ├── setup_database.py        # Database setup (run once)
│   └── verify_database.py       # Database verification
├── docs/                        # Extended documentation
└── tests/                       # Test suite
```

---

## Configuration

Key settings in `src/config.py`:

```python
# LLM Models
GENE_EXTRACTION_MODEL = "gpt-4.1-mini"
INTERPRETATION_MODEL = "gpt-4.1-mini"

# Database
DEFAULT_DB_PATH = "src/database/gene_database.sqlite"

# Retry Settings
MAX_INTERPRETATION_RETRIES = 2
MAX_EXTRACTION_RETRIES = 2
```

---

## Pipeline Overview

The pipeline executes 21 nodes organised into 5 phases:

| Phase | Nodes | Purpose | Key Outputs |
|-------|-------|---------|-------------|
| 1. Extraction | 2 | Parse query, fetch gene data | Gene profiles, database data |
| 2. Analysis | 3 | Cross-gene network & GO analysis | Shared partners, enriched terms |
| 3. Literature | 4 | PubMed search, BM25 pre-ranking, LLM ranking | Top papers per gene |
| 4. Interpretation & Validation | 11 | LLM insights + validation + refinement | Summaries, synthesis |
| 5. Report | 1 | Final report assembly | Comprehensive report |

State is persisted after each node — if a run is interrupted, use `--resume` to continue from the last checkpoint.


## Troubleshooting

### "ModuleNotFoundError: No module named 'X'"

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Database not found"

```bash
python scripts/setup_database.py
ls -lh src/database/gene_database.sqlite
```

### "Invalid API key" errors

Check your `.env` file — make sure there are no quotes around the key value:
```bash
# Correct:
OPENAI_API_KEY=sk-abc123...

# Incorrect:
OPENAI_API_KEY="sk-abc123..."
```

### "Port 8501 already in use"

```bash
streamlit run streamlit_app.py --server.port 8502
# or: lsof -i :8501 → kill -9 <PID>
```

### Database setup fails with 403 Forbidden

Some data sources (especially Gene Ontology) occasionally block automated downloads. Try again — temporary server issues are common. The setup script includes User-Agent headers and fallback URLs.

### PubMed returns 400 Bad Request

Your `NCBI_EMAIL` is likely still the placeholder, or `NCBI_API_KEY` is set but empty. Set a real email, and if you don't have an NCBI API key, **remove the line entirely** rather than leaving it blank.
