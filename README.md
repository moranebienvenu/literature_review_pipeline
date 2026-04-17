# Neuroscience Reproducibility Analysis

A two-step pipeline to analyze reproducibility practices in the neuroscience literature: data collection via the OpenAlex API, followed by semantic clustering on Hugging Face Spaces.

Adapted from **Karakuzu et al. (2025) — "Reproducible Research Practices in Magnetic Resonance Neuroimaging"** ([DOI: 10.55458/neurolibre.00021](https://doi.org/10.55458/neurolibre.00021)).

---

## Project Overview

The project is split into two independent scripts:

| Script | Role | Where to run |
|--------|------|--------------|
| `neuroscience_reproducibility_openalex.py` | Data collection and basic analysis via OpenAlex | **Locally** |
| `app.py` | SPECTER2 semantic clustering + figures | **Hugging Face Spaces** |

---

## Script 1 — OpenAlex Data Collection (local)

### What this script does

- Queries the OpenAlex API for open-access neuroscience articles (2015–2025)
- Detects code/data sharing (GitHub, Zenodo, OSF, Figshare, etc.)
- Generates figures and a summary report
- Produces a `data.csv` file used as input for Script 2

### Prerequisites

```bash
pip install requests pandas numpy matplotlib seaborn
```

### Quick setup

Open `neuroscience_reproducibility_openalex.py` and edit the parameters in the `main()` function:

```python
reviewer = OpenAlexNeuroscienceReview(
    email="your.email@example.com",   # ← REQUIRED for OpenAlex polite pool
    output_dir="./neuroscience_repro_output"
)
```

The email address is sent in the HTTP header (`mailto:`) — it is not stored by OpenAlex, but it grants faster, prioritized access to the API.

### Adjusting search filters

Filters are defined in the `build_query_for_year()` method:

```python
filters = [
    f"publication_year:{year}",
    "type:article",
    "is_oa:true",                              # Open Access only
    "concepts.id:C41008148",                   # OpenAlex concept: Neuroscience
    'title_and_abstract.search:(brain OR neuroscience)'  # ← edit here
]
```

**Example customizations:**
- Broaden search terms: `stroke OR fMRI OR EEG OR neurotransmitter`
- Change the year range in the `main()` loop: `for year in range(2018, 2026)`
- Remove the open access filter: delete the `"is_oa:true"` line
- Change the OpenAlex concept (e.g. C15744967 for a broader neuroscience definition)

### Running

```bash
python neuroscience_reproducibility_openalex.py
```

Estimated duration: **5–45 minutes** depending on the number of articles retrieved.

### Outputs

```
neuroscience_repro_output/
├── neuroscience_reproducibility_full.csv   # All articles with metadata
├── neuroscience_reproducibility_full_summary.txt                      # Full text report
├── fig_1_sharing_bars.png                     # Number of shared articles and total articles retrieved over time
├── fig_2_sharing_rate.png                     # Sharing rate over time
```

The file `neuroscience_reproducibility_full.csv` (renamed to `data.csv`) is used as input for Script 2.

---

## Script 2 — Semantic Clustering on Hugging Face (app.py)

### What this script does

- Encodes titles and abstracts with **SPECTER2** (AllenAI model optimized for scientific literature)
- Projects embeddings to 2D with **UMAP**
- Identifies thematic clusters with **HDBSCAN**
- Automatically names each cluster via the **Claude API**
- Generates 4 publication-ready figures

### Why Hugging Face?

Encoding ~10,000 articles with SPECTER2 takes **2–3 hours on CPU**. Hugging Face Spaces lets the analysis run in the background even after closing your computer. The **free CPU Basic tier** is sufficient.

### Setting up secrets

In your HF Space settings (Settings → Secrets), add:

| Secret name | Value |
|-------------|-------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key ([console.anthropic.com](https://console.anthropic.com)) |
| `HF_TOKEN` | Your Hugging Face token ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |

### Preparing the files

1. Rename your output CSV to `data.csv`
2. Upload it to the Space files (the "Files" tab in the HF interface)
3. Upload the `requirements.txt` file to HF interface.

### Estimated cost

- **Claude API**: ~$0.10 USD per full run (roughly 15–20 API calls, one per cluster)
- **Hugging Face**: free on CPU Basic

### Result persistence

The computed SPECTER2 embeddings are **saved to the HF repo** (`specter2_embeddings.npy`). On the next run, this step is skipped — only clustering and labeling are re-executed. This avoids recomputing 2–3 hours of encoding on every run, in case you want to modify the prompt sent to Claude.

### Customizing cluster names (Claude prompt)

The prompt sent to Claude to name each cluster is in the `name_clusters_with_claude()` function in `app.py`:

```python
prompt = (
    "You are a scientific librarian specialising in neuroscience.\n"
    "Below are 5 representative papers from a neuroscience research cluster identified "
    "by semantic similarity.\n\n"
    + "\n\n".join(snippets)
    + "\n\nGive a concise thematic label of 2 to 5 words that best "
    "summarises the neuroscience topic of this cluster.\n"
    "Reply with the label only — no explanation, no punctuation, "
    "no numbering."
)
```

You can modify this prompt to, for example:
- Change the label language (`Reply in French`)
- Request longer or shorter names
- Add domain-specific context to guide the model

### Outputs

```
outputs/
├── fig1_umap_all.png              # UMAP map — all articles
├── fig2_umap_sharing.png          # UMAP map — sharing articles only
├── fig3_cluster_sharing.png       # Sharing rate per cluster (bar chart)
├── fig4_temporal_clusters.png     # Temporal evolution per cluster
├── specter2_embeddings.npy        # Saved embeddings (reusable)
└── semantic_clusters_dataset.csv  # Enriched CSV with cluster labels
```

All files are automatically uploaded to the HF repo to persist across Space restarts.

---

## Differences from the Original Paper

| Aspect | Karakuzu et al. (2024) | This adaptation |
|--------|----------------------|-----------------|
| Literature API | Semantic Scholar | **OpenAlex** |
| Embeddings | SPECTER (pre-computed) | **SPECTER2** (computed on the fly) |
| Scope | MRI / MRM journal | **All neuroscience (open access)** |
| Interface | Jupyter Notebook | **Gradio (Hugging Face Spaces)** |

---

## Running the Full Pipeline

```bash
# Step 1 — locally
python neuroscience_reproducibility_openalex.py
# → produces data.csv in neuroscience_repro_output/

# Step 2 — on Hugging Face
# → upload data.csv to the Space files
# → set ANTHROPIC_API_KEY and HF_TOKEN in Space Secrets
# → the pipeline starts automatically when the Space launches
```
