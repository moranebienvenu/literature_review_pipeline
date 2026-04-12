"""
Semantic clustering of neuroscience reproducibility literature
Pipeline: SPECTER2 (CPU) → UMAP → HDBSCAN → Claude API cluster naming → figures
Designed for Hugging Face Spaces — CPU Basic (free tier)
"""

import os
import time
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import anthropic
from huggingface_hub import HfApi, hf_hub_download


from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Read secrets from HF environment (set in Space Settings → Secrets)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HF_TOKEN          = os.environ.get("HF_TOKEN", "")
 
# Space repo ID — automatically set by HF
SPACE_ID = os.environ.get("SPACE_ID", "")   # e.g. "Mobien/neuro-reproducibility"

CFG = {
    "embedding_model":          "allenai/specter2_base", #"allenai/specter2",
    "csv_filename":             "data.csv",  
    "batch_size":               32,        # smaller batch — CPU has no VRAM limit but is slow
    "max_length":               512,
    "umap_n_neighbors":         30,
    "umap_min_dist":            0.0,
    "umap_metric":              "cosine",
    "umap_random_state":        42,
    "hdbscan_min_cluster_size": 60,
    "hdbscan_min_samples":      20,
    "merge_threshold":          0.99,
    "claude_model":             "claude-sonnet-4-6",
    "output_dir":               "./outputs",
}

os.makedirs(CFG["output_dir"], exist_ok=True)

# Global log list — Gradio reads this to display progress
LOG_LINES = []
PIPELINE_DONE = False
PIPELINE_ERROR = None
 
 
def log(msg: str):
    """Append a message to the global log and print it."""
    print(msg)
    LOG_LINES.append(msg)

# ─────────────────────────────────────────────
# HF REPO HELPERS
# ─────────────────────────────────────────────
 
def upload_to_repo(local_path: str, repo_filename: str):
    """Upload a file to the HF Space repo so it survives restarts."""
    if not HF_TOKEN or not SPACE_ID:
        log(f"  (skipping HF upload — no token or SPACE_ID)")
        return
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_filename,
            repo_id=SPACE_ID,
            repo_type="space",
            token=HF_TOKEN,
        )
        log(f"  Saved to repo: {repo_filename}")
    except Exception as e:
        log(f"  Warning — HF upload failed: {e}")
 
 
def download_from_repo(repo_filename: str, local_path: str) -> bool:
    """Try to download a file from the HF repo. Returns True if found."""
    if not HF_TOKEN or not SPACE_ID:
        return False
    try:
        hf_hub_download(
            repo_id=SPACE_ID,
            filename=repo_filename,
            repo_type="space",
            token=HF_TOKEN,
            local_dir=CFG["output_dir"],
        )
        return os.path.exists(local_path)
    except Exception:
        return False
 
 
# ─────────────────────────────────────────────
# STEP 1 — LOAD CSV FROM REPO
# ─────────────────────────────────────────────
 
def load_csv() -> pd.DataFrame:
    """
    Load the input CSV from the HF repo.
    The file must be uploaded to the Space repo as data.csv
    """
    local_csv = os.path.join(CFG["output_dir"], CFG["csv_filename"])
 
    # Try to download from repo if not already local
    if not os.path.exists(local_csv):
        log("Downloading CSV from HF repo...")
        found = download_from_repo(CFG["csv_filename"], local_csv)
        if not found:
            # Try the app directory (uploaded directly to Space files)
            app_csv = os.path.join("/app", CFG["csv_filename"])
            if os.path.exists(app_csv):
                import shutil
                shutil.copy(app_csv, local_csv)
            else:
                raise FileNotFoundError(
                    f"Could not find {CFG['csv_filename']} in repo or /app. "
                    "Please upload your CSV to the Space files as data.csv"
                )
 
    df = pd.read_csv(local_csv)
    df = df[df["abstract"].notna() & (df["abstract"].str.len() > 50)].copy()
    df = df.reset_index(drop=True)
    log(f"Loaded {len(df)} articles with valid abstracts.")
    return df


# ─────────────────────────────────────────────
# STEP 2 — SPECTER2 EMBEDDINGS (CPU)
# ─────────────────────────────────────────────
#, progress=gr.Progress()
def encode_with_specter2(texts: list[str]) -> np.ndarray:
    """
    Encode texts with SPECTER2 on CPU.
    Input format: "<title> [SEP] <abstract>"  (recommended by AllenAI)
    Returns (N, 768) float32 array, L2-normalised.
    On CPU this takes ~3-4 hours for 13 000 articles.
    Embeddings are saved to disk after each batch so nothing is lost
    if the Space times out.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    #progress(0.01, desc="Loading SPECTER2 model...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["embedding_model"])
    model     = AutoModel.from_pretrained(CFG["embedding_model"])
    model.eval()
    # No .to("cuda") — pure CPU

    bs            = CFG["batch_size"]
    n_batches     = (len(texts) + bs - 1) // bs
    # Checkpoint path — lets us resume if interrupted
    checkpoint = os.path.join(CFG["output_dir"], "embeddings_checkpoint.npy")
    all_embeddings = []
    start_batch    = 0

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint):
        saved = np.load(checkpoint)
        n_saved = len(saved)
        start_batch = n_saved // bs
        all_embeddings = [saved]
        log(f"Resuming from checkpoint: {n_saved} articles already encoded "
            f"(batch {start_batch}/{n_batches})")
 
    log(f"Encoding {len(texts)} articles in {n_batches} batches "
        f"(batch_size={bs})...")

    for i in range(start_batch * bs, len(texts), bs):
        batch_idx = i // bs
        
        if batch_idx % 10 == 0:
            log(f"  Batch {batch_idx + 1}/{n_batches} "
                f"({100 * batch_idx / n_batches:.0f}%)")

        batch  = texts[i : i + bs]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CFG["max_length"],
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]           # CLS token
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.float().numpy())

        # Save checkpoint every 50 batches (~1600 articles)
        if (batch_idx + 1) % 50 == 0:
            np.save(checkpoint, np.vstack(all_embeddings))
            log(f"  Checkpoint saved at batch {batch_idx + 1}")

    embeddings = np.vstack(all_embeddings)
    log(f"SPECTER2 encoding done: {embeddings.shape}")

    # Save final embeddings
    final_path = os.path.join(CFG["output_dir"], "specter2_embeddings.npy")
    np.save(final_path, embeddings)

    # Clean up checkpoint
    # if os.path.exists(checkpoint):
    #     os.remove(checkpoint)

    return embeddings


# ─────────────────────────────────────────────
# STEP 2 — UMAP
# ─────────────────────────────────────────────
#, progress=gr.Progress()
def run_umap(embeddings: np.ndarray) -> np.ndarray:
    """Project (N, 768) → (N, 2)."""
    import umap
    #progress(0.52, desc="Running UMAP projection...")
    log("Running UMAP projection...")
    reducer = umap.UMAP(
        n_neighbors  = CFG["umap_n_neighbors"],
        min_dist     = CFG["umap_min_dist"],
        metric       = CFG["umap_metric"],
        random_state = CFG["umap_random_state"],
        verbose      = False,
    )
    coords = reducer.fit_transform(embeddings)
    log("UMAP done.")
    return coords

# ─────────────────────────────────────────────
# STEP 3 — HDBSCAN
# ─────────────────────────────────────────────
#, progress=gr.Progress()
def run_hdbscan(coords: np.ndarray) -> np.ndarray:
    """Cluster 2-D UMAP coords. Returns label array (-1 = noise)."""
    import hdbscan
    log("Clustering with HDBSCAN...")
    #progress(0.65, desc="Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = CFG["hdbscan_min_cluster_size"],
        min_samples      = CFG["hdbscan_min_samples"],
        metric           = "euclidean",
        core_dist_n_jobs = 1,
    )
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels[labels != -1]))
    n_noise    = int((labels == -1).sum())
    log(f"HDBSCAN: {n_clusters} clusters, "
        f"{n_noise} noise points ({100*n_noise/len(labels):.1f}%)")
    return labels
    
    # return clusterer.fit_predict(coords)


# ─────────────────────────────────────────────
# STEP 4 — MERGE SIMILAR CLUSTERS
# ─────────────────────────────────────────────

def merge_similar_clusters(labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Merge clusters whose SPECTER2 centroid cosine similarity > merge_threshold.
    Operates entirely in SPECTER2 embedding space.
    """
    unique_ids = sorted(set(labels[labels != -1]))
    if len(unique_ids) < 2:
        return labels

    centroids = {
        cid: embeddings[labels == cid].mean(axis=0)
        for cid in unique_ids
    }

    merge_map = {}
    threshold = CFG["merge_threshold"]

    for i, id_i in enumerate(unique_ids):
        if id_i in merge_map:
            continue
        for id_j in unique_ids[i + 1:]:
            if id_j in merge_map:
                continue
            sim = cosine_similarity(
                centroids[id_i].reshape(1, -1),
                centroids[id_j].reshape(1, -1),
            )[0, 0]
            if sim > threshold:
                merge_map[id_j] = id_i

    if not merge_map:
        return labels

    new_labels = labels.copy()
    for old, new in merge_map.items():
        new_labels[new_labels == old] = new

    remaining = len(set(new_labels[new_labels != -1]))
    log(f"Merged {len(merge_map)} similar clusters → {remaining} remaining")
    return new_labels


# ─────────────────────────────────────────────
# STEP 5 — CLAUDE API CLUSTER NAMING
# ─────────────────────────────────────────────
#api_key: str,
#progress=gr.Progress(),
def name_clusters_with_claude(
    df: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> dict[int, str]:
    """
    For each cluster:
      1. Find the 5 articles closest to the cluster centroid in SPECTER2 space.
      2. Send their titles + truncated abstracts to Claude.
      3. Ask for a 3-5 word thematic label.
    Claude is used only for naming — the clustering itself is done entirely
    by SPECTER2 + HDBSCAN. One API call per cluster (~15-20 calls total),
    so the cost is negligible (<$0.01).
    """
    if not ANTHROPIC_API_KEY:
        log("Warning — no ANTHROPIC_API_KEY found, using generic names.")
        return {cid: f"Cluster {i}"
                for i, cid in enumerate(sorted(set(labels[labels != -1])))}
 
    client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    unique_ids = sorted(set(labels[labels != -1]))
    cluster_names: dict[int, str] = {}

    log(f"Naming {len(unique_ids)} clusters with Claude...")

    for i, cid in enumerate(unique_ids):
        # progress(
        #     0.75 + 0.15 * i / len(unique_ids),
        #     desc=f"Naming cluster {i + 1}/{len(unique_ids)} with Claude...",
        # )

        # Select the 5 most representative articles (closest to centroid)
        cluster_mask = labels == cid
        cluster_embs = embeddings[cluster_mask]
        centroid     = cluster_embs.mean(axis=0)
        sims         = cosine_similarity(centroid.reshape(1, -1), cluster_embs)[0]
        top5_idx     = sims.argsort()[-5:][::-1]
        representative = df[cluster_mask].iloc[top5_idx]

        # Build context for Claude
        snippets = []
        for _, row in representative.iterrows():
            title    = str(row.get("title",    "")).strip()
            abstract = str(row.get("abstract", "")).strip()[:400]
            snippets.append(f"- {title}\n  {abstract}")

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

        try:
            message = client.messages.create(
                model      = CFG["claude_model"],
                max_tokens = 30,
                messages   = [{"role": "user", "content": prompt}],
            )
            label = message.content[0].text.strip().splitlines()[0].strip()
            log(f"  Raw Claude response for cluster {cid}: '{label}'")
            
            # Sanity check
            if len(label.split()) > 8 or len(label) < 3:
                raise ValueError(f"Unexpected output: {label}")

        except Exception as e:
            log(f"  Warning — cluster {cid} naming failed: {e}")
            #print(f"  Warning — cluster {cid} naming failed: {e}")
            label = f"Cluster {cid}"
        cluster_names[cid] = label
        log(f"  [{i}] {label}  (n={cluster_mask.sum()})")
 
    return cluster_names
 

    #     cluster_names[cid] = label
    #     print(f"  Cluster {cid}: {label}")

    # return cluster_names


# ─────────────────────────────────────────────
# STEP 6 — FIGURES
# ─────────────────────────────────────────────

def make_figures(
    df: pd.DataFrame,
    cluster_names: dict[int, str],
    seq_to_orig: dict[int, int],
) -> list[str]:
    """
    Three publication-ready figures:
      fig1 — UMAP map (all articles, sharing highlighted with black ring)
      fig2 — Sharing rate per cluster (bar chart, sorted descending)
      fig3 — Temporal evolution per cluster (line chart, 2015-2025)
    """
    paths      = []
    unique_seq = sorted(df.loc[df["cluster_label"] != -1, "cluster_label"].unique())
    n          = len(unique_seq)

    palette = (
        sns.color_palette("tab10", n) if n <= 10 else
        sns.color_palette("tab20", n) if n <= 20 else
        sns.color_palette("husl",  n)
    )
    color_map     = {lbl: palette[i] for i, lbl in enumerate(unique_seq)}
    color_map[-1] = "#cccccc"

    seq_name = {
        seq: cluster_names.get(orig, f"Cluster {seq}")
        for seq, orig in seq_to_orig.items()
    }

    from matplotlib import rcParams
    rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size":       9,
        "axes.labelsize":  10,
        "axes.titlesize":  11,
        "axes.linewidth":  0.8,
        "axes.edgecolor":  "#333333",
    })

    # ── Figure 1: UMAP map all articles ──────────────────────────────────────────────
    log("Generating figure 1 (UMAP all articles)...")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    noise = df[df["cluster_label"] == -1]
    if len(noise):
        ax.scatter(noise["x"], noise["y"], s=8, c="#cccccc",
                   alpha=0.3, linewidth=0)

    for lbl in unique_seq:
        sub = df[df["cluster_label"] == lbl]
        ax.scatter(sub["x"], sub["y"], s=14, color=color_map[lbl],
                   alpha=0.70, linewidth=0, label=seq_name[lbl])

    sharing = df[df["shares_code_or_data"]]
    if len(sharing):
        ax.scatter(sharing["x"], sharing["y"], s=25,
                   facecolors="none", edgecolors="black",
                   linewidth=0.5, alpha=0.8, label="With code / data")

    ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=max(1, n // 8))
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Semantic landscape of neuroscience reproducibility",
                 fontweight="semibold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.2)
    plt.tight_layout()

    p1 = os.path.join(CFG["output_dir"], "fig1_umap_all.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p1)
    upload_to_repo(p1, "fig1_umap_all.png")
    
    # ── Figure 2: UMAP — sharing articles only ──────────────────────────
    log("Generating figure 2 (UMAP sharing only)...")
    sharing_df = df[df["shares_code_or_data"]].copy()
    if len(sharing_df) > 0:
        x_lo, x_hi = np.percentile(sharing_df["x"], [2, 98])
        y_lo, y_hi = np.percentile(sharing_df["y"], [2, 98])
        mx = (x_hi - x_lo) * 0.08
        my = (y_hi - y_lo) * 0.08
 
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#fafafa")
 
        noise_s = sharing_df[sharing_df["cluster_label"] == -1]
        if len(noise_s):
            ax.scatter(noise_s["x"], noise_s["y"], s=18, c="#bbbbbb",
                       alpha=0.5, linewidth=0, zorder=1)
 
        handles = []
        for lbl in sorted(sharing_df.loc[sharing_df["cluster_label"] != -1,
                                         "cluster_label"].unique()):
            sub   = sharing_df[sharing_df["cluster_label"] == lbl]
            color = color_map[lbl]
            name  = seq_name[lbl]
            ax.scatter(sub["x"], sub["y"], c=[color], s=22, alpha=0.82,
                       linewidth=0.3, edgecolors="white", zorder=2)
            handles.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.5, markersize=8,
                label=f"{name}  (n={len(sub)})"))
 
        if len(noise_s):
            handles.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor="#bbbbbb", markersize=8,
                label=f"Unclustered  (n={len(noise_s)})"))
 
        ax.set_xlim(x_lo - mx, x_hi + mx)
        ax.set_ylim(y_lo - my, y_hi + my)
 
        legend = ax.legend(
            handles=handles, loc="center left",
            bbox_to_anchor=(1.01, 0.5), fontsize=7.5,
            frameon=True, framealpha=0.95, edgecolor="#dddddd",
            facecolor="white", borderpad=0.8, labelspacing=0.6,
            title="Semantic cluster", title_fontsize=8.5, ncol=1,
        )
        legend.get_title().set_fontweight("semibold")
 
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_title(
            "Sharing articles by thematic cluster (UMAP projection)",
            fontweight="semibold", pad=14, fontsize=10, loc="left")
        ax.annotate(f"n = {len(sharing_df)} articles",
                    xy=(0.99, 0.02), xycoords="axes fraction",
                    fontsize=7.5, color="#888888", ha="right", va="bottom")
 
        plt.tight_layout()
        p2 = os.path.join(CFG["output_dir"], "fig2_umap_sharing.png")
        fig.savefig(p2, dpi=300, bbox_inches="tight",
                    bbox_extra_artists=[legend], facecolor="#fafafa")
        plt.close(fig)
        paths.append(p2)
        upload_to_repo(p2, "fig2_umap_sharing.png")

    # ── Figure 3: Sharing rate per cluster ──────────────────────────────
    stats = (
        df[df["cluster_label"] != -1]
        .groupby("cluster_label")
        .agg(papers=("cluster_label", "count"),
             sharing_rate=("shares_code_or_data", "mean"))
        .reset_index()
        .sort_values("sharing_rate", ascending=False)
        .reset_index(drop=True)
    )
    stats["name"] = stats["cluster_label"].map(seq_name)

    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.8), 5), dpi=300)
    bar_colors = [color_map[lbl] for lbl in stats["cluster_label"]]
    ax.bar(range(len(stats)), stats["sharing_rate"],
           color=bar_colors, edgecolor="white", linewidth=0.5,
           alpha=0.85, width=0.75)

    for pos, row in stats.iterrows():
        ax.text(pos, row["sharing_rate"] + 0.012,
                f"n={int(row['papers'])}", ha="center", fontsize=7)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["name"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Sharing rate (code and/or data)")
    ax.set_title("Reproducibility culture per scientific community",
                 fontweight="semibold", pad=12)
    ax.set_ylim(0, min(1.0, stats["sharing_rate"].max() * 1.18))
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    p3 = os.path.join(CFG["output_dir"], "fig3_cluster_sharing.png")
    fig.savefig(p3, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p3)
    upload_to_repo(p3, "fig3_cluster_sharing.png")
    
    # ── Figure 4: Temporal evolution ────────────────────────────────────
    temporal = (
        df[df["cluster_label"] != -1]
        .groupby(["publication_year", "cluster_label"])["shares_code_or_data"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    for lbl in unique_seq:
        sub = temporal[temporal["cluster_label"] == lbl]
        if len(sub) < 2:
            continue
        ax.plot(sub["publication_year"], sub["shares_code_or_data"],
                marker="o", markersize=4, linewidth=1.5,
                markerfacecolor="white", markeredgewidth=1.2,
                color=color_map[lbl], label=seq_name[lbl])

    ax.set_xlabel("Publication year")
    ax.set_ylabel("Sharing rate (code and/or data)")
    ax.set_title("Adoption of reproducibility across communities",
                 fontweight="semibold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=7, ncol=min(4, n), framealpha=0.9)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    p4 = os.path.join(CFG["output_dir"], "fig4_temporal_clusters.png")
    fig.savefig(p4, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p4)
    upload_to_repo(p4, "fig4_temporal_clusters.png")
    
    return paths


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
#csv_file, api_key: str, hf_token: str, progress=gr.Progress()
def run_pipeline():
    global PIPELINE_DONE, PIPELINE_ERROR
    t0 = time.time()
    log(f"API key present: {bool(ANTHROPIC_API_KEY)} — prefix: {ANTHROPIC_API_KEY[:10] if ANTHROPIC_API_KEY else 'EMPTY'}")
 
    try:
        # ── Load CSV ─────────────────────────────────────────────────────
        df = load_csv()
 
        texts = (
            df["title"].fillna("").astype(str)
            + " [SEP] "
            + df["abstract"].fillna("").astype(str)
        ).tolist()
 
        # ── SPECTER2 — load from repo if already computed ─────────────────
        emb_local = os.path.join(CFG["output_dir"], "specter2_embeddings.npy")
 
        if not os.path.exists(emb_local):
            log("Looking for saved embeddings in HF repo...")
            found = download_from_repo("specter2_embeddings.npy", emb_local)
            if found:
                log("Embeddings found in repo — skipping encoding.")
 
        if os.path.exists(emb_local):
            embeddings = np.load(emb_local)
            log(f"Embeddings loaded: {embeddings.shape}")
        else:
            log("No saved embeddings found — starting SPECTER2 encoding.")
            log("This will take ~3-4 hours on CPU. You can close your computer.")
            embeddings = encode_with_specter2(texts)
            np.save(emb_local, embeddings)
            upload_to_repo(emb_local, "specter2_embeddings.npy")
 
        # ── UMAP ─────────────────────────────────────────────────────────
        coords   = run_umap(embeddings)
        df["x"]  = coords[:, 0]
        df["y"]  = coords[:, 1]
 
        # ── HDBSCAN ───────────────────────────────────────────────────────
        raw_labels    = run_hdbscan(coords)
        merged_labels = merge_similar_clusters(raw_labels, embeddings)
        df["cluster"] = merged_labels
 
        unique_orig = sorted(set(merged_labels[merged_labels != -1]))
        orig_to_seq = {orig: seq for seq, orig in enumerate(unique_orig)}
        seq_to_orig = {seq: orig for orig, seq in orig_to_seq.items()}
        df["cluster_label"] = (
            df["cluster"].map(orig_to_seq).fillna(-1).astype(int)
        )
        log(f"Final clusters: {len(unique_orig)}")
 
        # ── Claude naming ─────────────────────────────────────────────────
        cluster_names = name_clusters_with_claude(df, merged_labels, embeddings)
 
        # ── Figures ───────────────────────────────────────────────────────
        make_figures(df, cluster_names, seq_to_orig)
 
        # ── Save enriched CSV ─────────────────────────────────────────────
        out_csv = os.path.join(CFG["output_dir"], "semantic_clusters_dataset.csv")
        df.to_csv(out_csv, index=False)
        upload_to_repo(out_csv, "semantic_clusters_dataset.csv")
 
        elapsed = time.time() - t0
        log(f"\nPipeline complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        log(f"Sharing rate: {df['shares_code_or_data'].mean()*100:.1f}%")
        log("All figures and CSV saved to HF repo.")
 
    except Exception as e:
        import traceback
        PIPELINE_ERROR = traceback.format_exc()
        log(f"\nERROR: {e}")
        log(PIPELINE_ERROR)
 
    finally:
        PIPELINE_DONE = True
 
 
# ─────────────────────────────────────────────
# GRADIO UI  (display only — pipeline runs in background)
# ─────────────────────────────────────────────
 
def get_logs():
    """Called by Gradio every 5 seconds to refresh the log display."""
    text = "\n".join(LOG_LINES[-200:])   # show last 200 lines
    done = PIPELINE_DONE
 
    # Load figures if available
    figs = []
    for fname in ["fig1_umap_all.png", "fig2_umap_sharing.png",
                  "fig3_cluster_sharing.png", "fig4_temporal_clusters.png"]:
        path = os.path.join(CFG["output_dir"], fname)
        figs.append(path if os.path.exists(path) else None)
 
    return text, figs[0], figs[1], figs[2], figs[3]
 
 
with gr.Blocks(title="Neuro Reproducibility — Semantic Clustering") as demo:
    gr.Markdown(
        "## Neuroscience reproducibility — semantic clustering\n"
        "Pipeline runs automatically in the background. "
        "Logs refresh every 5 seconds.\n\n"
        "**You can close your computer** — the pipeline keeps running on HF servers.\n\n"
        "> Make sure `data.csv` is uploaded to the Space files and "
        "`ANTHROPIC_API_KEY` + `HF_TOKEN` are set in Space Secrets."
    )
 
    log_box = gr.Textbox(label="Pipeline logs", lines=30, interactive=False)
 
    with gr.Row():
        fig1 = gr.Image(label="UMAP — all articles")
        fig2 = gr.Image(label="UMAP — sharing articles only")
    with gr.Row():
        fig3 = gr.Image(label="Sharing rate per cluster")
        fig4 = gr.Image(label="Temporal evolution")
 
    # Auto-refresh every 5 seconds
    timer = gr.Timer(value=5)
    timer.tick(fn=get_logs, outputs=[log_box, fig1, fig2, fig3, fig4])
 
 
# ─────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    # Start pipeline in background thread immediately
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
 
    # Launch Gradio interface
    demo.launch(share=True)
