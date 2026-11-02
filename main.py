#!/usr/bin/env python3
"""
Project 54: Text-to-Image Embedding with CLIP
=============================================
Multimodal zero-shot text-to-image retrieval and cross-modal embedding alignment
utilizing OpenAI's CLIP (Contrastive Language-Image Pretraining) in PyTorch.

Key Architectural Highlights:
  - Hugging Face `openai/clip-vit-base-patch32` (151.3M parameters, 512-dim shared latent space).
  - Dual Vision Transformer (ViT-B/32) and Text Transformer encoders.
  - Symmetrically normalized hypersphere embedding projection:
      v_hat = v / ||v||_2,   u_hat = u / ||u||_2
  - Temperature-scaled cross-modal softmax matching probabilities:
      P(I_i | T_j) = softmax( <v_i, u_j> / tau )
  - Dual retrieval capabilities:
      * Text-to-Image Search (find image given natural language prompt)
      * Image-to-Text Classification (zero-shot classification)
  - 4-Panel diagnostic visualization artifact.
  - Interactive CLI text-to-image retrieval console.
"""

import os
import sys
import json
import time
import math
import argparse
from typing import List, Dict, Tuple, Any

# Suppress Hugging Face symlink warnings on Windows & enforce offline mode
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METADATA = os.path.join(SCRIPT_DIR, "data", "gallery_metadata.json")
DEFAULT_IMG_DIR = os.path.join(SCRIPT_DIR, "data", "images")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "outputs", "clip_text_image_results.png")

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")


# ==============================================================================
# 1. CLIP Multimodal Retrieval Engine
# ==============================================================================

class CLIPMatcher:
    """Encodes images and text into a joint 512-dimensional metric space."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[*] Initializing CLIP Multimodal Model on device: {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[+] Loaded '{model_name}' ({param_count:,} parameters, 512-dim latent space).")

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode natural language strings into L2-normalized 512D vectors."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_features = getattr(outputs, "pooler_output", outputs)
        return F.normalize(text_features, p=2, dim=-1)

    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode PIL images into L2-normalized 512D vectors."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            image_features = getattr(outputs, "pooler_output", outputs)
        return F.normalize(image_features, p=2, dim=-1)

    def match(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pairwise cosine similarity and temperature-scaled softmax probabilities.
        Returns: (similarity_matrix, softmax_probabilities)
        """
        # Cosine similarity: (num_texts, num_images)
        sim_matrix = torch.mm(text_embeds, image_embeds.t())

        # Temperature scaling from model's learnable logit_scale parameter
        logit_scale = self.model.logit_scale.exp().item()
        logits = sim_matrix * logit_scale
        probs = F.softmax(logits, dim=-1)

        return sim_matrix, probs


# ==============================================================================
# 2. Image Gallery Manager
# ==============================================================================

class ImageGallery:
    """Manages image collection, pre-computed embeddings, and retrieval."""

    def __init__(self, metadata_path: str = DEFAULT_METADATA, img_dir: str = DEFAULT_IMG_DIR):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.gallery_items = data["gallery"]
        self.benchmark_queries = data.get("benchmark_queries", [])
        self.img_dir = img_dir

        self.matcher = CLIPMatcher()

        # Load PIL images
        print(f"[*] Loading and encoding {len(self.gallery_items)} gallery images...")
        self.pil_images = []
        for item in self.gallery_items:
            path = os.path.join(img_dir, item["filename"])
            if not os.path.exists(path):
                raise FileNotFoundError(f"Gallery image not found: {path}")
            self.pil_images.append(Image.open(path).convert("RGB"))

        t0 = time.perf_counter()
        self.image_embeddings = self.matcher.encode_images(self.pil_images)
        dt = time.perf_counter() - t0
        print(f"[+] Encoded {len(self.gallery_items)} images in {dt:.3f} s ({len(self.gallery_items)/dt:.1f} img/s).")

    def search_text_to_image(self, prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-K images matching the input text prompt."""
        t0 = time.perf_counter()
        text_emb = self.matcher.encode_text([prompt])
        sim_matrix, probs = self.matcher.match(text_emb, self.image_embeddings)
        latency = time.perf_counter() - t0

        sims = sim_matrix[0]
        pr = probs[0]

        top_scores, top_indices = torch.topk(sims, k=min(top_k, len(self.gallery_items)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            item = self.gallery_items[idx]
            results.append({
                "id": item["id"],
                "title": item["title"],
                "category": item["category"],
                "filename": item["filename"],
                "cosine_similarity": float(score),
                "softmax_probability": float(pr[idx].item()) * 100.0,
                "latency_ms": latency * 1000.0,
            })
        return results

    def classify_image_to_text(self, image_path: str, candidate_labels: List[str]) -> List[Dict[str, Any]]:
        """Zero-shot image classification: match image against candidate text descriptions."""
        img = Image.open(image_path).convert("RGB")
        img_emb = self.matcher.encode_images([img])
        text_embs = self.matcher.encode_text(candidate_labels)

        sim_matrix, probs = self.matcher.match(text_embs, img_emb)
        sims = sim_matrix.squeeze(-1)
        pr = probs.squeeze(-1)

        ranked = sorted(
            zip(candidate_labels, sims.tolist(), pr.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        return [{"label": l, "similarity": s, "probability": p * 100.0} for l, s, p in ranked]


# ==============================================================================
# 3. Diagnostic Visualization Suite
# ==============================================================================

def generate_visualization(
    gallery: ImageGallery,
    sim_matrix: np.ndarray,
    probs_matrix: np.ndarray,
    query_records: List[Dict[str, Any]],
    output_path: str = DEFAULT_OUTPUT
):
    """
    Generate professional 4-panel diagnostic figure using pure Matplotlib:
      (a) Cross-Modal Cosine Similarity Heatmap (Text Queries vs Gallery Images)
      (b) Softmax Probability Distribution per Query
      (c) Gallery Image Thumbnails with Titles and IDs
      (d) Cosine Margin & Confidence Profile across Queries
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    image_titles = [item["title"] for item in gallery.gallery_items]
    query_labels = [f"Q{i+1}" for i in range(len(query_records))]

    # -------------------------------------------------------------------------
    # Panel 1: Cross-Modal Cosine Similarity Heatmap
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    im1 = ax1.imshow(sim_matrix, cmap="magma", aspect="auto", vmin=0.1, vmax=0.35)
    ax1.set_xticks(range(len(image_titles)))
    ax1.set_xticklabels(image_titles, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax1.set_yticks(range(len(query_labels)))
    ax1.set_yticklabels(query_labels, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Gallery Images ($N=6$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Text Queries ($M=6$)", fontsize=10, fontweight="bold")
    ax1.set_title("Cross-Modal Cosine Similarity Matrix ($S_{ij} = \\hat{\\mathbf{u}}_i \\cdot \\hat{\\mathbf{v}}_j$)", fontsize=11, fontweight="bold")

    # Annotate values on diagonal
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            val = sim_matrix[i, j]
            color = "black" if val > 0.28 else "white"
            fontw = "bold" if i == j else "normal"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight=fontw)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Cosine Similarity", fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: Softmax Match Probability Distribution
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    diag_probs = [probs_matrix[i, i] * 100.0 for i in range(len(query_records))]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    bars = ax2.bar(query_labels, diag_probs, color=colors, edgecolor="black", alpha=0.85)
    for bar, p in zip(bars, diag_probs):
        ax2.text(bar.get_x() + bar.get_width()/2.0, p + 1.5, f"{p:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Benchmark Query Identifier", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Target Image Softmax Probability (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Target Image Matching Confidence (Temperature-Scaled Softmax)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    # -------------------------------------------------------------------------
    # Panel 3: Image Gallery Montage
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.axis("off")
    ax3.set_title("Gallery Image Collection ($256 \\times 256$ RGB)", fontsize=11, fontweight="bold")

    # Create composite sub-grid in ax3
    n_imgs = len(gallery.pil_images)
    rows, cols = 2, 3
    sub_w = 1.0 / cols
    sub_h = 0.45

    for idx, (img, item) in enumerate(zip(gallery.pil_images, gallery.gallery_items)):
        r = idx // cols
        c = idx % cols
        left = c * sub_w + 0.02
        bottom = 0.52 - r * sub_h
        sub_ax = ax3.inset_axes([left, bottom, sub_w - 0.04, sub_h - 0.08])
        sub_ax.imshow(img)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        sub_ax.set_title(f"{item['title']}", fontsize=8, fontweight="bold", pad=2)

    # -------------------------------------------------------------------------
    # Panel 4: Cosine Margin & Latency Profile
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    margins = [q["cosine_margin"] for q in query_records]
    latencies = [q["latency_ms"] for q in query_records]

    x = np.arange(len(query_labels))
    width = 0.35
    b1 = ax4.bar(x - width/2, margins, width, label="Cosine Margin (Top1 - Top2)", color="#2a9d8f", edgecolor="black")
    ax4.set_ylabel("Cosine Similarity Margin", color="#2a9d8f", fontsize=10, fontweight="bold")
    ax4.tick_params(axis="y", labelcolor="#2a9d8f")

    ax4_twin = ax4.twinx()
    b2 = ax4_twin.bar(x + width/2, latencies, width, label="Inference Latency (ms)", color="#e76f51", edgecolor="black")
    ax4_twin.set_ylabel("Latency (milliseconds)", color="#e76f51", fontsize=10, fontweight="bold")
    ax4_twin.tick_params(axis="y", labelcolor="#e76f51")

    for bar, val in zip(b1, margins):
        ax4.text(bar.get_x() + bar.get_width()/2.0, val + 0.005, f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax4.set_xticks(x)
    ax4.set_xticklabels(query_labels, fontsize=9, fontweight="bold")
    ax4.set_title("Discriminative Cosine Margin & Latency Profile", fontsize=11, fontweight="bold")

    bars_all = [b1, b2]
    labels_all = ["Cosine Margin", "Latency (ms)"]
    ax4.legend(bars_all, labels_all, loc="upper right", framealpha=0.9)

    plt.suptitle("OpenAI CLIP (ViT-B/32): Zero-Shot Text-to-Image Cross-Modal Retrieval", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved diagnostic visualization figure: {output_path}")


# ==============================================================================
# 4. Benchmark Execution Pipeline
# ==============================================================================

def run_benchmark(metadata_path: str = DEFAULT_METADATA) -> None:
    """Execute complete CLIP text-to-image benchmark and cross-modal evaluation."""
    gallery = ImageGallery(metadata_path)
    queries = gallery.benchmark_queries

    print("\n" + "=" * 80)
    print("OPENAI CLIP TEXT-TO-IMAGE BENCHMARK (ViT-B/32 - 151M Parameters)")
    print("=" * 80)

    # Encode all queries
    q_texts = [q["query"] for q in queries]
    t0 = time.perf_counter()
    q_embeddings = gallery.matcher.encode_text(q_texts)
    q_enc_time = (time.perf_counter() - t0) * 1000.0

    # Cross-modal matching matrix
    sim_matrix, probs_matrix = gallery.matcher.match(q_embeddings, gallery.image_embeddings)
    sim_np = sim_matrix.cpu().numpy()
    probs_np = probs_matrix.cpu().numpy()

    query_records = []
    top1_hits = 0
    mrr_total = 0.0

    for i, q in enumerate(queries):
        query_text = q["query"]
        target_id = q["target_id"]

        row_sims = sim_np[i]
        sorted_indices = np.argsort(row_sims)[::-1]

        rank = next((r + 1 for r, idx in enumerate(sorted_indices) if gallery.gallery_items[idx]["id"] == target_id), len(sorted_indices))
        mrr_total += 1.0 / rank

        is_hit = (rank == 1)
        if is_hit:
            top1_hits += 1

        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        cosine_margin = float(row_sims[top_idx] - row_sims[second_idx])

        top_item = gallery.gallery_items[top_idx]
        target_item = next(item for item in gallery.gallery_items if item["id"] == target_id)

        rec = {
            "query": query_text,
            "target_id": target_id,
            "target_title": target_item["title"],
            "rank": rank,
            "top_title": top_item["title"],
            "top_similarity": float(row_sims[top_idx]),
            "cosine_margin": cosine_margin,
            "target_prob": float(probs_np[i, top_idx]) * 100.0,
            "latency_ms": q_enc_time / len(queries),
        }
        query_records.append(rec)

        status = "MATCH" if is_hit else "MISS"
        print(f"\n[Q{i+1:02d}] \"{query_text}\"")
        print(f"  Target: '{target_item['title']}' | Retrieved Rank: #{rank} (Sim: {rec['top_similarity']:.3f}, Margin: {cosine_margin:.3f}) [{status}]")
        print(f"  Top Match: '{top_item['title']}' | Softmax Probability: {rec['target_prob']:.1f}%")

    # Generate Visualization Figure
    generate_visualization(gallery, sim_np, probs_np, query_records)

    # -------------------------------------------------------------------------
    # Aggregate Metrics Summary
    # -------------------------------------------------------------------------
    top1_acc = (top1_hits / len(queries)) * 100.0
    mrr = (mrr_total / len(queries)) * 100.0
    avg_margin = float(np.mean([q["cosine_margin"] for q in query_records]))
    avg_latency = float(np.mean([q["latency_ms"] for q in query_records]))

    print("\n" + "=" * 80)
    print("CLIP CROSS-MODAL RETRIEVAL BENCHMARK SUMMARY (N=6 Queries)")
    print("=" * 80)
    print(f"  Top-1 Retrieval Accuracy : {top1_acc:.1f}% (Zero-shot text-to-image match)")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.1f}%")
    print(f"  Average Cosine Margin    : {avg_margin:.4f} (Separation over runner-up)")
    print(f"  Average Query Latency    : {avg_latency:.2f} ms")
    print("=" * 80 + "\n")


def interactive_search():
    """Interactive console prompt for user natural language text-to-image search."""
    gallery = ImageGallery()
    print("\n" + "=" * 80)
    print("Interactive CLIP Text-to-Image Search (Enter 'q' to quit)")
    print("=" * 80)

    while True:
        try:
            prompt = input("\nEnter image description prompt: ").strip()
            if prompt.lower() == "q":
                break
            if not prompt:
                continue

            results = gallery.search_text_to_image(prompt, top_k=3)
            print(f"\nTop-{len(results)} Matching Images in Gallery:")
            for rank, r in enumerate(results, 1):
                print(f"  #{rank} \033[92m{r['title']}\033[0m ({r['filename']})")
                print(f"      Similarity: {r['cosine_similarity']:.4f} | Softmax Prob: {r['softmax_probability']:.1f}% | Latency: {r['latency_ms']:.1f}ms")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive search.")
            break


def main():
    parser = argparse.ArgumentParser(description="CLIP Multimodal Text-to-Image Embedding & Retrieval")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA, help="Path to gallery metadata JSON")
    parser.add_argument("--prompt", type=str, default="", help="Natural language text prompt to find matching image")
    parser.add_argument("--top_k", type=int, default=3, help="Number of images to return")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive search console")
    args = parser.parse_args()

    if args.interactive:
        interactive_search()
    elif args.prompt:
        gallery = ImageGallery(args.metadata)
        results = gallery.search_text_to_image(args.prompt, top_k=args.top_k)
        print("\n" + "=" * 80)
        print(f"SEARCH PROMPT: \"{args.prompt}\"")
        print("=" * 80)
        for rank, r in enumerate(results, 1):
            print(f"#{rank} {r['title']} ({r['filename']}) - Category: {r['category']}")
            print(f"   Cosine Sim: {r['cosine_similarity']:.4f} | Confidence: {r['softmax_probability']:.1f}% | Latency: {r['latency_ms']:.1f}ms\n")
    else:
        run_benchmark(args.metadata)


if __name__ == "__main__":
    main()
