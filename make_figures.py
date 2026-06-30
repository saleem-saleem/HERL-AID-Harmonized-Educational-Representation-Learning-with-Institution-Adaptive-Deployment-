"""Generate publication-quality figures (>=300 DPI) from results/*.csv.

Uses the navy/blue/orange/green palette. Produces sensitivity plots and the SOTA
benchmark bar chart. Run the table scripts first to populate results/.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"navy": "#1F3864", "blue": "#2E74B5", "orange": "#C55A11", "green": "#538135"}
plt.rcParams.update({"font.size": 11, "font.family": "serif", "figure.dpi": 300})


def fig_sensitivity_lambda(path="results/sensitivity_lambda.csv", out="figures"):
    if not os.path.exists(path): return
    df = pd.read_csv(path); os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(df["lambda"], df["auc_cross_mean"], "-o", color=PALETTE["blue"], label="Cross-domain AUC")
    ax.set_xlabel(r"alignment weight $\lambda$"); ax.set_ylabel("Cross-domain AUC")
    ax.axvline(0.5, ls="--", color=PALETTE["orange"], label="default")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{out}/sensitivity_lambda.png", dpi=300)
    fig.savefig(f"{out}/sensitivity_lambda.pdf"); plt.close(fig)


def fig_sensitivity_alpha(path="results/sensitivity_alpha.csv", out="figures"):
    if not os.path.exists(path): return
    df = pd.read_csv(path); os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(df["alpha"], df["auc_cross_std"], "-o", color=PALETTE["green"], label="Cross-institution AUC std")
    ax.set_xlabel(r"stability weight $\alpha$"); ax.set_ylabel("Cross-institution AUC std")
    ax.axvline(0.3, ls="--", color=PALETTE["orange"], label="default")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{out}/sensitivity_alpha.png", dpi=300)
    fig.savefig(f"{out}/sensitivity_alpha.pdf"); plt.close(fig)


def fig_sota(path="results/table14_sota.csv", out="figures"):
    if not os.path.exists(path): return
    df = pd.read_csv(path); os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = [PALETTE["navy"] if m == "HERL-AID" else PALETTE["blue"] for m in df["model"]]
    ax.bar(df["model"], df["auc_cross"], color=colors)
    ax.set_ylabel("Cross-domain AUC"); ax.set_ylim(0.7, 0.95)
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(f"{out}/sota_cross_auc.png", dpi=300); fig.savefig(f"{out}/sota_cross_auc.pdf"); plt.close(fig)


def main():
    fig_sensitivity_lambda(); fig_sensitivity_alpha(); fig_sota()
    print("figures written to figures/ (300 DPI PNG + vector PDF)")


if __name__ == "__main__":
    main()
