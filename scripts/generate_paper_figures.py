"""Generate publication-vector figures from values reported in paper_mtc.tex.

The script deliberately contains only already reported aggregate statistics; it
does not reconstruct or alter unavailable record-level experimental data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
BLUE = "#2b65a0"
LIGHT_BLUE = "#d6e4f1"
GREEN = "#2f7d32"
ORANGE = "#ef7d00"
RED = "#bd1f20"
PURPLE = "#5e50a1"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def family_distribution():
    labels = [
        "Formation energy", "Band gap", "Superconductivity", "Chemical formula",
        "Temperature", "Stability", "Pressure",
    ]
    values = [22, 21, 4, 2, 2, 2, 1]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.barh(labels[::-1], values[::-1], color=[BLUE] + [LIGHT_BLUE] * 6,
                   edgecolor=BLUE, linewidth=0.9)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=10)
    ax.set_xlabel("Diagnostic divergence cases")
    ax.set_xlim(0, 25)
    ax.set_title("Constraint-family distribution of 54 divergence cases", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig3_divergence_cases_by_family")


def constraint_ablation():
    labels = ["Full", "L0", "L1", "-FE", "-BG", "-Exc."]
    divergence = [54, 29, 25, 32, 34, 69]
    kappas = [0.773, 0.185, 0.580, 0.597, 0.488, 0.712]
    colors = [BLUE, LIGHT_BLUE, GREEN, ORANGE, PURPLE, RED]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45), constrained_layout=True)
    for ax, values, title, ylabel, fmt in [
        (axes[0], divergence, "Rule-ablation effects on semantic-scientific divergence",
         "High-support/low-constraint cases", "{:d}"),
        (axes[1], kappas, "Concordance with the rule-aligned\nQwen-plus reviewer",
         "Cohen's kappa (development set)", "{:.3f}"),
    ]:
        bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor="#334155", linewidth=0.55)
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=7.5)
        ax.set_title(title, fontsize=10, pad=7)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (1.2 if max(values) > 1 else 0.025),
                    fmt.format(value), ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 0.9)
    save(fig, "fig10_constraint_ablation")


def qwen_disagreements():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.85, 1.15]})
    labels = ["Qwen-plus positive /\nPI-SRET negative", "PI-SRET positive /\nQwen-plus negative"]
    values = [218, 63]
    bars = axes[0].bar(labels, values, color=[RED, ORANGE], edgecolor="#334155", linewidth=0.6)
    axes[0].set_title("Development-set disagreement patterns\n($n=281$)", fontsize=10, pad=7)
    axes[0].set_ylabel("Disagreement count")
    axes[0].spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 7, f"{value}\n{value / 281:.1%}",
                     ha="center", va="bottom", fontsize=8)

    source_labels = [
        "Formation energy\nQwen+ / PI-SRET-", "Pressure\nQwen+ / PI-SRET-",
        "Band gap\nQwen+ / PI-SRET-", "Stability\nPI-SRET+ / Qwen-",
        "Superconductivity\nPI-SRET+ / Qwen-", "Temperature\nQwen+ / PI-SRET-",
        "Chemical formula\nQwen+ / PI-SRET-",
    ]
    source_values = [123, 34, 27, 21, 15, 10, 10]
    source_colors = [RED, RED, RED, ORANGE, ORANGE, RED, RED]
    ypos = np.arange(len(source_labels))[::-1]
    bars = axes[1].barh(ypos, source_values, color=source_colors, edgecolor="#334155", linewidth=0.6)
    axes[1].set_yticks(ypos, source_labels)
    axes[1].set_xlabel("Count")
    axes[1].set_title("Largest disagreement sources", fontsize=10, pad=7)
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, source_values):
        axes[1].text(value + 3, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=8)
    axes[1].set_xlim(0, 145)
    save(fig, "fig11_llm_judge_error_analysis")


def human_reference():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.8, 1.2]})
    labels = ["A\n($n=200$)", "B\n($n=200$)", "Adjudicated\nreference"]
    violations = np.array([108, 81, 85])
    nonviolations = np.array([92, 119, 115])
    axes[0].bar(labels, violations, label="Violation", color=RED, edgecolor="#334155", linewidth=0.55)
    axes[0].bar(labels, nonviolations, bottom=violations, label="No violation", color=LIGHT_BLUE,
                edgecolor="#2b65a0", linewidth=0.55)
    axes[0].set_ylabel("Cases")
    axes[0].set_ylim(0, 220)
    axes[0].set_title("Blinded reviewers and human-adjudicated\nreference", fontsize=10, pad=7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    matrix = np.array([[100, 15], [3, 82]])
    image = axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=100)
    axes[1].set_xticks([0, 1], ["PI-SRET:\nno violation", "PI-SRET:\nviolation"])
    axes[1].set_yticks([0, 1], ["Human-adjudicated:\nno violation", "Human-adjudicated:\nviolation"])
    axes[1].set_title("PI-SRET evaluation against the\nhuman-adjudicated reference", fontsize=10, pad=7)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12,
                         color="white" if matrix[i, j] > 50 else "#263238", fontweight="bold")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    save(fig, "fig12_expert_validation")


def fixed_context_stress_test():
    labels = ["Local-rule RAG", "Qwen-max RAG", "DeepSeek-R1 RAG"]
    lexical = [0.512, 0.481, 0.689]
    nli = [0.585, 0.456, 0.736]
    constraint = [0.333, 0.389, 0.325]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), constrained_layout=True)
    xpos = np.arange(len(labels))
    width = 0.36
    for ax, first, second, title, first_label, second_label, first_color, second_color in [
        (axes[0], lexical, nli, "Semantic-support proxies on fixed\nretrieved contexts", "Lexical support", "NLI proxy", BLUE, ORANGE),
    ]:
        ax.bar(xpos - width / 2, first, width, label=first_label, color=first_color)
        ax.bar(xpos + width / 2, second, width, label=second_label, color=second_color)
        ax.set_xticks(xpos, labels, rotation=16, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean score / rate")
        ax.set_title(title, fontsize=10, pad=7)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].bar(xpos, constraint, width=0.58, color=GREEN, label="Constraint-pass rate")
    axes[1].set_xticks(xpos, labels, rotation=16, ha="right")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Mean rate")
    axes[1].set_title("PI-SRET constraint-pass rate on fixed\nretrieved contexts", fontsize=10, pad=7)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].spines[["top", "right"]].set_visible(False)
    save(fig, "fig13_frozen_heldout_three_system")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    family_distribution()
    constraint_ablation()
    qwen_disagreements()
    human_reference()
    fixed_context_stress_test()
