"""
generate_plots.py
=================
Generate all publication-ready figures from experiment results.
"""
import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.size': 10,
})

OUTPUT_DIR = "/mnt/agents/output/resonance"


def plot_exp1_curves():
    path = os.path.join(OUTPUT_DIR, "results", "exp1_real.json")
    if not os.path.exists(path):
        print("exp1_real.json not found, skipping Exp1 plots")
        return
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Training loss curves
    ax = axes[0]
    for key, val in data.items():
        if isinstance(val, dict) and "history" in val:
            hist = val["history"]
            epochs = hist.get("epochs", list(range(1, len(hist["train_loss"])+1)))
            label = key.replace("exp1_", "").replace("_", " ").title()
            style = "-" if "standard" in key else "--"
            ax.plot(epochs, hist["train_loss"], style, label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Experiment 1: Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Val perplexity curves
    ax = axes[1]
    for key, val in data.items():
        if isinstance(val, dict) and "history" in val:
            hist = val["history"]
            epochs = hist.get("epochs", list(range(1, len(hist["val_ppl"])+1)))
            label = key.replace("exp1_", "").replace("_", " ").title()
            style = "-" if "standard" in key else "--"
            ax.plot(epochs, hist["val_ppl"], style, label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Experiment 1: Validation Perplexity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "exp1_training_curves.png"))
    plt.close(fig)
    print("Saved figures/exp1_training_curves.png")


def plot_exp2_scaling():
    path = os.path.join(OUTPUT_DIR, "results", "exp2_scaling.json")
    if not os.path.exists(path):
        print("exp2_scaling.json not found, skipping Exp2 plots")
        return
    with open(path) as f:
        data = json.load(f)

    standard_points = []
    resonance_points = []

    for key, val in data.items():
        if isinstance(val, dict) and "params" in val:
            params = val["params"]
            ppl = val["final_val_ppl"]
            if "standard" in key:
                standard_points.append((params, ppl))
            else:
                resonance_points.append((params, ppl))

    fig, ax = plt.subplots(figsize=(7, 5))
    if standard_points:
        xs, ys = zip(*sorted(standard_points))
        ax.plot(np.log10(xs), np.log10(ys), 'o-', label='Standard', linewidth=2, markersize=8, color='#1f77b4')
    if resonance_points:
        xs, ys = zip(*sorted(resonance_points))
        ax.plot(np.log10(xs), np.log10(ys), 's--', label='Resonance', linewidth=2, markersize=8, color='#ff7f0e')

    ax.set_xlabel("log10(Parameter Count)")
    ax.set_ylabel("log10(Validation Perplexity)")
    ax.set_title("Scaling Law: Parameters vs Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "scaling_law.png"))
    plt.close(fig)
    print("Saved figures/scaling_law.png")


def plot_exp3_compressibility():
    path = os.path.join(OUTPUT_DIR, "results", "exp3_compressibility.json")
    if not os.path.exists(path):
        print("exp3_compressibility.json not found, skipping Exp3 plots")
        return
    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'standard': '#1f77b4', 'resonance': '#ff7f0e'}
    markers = {'standard': 'o', 'resonance': 's'}

    for variant in ["standard", "resonance"]:
        if variant not in data:
            continue
        tests = data[variant]["compression_tests"]
        crs = [t["compression_ratio"] for t in tests]
        ppls = [t["ppl"] for t in tests]
        labels = [t["method"] for t in tests]
        ax.scatter(crs, ppls, label=variant.title(), color=colors[variant],
                   marker=markers[variant], s=80, alpha=0.8)
        for cr, ppl, lab in zip(crs, ppls, labels):
            ax.annotate(lab, (cr, ppl), fontsize=7, alpha=0.7)

    ax.set_xlabel("Compression Ratio (higher = more compressed)")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Compressibility: Compression vs Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "compressibility_real.png"))
    plt.close(fig)
    print("Saved figures/compressibility_real.png")


def plot_exp4_gestalt():
    path = os.path.join(OUTPUT_DIR, "results", "exp4_gestalt.json")
    if not os.path.exists(path):
        print("exp4_gestalt.json not found, skipping Exp4 plots")
        return
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: PCA dimensionality comparison
    ax = axes[0]
    categories = ["semantic", "phase_proj", "phase_raw"]
    values = [data["pca_dim_95"].get(c, 0) for c in categories]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar([c.replace("_", "\n") for c in categories], values, color=colors, edgecolor='black')
    ax.set_ylabel("Dimensions (95% variance)")
    ax.set_title("Stream Dimensionality")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, str(v), ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Cross-predictability and compressibility
    ax = axes[1]
    metrics = []
    vals = []
    if "cross_predictability_r2" in data:
        metrics.extend(["Phase→Sem R²", "Sem→Phase R²"])
        vals.extend([
            data["cross_predictability_r2"]["phase_to_semantic"],
            data["cross_predictability_r2"]["semantic_to_phase"],
        ])
    if "independent_compressibility" in data:
        metrics.extend(["Phase ΔPPL", "Semantic ΔPPL"])
        vals.extend([
            data["independent_compressibility"]["phase_delta"],
            data["independent_compressibility"]["semantic_delta"],
        ])
    colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(metrics, vals, color=colors2[:len(vals)], edgecolor='black')
    ax.set_ylabel("Value")
    ax.set_title("Stream Interactions & Compressibility")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}", ha='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "gestalt_real.png"))
    plt.close(fig)
    print("Saved figures/gestalt_real.png")


def generate_report():
    """Generate markdown summary report."""
    lines = ["# Resonance Transformer: Experiment Summary\n"]
    lines.append("## Hyperparameters & Compute Notes\n")
    lines.append("- **Device**: CPU (torch.set_num_threads(4))")
    lines.append("- **Dataset**: Structured synthetic tokens with 10 modes, local Markov structure, and long-range (every 8th) dependencies")
    lines.append("- **Vocab size**: 5000")
    lines.append("- **Sequence length**: 64 (reduced from 128 for CPU compute; noted in analysis)")
    lines.append("- **Optimizer**: AdamW, lr=3e-4, weight_decay=0.01, cosine schedule")
    lines.append("- **Batch size**: 32")
    lines.append("- **Epochs**: 5 for Small, 3 for Medium (Exp 1); 2 for all scales (Exp 2)")
    lines.append("- **Note**: Runs calibrated to fit within ~10 min per shell call on 2-thread CPU\n")

    # Exp 1
    path = os.path.join(OUTPUT_DIR, "results", "exp1_real.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        lines.append("## Experiment 1: Real Data Comparison\n")
        lines.append("| Model | Params | Final Val PPL | Val Acc@1 | Val Acc@5 | Train Time |")
        lines.append("|-------|--------|---------------|-----------|-----------|------------|")
        for key, val in data.items():
            if isinstance(val, dict) and "params" in val:
                name = key.replace("exp1_", "").replace("_", " ")
                lines.append(f"| {name} | {val['params']:,} | {val['final_val_ppl']:.2f} | {val['final_val_acc@1']:.4f} | {val['final_val_acc@5']:.4f} | {val['train_time_sec']:.0f}s |")
        if "perturbation_test" in data:
            lines.append(f"\n**Perturbation Test**: KL divergence at position 17 = {data['perturbation_test']['kl_divergence_at_pos17']:.4f}")
        lines.append("")

    # Exp 2
    path = os.path.join(OUTPUT_DIR, "results", "exp2_scaling.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        lines.append("## Experiment 2: Mini Scale Study\n")
        lines.append("| Scale | Variant | Params | Final Val PPL | Val Acc@1 | Train Time |")
        lines.append("|-------|---------|--------|---------------|-----------|------------|")
        for key, val in data.items():
            if isinstance(val, dict) and "params" in val:
                lines.append(f"| {val['scale']} | {val['variant']} | {val['params']:,} | {val['final_val_ppl']:.2f} | {val['final_val_acc@1']:.4f} | {val['train_time_sec']:.0f}s |")
        lines.append("")

    # Exp 3
    path = os.path.join(OUTPUT_DIR, "results", "exp3_compressibility.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        lines.append("## Experiment 3: Compressibility\n")
        for variant in ["standard", "resonance"]:
            if variant in data:
                lines.append(f"\n### {variant.title()}\n")
                lines.append(f"- Baseline FP32 PPL: {data[variant]['baseline_ppl']:.2f}")
                lines.append("| Method | Compression Ratio | PPL |")
                lines.append("|--------|-------------------|-----|")
                for t in data[variant]["compression_tests"]:
                    lines.append(f"| {t['method']} | {t['compression_ratio']:.2f}x | {t['ppl']:.2f} |")
        lines.append("")

    # Exp 4
    path = os.path.join(OUTPUT_DIR, "results", "exp4_gestalt.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        lines.append("## Experiment 4: Gestalt Stream Analysis\n")
        lines.append(f"- **PCA 95% dims**: Semantic={data['pca_dim_95']['semantic']}, Phase(proj)={data['pca_dim_95']['phase_proj']}, Phase(raw)={data['pca_dim_95']['phase_raw']}")
        lines.append(f"- **Cross-predictability**: Phase→Sem R²={data['cross_predictability_r2']['phase_to_semantic']:.4f}, Sem→Phase R²={data['cross_predictability_r2']['semantic_to_phase']:.4f}")
        lines.append(f"- **Resonance matrix**: Entropy={data['resonance_matrix']['entropy']:.4f}, Effective rank={data['resonance_matrix']['effective_rank']:.2f}")
        lines.append(f"- **Alpha blend**: mean={data['alpha_stats']['mean']:.4f}, std={data['alpha_stats']['std']:.4f}")
        lines.append(f"- **Independent compressibility**: Phase INT4 ΔPPL={data['independent_compressibility']['phase_delta']:.4f}, Semantic INT4 ΔPPL={data['independent_compressibility']['semantic_delta']:.4f}")
        lines.append("")

    # Assessment
    lines.append("## Assessment: Does Resonance Show Advantages?\n")
    lines.append("Based on the experiments above, we evaluate the Resonance Transformer against the Standard baseline:\n")

    # Read results for comparison
    exp1_path = os.path.join(OUTPUT_DIR, "results", "exp1_real.json")
    exp2_path = os.path.join(OUTPUT_DIR, "results", "exp2_scaling.json")
    
    has_data = False
    if os.path.exists(exp1_path) and os.path.exists(exp2_path):
        with open(exp1_path) as f:
            d1 = json.load(f)
        with open(exp2_path) as f:
            d2 = json.load(f)
        
        # Compare per scale
        for scale in ["small", "medium"]:
            std_key = f"exp1_{scale}_standard"
            res_key = f"exp1_{scale}_resonance"
            if std_key in d1 and res_key in d1:
                std_ppl = d1[std_key]["final_val_ppl"]
                res_ppl = d1[res_key]["final_val_ppl"]
                delta = std_ppl - res_ppl
                lines.append(f"- **{scale.title()}**: Resonance PPL={res_ppl:.2f} vs Standard={std_ppl:.2f} (Δ={delta:+.2f})")
                has_data = True
        
        if has_data:
            lines.append("")
            lines.append("**Key Observations**:")
            lines.append("1. The Resonance model incorporates phase-aware attention bias, which may capture long-range structural patterns in the synthetic data.")
            lines.append("2. If Resonance achieves lower PPL, this suggests the phase embeddings and resonance bias provide useful inductive bias.")
            lines.append("3. The cross-predictability and compressibility metrics reveal whether the dual-stream architecture factorizes information effectively.")
    
    if not has_data:
        lines.append("- Results pending experiment completion.\n")

    lines.append("\n---\n*Report generated automatically from experiment outputs.*")

    report_path = os.path.join(OUTPUT_DIR, "results", "experiment_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)
    plot_exp1_curves()
    plot_exp2_scaling()
    plot_exp3_compressibility()
    plot_exp4_gestalt()
    generate_report()
    print("\nAll plots and report generated.")
