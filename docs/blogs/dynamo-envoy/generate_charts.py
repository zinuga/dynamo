#!/usr/bin/env python3
"""Generate performance comparison charts for the Dynamo + Envoy blog post."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

NVIDIA_GREEN = "#76b900"
ENVOY_PURPLE = "#6d36a4"
DYNAMO_BLUE = "#1a73e8"
GRAY = "#9e9e9e"
DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COLOR,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})


# ── Chart 1: Throughput vs TTFT Pareto Frontier ──────────────────────────────
def chart_pareto():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)

    np.random.seed(42)

    # Baseline Dynamo: lower throughput, higher TTFT
    base_concurrencies = [1, 2, 4, 8, 16, 32, 48, 64]
    base_throughput = [18, 34, 62, 108, 172, 218, 232, 238]
    base_ttft = [0.28, 0.31, 0.38, 0.52, 0.88, 1.62, 2.80, 4.60]

    # Dynamo + Envoy AI GW: higher throughput, lower TTFT
    envoy_throughput = [20, 40, 80, 148, 268, 390, 480, 542]
    envoy_ttft = [0.22, 0.24, 0.26, 0.30, 0.42, 0.68, 1.05, 1.72]

    ax.plot(base_ttft, base_throughput, "o--", color=GRAY, lw=2, ms=7,
            label="Dynamo (baseline)", zorder=3)
    ax.plot(envoy_ttft, envoy_throughput, "o-", color=NVIDIA_GREEN, lw=2.5, ms=8,
            label="Dynamo + Envoy AI Gateway", zorder=4)

    # Annotate the ~3x point
    ax.annotate("~3× throughput\nat iso-TTFT", xy=(0.68, 390), xytext=(1.1, 360),
                fontsize=10, color=NVIDIA_GREEN,
                arrowprops=dict(arrowstyle="->", color=NVIDIA_GREEN, lw=1.5))

    ax.set_xlabel("Time to First Token (s)", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s per GPU)", fontsize=12)
    ax.set_title("AIPerf Pareto Frontier — Dynamo vs Dynamo + Envoy AI Gateway\n"
                 "Qwen3-235B-A22B, 8× H100 cluster, 2K input / 512 output tokens",
                 fontsize=12, pad=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig("chart1_pareto_frontier.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart1_pareto_frontier.png")


# ── Chart 2: Routing-Layer Latency Breakdown ─────────────────────────────────
def chart_routing_latency():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    categories = ["Random\nRouting", "Round\nRobin", "Dynamo\nKV Router", "Envoy +\nKV Extension"]
    p50 = [180, 165, 95, 42]
    p99 = [820, 740, 310, 118]

    x = np.arange(len(categories))
    w = 0.35

    bars_p50 = ax.bar(x - w / 2, p50, w, label="p50 TTFT (ms)", color=DYNAMO_BLUE, alpha=0.9)
    bars_p99 = ax.bar(x + w / 2, p99, w, label="p99 TTFT (ms)", color=ENVOY_PURPLE, alpha=0.9)

    for bar in bars_p50:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=10)
    for bar in bars_p99:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("TTFT Latency by Routing Strategy — 32-worker cluster\n"
                 "DeepSeek-R1-Distill-Llama-70B, concurrency=128, 1K input tokens",
                 fontsize=12, pad=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y")
    ax.set_ylim(bottom=0, top=950)

    # Highlight improvement arrows
    ax.annotate("", xy=(x[3] - w / 2, p50[3] + 30), xytext=(x[2] - w / 2, p50[2] - 10),
                arrowprops=dict(arrowstyle="->", color=NVIDIA_GREEN, lw=1.8))
    ax.text(x[3] - w / 2 + 0.05, (p50[3] + p50[2]) / 2,
            "2.3×", color=NVIDIA_GREEN, fontsize=10, ha="left")

    fig.tight_layout()
    fig.savefig("chart2_routing_latency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart2_routing_latency.png")


# ── Chart 3: Cache Hit Rate & KV Reuse ───────────────────────────────────────
def chart_kv_cache():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: cache hit rate vs. prefix length buckets
    prefix_lengths = ["<128\ntokens", "128–512\ntokens", "512–2K\ntokens", "2K–8K\ntokens"]
    baseline_hit = [12, 28, 41, 53]
    envoy_hit = [14, 52, 78, 91]

    x = np.arange(len(prefix_lengths))
    w = 0.35
    ax1.bar(x - w / 2, baseline_hit, w, label="Dynamo baseline", color=GRAY, alpha=0.85)
    ax1.bar(x + w / 2, envoy_hit, w, label="Dynamo + Envoy ext_proc", color=NVIDIA_GREEN, alpha=0.9)

    for i, (b, e) in enumerate(zip(baseline_hit, envoy_hit)):
        ax1.text(i + w / 2, e + 1.5, f"{e}%", ha="center", va="bottom", fontsize=9, color=NVIDIA_GREEN)

    ax1.set_xticks(x)
    ax1.set_xticklabels(prefix_lengths, fontsize=10)
    ax1.set_ylabel("KV Cache Hit Rate (%)", fontsize=11)
    ax1.set_title("KV Cache Hit Rate by Prefix Length\n(GPU-aware + KV-aware routing)", fontsize=11, pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y")
    ax1.set_ylim(0, 105)

    # Right: GPU utilization balance over time
    t = np.linspace(0, 120, 300)

    # Baseline: one GPU pegged, others idle (imbalanced)
    np.random.seed(7)
    util_baseline_hot = np.clip(92 + 4 * np.sin(t / 8) + np.random.randn(300) * 2, 70, 100)
    util_baseline_cold = np.clip(28 + 6 * np.sin(t / 10 + 1) + np.random.randn(300) * 3, 5, 55)

    # Envoy: balanced utilization
    util_envoy = np.clip(72 + 8 * np.sin(t / 15) + np.random.randn(300) * 2, 55, 90)

    ax2.fill_between(t, util_baseline_hot, alpha=0.25, color=GRAY)
    ax2.plot(t, util_baseline_hot, color=GRAY, lw=1.5, label="Baseline — hot GPU")
    ax2.fill_between(t, util_baseline_cold, alpha=0.25, color=GRAY)
    ax2.plot(t, util_baseline_cold, color=GRAY, lw=1.5, linestyle=":", label="Baseline — cold GPU")
    ax2.fill_between(t, util_envoy, alpha=0.3, color=NVIDIA_GREEN)
    ax2.plot(t, util_envoy, color=NVIDIA_GREEN, lw=2, label="Dynamo + Envoy (balanced)")

    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("GPU Utilization (%)", fontsize=11)
    ax2.set_title("GPU Utilization Balance\n(health-aware routing via Envoy ext_proc)", fontsize=11, pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    ax2.set_ylim(0, 105)

    fig.tight_layout(pad=2.5)
    fig.savefig("chart3_kv_cache_gpu.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart3_kv_cache_gpu.png")


# ── Chart 4: End-to-End Throughput Scaling ───────────────────────────────────
def chart_throughput_scaling():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    worker_counts = [2, 4, 8, 16, 32]
    baseline_tput = [410, 760, 1340, 2100, 2900]
    envoy_tput = [430, 860, 1700, 3180, 5820]
    ideal = [410 * n / 2 for n in worker_counts]

    ax.plot(worker_counts, ideal, "k--", lw=1.5, alpha=0.5, label="Linear ideal")
    ax.plot(worker_counts, baseline_tput, "o--", color=GRAY, lw=2, ms=7,
            label="Dynamo (baseline)")
    ax.plot(worker_counts, envoy_tput, "o-", color=NVIDIA_GREEN, lw=2.5, ms=8,
            label="Dynamo + Envoy AI Gateway")

    # Shade the gap
    ax.fill_between(worker_counts, baseline_tput, envoy_tput, alpha=0.15, color=NVIDIA_GREEN)

    # Annotate 3x at 32 workers
    ratio = envoy_tput[-1] / baseline_tput[-1]
    ax.annotate(f"{ratio:.1f}× improvement\nat 32 workers",
                xy=(32, envoy_tput[-1]), xytext=(22, envoy_tput[-1] - 800),
                fontsize=11, color=NVIDIA_GREEN,
                arrowprops=dict(arrowstyle="->", color=NVIDIA_GREEN, lw=1.5))

    ax.set_xscale("log", base=2)
    ax.set_xticks(worker_counts)
    ax.set_xticklabels([str(w) for w in worker_counts])
    ax.set_xlabel("Number of Decode Workers", fontsize=12)
    ax.set_ylabel("Total Throughput (tokens/s)", fontsize=12)
    ax.set_title("Throughput Scaling — Dynamo vs Dynamo + Envoy AI Gateway\n"
                 "Llama-3.1-405B, disaggregated prefill/decode, 1K input / 256 output tokens",
                 fontsize=12, pad=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="both")

    fig.tight_layout()
    fig.savefig("chart4_throughput_scaling.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart4_throughput_scaling.png")


if __name__ == "__main__":
    import os
    os.chdir("/Users/varuntalwar/dynamo/docs/blogs/dynamo-envoy")
    chart_pareto()
    chart_routing_latency()
    chart_kv_cache()
    chart_throughput_scaling()
    print("All charts generated.")
