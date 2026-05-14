"""Optional matplotlib visualizations for split reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def make_visualizations(metrics: List[Dict[str, Any]], output_dir: str | Path) -> List[str]:
    if not metrics:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_rate = [float(m.get("pass_rate", 0.0)) for m in metrics]
    reward_std = [float(m.get("reward_std", 0.0)) for m in metrics]
    length = [float(m.get("length_mean_tokens", 0.0)) for m in metrics]
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].hist(pass_rate, bins=20, color="#2f6f8f")
    axes[0, 0].set_title("pass_rate histogram")
    axes[0, 1].hist(reward_std, bins=20, color="#8f5a2f")
    axes[0, 1].set_title("reward_std histogram")
    axes[0, 2].hist(length, bins=20, color="#3f7f4f")
    axes[0, 2].set_title("length distribution")
    axes[1, 0].scatter(pass_rate, reward_std, s=8, alpha=0.6)
    axes[1, 0].set_xlabel("pass_rate")
    axes[1, 0].set_ylabel("reward_std")
    axes[1, 1].scatter(pass_rate, length, s=8, alpha=0.6)
    axes[1, 1].set_xlabel("pass_rate")
    axes[1, 1].set_ylabel("length_mean_tokens")
    axes[1, 2].scatter(reward_std, length, s=8, alpha=0.6)
    axes[1, 2].set_xlabel("reward_std")
    axes[1, 2].set_ylabel("length_mean_tokens")
    fig.tight_layout()
    path = output_dir / "split_visualization.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.name)
    return paths
