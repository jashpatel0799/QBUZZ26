"""
Plotting utilities.
Generates all 6 required plots and saves them as PNGs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from typing import Dict, List, Any
import logging

from config import EXIT_CHECKPOINTS, PLOT_DIR

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")


def plot_train_losses(history: List[Dict[str, float]], save_path: str = None) -> plt.Figure:
    """Plot training loss for all 4 exits + total over steps."""
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=df, x="step", y="train_loss_total", label="Total Loss", linewidth=2, color="black", ax=ax)
    
    colors = sns.color_palette("viridis", len(EXIT_CHECKPOINTS))
    for i, k in enumerate(EXIT_CHECKPOINTS):
        col = f"train_loss_exit_{k}"
        if col in df.columns:
            sns.lineplot(data=df, x="step", y=col, label=f"Exit {k}", color=colors[i], alpha=0.7, ax=ax)
            
    ax.set_title("Training Loss per Exit Checkpoint")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_yscale("log")
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_val_losses(history: List[Dict[str, float]], save_path: str = None) -> plt.Figure:
    """Plot validation loss for all 4 exits."""
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("viridis", len(EXIT_CHECKPOINTS))
    for i, k in enumerate(EXIT_CHECKPOINTS):
        col = f"val_loss_exit_{k}"
        if col in df.columns:
            sns.lineplot(data=df, x="step", y=col, label=f"Exit {k} Val Loss", color=colors[i], marker="o", ax=ax)
            
    ax.set_title("Validation Loss per Exit Checkpoint")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross Entropy Loss")
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_perplexity(history: List[Dict[str, float]], save_path: str = None) -> plt.Figure:
    """Plot validation perplexity for all 4 exits."""
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("viridis", len(EXIT_CHECKPOINTS))
    for i, k in enumerate(EXIT_CHECKPOINTS):
        loss_col = f"val_loss_exit_{k}"
        if loss_col in df.columns:
            # PPL = exp(loss)
            import numpy as np
            ppl = np.exp(df[loss_col])
            sns.lineplot(x=df["step"], y=ppl, label=f"Exit {k} PPL", color=colors[i], marker="s", ax=ax)
            
    ax.set_title("Validation Perplexity per Exit")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_tradeoff_curve(tau_results: Dict[float, Dict[str, float]], save_path: str = None) -> plt.Figure:
    """Plot Speed vs Quality tradeoff curve (Avg Exit Layer vs Perplexity)."""
    taus = list(tau_results.keys())
    ppls = [res["ppl"] for res in tau_results.values()]
    avg_exits = [res["avg_exit_layer"] for res in tau_results.values()]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(avg_exits, ppls, c=taus, cmap="coolwarm", s=100, edgecolor="black")
    ax.plot(avg_exits, ppls, 'k--', alpha=0.5)
    
    for i, txt in enumerate(taus):
        ax.annotate(f"τ={txt}", (avg_exits[i], ppls[i]), xytext=(5, 5), textcoords="offset points")
        
    cbar = fig.colorbar(scatter)
    cbar.set_label("Entropy Threshold (τ)")
    
    ax.set_title("Elastic Tradeoff: Speed vs Quality")
    ax.set_xlabel("Average Exit Layer (Lower = Faster)")
    ax.set_ylabel("Perplexity (Lower = Better)")
    
    # Invert x-axis so faster is on the right? Or keep left. 
    # Usually we want right and up to be better, but here left and down is better.
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_phase_curriculum(phase1_weights, phase2_weights, phase1_steps, total_steps, save_path: str = None) -> plt.Figure:
    """Plot the curriculum phase transition of loss weights."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    steps = [0, phase1_steps, phase1_steps + 1, total_steps]
    
    colors = sns.color_palette("viridis", len(EXIT_CHECKPOINTS))
    for i, k in enumerate(EXIT_CHECKPOINTS):
        w1 = phase1_weights[i]
        w2 = phase2_weights[i]
        weights = [w1, w1, w2, w2]
        ax.plot(steps, weights, label=f"Exit {k} weight", color=colors[i], linewidth=2)
        
    ax.axvline(x=phase1_steps, color='red', linestyle='--', alpha=0.5, label="Phase Transition")
    
    ax.set_title("Curriculum Training Phase Transition")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss Weight (λ)")
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_all(train_history, val_history, tau_results, config):
    """Generate and save all plots."""
    logger.info("Generating plots...")
    
    plot_train_losses(train_history, os.path.join(PLOT_DIR, "train_losses.png"))
    plot_val_losses(val_history, os.path.join(PLOT_DIR, "val_losses.png"))
    plot_perplexity(val_history, os.path.join(PLOT_DIR, "val_perplexity.png"))
    
    if tau_results:
        plot_tradeoff_curve(tau_results, os.path.join(PLOT_DIR, "tradeoff_curve.png"))
        
    plot_phase_curriculum(
        config["PHASE1_EXIT_WEIGHTS"],
        config["PHASE2_EXIT_WEIGHTS"],
        config["TRAIN_CONFIG"]["phase1_steps"],
        config["TRAIN_CONFIG"]["total_steps"],
        os.path.join(PLOT_DIR, "curriculum.png")
    )
