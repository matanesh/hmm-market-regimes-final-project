"""
Analysis module for HMM market regimes project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional


def analyze_transitions(transmat: np.ndarray, state_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Analyze the transition matrix of an HMM.

    Parameters:
    -----------
    transmat : np.ndarray
        Transition matrix of shape (n_states, n_states) where transmat[i, j] 
        is the probability of transitioning from state i to state j.
    state_names : list, optional
        Names for the states. If None, states are labeled 0, 1, ..., n_states-1.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing transition analysis metrics.
    """
    n_states = transmat.shape[0]
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]
    elif len(state_names) != n_states:
        raise ValueError("Length of state_names must match number of states")
    
    # 1. Average diagonal probability (persistence)
    avg_diagonal = np.mean(np.diag(transmat))
    
    # 2. Entropy of each row (uncertainty in next state given current state)
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-10
    row_entropies = -np.sum(transmat * np.log(transmat + epsilon), axis=1)
    avg_entropy = np.mean(row_entropies)
    
    # 3. Mean first passage time (approximate)
    # For ergodic chains, mean recurrence time for state i is 1/pi_i where pi is stationary distribution
    # We'll compute the stationary distribution by solving pi = pi * transmat
    # Using the eigenvector corresponding to eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    # Find the eigenvector corresponding to eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    # Normalize so that sum is 1
    stationary = stationary / np.sum(stationary)
    # Mean recurrence time
    mean_recurrence = 1.0 / stationary
    
    # 4. Percentage of variance explained by the first eigenvalue (for reversible chains)
    # Sort eigenvalues by absolute value in descending order
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    if len(sorted_eigenvalues) > 1:
        spectral_gap = sorted_eigenvalues[0] - sorted_eigenvalues[1]
    else:
        spectral_gap = 0.0
    
    return {
        "n_states": n_states,
        "state_names": state_names,
        "transition_matrix": transmat.copy(),
        "average_diagonal_probability": avg_diagonal,
        "average_row_entropy": avg_entropy,
        "stationary_distribution": stationary,
        "mean_recurrence_time": mean_recurrence,
        "spectral_gap": spectral_gap,
        "eigenvalues": eigenvalues,
    }


def plot_transition_heatmap(transmat: np.ndarray, 
                           state_names: Optional[list] = None,
                           title: str = "Transition Matrix Heatmap",
                           cmap: str = "Blues",
                           fmt: str = ".2f",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a heatmap of the transition matrix.

    Parameters:
    -----------
    transmat : np.ndarray
        Transition matrix of shape (n_states, n_states).
    state_names : list, optional
        Names for the states. If None, states are labeled 0, 1, ..., n_states-1.
    title : str, default "Transition Matrix Heatmap"
        Title for the plot.
    cmap : str, default "Blues"
        Colormap for the heatmap.
    fmt : str, default ".2f"
        Format string for the annotations.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns:
    --------
    plt.Figure
        The matplotlib figure object.
    """
    n_states = transmat.shape[0]
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(transmat, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=state_names, yticklabels=state_names,
                ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_xlabel('To State', fontsize=12)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_state_characteristics(state_summary: pd.DataFrame,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot characteristics of the hidden states.

    Parameters:
    -----------
    state_summary : pd.DataFrame
        Output from summarize_states function.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns:
    --------
    plt.Figure
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: State frequencies
    axes[0, 0].bar(state_summary['state'], state_summary['frequency_%'])
    axes[0, 0].set_xlabel('State')
    axes[0, 0].set_ylabel('Frequency (%)')
    axes[0, 0].set_title('State Frequencies')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean daily return
    axes[0, 1].bar(state_summary['state'], state_summary['mean_daily_return_%'])
    axes[0, 1].set_xlabel('State')
    axes[0, 1].set_ylabel('Mean Daily Return (%)')
    axes[0, 1].set_title('Mean Daily Return by State')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Volatility
    axes[1, 0].bar(state_summary['state'], state_summary['volatility_daily_%'])
    axes[1, 0].set_xlabel('State')
    axes[1, 0].set_ylabel('Volatility (%)')
    axes[1, 0].set_title('Return Volatility by State')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average duration
    axes[1, 1].bar(state_summary['state'], state_summary['avg_duration_days'])
    axes[1, 1].set_xlabel('State')
    axes[1, 1].set_ylabel('Average Duration (days)')
    axes[1, 1].set_title('Average State Duration')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Hidden State Characteristics', fontsize=16)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig