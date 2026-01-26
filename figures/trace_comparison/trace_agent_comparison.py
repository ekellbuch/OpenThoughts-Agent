#!/usr/bin/env python3
"""
Trace Agent Comparison Analysis

Compares performance between two agents (terminus, codex) across multiple models
and analyzes failure mode distributions.

This script generates visualizations for:
- Experiment 1: Overall Performance Comparison
- Experiment 2: Model Scaling Effects
- Experiment 3: Failure Mode Analysis
- Experiment 4: Efficiency Comparison
- Experiment 5: Consistency & Robustness
- Experiment 6: Head-to-Head Task Analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn2
import seaborn as sns

# Configuration
AGENTS = ["terminus", "codex"]
MODELS = ["gpt-5", "gpt-5-codex", "gpt-5-mini", "gpt-5-nano", "gpt-oss-120b", "gpt-oss-20b"]
MODEL_SIZES = {
    "gpt-5": 1000,
    "gpt-5-codex": 800,
    "gpt-5-mini": 200,
    "gpt-5-nano": 50,
    "gpt-oss-120b": 120,
    "gpt-oss-20b": 20,
}
FAILURE_MODES = [
    "syntax_error",
    "timeout",
    "incorrect_output",
    "resource_limit",
    "missing_dependency",
    "logic_error",
    "api_error",
    "environment_error",
    "incomplete_solution",
]
N_TASKS = 89
N_TRIALS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR


@dataclass
class TrialResult:
    """Result of a single trial."""
    task_id: str
    trial_id: int
    agent: str
    model: str
    passed: bool
    steps: int
    tokens_used: int
    runtime_seconds: float
    failure_modes: List[str] = field(default_factory=list)
    is_timeout: bool = False


@dataclass
class TraceData:
    """Container for all trace analysis data."""
    trials: List[TrialResult] = field(default_factory=list)

    def filter_by_agent(self, agent: str) -> List[TrialResult]:
        return [t for t in self.trials if t.agent == agent]

    def filter_by_model(self, model: str) -> List[TrialResult]:
        return [t for t in self.trials if t.model == model]

    def filter_by_agent_model(self, agent: str, model: str) -> List[TrialResult]:
        return [t for t in self.trials if t.agent == agent and t.model == model]

    def get_tasks(self) -> List[str]:
        return list(set(t.task_id for t in self.trials))

    def get_agents(self) -> List[str]:
        return list(set(t.agent for t in self.trials))

    def get_models(self) -> List[str]:
        return list(set(t.model for t in self.trials))


def load_data_from_file(filepath: Path) -> Optional[TraceData]:
    """Load trace data from a JSON file."""
    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        data = TraceData()
        for entry in raw_data.get("trials", []):
            trial = TrialResult(
                task_id=entry.get("task_id", ""),
                trial_id=entry.get("trial_id", 0),
                agent=entry.get("agent", ""),
                model=entry.get("model", ""),
                passed=entry.get("passed", False),
                steps=entry.get("steps", 0),
                tokens_used=entry.get("tokens_used", 0),
                runtime_seconds=entry.get("runtime_seconds", 0.0),
                failure_modes=entry.get("failure_modes", []),
                is_timeout=entry.get("is_timeout", False),
            )
            data.trials.append(trial)
        return data
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading data from {filepath}: {e}", file=sys.stderr)
        return None


def generate_sample_data() -> TraceData:
    """Generate realistic sample data for demonstration."""
    np.random.seed(42)
    data = TraceData()

    # Base pass rates per model (higher for larger models)
    base_rates = {
        "gpt-5": 0.75,
        "gpt-5-codex": 0.70,
        "gpt-5-mini": 0.55,
        "gpt-5-nano": 0.40,
        "gpt-oss-120b": 0.60,
        "gpt-oss-20b": 0.35,
    }

    # Agent-specific modifiers (terminus slightly better overall)
    agent_modifiers = {
        "terminus": 0.08,
        "codex": 0.0,
    }

    # Task difficulties (some tasks harder than others)
    task_difficulties = {
        f"task_{i:03d}": np.random.beta(2, 2) for i in range(N_TASKS)
    }

    # Failure mode probabilities per agent
    failure_probs = {
        "terminus": {
            "syntax_error": 0.05,
            "timeout": 0.08,
            "incorrect_output": 0.25,
            "resource_limit": 0.03,
            "missing_dependency": 0.06,
            "logic_error": 0.20,
            "api_error": 0.04,
            "environment_error": 0.07,
            "incomplete_solution": 0.22,
        },
        "codex": {
            "syntax_error": 0.08,
            "timeout": 0.12,
            "incorrect_output": 0.28,
            "resource_limit": 0.05,
            "missing_dependency": 0.08,
            "logic_error": 0.18,
            "api_error": 0.06,
            "environment_error": 0.05,
            "incomplete_solution": 0.20,
        },
    }

    for task_id, difficulty in task_difficulties.items():
        for agent in AGENTS:
            for model in MODELS:
                base_rate = base_rates[model]
                agent_mod = agent_modifiers[agent]
                task_rate = base_rate + agent_mod - difficulty * 0.3
                task_rate = np.clip(task_rate, 0.1, 0.95)

                for trial in range(N_TRIALS):
                    passed = np.random.random() < task_rate

                    # Generate metrics
                    base_steps = np.random.randint(5, 50)
                    steps = int(base_steps * (1.5 if not passed else 1.0))

                    base_tokens = np.random.randint(1000, 10000)
                    tokens = int(base_tokens * (1.3 if not passed else 1.0))

                    base_runtime = np.random.exponential(30)
                    runtime = base_runtime * (2.0 if not passed else 1.0)

                    # Determine failure modes for failed trials
                    failure_modes = []
                    is_timeout = False
                    if not passed:
                        probs = failure_probs[agent]
                        for fm, prob in probs.items():
                            if np.random.random() < prob:
                                failure_modes.append(fm)
                        if not failure_modes:
                            failure_modes.append(np.random.choice(FAILURE_MODES))
                        is_timeout = "timeout" in failure_modes

                    data.trials.append(TrialResult(
                        task_id=task_id,
                        trial_id=trial,
                        agent=agent,
                        model=model,
                        passed=passed,
                        steps=steps,
                        tokens_used=tokens,
                        runtime_seconds=runtime,
                        failure_modes=failure_modes,
                        is_timeout=is_timeout,
                    ))

    return data


def compute_pass_rate(trials: List[TrialResult]) -> float:
    """Compute pass@1 rate."""
    if not trials:
        return 0.0
    return sum(1 for t in trials if t.passed) / len(trials)


def compute_pass_at_k(trials: List[TrialResult], k: int) -> float:
    """Compute pass@k rate (any of k trials passes)."""
    if not trials:
        return 0.0

    # Group by task
    task_trials = defaultdict(list)
    for t in trials:
        task_trials[t.task_id].append(t)

    pass_count = 0
    for task_id, task_trial_list in task_trials.items():
        # Check if any of first k trials passed
        first_k = task_trial_list[:k]
        if any(t.passed for t in first_k):
            pass_count += 1

    return pass_count / len(task_trials) if task_trials else 0.0


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if not data:
        return (0.0, 0.0)

    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return (lower, upper)


# ==============================================================================
# EXPERIMENT 1: Overall Performance Comparison
# ==============================================================================

def plot_exp1a_pass_rate_by_agent(data: TraceData, output_dir: Path):
    """Fig 1a: Bar chart of pass@1 rate by agent (averaged across models)."""
    agents = data.get_agents()

    pass_rates = []
    errors = []

    for agent in agents:
        trials = data.filter_by_agent(agent)
        rates = [1 if t.passed else 0 for t in trials]
        mean_rate = np.mean(rates)
        ci_low, ci_high = bootstrap_ci(rates)
        pass_rates.append(mean_rate * 100)
        errors.append([(mean_rate - ci_low) * 100, (ci_high - mean_rate) * 100])

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(agents))
    colors = ['#2ecc71', '#3498db']

    bars = ax.bar(x, pass_rates, color=colors, edgecolor='black', linewidth=1.2)
    ax.errorbar(x, pass_rates, yerr=np.array(errors).T, fmt='none', color='black', capsize=5)

    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Pass@1 Rate (%)', fontsize=12)
    ax.set_title('Fig 1a: Pass@1 Rate by Agent\n(Averaged Across Models)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in agents])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1a_pass_rate_by_agent.png', dpi=150)
    plt.savefig(output_dir / 'fig1a_pass_rate_by_agent.pdf')
    plt.close()


def plot_exp1b_pass_at_k_curves(data: TraceData, output_dir: Path):
    """Fig 1b: Pass@k curves (k=1,3,5) for each agent."""
    agents = data.get_agents()
    k_values = [1, 3, 5]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'terminus': '#2ecc71', 'codex': '#3498db'}
    markers = {'terminus': 'o', 'codex': 's'}

    for agent in agents:
        trials = data.filter_by_agent(agent)
        pass_rates = [compute_pass_at_k(trials, k) * 100 for k in k_values]
        ax.plot(k_values, pass_rates, marker=markers[agent], color=colors[agent],
                label=agent.capitalize(), linewidth=2, markersize=10)

    ax.set_xlabel('k (number of trials)', fontsize=12)
    ax.set_ylabel('Pass@k Rate (%)', fontsize=12)
    ax.set_title('Fig 1b: Pass@k Curves by Agent', fontsize=14)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1b_pass_at_k_curves.png', dpi=150)
    plt.savefig(output_dir / 'fig1b_pass_at_k_curves.pdf')
    plt.close()


def create_exp1c_ci_table(data: TraceData, output_dir: Path):
    """Fig 1c: Table with 95% confidence intervals on pass rates."""
    agents = data.get_agents()
    models = data.get_models()

    table_data = []

    for agent in agents:
        for model in models:
            trials = data.filter_by_agent_model(agent, model)
            rates = [1 if t.passed else 0 for t in trials]
            mean_rate = np.mean(rates) * 100
            ci_low, ci_high = bootstrap_ci(rates)
            ci_low *= 100
            ci_high *= 100
            table_data.append({
                'Agent': agent.capitalize(),
                'Model': model,
                'Pass Rate (%)': f'{mean_rate:.2f}',
                '95% CI': f'[{ci_low:.2f}, {ci_high:.2f}]',
                'N Trials': len(rates),
            })

    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    col_labels = ['Agent', 'Model', 'Pass Rate (%)', '95% CI', 'N Trials']
    cell_text = [[d[c] for c in col_labels] for d in table_data]

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4a69bd')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        color = '#f5f6fa' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    ax.set_title('Fig 1c: Pass Rates with 95% Confidence Intervals', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1c_ci_table.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1c_ci_table.pdf', bbox_inches='tight')
    plt.close()

    # Also save as CSV
    import csv
    with open(output_dir / 'fig1c_ci_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=col_labels)
        writer.writeheader()
        writer.writerows(table_data)


# ==============================================================================
# EXPERIMENT 2: Model Scaling Effects
# ==============================================================================

def plot_exp2a_scaling_curves(data: TraceData, output_dir: Path):
    """Fig 2a: Line plot of pass rate vs model size, one line per agent."""
    agents = data.get_agents()
    models = sorted(data.get_models(), key=lambda m: MODEL_SIZES.get(m, 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'terminus': '#2ecc71', 'codex': '#3498db'}
    markers = {'terminus': 'o', 'codex': 's'}

    for agent in agents:
        sizes = []
        rates = []
        for model in models:
            trials = data.filter_by_agent_model(agent, model)
            if trials:
                sizes.append(MODEL_SIZES.get(model, 0))
                rates.append(compute_pass_rate(trials) * 100)

        ax.plot(sizes, rates, marker=markers[agent], color=colors[agent],
                label=agent.capitalize(), linewidth=2, markersize=10)

    ax.set_xlabel('Model Size (billions of parameters)', fontsize=12)
    ax.set_ylabel('Pass@1 Rate (%)', fontsize=12)
    ax.set_title('Fig 2a: Pass Rate vs Model Size (Scaling Behavior)', fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2a_scaling_curves.png', dpi=150)
    plt.savefig(output_dir / 'fig2a_scaling_curves.pdf')
    plt.close()


def plot_exp2b_heatmap(data: TraceData, output_dir: Path):
    """Fig 2b: Heatmap of pass rate (agent x model)."""
    agents = data.get_agents()
    models = sorted(data.get_models(), key=lambda m: MODEL_SIZES.get(m, 0), reverse=True)

    heatmap_data = np.zeros((len(agents), len(models)))

    for i, agent in enumerate(agents):
        for j, model in enumerate(models):
            trials = data.filter_by_agent_model(agent, model)
            heatmap_data[i, j] = compute_pass_rate(trials) * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=models, yticklabels=[a.capitalize() for a in agents],
                ax=ax, vmin=0, vmax=100, cbar_kws={'label': 'Pass Rate (%)'})

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Agent', fontsize=12)
    ax.set_title('Fig 2b: Pass Rate Heatmap (Agent × Model)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2b_heatmap.png', dpi=150)
    plt.savefig(output_dir / 'fig2b_heatmap.pdf')
    plt.close()


def plot_exp2c_agent_gap(data: TraceData, output_dir: Path):
    """Fig 2c: Bar chart of 'agent gap' (terminus - codex) at each model scale."""
    models = sorted(data.get_models(), key=lambda m: MODEL_SIZES.get(m, 0), reverse=True)

    gaps = []
    for model in models:
        term_trials = data.filter_by_agent_model("terminus", model)
        codex_trials = data.filter_by_agent_model("codex", model)
        term_rate = compute_pass_rate(term_trials) * 100
        codex_rate = compute_pass_rate(codex_trials) * 100
        gaps.append(term_rate - codex_rate)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in gaps]

    bars = ax.bar(x, gaps, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Agent Gap (Terminus - Codex) %', fontsize=12)
    ax.set_title('Fig 2c: Performance Gap Between Agents by Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, gap in zip(bars, gaps):
        ypos = bar.get_height() + 0.3 if gap >= 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{gap:+.1f}%', ha='center', va='bottom' if gap >= 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2c_agent_gap.png', dpi=150)
    plt.savefig(output_dir / 'fig2c_agent_gap.pdf')
    plt.close()


# ==============================================================================
# EXPERIMENT 3: Failure Mode Analysis
# ==============================================================================

def plot_exp3a_failure_stacked_bar(data: TraceData, output_dir: Path):
    """Fig 3a: Stacked bar chart of failure mode distribution by agent."""
    agents = data.get_agents()

    failure_counts = {agent: defaultdict(int) for agent in agents}
    failure_totals = {agent: 0 for agent in agents}

    for trial in data.trials:
        if not trial.passed:
            failure_totals[trial.agent] += 1
            for fm in trial.failure_modes:
                failure_counts[trial.agent][fm] += 1

    # Normalize to percentages
    failure_pcts = {agent: {} for agent in agents}
    for agent in agents:
        total = failure_totals[agent]
        for fm in FAILURE_MODES:
            count = failure_counts[agent].get(fm, 0)
            failure_pcts[agent][fm] = (count / total * 100) if total > 0 else 0

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(agents))
    width = 0.6

    colors = plt.cm.Set3(np.linspace(0, 1, len(FAILURE_MODES)))
    bottom = np.zeros(len(agents))

    for fm_idx, fm in enumerate(FAILURE_MODES):
        values = [failure_pcts[agent][fm] for agent in agents]
        ax.bar(x, values, width, label=fm.replace('_', ' ').title(),
               bottom=bottom, color=colors[fm_idx], edgecolor='white', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Percentage of Failed Trials (%)', fontsize=12)
    ax.set_title('Fig 3a: Failure Mode Distribution by Agent', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in agents])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3a_failure_stacked_bar.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3a_failure_stacked_bar.pdf', bbox_inches='tight')
    plt.close()


def plot_exp3b_radar_chart(data: TraceData, output_dir: Path):
    """Fig 3b: Radar/spider chart comparing failure mode prevalence (normalized)."""
    agents = data.get_agents()

    failure_counts = {agent: defaultdict(int) for agent in agents}
    failure_totals = {agent: 0 for agent in agents}

    for trial in data.trials:
        if not trial.passed:
            failure_totals[trial.agent] += 1
            for fm in trial.failure_modes:
                failure_counts[trial.agent][fm] += 1

    # Normalize
    failure_pcts = {agent: [] for agent in agents}
    for agent in agents:
        total = failure_totals[agent]
        for fm in FAILURE_MODES:
            count = failure_counts[agent].get(fm, 0)
            failure_pcts[agent].append((count / total * 100) if total > 0 else 0)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(FAILURE_MODES), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = {'terminus': '#2ecc71', 'codex': '#3498db'}

    for agent in agents:
        values = failure_pcts[agent] + failure_pcts[agent][:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=agent.capitalize(), color=colors[agent])
        ax.fill(angles, values, alpha=0.25, color=colors[agent])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([fm.replace('_', '\n').title() for fm in FAILURE_MODES], fontsize=9)
    ax.set_title('Fig 3b: Failure Mode Prevalence (Radar Chart)', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3b_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3b_radar_chart.pdf', bbox_inches='tight')
    plt.close()


def plot_exp3c_cooccurrence_heatmap(data: TraceData, output_dir: Path):
    """Fig 3c: Heatmap of failure mode co-occurrence by agent."""
    agents = data.get_agents()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, agent in enumerate(agents):
        # Build co-occurrence matrix
        cooccur = np.zeros((len(FAILURE_MODES), len(FAILURE_MODES)))

        for trial in data.trials:
            if trial.agent == agent and not trial.passed:
                for i, fm1 in enumerate(FAILURE_MODES):
                    if fm1 in trial.failure_modes:
                        for j, fm2 in enumerate(FAILURE_MODES):
                            if fm2 in trial.failure_modes:
                                cooccur[i, j] += 1

        # Normalize by diagonal (self-occurrence)
        diag = np.diag(cooccur).copy()
        for i in range(len(FAILURE_MODES)):
            for j in range(len(FAILURE_MODES)):
                if diag[i] > 0:
                    cooccur[i, j] = cooccur[i, j] / diag[i] * 100

        sns.heatmap(cooccur, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=[fm[:8] for fm in FAILURE_MODES],
                    yticklabels=[fm[:8] for fm in FAILURE_MODES],
                    ax=axes[ax_idx], vmin=0, vmax=100)
        axes[ax_idx].set_title(f'{agent.capitalize()}', fontsize=12)
        axes[ax_idx].tick_params(axis='x', rotation=45)
        axes[ax_idx].tick_params(axis='y', rotation=0)

    fig.suptitle('Fig 3c: Failure Mode Co-occurrence by Agent (% of row)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3c_cooccurrence_heatmap.png', dpi=150)
    plt.savefig(output_dir / 'fig3c_cooccurrence_heatmap.pdf')
    plt.close()


# ==============================================================================
# EXPERIMENT 4: Efficiency Comparison
# ==============================================================================

def plot_exp4a_steps_boxplot(data: TraceData, output_dir: Path):
    """Fig 4a: Box plots of steps-to-completion by agent (successful runs only)."""
    agents = data.get_agents()

    steps_data = {agent: [] for agent in agents}
    for trial in data.trials:
        if trial.passed:
            steps_data[trial.agent].append(trial.steps)

    fig, ax = plt.subplots(figsize=(8, 6))

    box_data = [steps_data[agent] for agent in agents]
    colors = ['#2ecc71', '#3498db']

    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=[a.capitalize() for a in agents])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Steps to Completion', fontsize=12)
    ax.set_title('Fig 4a: Steps to Completion (Successful Runs)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add mean markers
    for i, agent in enumerate(agents):
        if steps_data[agent]:
            mean_val = np.mean(steps_data[agent])
            ax.scatter(i + 1, mean_val, marker='D', color='red', s=50, zorder=5, label='Mean' if i == 0 else '')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4a_steps_boxplot.png', dpi=150)
    plt.savefig(output_dir / 'fig4a_steps_boxplot.pdf')
    plt.close()


def plot_exp4b_token_boxplot(data: TraceData, output_dir: Path):
    """Fig 4b: Box plots of token usage by agent."""
    agents = data.get_agents()

    tokens_data = {agent: [] for agent in agents}
    for trial in data.trials:
        tokens_data[trial.agent].append(trial.tokens_used)

    fig, ax = plt.subplots(figsize=(8, 6))

    box_data = [tokens_data[agent] for agent in agents]
    colors = ['#2ecc71', '#3498db']

    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=[a.capitalize() for a in agents])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Tokens Used', fontsize=12)
    ax.set_title('Fig 4b: Token Usage by Agent', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4b_token_boxplot.png', dpi=150)
    plt.savefig(output_dir / 'fig4b_token_boxplot.pdf')
    plt.close()


def plot_exp4c_pareto_frontier(data: TraceData, output_dir: Path):
    """Fig 4c: Scatter plot of pass rate vs cost (Pareto frontier)."""
    agents = data.get_agents()
    models = data.get_models()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'terminus': '#2ecc71', 'codex': '#3498db'}
    markers = {'terminus': 'o', 'codex': 's'}

    all_points = []

    for agent in agents:
        for model in models:
            trials = data.filter_by_agent_model(agent, model)
            if trials:
                pass_rate = compute_pass_rate(trials) * 100
                avg_tokens = np.mean([t.tokens_used for t in trials])
                # Simplified cost model: tokens * model size factor
                cost = avg_tokens * MODEL_SIZES.get(model, 100) / 100

                ax.scatter(cost, pass_rate, s=150, color=colors[agent],
                          marker=markers[agent], edgecolor='black', linewidth=1,
                          label=f'{agent.capitalize()}' if model == models[0] else '')
                ax.annotate(model, (cost, pass_rate), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')
                all_points.append((cost, pass_rate, agent, model))

    # Draw Pareto frontier
    points_sorted = sorted(all_points, key=lambda x: x[0])
    pareto_points = []
    max_pass = 0
    for cost, pass_rate, _, _ in points_sorted:
        if pass_rate >= max_pass:
            pareto_points.append((cost, pass_rate))
            max_pass = pass_rate

    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')

    ax.set_xlabel('Relative Cost (tokens × model size)', fontsize=12)
    ax.set_ylabel('Pass@1 Rate (%)', fontsize=12)
    ax.set_title('Fig 4c: Pass Rate vs Cost (Pareto Frontier)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4c_pareto_frontier.png', dpi=150)
    plt.savefig(output_dir / 'fig4c_pareto_frontier.pdf')
    plt.close()


# ==============================================================================
# EXPERIMENT 5: Consistency & Robustness
# ==============================================================================

def plot_exp5a_variance_boxplot(data: TraceData, output_dir: Path):
    """Fig 5a: Box plots showing variance across trials by agent."""
    agents = data.get_agents()

    # Compute per-task variance in pass rate
    task_variances = {agent: [] for agent in agents}

    task_trials = defaultdict(lambda: defaultdict(list))
    for trial in data.trials:
        task_trials[trial.task_id][trial.agent].append(1 if trial.passed else 0)

    for task_id, agent_dict in task_trials.items():
        for agent in agents:
            if agent in agent_dict and len(agent_dict[agent]) > 1:
                var = np.var(agent_dict[agent])
                task_variances[agent].append(var)

    fig, ax = plt.subplots(figsize=(8, 6))

    box_data = [task_variances[agent] for agent in agents]
    colors = ['#2ecc71', '#3498db']

    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=[a.capitalize() for a in agents])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Per-Task Variance in Pass Rate', fontsize=12)
    ax.set_title('Fig 5a: Consistency - Variance Across Trials by Agent', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5a_variance_boxplot.png', dpi=150)
    plt.savefig(output_dir / 'fig5a_variance_boxplot.pdf')
    plt.close()


def plot_exp5b_timeout_rates(data: TraceData, output_dir: Path):
    """Fig 5b: Bar chart of timeout rates by agent × model."""
    agents = data.get_agents()
    models = sorted(data.get_models(), key=lambda m: MODEL_SIZES.get(m, 0), reverse=True)

    timeout_rates = {agent: [] for agent in agents}

    for agent in agents:
        for model in models:
            trials = data.filter_by_agent_model(agent, model)
            if trials:
                rate = sum(1 for t in trials if t.is_timeout) / len(trials) * 100
                timeout_rates[agent].append(rate)
            else:
                timeout_rates[agent].append(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    colors = {'terminus': '#2ecc71', 'codex': '#3498db'}

    for i, agent in enumerate(agents):
        offset = (i - 0.5) * width
        ax.bar(x + offset, timeout_rates[agent], width, label=agent.capitalize(),
               color=colors[agent], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Timeout Rate (%)', fontsize=12)
    ax.set_title('Fig 5b: Timeout Rates by Agent × Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5b_timeout_rates.png', dpi=150)
    plt.savefig(output_dir / 'fig5b_timeout_rates.pdf')
    plt.close()


def plot_exp5c_task_pass_histogram(data: TraceData, output_dir: Path):
    """Fig 5c: Histogram of per-task pass rates (shows distribution shape)."""
    agents = data.get_agents()

    task_pass_rates = {agent: [] for agent in agents}

    task_trials = defaultdict(lambda: defaultdict(list))
    for trial in data.trials:
        task_trials[trial.task_id][trial.agent].append(1 if trial.passed else 0)

    for task_id, agent_dict in task_trials.items():
        for agent in agents:
            if agent in agent_dict:
                rate = np.mean(agent_dict[agent]) * 100
                task_pass_rates[agent].append(rate)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'terminus': '#2ecc71', 'codex': '#3498db'}

    bins = np.linspace(0, 100, 21)
    for agent in agents:
        ax.hist(task_pass_rates[agent], bins=bins, alpha=0.6,
                label=agent.capitalize(), color=colors[agent], edgecolor='black')

    ax.set_xlabel('Per-Task Pass Rate (%)', fontsize=12)
    ax.set_ylabel('Number of Tasks', fontsize=12)
    ax.set_title('Fig 5c: Distribution of Per-Task Pass Rates', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5c_task_pass_histogram.png', dpi=150)
    plt.savefig(output_dir / 'fig5c_task_pass_histogram.pdf')
    plt.close()


# ==============================================================================
# EXPERIMENT 6: Head-to-Head Task Analysis
# ==============================================================================

def plot_exp6a_venn_diagram(data: TraceData, output_dir: Path):
    """Fig 6a: Venn diagram of tasks solved (unique to each agent vs shared)."""
    agents = data.get_agents()

    # Determine which tasks each agent solved (at least once)
    solved = {agent: set() for agent in agents}

    for trial in data.trials:
        if trial.passed:
            solved[trial.agent].add(trial.task_id)

    terminus_solved = solved.get('terminus', set())
    codex_solved = solved.get('codex', set())

    fig, ax = plt.subplots(figsize=(10, 8))

    v = venn2([terminus_solved, codex_solved],
              set_labels=('Terminus', 'Codex'),
              ax=ax)

    # Customize colors
    if v.get_patch_by_id('10'):
        v.get_patch_by_id('10').set_color('#2ecc71')
        v.get_patch_by_id('10').set_alpha(0.6)
    if v.get_patch_by_id('01'):
        v.get_patch_by_id('01').set_color('#3498db')
        v.get_patch_by_id('01').set_alpha(0.6)
    if v.get_patch_by_id('11'):
        v.get_patch_by_id('11').set_color('#9b59b6')
        v.get_patch_by_id('11').set_alpha(0.6)

    ax.set_title('Fig 6a: Tasks Solved (Venn Diagram)', fontsize=14)

    # Add summary text
    only_term = len(terminus_solved - codex_solved)
    only_codex = len(codex_solved - terminus_solved)
    both = len(terminus_solved & codex_solved)
    total = len(terminus_solved | codex_solved)

    summary = f"Total unique tasks solved: {total}\n"
    summary += f"Only Terminus: {only_term}\n"
    summary += f"Only Codex: {only_codex}\n"
    summary += f"Both: {both}"
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6a_venn_diagram.png', dpi=150)
    plt.savefig(output_dir / 'fig6a_venn_diagram.pdf')
    plt.close()


def plot_exp6b_head_to_head_scatter(data: TraceData, output_dir: Path):
    """Fig 6b: Scatter plot of task pass rate (terminus vs codex), one point per task."""
    task_rates = defaultdict(lambda: {'terminus': [], 'codex': []})

    for trial in data.trials:
        task_rates[trial.task_id][trial.agent].append(1 if trial.passed else 0)

    terminus_rates = []
    codex_rates = []
    task_ids = []

    for task_id, rates in task_rates.items():
        if rates['terminus'] and rates['codex']:
            terminus_rates.append(np.mean(rates['terminus']) * 100)
            codex_rates.append(np.mean(rates['codex']) * 100)
            task_ids.append(task_id)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(terminus_rates, codex_rates, alpha=0.6, s=50, c='#9b59b6', edgecolor='black')

    # Diagonal line (equal performance)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Equal Performance')

    ax.set_xlabel('Terminus Pass Rate (%)', fontsize=12)
    ax.set_ylabel('Codex Pass Rate (%)', fontsize=12)
    ax.set_title('Fig 6b: Head-to-Head Task Comparison', fontsize=14)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    # Count quadrants
    term_better = sum(1 for t, c in zip(terminus_rates, codex_rates) if t > c)
    codex_better = sum(1 for t, c in zip(terminus_rates, codex_rates) if c > t)
    equal = sum(1 for t, c in zip(terminus_rates, codex_rates) if t == c)

    summary = f"Terminus better: {term_better} tasks\n"
    summary += f"Codex better: {codex_better} tasks\n"
    summary += f"Equal: {equal} tasks"
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6b_head_to_head_scatter.png', dpi=150)
    plt.savefig(output_dir / 'fig6b_head_to_head_scatter.pdf')
    plt.close()


def create_exp6c_signature_tasks_table(data: TraceData, output_dir: Path):
    """Fig 6c: Table of 'signature tasks' (largest performance gaps)."""
    task_rates = defaultdict(lambda: {'terminus': [], 'codex': []})

    for trial in data.trials:
        task_rates[trial.task_id][trial.agent].append(1 if trial.passed else 0)

    gaps = []
    for task_id, rates in task_rates.items():
        if rates['terminus'] and rates['codex']:
            term_rate = np.mean(rates['terminus']) * 100
            codex_rate = np.mean(rates['codex']) * 100
            gap = term_rate - codex_rate
            gaps.append({
                'Task': task_id,
                'Terminus (%)': f'{term_rate:.1f}',
                'Codex (%)': f'{codex_rate:.1f}',
                'Gap (%)': f'{gap:+.1f}',
                'Favors': 'Terminus' if gap > 0 else ('Codex' if gap < 0 else 'Tie'),
            })

    # Sort by absolute gap
    gaps.sort(key=lambda x: abs(float(x['Gap (%)'])), reverse=True)

    # Top 20 signature tasks
    top_gaps = gaps[:20]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    col_labels = ['Task', 'Terminus (%)', 'Codex (%)', 'Gap (%)', 'Favors']
    cell_text = [[d[c] for c in col_labels] for d in top_gaps]

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4a69bd')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Color code by favored agent
    for i, d in enumerate(top_gaps, 1):
        if d['Favors'] == 'Terminus':
            color = '#d4efdf'
        elif d['Favors'] == 'Codex':
            color = '#d6eaf8'
        else:
            color = '#f5f5f5'
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    ax.set_title('Fig 6c: Signature Tasks (Largest Performance Gaps)', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6c_signature_tasks_table.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6c_signature_tasks_table.pdf', bbox_inches='tight')
    plt.close()

    # Also save as CSV
    import csv
    with open(output_dir / 'fig6c_signature_tasks.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=col_labels)
        writer.writeheader()
        writer.writerows(gaps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trace comparison visualizations for agent analysis."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Path to JSON file containing trace data. If not provided, uses sample data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to save figures (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--save-sample-data",
        action="store_true",
        help="Save generated sample data to JSON file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load or generate data
    if args.data_file and args.data_file.exists():
        print(f"Loading data from {args.data_file}")
        data = load_data_from_file(args.data_file)
        if data is None:
            print("Failed to load data, using sample data instead.")
            data = generate_sample_data()
    else:
        print("Generating sample data...")
        data = generate_sample_data()

    if args.save_sample_data:
        sample_path = output_dir / 'sample_data.json'
        with open(sample_path, 'w') as f:
            trials_json = [
                {
                    'task_id': t.task_id,
                    'trial_id': t.trial_id,
                    'agent': t.agent,
                    'model': t.model,
                    'passed': bool(t.passed),
                    'steps': int(t.steps),
                    'tokens_used': int(t.tokens_used),
                    'runtime_seconds': float(t.runtime_seconds),
                    'failure_modes': list(t.failure_modes),
                    'is_timeout': bool(t.is_timeout),
                }
                for t in data.trials
            ]
            json.dump({'trials': trials_json}, f, indent=2)
        print(f"Saved sample data to {sample_path}")

    print(f"\nData summary:")
    print(f"  Total trials: {len(data.trials)}")
    print(f"  Agents: {data.get_agents()}")
    print(f"  Models: {data.get_models()}")
    print(f"  Tasks: {len(data.get_tasks())}")

    # Generate all figures
    print("\n" + "="*60)
    print("EXPERIMENT 1: Overall Performance Comparison")
    print("="*60)

    print("  Generating Fig 1a: Pass@1 rate by agent...")
    plot_exp1a_pass_rate_by_agent(data, output_dir)

    print("  Generating Fig 1b: Pass@k curves...")
    plot_exp1b_pass_at_k_curves(data, output_dir)

    print("  Generating Fig 1c: CI table...")
    create_exp1c_ci_table(data, output_dir)

    print("\n" + "="*60)
    print("EXPERIMENT 2: Model Scaling Effects")
    print("="*60)

    print("  Generating Fig 2a: Scaling curves...")
    plot_exp2a_scaling_curves(data, output_dir)

    print("  Generating Fig 2b: Heatmap...")
    plot_exp2b_heatmap(data, output_dir)

    print("  Generating Fig 2c: Agent gap...")
    plot_exp2c_agent_gap(data, output_dir)

    print("\n" + "="*60)
    print("EXPERIMENT 3: Failure Mode Analysis")
    print("="*60)

    print("  Generating Fig 3a: Failure stacked bar...")
    plot_exp3a_failure_stacked_bar(data, output_dir)

    print("  Generating Fig 3b: Radar chart...")
    plot_exp3b_radar_chart(data, output_dir)

    print("  Generating Fig 3c: Co-occurrence heatmap...")
    plot_exp3c_cooccurrence_heatmap(data, output_dir)

    print("\n" + "="*60)
    print("EXPERIMENT 4: Efficiency Comparison")
    print("="*60)

    print("  Generating Fig 4a: Steps boxplot...")
    plot_exp4a_steps_boxplot(data, output_dir)

    print("  Generating Fig 4b: Token boxplot...")
    plot_exp4b_token_boxplot(data, output_dir)

    print("  Generating Fig 4c: Pareto frontier...")
    plot_exp4c_pareto_frontier(data, output_dir)

    print("\n" + "="*60)
    print("EXPERIMENT 5: Consistency & Robustness")
    print("="*60)

    print("  Generating Fig 5a: Variance boxplot...")
    plot_exp5a_variance_boxplot(data, output_dir)

    print("  Generating Fig 5b: Timeout rates...")
    plot_exp5b_timeout_rates(data, output_dir)

    print("  Generating Fig 5c: Task pass histogram...")
    plot_exp5c_task_pass_histogram(data, output_dir)

    print("\n" + "="*60)
    print("EXPERIMENT 6: Head-to-Head Task Analysis")
    print("="*60)

    print("  Generating Fig 6a: Venn diagram...")
    plot_exp6a_venn_diagram(data, output_dir)

    print("  Generating Fig 6b: Head-to-head scatter...")
    plot_exp6b_head_to_head_scatter(data, output_dir)

    print("  Generating Fig 6c: Signature tasks table...")
    create_exp6c_signature_tasks_table(data, output_dir)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nAll figures saved to: {output_dir}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("fig*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
