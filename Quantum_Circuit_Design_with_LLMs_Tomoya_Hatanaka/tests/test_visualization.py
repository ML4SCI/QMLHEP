"""Tests for exploration visualization helpers."""

from __future__ import annotations

from pathlib import Path

from llm_vqc.circuit_explorer import explore_circuit_space
from llm_vqc.visualization import (
    generate_exploration_visualizations,
    plot_exploration_trajectory,
    plot_gate_distribution,
    plot_pruning_efficiency,
    plot_state_probabilities,
    save_best_circuit_visualization,
)


def test_plot_exploration_trajectory(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)
    output_path = tmp_path / "exploration_trajectory.png"

    saved_path = plot_exploration_trajectory(result, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(result["exploration_trajectory"]) == result["total_unique_states"]


def test_plot_gate_distribution(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)
    output_path = tmp_path / "gate_distribution.png"

    saved_path = plot_gate_distribution(result, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert sum(result["gate_distribution"].values()) > 0


def test_save_best_circuit_visualization(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)

    outputs = save_best_circuit_visualization(result, tmp_path)

    assert outputs["best_circuit_text"].exists()
    text = outputs["best_circuit_text"].read_text(encoding="utf-8")
    assert result["best_circuit"]["circuit_str"] in text
    if "best_circuit_image" in outputs:
        assert outputs["best_circuit_image"].exists()


def test_plot_pruning_efficiency(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)
    output_path = tmp_path / "pruning_efficiency.png"

    saved_path = plot_pruning_efficiency(result, output_path)

    assert saved_path.exists()
    assert result["explored_states_at_depth"]
    assert int(result["num_actions_per_step"]) > 0


def test_plot_state_probabilities(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)
    output_path = tmp_path / "state_probabilities.png"

    saved_path = plot_state_probabilities(result, output_path)

    assert saved_path.exists()
    assert result["most_complex_state"]["measurement_probabilities"]


def test_generate_exploration_visualizations(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)

    outputs = generate_exploration_visualizations(result, tmp_path)

    assert outputs["exploration_trajectory"].name == "exploration_trajectory.png"
    assert outputs["gate_distribution"].name == "gate_distribution.png"
    assert outputs["pruning_efficiency"].name == "pruning_efficiency.png"
    assert outputs["state_probabilities"].name == "state_probabilities.png"
    assert outputs["best_circuit_text"].name == "best_circuit.txt"
    assert outputs["exploration_trajectory"].exists()
    assert outputs["gate_distribution"].exists()
    assert outputs["pruning_efficiency"].exists()
    assert outputs["state_probabilities"].exists()
