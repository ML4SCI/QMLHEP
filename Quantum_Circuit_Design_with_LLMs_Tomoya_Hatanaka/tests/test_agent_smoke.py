"""Smoke tests for agent tool round-trip without calling OpenAI."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_vqc.agent import Agent


def _tool_call(arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_test",
        type="function",
        function=SimpleNamespace(
            name="explore_circuit_space",
            arguments=json.dumps(arguments),
        ),
    )


def test_agent_tool_round_trip() -> None:
    tool_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[_tool_call({"num_qubits": 2, "max_depth": 2})],
                ),
            )
        ]
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="Found unique stabilizer states.",
                    tool_calls=None,
                ),
            )
        ]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [tool_response, final_response]

    with patch("llm_vqc.agent.OpenAI", return_value=mock_client):
        agent = Agent(api_key="test-key")
        result = agent.run("Explore N=2, G=2.")

    assert result == "Found unique stabilizer states."
    assert mock_client.chat.completions.create.call_count == 2

    tool_messages = [msg for msg in agent.messages if msg.get("role") == "tool"]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload["num_qubits"] == 2
    assert payload["max_depth"] == 2
    assert payload["total_unique_states"] > 0


if __name__ == "__main__":
    test_agent_tool_round_trip()
    print("OK agent round-trip")
