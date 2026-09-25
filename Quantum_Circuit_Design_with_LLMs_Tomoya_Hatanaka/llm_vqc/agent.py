"""Custom OpenAI function-calling agent loop."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from openai import (
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from llm_vqc.circuit_explorer import CircuitExplorerError, explore_circuit_space

logger = logging.getLogger(__name__)

EXPLORE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "explore_circuit_space",
        "description": (
            "Explore all unique stabilizer states reachable from |0...0> by breadth-first "
            "search over Clifford circuits up to max_depth gates. Uses gate set "
            "{H, X, Y, Z, CX, CY, CZ} and prunes equivalent stabilizer states. "
            "Returns total unique state count, per-depth discovery counts, and sample "
            "minimum-depth circuits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_qubits": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of qubits N.",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Maximum circuit depth G (gate count) to explore.",
                },
            },
            "required": ["num_qubits", "max_depth"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "explore_circuit_space": explore_circuit_space,
}


class AgentError(Exception):
    """Raised when the agent loop cannot complete successfully."""


class Agent:
    """Minimal LLM agent with OpenAI tool calling."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.messages: list[ChatCompletionMessageParam] = []
        self.last_tool_result: dict[str, Any] | None = None

        if system_prompt is None:
            system_prompt = (
                "You are a quantum software engineer assistant. When asked about "
                "reachable stabilizer states or circuit equivalence classes, call "
                "explore_circuit_space with the requested num_qubits and max_depth. "
                "Interpret results carefully: total_unique_states counts all states "
                "with minimum depth <= G, while states_at_exact_depth_G counts states "
                "first discovered at exactly depth G."
            )

        self.messages.append({"role": "system", "content": system_prompt})

    def _call_model(self) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=[EXPLORE_TOOL],
            )
        except AuthenticationError as exc:
            raise AgentError(
                "OpenAI authentication failed. Check OPENAI_API_KEY."
            ) from exc
        except RateLimitError as exc:
            raise AgentError(
                "OpenAI rate limit exceeded. Retry after a short delay."
            ) from exc
        except APIError as exc:
            raise AgentError(f"OpenAI API error: {exc}") from exc

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in TOOL_REGISTRY:
            return {"error": f"Unknown tool: {tool_name}"}

        if tool_name == "explore_circuit_space":
            if "num_qubits" not in arguments or "max_depth" not in arguments:
                return {
                    "error": "Missing required arguments: num_qubits and max_depth."
                }
            try:
                num_qubits = int(arguments["num_qubits"])
                max_depth = int(arguments["max_depth"])
            except (TypeError, ValueError) as exc:
                return {"error": f"Invalid argument types: {exc}"}

            try:
                result = TOOL_REGISTRY[tool_name](
                    num_qubits=num_qubits,
                    max_depth=max_depth,
                )
                self.last_tool_result = result
                return result
            except CircuitExplorerError as exc:
                return {"error": str(exc)}
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Tool execution failed for %s", tool_name)
                return {"error": f"Tool execution failed: {exc}"}

        return {"error": f"Tool not implemented: {tool_name}"}

    def _handle_tool_calls(self, assistant_message: Any) -> None:
        tool_calls = assistant_message.tool_calls or []
        self.messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result = {"error": f"Invalid JSON arguments: {exc}"}
            else:
                result = self._execute_tool(tool_name, arguments)

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    def run(self, user_prompt: str) -> str:
        """Run the agent loop until the model returns a final text response."""
        self.messages.append({"role": "user", "content": user_prompt})

        for _iteration in range(self.max_iterations):
            response = self._call_model()
            choice = response.choices[0]
            assistant_message = choice.message

            if choice.finish_reason == "tool_calls":
                self._handle_tool_calls(assistant_message)
                continue

            content = assistant_message.content
            if not content:
                raise AgentError("Model returned an empty response.")

            self.messages.append({"role": "assistant", "content": content})
            return content

        raise AgentError(
            f"Agent exceeded max_iterations ({self.max_iterations}) without a final answer."
        )
