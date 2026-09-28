"""Entry point for the LLM quantum circuit exploration agent."""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from llm_vqc.agent import Agent, AgentError
from llm_vqc.visualization import VisualizationError, generate_exploration_visualizations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    agent = Agent(api_key=api_key, model=model)

    prompt = (
        "Please find all unique quantum states reachable with N=3 qubits and "
        "exactly G=5 gates using our standard gate set, and tell me how many "
        "unique states exist."
    )

    try:
        response = agent.run(prompt)
    except AgentError as exc:
        print(f"Agent error: {exc}", file=sys.stderr)
        sys.exit(1)

    if agent.last_tool_result:
        try:
            outputs = generate_exploration_visualizations(agent.last_tool_result)
            print("\nVisualizations saved to outputs/:")
            for name, path in sorted(outputs.items()):
                print(f"  {name}: {path}")
        except VisualizationError as exc:
            logger.warning("Visualization skipped: %s", exc)

    print(f"\n{response}")


if __name__ == "__main__":
    main()
