"""Plot Agent: generate SQL-enabled Python notebooks that render visualizations."""

import os

from google.adk.agents import Agent

from ...code_executors.local_matplotlib_code_executor import (
    LocalMatplotlibCodeExecutor,
)

from .prompts import return_instructions_plot


plot_agent = Agent(
    model=os.getenv(
        "PLOT_AGENT_MODEL",
        os.getenv("ANALYTICS_AGENT_MODEL", ""),
    ),
    name="plot_agent",
    instruction=return_instructions_plot(),
    code_executor=LocalMatplotlibCodeExecutor(),
)

