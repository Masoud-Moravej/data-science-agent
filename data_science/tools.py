# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for the ADK Samples Data Science Agent."""

import logging
import os
from typing import Dict, Iterable, List, Tuple

from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import alloydb_agent, analytics_agent, bigquery_agent, plot_agent

logger = logging.getLogger(__name__)


async def call_bigquery_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call bigquery database (nl2sql) agent."""
    logger.debug("call_bigquery_agent: %s", question)

    agent_tool = AgentTool(agent=bigquery_agent)

    bigquery_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["bigquery_agent_output"] = bigquery_agent_output
    return bigquery_agent_output


async def call_alloydb_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call alloydb database (nl2sql) agent."""
    logger.debug("call_alloydb_agent: %s", question)

    agent_tool = AgentTool(agent=alloydb_agent)

    alloydb_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["alloydb_agent_output"] = alloydb_agent_output
    return alloydb_agent_output


async def call_analytics_agent(
    question: str,
    tool_context: ToolContext,
):
    """
    This tool can generate Python code to process and analyze a dataset.

    Some of the tasks it can do in Python include:
    * Creating graphics for data visualization;
    * Processing or filtering existing datasets;
    * Combining datasets to create a joined dataset for further analysis.

    The Python modules available to it are:
    * io
    * math
    * re
    * matplotlib.pyplot
    * numpy
    * pandas

    The tool DOES NOT have the ability to retrieve additional data from
    a database. Only the data already retrieved will be analyzed.

    Args:
        question (str): Natural language question or analytics request.
        tool_context (ToolContext): The tool context to use for generating the
            SQL query.

    Returns:
        Response from the analytics agent.

    """
    logger.debug("call_analytics_agent: %s", question)

    # if question == "N/A":
    #    return tool_context.state["db_agent_output"]

    bigquery_data = ""
    alloydb_data = ""

    if "bigquery_query_result" in tool_context.state:
        bigquery_data = tool_context.state["bigquery_query_result"]
    if "alloydb_query_result" in tool_context.state:
        alloydb_data = tool_context.state["alloydb_query_result"]

    question_with_data = f"""
  Question to answer: {question}

  Actual data to analyze this question is available in the following data
  tables:

  <BIGQUERY>
  {bigquery_data}
  </BIGQUERY>

  <ALLOYDB>
  {alloydb_data}
  </ALLOYDB>

  """

    agent_tool = AgentTool(agent=analytics_agent)

    analytics_agent_output = await agent_tool.run_async(
        args={"request": question_with_data}, tool_context=tool_context
    )
    tool_context.state["analytics_agent_output"] = analytics_agent_output
    return analytics_agent_output


def _format_table_schema(schema: Iterable[Tuple[str, str]], max_columns: int = 12) -> str:
    """Summarize a table schema as a single line string."""
    columns = list(schema)
    display_columns = columns[:max_columns]
    formatted = ", ".join(f"{name} ({type_})" for name, type_ in display_columns)
    if len(columns) > max_columns:
        formatted += ", ..."
    return formatted


def _format_schema(schema: Dict[str, Dict[str, Iterable[Tuple[str, str]]]]) -> str:
    """Render the dataset schema in a compact human-readable format."""
    lines: List[str] = []
    for table_name in sorted(schema):
        table_info = schema.get(table_name, {})
        table_schema = table_info.get("table_schema") or []
        summary = _format_table_schema(table_schema)
        lines.append(f"- {table_name}: {summary}")
    return "\n  ".join(lines)


async def call_plot_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call the plot agent that can query BigQuery from Python and plot results."""
    logger.debug("call_plot_agent: %s", question)

    db_settings = tool_context.state.get("database_settings") or {}
    bigquery_settings: Dict[str, Dict[str, Iterable[Tuple[str, str]]]] | None = (
        db_settings.get("bigquery") if isinstance(db_settings, dict) else None
    )

    compute_project_id = os.getenv("BQ_COMPUTE_PROJECT_ID", "")
    data_project_id = os.getenv("BQ_DATA_PROJECT_ID", "")
    dataset_id = os.getenv("BQ_DATASET_ID", "")
    schema_text = ""

    if isinstance(bigquery_settings, dict):
        data_project_id = bigquery_settings.get("data_project_id", data_project_id)
        dataset_id = bigquery_settings.get("dataset_id", dataset_id)
        schema = bigquery_settings.get("schema")
        if isinstance(schema, dict) and schema:
            schema_text = _format_schema(schema)

    question_with_context = f"""
  Question to answer: {question}

  BigQuery configuration:
    compute_project_id: {compute_project_id}
    data_project_id: {data_project_id}
    dataset_id: {dataset_id}

  Available tables:
  {schema_text or "Schema unavailable"}

  """

    agent_tool = AgentTool(agent=plot_agent)

    plot_agent_output = await agent_tool.run_async(
        args={"request": question_with_context}, tool_context=tool_context
    )
    tool_context.state["plot_agent_output"] = plot_agent_output
    return plot_agent_output
