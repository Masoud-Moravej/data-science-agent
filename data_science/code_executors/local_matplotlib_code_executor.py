"""Local code executor that runs Python in-process and captures matplotlib figures."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import ExitStack, redirect_stdout
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to render off-screen
import matplotlib.pyplot as plt
from pydantic import Field, PrivateAttr
from typing_extensions import override

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import (
    CodeExecutionInput,
    CodeExecutionResult,
    CodeExecutionUtils,
    File,
)


_PRELUDE = """\
import io
import math
import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
  import scipy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
  scipy = None


def crop(s: str, max_chars: int = 64) -> str:
  \"\"\"Crops a string to max_chars characters.\"\"\"
  return s[: max_chars - 3] + '...' if len(s) > max_chars else s


def explore_df(df: pd.DataFrame) -> None:
  \"\"\"Prints some information about a pandas DataFrame.\"\"\"

  with pd.option_context(
      'display.max_columns', None, 'display.expand_frame_repr', False
  ):
    df_dtypes = df.dtypes
    df_nulls = (len(df) - df.isnull().sum()).apply(
        lambda x: f'{x} / {df.shape[0]} non-null'
    )
    df_unique_count = df.apply(lambda x: len(x.unique()))
    df_unique = df.apply(lambda x: crop(str(list(x.unique()))))

    df_info = pd.concat(
        (
            df_dtypes.rename('Dtype'),
            df_nulls.rename('Non-Null Count'),
            df_unique_count.rename('Unique Values Count'),
            df_unique.rename('Unique Values'),
        ),
        axis=1,
    )
    df_info.index.name = 'Columns'
    print(f\"\"\"Total rows: {df.shape[0]}
Total columns: {df.shape[1]}

{df_info}\"\"\")
"""


class LocalMatplotlibCodeExecutor(BaseCodeExecutor):
  """Executes code locally and returns captured stdout and matplotlib figures."""

  # Disable statefulness by default so each execution starts fresh.
  stateful: bool = Field(default=False)

  # Local execution cannot safely optimize request data files automatically.
  optimize_data_file: bool = Field(default=False, frozen=True)

  _globals: dict[str, Any] = PrivateAttr(default_factory=dict)

  def _build_globals(self) -> dict[str, Any]:
    """Create the globals dict and execute the prelude."""
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    exec(_PRELUDE, namespace)
    return namespace

  def _ensure_state(self) -> dict[str, Any]:
    """Return the globals dict respecting the executor statefulness."""
    if self.stateful:
      if not self._globals:
        self._globals = self._build_globals()
      return self._globals
    return self._build_globals()

  def _reset_figures(self) -> list[File]:
    """Capture active matplotlib figures and return them as ADK files."""
    files: list[File] = []
    for idx, figure_number in enumerate(list(plt.get_fignums()), start=1):
      figure = plt.figure(figure_number)
      buffer = io.BytesIO()
      figure.savefig(buffer, format="png", bbox_inches="tight")
      buffer.seek(0)
      encoded = CodeExecutionUtils.get_encoded_file_content(
          buffer.read()
      ).decode("utf-8")
      files.append(
          File(
              name=f"figure_{idx}.png",
              content=encoded,
              mime_type="image/png",
          )
      )
    plt.close("all")
    return files

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    globals_dict = self._ensure_state()
    stdout_capture = io.StringIO()
    stderr_message = ""
    files: list[File] = []

    with ExitStack() as stack:
      stack.enter_context(redirect_stdout(stdout_capture))
      temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
      globals_dict.setdefault("TEMP_DIR", temp_dir)
      # Materialize input files under temp dir for convenience.
      if code_execution_input.input_files:
        input_paths = []
        for input_file in code_execution_input.input_files:
          path = os.path.join(temp_dir, input_file.name)
          with open(path, "wb") as handle:
            handle.write(input_file.content)
          input_paths.append(path)
        globals_dict["INPUT_FILE_PATHS"] = input_paths

      try:
        exec(code_execution_input.code, globals_dict)
      except Exception as exc:  # pylint: disable=broad-except
        stderr_message = str(exc)

    # Collect generated figures and reset between executions.
    files.extend(self._reset_figures())

    if not self.stateful:
      self._globals.clear()

    return CodeExecutionResult(
        stdout=stdout_capture.getvalue(),
        stderr=stderr_message,
        output_files=files,
    )
