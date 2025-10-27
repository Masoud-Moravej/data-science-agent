"""Unsafe local executor that captures matplotlib figures as ADK artifacts."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import ExitStack, redirect_stdout
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Ensure headless backend
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


class UnsafeMatplotlibCodeExecutor(BaseCodeExecutor):
  """Executes Python locally while capturing generated matplotlib figures."""

  stateful: bool = Field(default=False)
  optimize_data_file: bool = Field(default=False, frozen=True)

  _globals: dict[str, Any] = PrivateAttr(default_factory=dict)

  def _globals_for_run(self) -> dict[str, Any]:
    """Return the globals mapping to use for this execution."""
    if self.stateful:
      if not self._globals:
        self._globals = {"__builtins__": __builtins__}
      return self._globals
    return {"__builtins__": __builtins__}

  def _reset_if_needed(self) -> None:
    if not self.stateful:
      self._globals.clear()

  def _capture_figures(self) -> list[File]:
    """Serialize active matplotlib figures into ADK File artifacts."""
    files: list[File] = []
    for idx, figure_number in enumerate(list(plt.get_fignums()), start=1):
      figure = plt.figure(figure_number)
      buffer = io.BytesIO()
      figure.savefig(buffer, format="png", bbox_inches="tight")
      buffer.seek(0)
      encoded = CodeExecutionUtils.get_encoded_file_content(buffer.read()).decode(
          "utf-8"
      )
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
    globals_dict = self._globals_for_run()
    stdout_capture = io.StringIO()
    stderr_message = ""
    files: list[File] = []

    with ExitStack() as stack:
      stack.enter_context(redirect_stdout(stdout_capture))
      temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
      globals_dict.setdefault("TEMP_DIR", temp_dir)

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

    files.extend(self._capture_figures())
    self._reset_if_needed()

    return CodeExecutionResult(
        stdout=stdout_capture.getvalue(),
        stderr=stderr_message,
        output_files=files,
    )

