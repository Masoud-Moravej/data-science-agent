from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent

# Ensure required environment variables (e.g., BigQuery settings) are loaded
# before importing the main agent module, which expects them at import time.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from data_science.agent import root_agent


_FIGURE_DATA_PATTERN = re.compile(
    r"FIGURE(?:\[(?P<title>[^\]]+)\])?:\s*(?P<data>data:image/[a-zA-Z0-9.+\-]+;base64,[A-Za-z0-9+/=\s]+)"
)


class AgentChatService:
    """Lightweight asynchronous wrapper around the ADK runner for chat workflows."""

    def __init__(self) -> None:
        self._load_env()
        app_name = os.getenv("APP_NAME", "agents")
        self._runner = InMemoryRunner(agent=root_agent, app_name=app_name)
        self._app_name = self._runner.app_name
        self._sessions: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def _ensure_session(self, user_id: str) -> Any:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = await self._runner.session_service.create_session(
                    app_name=self._app_name,
                    user_id=user_id,
                )
                self._sessions[user_id] = session
            return session

    async def send_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Send a user message to the agent and collate the structured response."""
        session = await self._ensure_session(user_id)
        user_content = UserContent(parts=[Part(text=message)])

        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_responses: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        seen_artifacts: set[tuple[str, str]] = set()

        async for event in self._runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=user_content,
        ):
            content = getattr(event, "content", None)
            if not content or not getattr(content, "parts", None):
                continue

            for part in content.parts:
                if getattr(part, "text", None):
                    text_chunks.append(part.text)
                if getattr(part, "function_call", None):
                    tool_calls.append(
                        {
                            "name": part.function_call.name,
                            "args": part.function_call.args or {},
                        }
                    )
                if getattr(part, "function_response", None):
                    tool_responses.append(
                        {
                            "name": part.function_response.name,
                            "response": part.function_response.response or {},
                        }
                    )
                code_result = getattr(part, "code_execution_result", None)
                if code_result:
                    for output_file in getattr(code_result, "output_files", []) or []:
                        encoded = getattr(output_file, "content", None)
                        if not encoded:
                            continue
                        name = getattr(output_file, "name", None) or "artifact"
                        key = (name, encoded)
                        if key in seen_artifacts:
                            continue
                        artifacts.append(
                            {
                                "name": name,
                                "mime_type": getattr(output_file, "mime_type", None)
                                or "application/octet-stream",
                                "data": encoded,
                                "display_name": getattr(
                                    output_file, "display_name", None
                                )
                                or name,
                            }
                        )
                        seen_artifacts.add(key)
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    artifacts.append(
                        {
                            "mime_type": getattr(inline, "mime_type", None)
                            or "application/octet-stream",
                            "data": data,
                            "display_name": getattr(inline, "display_name", None),
                        }
                    )

                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    raw_bytes = inline.data
                    if isinstance(raw_bytes, str):
                        try:
                            base64.b64decode(raw_bytes, validate=True)
                            encoded = raw_bytes
                        except binascii.Error:
                            encoded = base64.b64encode(
                                raw_bytes.encode("utf-8")
                            ).decode("utf-8")
                    else:
                        encoded = base64.b64encode(raw_bytes).decode("utf-8")
                    name = getattr(inline, "display_name", None) or "artifact"
                    key = (name, encoded)
                    if key not in seen_artifacts:
                        artifacts.append(
                            {
                                "name": name,
                                "mime_type": getattr(inline, "mime_type", None)
                                or "application/octet-stream",
                                "data": encoded,
                                "display_name": getattr(inline, "display_name", None),
                            }
                        )
                        seen_artifacts.add(key)

            actions = getattr(event, "actions", None)
            artifact_delta = getattr(actions, "artifact_delta", None)
            if artifact_delta and self._runner.artifact_service:
                for filename, version in artifact_delta.items():
                    artifact_part = await self._runner.artifact_service.load_artifact(
                        app_name=self._app_name,
                        user_id=session.user_id,
                        session_id=session.id,
                        filename=filename,
                        version=version,
                    )
                    if not artifact_part:
                        continue
                    inline = getattr(artifact_part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        raw_bytes = inline.data
                        if isinstance(raw_bytes, str):
                            try:
                                base64.b64decode(raw_bytes, validate=True)
                                encoded = raw_bytes
                            except binascii.Error:
                                encoded = base64.b64encode(
                                    raw_bytes.encode("utf-8")
                                ).decode("utf-8")
                        else:
                            encoded = base64.b64encode(raw_bytes).decode("utf-8")
                        key = (filename, encoded)
                        if key not in seen_artifacts:
                            artifacts.append(
                                {
                                    "name": filename,
                                    "mime_type": getattr(inline, "mime_type", None)
                                    or "application/octet-stream",
                                    "data": encoded,
                                    "display_name": getattr(inline, "display_name", None)
                                    or filename,
                                }
                            )
                            seen_artifacts.add(key)

        combined_text = "".join(text_chunks).strip()
        (
            cleaned_text,
            embedded_artifacts,
        ) = self._extract_embedded_figures(combined_text)
        artifacts.extend(embedded_artifacts)

        return {
            "text": cleaned_text,
            "tool_calls": tool_calls,
            "tool_responses": tool_responses,
            "artifacts": artifacts,
        }

    @staticmethod
    def _load_env() -> None:
        """Load environment variables from the project .env files if present."""
        # Load repo-level defaults from the working directory.
        load_dotenv(override=False)

        # Explicitly load the agent-specific .env file.
        # agent_env = Path(__file__).resolve().parent.parent / "first_agent" / ".env"
        # if agent_env.exists():
        #     load_dotenv(agent_env, override=False)

    @staticmethod
    def _extract_embedded_figures(text: str) -> tuple[str, List[Dict[str, Any]]]:
        """Convert FIGURE[data-uri] markers into artifact entries."""

        if not text:
            return text, []

        artifacts: List[Dict[str, Any]] = []

        def _replace(match: re.Match[str]) -> str:
            data_uri = match.group("data") or ""
            title = match.group("title") or "figure"
            try:
                prefix, encoded = data_uri.split(",", 1)
            except ValueError:
                return match.group(0)

            mime_type = prefix[len("data:") :].split(";", 1)[0] or "image/png"
            encoded_clean = "".join(encoded.split())

            try:
                base64.b64decode(encoded_clean, validate=True)
            except binascii.Error:
                return match.group(0)

            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", title.strip() or "figure")
            artifacts.append(
                {
                    "name": f"{safe_name}.png"
                    if not safe_name.lower().endswith(".png")
                    else safe_name,
                    "mime_type": mime_type,
                    "data": encoded_clean,
                    "display_name": title.strip() or safe_name,
                }
            )
            return f"[See figure: {title.strip() or safe_name}]"

        cleaned_text = _FIGURE_DATA_PATTERN.sub(_replace, text)
        return cleaned_text, artifacts


agent_chat_service = AgentChatService()
