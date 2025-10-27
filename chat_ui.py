from __future__ import annotations

import base64
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import gradio as gr
from gradio_client import utils as gr_utils

from services.agent_service import agent_chat_service


Message = Dict[str, Any]

_FIGURE_PATTERN = re.compile(
    r"FIGURE(?:\[(?P<title>[^\]]+)\])?:\s*(?P<data>data:image/[A-Za-z0-9.+\-]+;base64,[A-Za-z0-9+/=\s]+)",
    re.IGNORECASE,
)


def _format_tool_summary(tool_payload: dict) -> str:
    """Render a simple bulleted summary for tool responses."""
    if not tool_payload:
        return ""
    lines = "\n".join(f"- {key.title()}: {value}" for key, value in tool_payload.items())
    return f"Last tool response:\n{lines}"


def _format_session_label(session_id: str) -> str:
    return f"**Session ID:** `{session_id}`"


def _build_data_uri(mime_type: str, data: str) -> str:
    """Convert raw base64 data into a data URI consumable by Gradio."""
    if data.startswith("data:"):
        return data
    return f"data:{mime_type};base64,{data}"


def _extract_inline_figures(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Find FIGURE[...] markers in text and return cleaned text + data URIs."""

    if not text:
        return text, []

    figures: List[Dict[str, str]] = []

    def _replace(match: re.Match[str]) -> str:
        title = (match.group("title") or "figure").strip()
        data_uri = match.group("data") or ""
        if not data_uri.startswith("data:"):
            return match.group(0)
        figures.append({"title": title or "figure", "data_uri": "".join(data_uri.split())})
        return f"[See figure: {title or 'figure'}]"

    cleaned_text = _FIGURE_PATTERN.sub(_replace, text)
    return cleaned_text, figures


async def _handle_message(
    message: str,
    history: List[Message],
    session_id: str,
) -> Tuple[List[Message], str, str, str]:
    """Send a chat message and update the conversation history."""
    cleaned_message = (message or "").strip()
    if not cleaned_message:
        return history, "", session_id, _format_session_label(session_id)

    result = await agent_chat_service.send_message(user_id=session_id, message=cleaned_message)

    text = (result.get("text") or "").strip()
    tool_responses = result.get("tool_responses") or []
    artifacts = result.get("artifacts") or []

    reply = text or "I wasn't able to generate a response. Please try asking in a different way."
    if not text and tool_responses:
        summary = _format_tool_summary(tool_responses[-1].get("response") or {})
        if summary:
            reply = f"{reply}\n\n{summary}"

    inline_figures: List[Dict[str, str]] = []
    if reply:
        reply, inline_figures = _extract_inline_figures(reply)

    assistant_messages: List[Message] = []
    if reply:
        assistant_messages.append({"role": "assistant", "content": reply})

    for artifact in artifacts:
        data = artifact.get("data")
        if not data:
            continue
        mime_type = artifact.get("mime_type") or "application/octet-stream"
        alt_text = artifact.get("display_name") or artifact.get("name") or "Generated artifact"
        try:
            file_obj = gr_utils.decode_base64_to_file(
                _build_data_uri(mime_type, data),
                prefix=artifact.get("name") or "artifact",
            )
        except ValueError:
            continue
        assistant_messages.append(
            {
                "role": "assistant",
                "content": {"path": file_obj.name, "alt_text": alt_text},
            }
        )

    for figure in inline_figures:
        try:
            file_obj = gr_utils.decode_base64_to_file(
                figure["data_uri"], prefix=re.sub(r"[^A-Za-z0-9._-]+", "_", figure["title"]) or "figure"
            )
        except ValueError:
            continue
        assistant_messages.append(
            {
                "role": "assistant",
                "content": {"path": file_obj.name, "alt_text": figure["title"] or "Generated figure"},
            }
        )

    if not assistant_messages:
        assistant_messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

    updated_history: List[Message] = history + [
        {
            "role": "user",
            "content": cleaned_message,
        },
    ]
    updated_history.extend(assistant_messages)

    return updated_history, "", session_id, _format_session_label(session_id)


def _reset_session() -> Tuple[List[Message], str, str, str]:
    """Start a new chat session."""
    new_session_id = str(uuid4())
    return [], "", new_session_id, _format_session_label(new_session_id)


def build_chat_ui() -> gr.Blocks:
    """Construct the Gradio Blocks application with custom styling."""
    DEFAULT_BADGE_DATA_URI = (
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'>"
        "<rect width='120' height='120' rx='24' fill='#05070d'/>"
        "<rect x='14' y='14' width='92' height='92' rx='20' fill='none' stroke='#f2c94c' stroke-width='2' opacity='0.65'/>"
        "<text x='60' y='70' font-size='34' text-anchor='middle' fill='#f2c94c' font-family='Arial Black' font-weight='700'>007</text>"
        "</svg>"
    )

    assets_dir = Path(__file__).resolve().parent / "assets"
    agent_badge = assets_dir / "agent_007.svg"
    badge_data_url = ""
    if agent_badge.exists():
        badge_data_url = "data:image/svg+xml;base64," + base64.b64encode(agent_badge.read_bytes()).decode("ascii")
    if not badge_data_url:
        badge_data_url = DEFAULT_BADGE_DATA_URI
    avatar_path = str(agent_badge) if agent_badge.exists() else None

    css = """
    :root {
        --agent-bg: #05070d;
        --agent-panel: rgba(13, 17, 23, 0.85);
        --agent-border: rgba(255, 215, 0, 0.35);
        --agent-gold: #f2c94c;
        --agent-accent: #0d1117;
        --bubble-user: linear-gradient(135deg, #f2c94c, #b8860b);
        --bubble-agent: linear-gradient(135deg, #1f2937, #0d1117);
    }
    body {
        background: var(--agent-bg) !important;
    }
    #agent-header {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        background: var(--agent-panel);
        border-radius: 18px;
        padding: 1.5rem;
        border: 1px solid var(--agent-border);
        color: #f5f5f5;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }
    #agent-header h1 {
        font-size: 1.9rem;
        margin: 0;
        letter-spacing: 0.08em;
    }
    #agent-header p {
        margin: 0.4rem 0 0;
        font-size: 0.95rem;
        color: rgba(229, 231, 235, 0.78);
    }
    #agent-figure {
        width: 92px;
        height: 92px;
        border-radius: 16px;
        background: rgba(8, 11, 19, 0.9);
        padding: 0.75rem;
        border: 1px solid rgba(242, 201, 76, 0.4);
        box-shadow: inset 0 0 18px rgba(242, 201, 76, 0.15);
    }
    #chatbot-container {
        border-radius: 18px;
        background: rgba(10, 15, 24, 0.85);
        border: 1px solid rgba(242, 201, 76, 0.1);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
    }
    #chatbot-container .message.user .bubble {
        background: var(--bubble-user);
        color: #161616;
        font-weight: 500;
    }
    #chatbot-container .message.bot .bubble {
        background: var(--bubble-agent);
        color: #eef2ff;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    #chatbot-container .message .avatar {
        border: none;
        background: rgba(0,0,0,0.25);
    }
    #session-label {
        text-align: right;
        color: var(--agent-gold);
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    }
    .gradio-button.primary {
        background: linear-gradient(135deg, #f2c94c, #b5860f) !important;
        color: #111827 !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(178, 134, 15, 0.35) !important;
    }
    .gradio-button.secondary {
        background: rgba(242, 201, 76, 0.1) !important;
        color: var(--agent-gold) !important;
        border: 1px solid rgba(242, 201, 76, 0.35) !important;
    }
    .gradio-textbox textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(242, 201, 76, 0.2) !important;
        color: #f9fafb !important;
    }
    """

    theme = (
        gr.themes.Soft(primary_hue="amber", secondary_hue="gray", neutral_hue="slate")
        .set(
            body_background_fill="#05070d",
            body_text_color="#f9fafb",
            block_background_fill="#060a12",
            block_title_text_color="#f2c94c",
            block_border_color="#101826",
        )
    )

    with gr.Blocks(title="Agent 007 Control Room", css=css, theme=theme) as demo:
        initial_session_id = str(uuid4())
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(
                    """
                    <div id="agent-header">
                        <img id="agent-figure" src="{}" alt="Agent 007 badge">
                        <div>
                            <h1>Agent 007</h1>
                            <p>Temporal intelligence division. Issue precise time briefings for any city on request.</p>
                        </div>
                    </div>
                    """.format(badge_data_url)
                )
            with gr.Column(scale=1, min_width=0):
                session_indicator = gr.Markdown(
                    _format_session_label(initial_session_id), elem_id="session-label"
                )

        chatbot = gr.Chatbot(
            label="Secure Channel",
            elem_id="chatbot-container",
            height=500,
            bubble_full_width=False,
            avatar_images=(None, avatar_path),
            type="messages",
        )

        with gr.Row():
            message_box = gr.Textbox(
                placeholder="Ask Agent 007 for the time in any city...",
                autofocus=True,
                scale=4,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)
            new_session_button = gr.Button("New Session", variant="secondary", scale=1)

        session_state = gr.State(initial_session_id)

        send_button.click(
            _handle_message,
            inputs=[message_box, chatbot, session_state],
            outputs=[chatbot, message_box, session_state, session_indicator],
        )
        message_box.submit(
            _handle_message,
            inputs=[message_box, chatbot, session_state],
            outputs=[chatbot, message_box, session_state, session_indicator],
        )
        new_session_button.click(
            _reset_session,
            inputs=None,
            outputs=[chatbot, message_box, session_state, session_indicator],
        )

    return demo
