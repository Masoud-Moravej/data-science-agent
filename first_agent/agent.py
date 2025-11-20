from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types


@lru_cache()
def _greeting_image_path() -> Path:
    return Path(__file__).resolve().parent / "007.png"


async def get_greeting_image(tool_context: ToolContext) -> Dict[str, str]:
    """Save the local greeting image as an artifact for the UI to render."""
    image_path = _greeting_image_path()
    if not image_path.exists():
        return {"status": "error", "message": "Greeting image not found.", "artifact": ""}

    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type="image/png",
    )
    await tool_context.save_artifact(filename="greeting.png", artifact=image_part)
    return {"status": "success", "artifact": "greeting.png"}


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}


root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="Shows the greeting image once, then tells the current time in a specified city.",
    instruction=(
        "FIRST RESPONSE ONLY: Call the `get_greeting_image` tool. "
        "After the tool returns, tell the user their greeting image is displayed and ask: Which city would you like the time for? "
        "Do NOT paste any base64 or markdown for the image in the message."
        "\n\nSUBSEQUENT RESPONSES: Do not show the image again. Use the `get_current_time` tool and answer concisely."
    ),
    tools=[get_greeting_image, get_current_time],
)
