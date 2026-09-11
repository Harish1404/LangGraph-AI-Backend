"""
The tool agent's toolbelt — the single place that says which tools exist.

Everything that needs the list reads it from here: the ToolNode in
tool_graph.py, and `_build_models` in app/ai/chat.py which binds them to the
model. Adding a tool is a one-line change in this file.

Kept in its own module rather than in tool_graph.py on purpose. chat.py imports
this list, and tool_graph.py's nodes import chat.py — going through the graph
module would close that loop into a circular import. This one imports nothing
from app.ai, so it is always safe to import.
"""

from app.ai.agents.tool_agent.tools.weather import get_weather

TOOLS = [get_weather]
