"""Wire models for the MCP endpoint's settings (ADR-066 choice 9, A4).

The endpoint itself speaks JSON-RPC (`doc_assistant.mcp_server`); these are what the window reads
and writes about it. ``core/types/mcp.ts`` mirrors these names.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class McpCallPayload(BaseModel):
    """One call an outside program made: the tool, how many items it returned, how long it took,
    when. Never the call's arguments."""

    tool: str
    items: int
    milliseconds: int
    at: datetime


class McpStatusPayload(BaseModel):
    """What Settings shows. ``token_hint`` is the token's last characters, ``None`` before the
    endpoint was first turned on; the token itself comes only from ``POST /api/mcp/setup``."""

    enabled: bool
    conversations: bool
    token_hint: str | None
    url: str
    tools: list[str]
    surface_version: int
    recent_calls: list[McpCallPayload]


class McpSettingsUpdate(BaseModel):
    enabled: bool | None = None
    conversations: bool | None = None


class McpCommandSetup(BaseModel):
    """How a client that starts a command runs the bridge."""

    command: str
    args: list[str]
    env: dict[str, str]


class McpSetupPayload(BaseModel):
    """The token and the two setups to paste: one for a client that connects over HTTP, one for
    a client that starts a command (the bridge). ``config_json`` is the second as the common
    ``mcpServers`` file entry, ready to paste."""

    token: str
    url: str
    header: str
    command: McpCommandSetup
    config_json: str
