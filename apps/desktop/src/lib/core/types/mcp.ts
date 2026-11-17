// TypeScript mirror of the desktop-API payloads (apps/api/models/mcp.py).
// Keep in sync with the pydantic models — this is the wire contract; a change to the
// model and a change here belong in the same commit (apps/desktop/CLAUDE.md).
//
// The MCP endpoint's settings (ADR-066 choice 9, A4): GET /api/mcp, POST /api/mcp/settings,
// POST /api/mcp/setup, POST /api/mcp/token/replace.

/** One call an outside program made. Never its arguments. */
export interface McpCall {
  tool: string
  items: number
  milliseconds: number
  at: string
}

/** What Settings shows. `token_hint` is the token's last characters; the token itself comes
 *  only from the setup. */
export interface McpStatus {
  enabled: boolean
  conversations: boolean
  token_hint: string | null
  url: string
  tools: string[]
  surface_version: number
  recent_calls: McpCall[]
}

export interface McpCommandSetup {
  command: string
  args: string[]
  env: Record<string, string>
}

/** The token and what to paste into a client: an HTTP address with its header, or a command
 *  entry (`config_json`) for a client that starts a program. */
export interface McpSetup {
  token: string
  url: string
  header: string
  command: McpCommandSetup
  config_json: string
}
