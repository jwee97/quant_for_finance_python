---
name: GitHub CLI authentication in Replit
description: Distinguishes project-level GitHub connections from authentication available to agent-shell Git commands.
---

An active GitHub connection does not necessarily mean `git push` or `gh` is authenticated in the agent shell. Verify CLI authentication directly before relying on the connection; never inspect or copy credentials as a workaround. If it fails, use Replit's Git pane or have the user refresh its GitHub authorization.

**Why:** In this workspace the GitHub connection was reported active, but GitHub rejected a push and `gh auth status` reported no signed-in host.

**How to apply:** Before committing to an agent-shell push, run a safe authentication/status check. Keep the local commit, report push failure accurately, and direct the user to the Git pane rather than requesting a token.
