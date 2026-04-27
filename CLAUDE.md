# CLAUDE.md — Project Rules & Agent Behavior

## AgentMemory MCP — Mandatory Usage Rules

These rules are non-negotiable and apply to every conversation in this project.

### 1. Session Start — ALWAYS recall first

At the start of every conversation, before doing any work:
1. Run `memory_sessions` to see recent sessions.
2. Run `memory_smart_search` or `memory_recall` with the topic at hand (e.g., "diarization", "enrollment", "transcription pipeline") to load relevant context.

Do NOT skip this step even if the user's request seems simple or self-contained.

### 2. During Work — Save decisions as they happen

Save to agentmemory immediately (not at the end) when:
- A non-obvious architectural decision is made (type: `architecture`)
- A bug root cause is found or fixed (type: `bug`)
- A pattern or convention is established (type: `pattern`)
- A user preference or workflow is confirmed (type: `preference`)
- A key fact about the system is learned (type: `fact`)
- A repeatable workflow step is identified (type: `workflow`)

Use `memory_save` with:
- `content`: clear, self-contained description of the insight
- `type`: one of pattern | preference | architecture | bug | workflow | fact
- `concepts`: comma-separated keywords for future retrieval
- `files`: relevant file paths when applicable

### 3. Session End — Save a summary

Before ending any substantive session, save a session summary:
```
type: workflow
content: "Session summary: <what was done, key decisions, open items>"
concepts: "session-summary, <topic keywords>"
```

### 4. Search before asking

Before asking the user a question that might have been answered in a prior session, run `memory_smart_search` first. If the answer is in memory, use it without asking.

### 5. Retrieval strategy

- Use `memory_smart_search` for open-ended queries (semantic + keyword hybrid).
- Use `memory_recall` when you know specific keywords or file names.
- Use `memory_sessions` to get session IDs, then expand with `memory_smart_search` `expandIds`.
- Keep `limit` at 10–20 for broad queries; reduce for focused ones.
- Use `token_budget` on `memory_recall` when context is tight.

---

## Project Context

- **Project**: SST-models — call transcription & speaker diarization pipeline
- **Hardware**: RTX 4050 6GB VRAM (GPU memory is a constraint)
- **Domain**: Car Planet dealership call processing (agent vs. customer diarization)
- **Key rule**: Enrollment audio must contain only clean agent speech — never customer windows

## Auto-Memory (file-based)

The file-based memory at `C:\Users\abhis\.claude\projects\C--Users-abhis-Desktop-SST-models\memory\` is a separate system. Both systems are used:
- **File-based memory** (`MEMORY.md`): user profile, feedback, project-level facts, references
- **AgentMemory MCP**: session observations, code-level decisions, bug fixes, workflow steps

Do not duplicate between the two; use file-based memory for durable cross-project user preferences and agentmemory for session-level technical context.
