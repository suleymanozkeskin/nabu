type HarnessEvent = Record<string, unknown>

const EVENT_NAMES = [
  "message.updated",
  "message.part.updated",
  "message.removed",
  "message.part.removed",
  "session.created",
  "session.updated",
  "session.compacted",
  "session.idle",
  "session.error",
  "tool.execute.before",
  "tool.execute.after",
  "command.executed",
  "file.edited",
] as const

async function capture(eventName: string, payload: HarnessEvent) {
  try {
    const command = process.env.NABU_BIN || "nabu"
    const args = ["ingest", "hook", "--tool", "opencode"]

    if (process.env.NABU_HOME) {
      args.push("--home", process.env.NABU_HOME)
    }

    const proc = Bun.spawn([command, ...args], {
      stdin: "pipe",
      stdout: "ignore",
      stderr: "pipe",
    })
    const stdin = JSON.stringify({
      ...payload,
      hook_event_name: eventName,
      type: typeof payload.type === "string" ? payload.type : eventName,
    })

    await proc.stdin.write(stdin)
    proc.stdin.end()

    const exitCode = await proc.exited
    if (exitCode !== 0) {
      console.error(`[nabu] ingest exited with status ${exitCode}`)
    }
  } catch (error) {
    console.error("[nabu] failed to capture OpenCode event", error)
  }
}

export default async function harnessHistoryPlugin() {
  const hooks: Record<string, (payload: HarnessEvent) => Promise<void>> = {}

  for (const eventName of EVENT_NAMES) {
    hooks[eventName] = async (payload: HarnessEvent) => {
      await capture(eventName, payload)
    }
  }

  return hooks
}
