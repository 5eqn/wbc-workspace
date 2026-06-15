Behavioral guidelines to reduce common LLM coding mistakes.

Tradeoff: these guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

State assumptions explicitly.
Surface tradeoffs early.
Present multiple reasonable interpretations when they exist.
Ask when meaning, scope, or risk needs user input.
Call out simpler approaches when they fit.

## 2. Simplicity First

Implement the minimum code that solves the request.
Keep abstractions single-purpose.
Add configuration only when the request asks for it.
Keep error handling aligned with real failure modes.
Review large diffs and compress them when a shorter path exists.

## 3. Surgical Changes

Touch only lines tied directly to the request.
Match existing style, structure, and naming.
Mention unrelated dead code or cleanup candidates in the closeout.
Remove imports, variables, or helpers that become unused through your edit.
Preserve user changes and adjacent behavior.

## 4. Goal-Driven Execution

Define a concrete outcome and verification surface before large changes.
Prefer tests, reports, logs, artifacts, or command outputs that prove the result.
For multi-step tasks, write a brief plan with verification at each step.

Example:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong goals usually describe:

- Outcome
- Verification surface
- Constraints
- Boundaries
- Iteration policy
- Blocked stop condition

When the task produces artifacts, prefer a clear tree such as:

```text
wbc-workspace/
├── docker/
├── scripts/
├── assets/
├── thirdparties/
├── logs/
└── artifacts/
```

## 5. Repo-Level Skills

Keep repo skills under `.agents/skills/`.

Each skill lives in `.agents/skills/<skill-name>/` and includes:

- `SKILL.md`
- `agents/openai.yaml`

Add a skill with:

```bash
python /home/seqn/.codex/skills/.system/skill-creator/scripts/init_skill.py <skill-name> \
  --path .agents/skills \
  --interface display_name="..." \
  --interface short_description="..." \
  --interface default_prompt="Use $<skill-name> ..."
```

Validate a skill with:

```bash
python /home/seqn/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/<skill-name>
```

Use a skill by:

- Naming `$skill-name` directly in the prompt
- Describing a task that matches the `description` field in `SKILL.md`
- Adding `scripts/`, `references/`, or `assets/` when the skill benefits from helpers, reference material, or reusable output assets

When a task in this repo becomes repeated, stable, or multi-step, capture it as a repo skill and add the skill to `STATE.md`.

## 6. FRESH Environment Management

Use FRESH when the user asks for FRESH environment management.
Prefer host-level environment setup from a base distro, conda configuration, and read-only thirdparty dependencies.
Install dependencies via conda or pip when available.
Place source-built thirdparty dependencies in `$HOME`.
Keep major execution logic on host.
When a task asks for patching a thirdparty under FRESH, keep that dependency in project `thirdparties/` and record every changed byte in a sibling HTML file.
Mention planned thirdparty patches during planning.
Use reachable China mirrors after a quick availability check, especially for CUDA or NVIDIA channels.

## Lessons Learned

- Prefer least-surprise paths.
- Prefer official download paths.
- For long waits, use longer polling intervals around five minutes.

## State Tracking

Use `STATE.md` to track state.

Track:

- Eras: datetime, codename, description
- Files: path, era, unique description
- Skills: skill name, path, trigger, purpose

Read `STATE.md` at session start.
Use the latest related era to frame new work.
Use the file and skill lists to find existing assets and entrypoints.
Update `STATE.md` at session end by adding or refining the relevant era, file, and skill entries.
