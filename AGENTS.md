Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### Details about goal design

The strongest Goals usually define six things:

Outcome: what should be true when the work is done.
Verification surface: the test, benchmark, report, artifact, command output, or source material that proves it.
Constraints: what must not regress while Codex works.
Boundaries: which files, tools, data, repositories, or resources Codex may use.
Iteration policy: how Codex should decide what to try next after each attempt.
Blocked stop condition: when Codex should stop and report that no defensible path remains under the current limits.

We aim for very high standards in verification surface, and prefer a test-generate-report workflow, so that verification surface gets unified to a single file tree. For example, to prove that a humanoid robot motion tracking Sim2Sim is correct, you output a raw log file, a report that calculates the RMSE in joint space and verify it's below 0.2 for majority of motions, a report that calculates the phase time & wall time & sim time and verify that they're the same, a report that measures inference speed of policy and verify that it's below control rate, some smoke tests with raw logs and a report that proves the behaviour that: no policy control and release causes the robot to fall in 2 seconds, add control then release can make robot stand for more than 5 seconds, releasing control then causes the robot to fall in 2 seconds, and event logs proving that in actual Sim2Sim you've released the robot before activating motion tracking. Most importantly, we asked agents to generate side-by-side comparison video for two motion tracking methods, rendering robot inside MuJoCo with collected raw log artifact, so that human can verify if you're doing things correctly. When designing goals, you must formalize verification surface as a concrete file tree.

We also include our final goal & vision inside the outcome section. For example, we do the Sim2Sim comparison of multiple policies so that we can immediately swap simulator docker container to real robot, and the deployment will be immediately used to make real robot perform loco-manipulation tasks, assessing the performance of different policies. Informing agents our vision can align them with our actual goal, so they no longer treat the verification surface as rules to hack through, instead they make real productional progress. In every outcome section we highlight that both Claude Opus 4.6 and human experts in embodied intelligence will review your work, and that they're satisfied should be true when the work is done.

For constraints, we ask the agents to not change host environment whenever possible. We allow installing lightweight Python packages on host, but all major logic should be run inside Docker containers. Below is a self-explanatory example:

```
wbc-workspace/
├── docker/                   # Dockerfiles — pure env build, NO scripts, NO entrypoint logic
│   ├── holomotion.Dockerfile
│   ├── gear-sonic.Dockerfile
│   └── unitree_mujoco.Dockerfile
├── scripts/
│   ├── run.sh                # Entrypoint 1: build, deploy, run a policy+motion pair
│   ├── report.sh             # Entrypoint 2: generate all reports from collected logs
│   └── download.sh           # Asset downloader (excluded from script count)
├── assets/                   # Pre-formatted motion clips + model weights (build context)
├── thirdparties/             # Upstream repos (build context)
├── logs/                     # Raw runtime output per run
└── artifacts/                # Final reports & video covering all 10 motions
```

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
