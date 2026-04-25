# Suyash → Roopal Handoff Template

Copy, fill, and send this as one message after the short run finishes.

---

## If The Run Succeeded

```text
Short run complete.

Commit hash trained from: [FILL]
Exact command used: [FILL]
Platform: [FILL]
Output directory: [FILL]
Runtime: [FILL]

Artifacts:
- training_summary.json: [attached/path]
- reward_log.csv: [attached/path]
- reward curve PNG: [attached/path]
- optional eval JSON: [attached/path or N/A]

Key metrics:
- mean reward: [FILL]
- final reward: [FILL]
- best reward: [FILL]
- worst reward: [FILL]
- success rate: [FILL]
- avg steps/episode: [FILL]
- final curriculum tier: [FILL]

Qualitative evidence:
- early bad episode ID: [FILL]
- best late episode ID: [FILL]
- clearest learned behavior: [FILL]
- one bug/surprise from the run: [FILL]
```

## If The Run Failed

```text
Short run failed.

Commit hash trained from: [FILL]
Exact command used: [FILL]
Platform: [FILL]
Failure stage: [FILL]

Preflight:
- tests passed: [FILL]
- dry-run passed: [FILL]

Logs:
- last relevant log lines: [paste]

Diagnosis:
- likely cause: [FILL]
- fix attempted: [FILL]
- next recommended action: [FILL]

Why this was still a good use of time:
- [FILL]
```
