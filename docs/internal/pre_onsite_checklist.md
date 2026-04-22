# Final Task Checklist Before Apr 25 (Onsite)

> **Rule:** NO training before Apr 25. Only dry-run pipeline validation and notebook testing.

## Verification Status (Apr 22)

| Check | Status | Details |
|-------|--------|---------|
| Pytest | 249/249 passed | 17s, no failures |
| Ruff lint | All passed | Python clean; notebooks excluded (not standard JSON) |
| Docker | Dockerfile valid | HF Space proves it builds correctly |
| HF Space `/ping` | `{"status":"ok"}` | Live |
| HF Space `/schema` | Full schema returned | 19 action types, TrialObservation correct |
| HF Space `/reset` | Works | Returns warmup scenario, 2 available actions |
| HF Space `/step` | Works | Returns decomposed reward (7 components) |
| `train.py --dry-run` | Works | 2 episodes, CSV + JSON summary generated |
| `eval_compare.py --base-only` | Works | Baseline report generated, mean_reward=152.13 |
| Git | Clean | main at `ca765cf`, pushed |

---

## Roopal

- [ ] **Test `train_colab.ipynb` on Colab free tier** — open in Colab, run with `--dry-run`. Verify it connects to HF Space, runs 2 episodes, produces CSV. If Qwen2.5-7B setup cell fails on T4, add a 1.5B + Unsloth 4-bit fallback cell.
- [ ] **Create proper `train_kaggle.ipynb`** — currently empty (0 bytes). Copy structure from `train_colab.ipynb`, adapt for Kaggle (different pip install, different env vars). Test with `--dry-run` on Kaggle.
- [ ] **Prepare deliverable templates** — add `[FILL ONSITE]` placeholders in `docs/mini_blog_draft.md` and `docs/internal/pitch_notes.md` for reward curves, episode transcripts, and real numbers.
- [ ] **Create `docs/onsite_checklist.md`** — exact terminal commands for onsite:
  - `git clone` + `pip install` sequence for H100 environment
  - `HF_TOKEN` and env var setup
  - Model size fallback ladder: try 7B → 3B → 1.5B
  - Expected runtimes per model size on H100
  - Checkpoint frequency and disk budget
- [ ] **Prepare `docs/training_log.md` template** — columns ready for Statement 4 bugs (timestamp, bug, fix, impact on reward).
- [ ] **Study `docs/internal/resources.md`** — reward engineering best practices, common GRPO pitfalls, organiser Q&A.

## Suyash

- [ ] **Run `train.py --dry-run --episodes 5 --model-size 1.5b`** on merged main — verify CSV, JSON, all correct.
- [ ] **Run `train.py --dry-run --episodes 5 --model-size 3b`** and `7b` — verify presets apply correctly.
- [ ] **Run `eval_compare.py --base-only --episodes 5`** — verify baseline JSON report.
- [ ] **Stress-test resume logic** — run with `--episodes 1`, check checkpoint files, verify `--episodes 100` dry-run handles large count.
- [ ] **Verify HF Space** — hit `/ping`, `/reset`, `/step`, `/state` from a script. Confirm all endpoints respond.
- [ ] **Install matplotlib locally** (or verify it's in the pip install for onsite) — dry-run warned `No module named 'matplotlib'` for plot generation.

## Joint

- [ ] **Review ROADMAP Push 8 plan** — ensure both understand the onsite flow: Run 1 (1.5B fast) → Run 2 (3B scale) → eval → deliverables → pitch.
- [ ] **Agree on HF Hub repo name** for checkpoint upload (e.g., `roopal-gn/clinical-trial-agent-lora`).
- [ ] **Pack IDs** — valid government-issued ID + college/company ID for venue entry.
