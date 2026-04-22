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

- [x] **Fix `train_colab.ipynb`** — fixed API signatures (seed not tier), added DRY_RUN toggle, MODEL_PRESETS, correct ENV_URL, reward dict handling. Now matches train_kaggle.ipynb structure.
- [ ] **Test `train_colab.ipynb` on Colab free tier** — open in Colab, set `DRY_RUN = True`, run all cells. Verify env connection + dry-run CSV output.
- [x] **Create proper `train_kaggle.ipynb`** — 22-cell notebook with DRY_RUN toggle, MODEL_PRESETS, kaggle_secrets for HF auth.
- [ ] **Test `train_kaggle.ipynb` on Kaggle** — set `DRY_RUN = True`, run all cells. Verify env connection + dry-run CSV output.
- [x] **Prepare deliverable templates** — `[FILL ONSITE]` placeholders added to `docs/mini_blog_draft.md` and `docs/internal/pitch_notes.md`.
- [x] **Create `docs/onsite_checklist.md`** — Phase 0-5 terminal commands, model fallback ladder, runtimes, disk budget.
- [x] **Prepare `docs/training_log.md` template** — updated for onsite H100, `[FILL ONSITE]` placeholders.
- [ ] **Study `docs/internal/resources.md`** — reward engineering best practices, common GRPO pitfalls, organiser Q&A.

## Suyash

- [x] **Run `train.py --dry-run --episodes 5 --model-size 1.5b`** on merged main — verified CSV, JSON correct.
- [x] **Run `train.py --dry-run --episodes 5 --model-size 3b`** and `7b` — verified presets apply correctly.
- [x] **Run `eval_compare.py --base-only --episodes 5`** — verified baseline JSON report.
- [x] **Stress-test resume logic** — verified checkpoint files and large episode count.
- [x] **Verify HF Space** — all endpoints (`/ping`, `/reset`, `/step`, `/state`) respond.
- [x] **Install matplotlib locally** — verified available for onsite.

## Joint

- [ ] **Review ROADMAP Push 8 plan** — ensure both understand the onsite flow: Run 1 (1.5B fast) → Run 2 (3B scale) → eval → deliverables → pitch.
- [ ] **Agree on HF Hub repo name** for checkpoint upload (e.g., `roopal-gn/clinical-trial-agent-lora`).
- [ ] **Pack IDs** — valid government-issued ID + college/company ID for venue entry.
