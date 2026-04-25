# Submission Checklist — Post-Run Intake

Use this the moment Suyash sends the short-run artifacts. Goal: update every submission-facing surface in under 10 minutes.

---

## Link Set

Verify these before submission:

- Space URL: `https://roopalgn-openenv-clinical-trial.hf.space`
- Notebook link: `train_colab.ipynb`
- Repo link: `https://github.com/Roopalgn/openenv-clinical-trial`
- Planned blog destination: Hugging Face blog post published from `docs/mini_blog_draft.md`

---

## Artifact Intake

Canonical sources of truth:

- Metrics: `training_summary.json`
- Reward curve: reward PNG generated from `reward_log.csv`
- Qualitative evidence: early-bad episode ID + best-late episode ID + transcript/log snippets

Required from Suyash:

- Commit hash trained from
- Exact command used
- `training_summary.json`
- `reward_log.csv`
- Reward curve PNG
- Early bad episode ID
- Best late episode ID
- Optional eval JSON

---

## Fill Order

1. Update `README.md`
   - Fill trained-policy row
   - Fill run summary table
   - Add reward curve reference
   - Fill early/late episode comparison
2. Update `docs/mini_blog_draft.md`
   - Fill Onsite Fill Sheet
   - Fill setup + drop-in metrics
   - Fill the one-paragraph “what the agent learned” section
3. Update `docs/internal/pitch_notes.md`
   - Fill top metric sheet
   - Fill training-results script shell
   - Fill early and late episode placeholders
4. Update `docs/training_log.md`
   - Fill Fast Fill Sheet
   - Record one bug or surprise
   - Record the early and late episode pair

---

## If The Run Failed

Collect and document:

- Failure stage
- Exact command used
- Dry-run status
- Test status
- Last relevant logs
- Fix attempted
- Why preserving HF credits and using Colab was still the right call
