# HF Credit Run Guide

Use this after the Colab validation run to launch the stronger judged run on your Hugging Face account.

---

## Goal

Use your HF credits for a longer, more reliable run than Colab so you can produce:

- a clearer reward curve
- a stronger trained-vs-random gap
- a fuller artifact bundle for README, blog, and pitch

---

## Recommended First Paid Run

- Account: `Roopalgn`
- Hardware: `1x Nvidia A100 - large`
- Model: `Qwen2.5-1.5B-Instruct`
- Episodes: `50`
- Seed: `42`
- Output dir: `outputs/run_hf_1`

Why this choice:

- more reliable than Colab
- cheaper and safer than jumping straight to 7B
- enough room to produce a meaningful second evidence run if needed

---

## Step-by-Step

1. Log in to Hugging Face on the machine where you will launch the job.
   - `huggingface-cli login` or `hf auth login`

2. Make sure the repo is up to date.
   - `git checkout main`
   - `git pull github main`

3. Install the training extras.
   - `pip install -e ".[train]"`
   - `pip install matplotlib`

4. Set your environment variables.
   - `HF_TOKEN`
   - `PYTHONUNBUFFERED=1`
   - optional cache path if the machine supports it

5. Run one last preflight locally or on the launch environment.
   - `python train.py --dry-run --episodes 2 --model-size 1.5b --output-dir outputs/verify_hf`

6. Launch the first paid run with:
   - `python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 50 --seed 42 --output-dir outputs/run_hf_1`

7. After training finishes, generate the curve:
   - `python plot_rewards.py --csv outputs/run_hf_1/reward_log.csv --out results/reward_curve_hf_run1.png`

8. Run eval:
   - `python eval_compare.py --model-path outputs/run_hf_1/checkpoint --episodes 20 --output-dir outputs/eval_hf_run1`

9. If the run looks good, upload the model artifact under your account.

10. Fill docs in this order:
   - `README.md`
   - `docs/mini_blog_draft.md`
   - `docs/internal/pitch_notes.md`
   - `docs/training_log.md`

---

## Stop / Continue Rule

- Continue to a second paid run if the first HF run is stable but the curve/story is still weak.
- Stop and use the first HF run if:
  - the trained-vs-random gap is clearly better than Colab
  - the curve is more obviously improving
  - you have usable transcripts and artifacts

---

## Artifacts To Keep

- `training_summary.json`
- `reward_log.csv`
- eval JSON
- reward curve PNG
- model repo link
- one early failure episode
- one best late episode
