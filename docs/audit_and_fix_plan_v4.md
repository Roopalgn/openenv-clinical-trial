# Codebase Audit & Fix Plan — V4

**Date:** 2026-04-26  
**Status:** ALL FIXES IMPLEMENTED — 249/249 tests pass

**Context:** V3.1 training showed reward starting at 5.x and plateauing at 6.0 with std=0 by step 12. Deep audit revealed 5 critical, 8 major, and 6 moderate issues in the reward system, simulator, judge, and episode manager.

---

## Issues Detected & Status

### CRITICAL — Blocks Training (all FIXED)

| ID | Issue | File | Status |
|---|---|---|---|
| C1 | Power calculator NCP off by √2 | `power_calculator.py` | ✅ FIXED |
| C2 | Shaping bonus contaminates r_ordering | `episode_manager.py` | ✅ FIXED |
| C3 | No terminal failure penalty (was 0.0, spec says -1.0) | `reward_computer.py` | ✅ FIXED |
| C4 | NaN vulnerability in calibration + info_gain reward | `reward_computer.py` | ✅ FIXED |
| C5 | Phase mapping difference TransitionEngine vs PhaseDetector | N/A | ❌ NOT A BUG |

### MAJOR — Significant Quality Gaps (all FIXED)

| ID | Issue | File | Status |
|---|---|---|---|
| M1 | `r_novelty` permanently zero (checked post-transition history) | `reward_computer.py` | ✅ FIXED |
| M2 | `_milestone_reward` fires on every repeat (not first-time only) | `reward_computer.py` | ✅ FIXED |
| M3 | `SUBMIT_TO_FDA_REVIEW` unreachable (chicken-and-egg prereq) | `fda_rules.py` | ✅ FIXED |
| M4 | `initial_budget` uses pre-noise scenario value | `episode_manager.py` | ✅ FIXED |
| M5 | Soft-violation degradation dead code (multiplicative × 0) | `transition_engine.py` | ✅ FIXED |
| M6 | No `done` guard — agent can step after episode ends | `episode_manager.py` | ✅ FIXED |
| M7 | Judge overconfidence fires on every step (power/p-value always fail early) | `judge.py` | ✅ FIXED |
| M8 | Invalid steps don't count toward step limit (infinite loop) | `episode_manager.py` | ✅ FIXED |

### MODERATE — Quality Improvements (FIXED where impactful)

| ID | Issue | File | Status |
|---|---|---|---|
| M9 | `SYNTHESIZE_CONCLUSION` unreachable after `trial_complete` triggers done | N/A | ⏭ DEFERRED (dead code, not training-critical) |
| M10 | Global `_manager` singleton not thread-safe | `app.py` | ⏭ DEFERRED (single-worker uvicorn) |
| M11 | Shaping bonus folded into `r_info_gain` loses interpretability | N/A | ⏭ DEFERRED (would break 8-field assertions) |
| M12 | Phase mapping inconsistencies (5 actions differ between phase_detector & transition_engine) | N/A | ❌ NOT A BUG (different purposes) |
| M13 | No timeout penalty for timed-out episodes | `episode_manager.py` | ✅ FIXED |
| M14 | Adverse events don't scale with patient enrollment count | `transition_engine.py` | ✅ FIXED |

---

## Detailed Fix Descriptions

### C1. Power Calculator NCP Off by √2
**File:** `server/simulator/power_calculator.py` L39  
**Before:** `ncp = abs(effect_size) * math.sqrt(n_per_arm / 2.0)`  
**After:** `ncp = abs(effect_size) * math.sqrt(n_per_arm)`  
**Verification:** `calculate_power(0.5, 100, 0.05)` = 0.9424 (was 0.38)

### C2. Shaping Bonus Contaminates r_ordering
**File:** `server/episode_manager.py`  
**Before:** Shaping bonus added to `r_ordering`  
**After:** Shaping bonus added to `r_info_gain` (semantically closer: milestone progress)

### C3. No Terminal Failure Penalty
**File:** `server/reward/reward_computer.py`  
**Before:** `_terminal_success_reward()` returned 0.0 on trial failure  
**After:** Returns -1.0 when `trial_complete=True` and `not result.success`

### C4. NaN Vulnerability
**File:** `server/reward/reward_computer.py`  
**Fix:** Added `math.isfinite()` guards in `_terminal_calibration_reward()` and `_info_gain_reward()`

### C5. Phase Mapping Difference — NOT A BUG
TransitionEngine maps `SET_PRIMARY_ENDPOINT → "hypothesis"` (workflow state). PhaseDetector maps it to `"design"` (r_ordering category). Different purposes. Changing TransitionEngine broke FDA compliance checks. Reverted.

### M1. r_novelty Permanently Zero
**File:** `server/reward/reward_computer.py`  
**Root cause:** TransitionEngine appends action to `action_history` before `compute_reward` runs, so the action was always found.  
**Fix:** Check `action_history[:-1]` (exclude current action)

### M2. Milestone Reward Repeat Exploit
**File:** `server/reward/reward_computer.py`  
**Root cause:** Milestone flags stay True forever → bonus fires on every repeat  
**Fix:** Check `action_count == 1` in `action_history` (only reward first occurrence)

### M3. SUBMIT_TO_FDA_REVIEW Unreachable
**File:** `server/rules/fda_rules.py`  
**Root cause:** `check_fda_compliance` required `protocol_submitted=True` to allow `SUBMIT_TO_FDA_REVIEW`, but that flag is only set BY the action itself  
**Fix:** Removed `protocol_submitted` prerequisite from `SUBMIT_TO_FDA_REVIEW` (kept `phase_i_complete` only). `REQUEST_PROTOCOL_AMENDMENT` still requires `protocol_submitted`.

### M4. initial_budget Mismatch
**File:** `server/episode_manager.py`  
**Root cause:** `initial_budget=float(self._scenario.budget_usd)` used pre-noise budget, but `latent.budget_remaining` starts from randomized budget  
**Fix:** Store `self._initial_budget = float(randomized.budget_usd)` at reset and use it for reward/shaping

### M5. Soft-Violation Degradation Dead Code
**File:** `server/simulator/transition_engine.py`  
**Root cause:** `measurement_noise` and `site_variability` start at 0.0, multiplicative degradation `0 × factor = 0`  
**Fix:** Changed to additive degradation: `+= 0.05 × (0.5 - confidence)` for noise, `+= 0.03` for budget overrun

### M6. No Done Guard
**File:** `server/episode_manager.py`  
**Fix:** Added `self._episode_done` flag set when `done=True`, checked at top of `step()` → raises `RuntimeError` if called after episode ends

### M7. Judge Overconfidence Fires Every Step
**File:** `server/judge.py`  
**Root cause:** Power/p-value checks always fail early in episode → overconfidence penalty fires on every step if confidence ≥ 0.8  
**Fix:** Only count FDA compliance violations and budget violations as "actionable" for overconfidence penalty. Power/p-value violations still appear in `violations` list for feedback but don't trigger the penalty.

### M8. Invalid Steps Don't Count Toward Limit
**File:** `server/episode_manager.py`  
**Root cause:** Invalid actions don't advance `action_history` → `step_idx >= _MAX_STEPS` never triggers  
**Fix:** Added `self._step_count` that increments on every `step()` call (valid or invalid). Done check uses `_step_count >= _MAX_STEPS`.

### M13. Timeout Penalty
**File:** `server/episode_manager.py`  
**Fix:** When `done=True` from step limit (not `trial_complete`), replace reward with flat penalty: `r_validity=-0.5, r_penalty=-1.5` (total = -2.0, matching spec)

### M14. Adverse Events Don't Scale with Enrollment
**File:** `server/simulator/transition_engine.py`  
**Root cause:** `ENROLL_PATIENTS` with 200 patients rolls a single AE check  
**Fix:** Each enrolled patient gets an independent `rng.random() < side_effect_rate` check. Other actions (OBSERVE_SAFETY_SIGNAL, RUN_DOSE_ESCALATION) still use single check.

---

## Test Changes

| Test | Change | Reason |
|---|---|---|
| `test_integration.py::test_submit_fda_review_without_protocol` | Now asserts `valid=True` | M3: removed protocol_submitted prereq |
| `test_integration.py::test_novelty_reward_zero_for_repeated_action` | History needs 2 entries | M1: novelty checks `[:-1]` |
| `test_judge.py::test_overconfidence_penalty_scales_with_violation_count` | Asserts `<= -0.5` (was -1.0) | M7: only FDA+budget violations count |

---

## Files Modified

| File | Changes |
|---|---|
| `server/simulator/power_calculator.py` | C1: Fix NCP formula |
| `server/episode_manager.py` | C2: shaping → r_info_gain, M4: initial_budget from randomized, M6: done guard, M8: step counter, M13: timeout penalty |
| `server/reward/reward_computer.py` | C3: terminal failure -1.0, C4: NaN guards, M1: novelty checks [:-1], M2: milestone first-time only |
| `server/rules/fda_rules.py` | M3: removed protocol_submitted prereq from SUBMIT_TO_FDA_REVIEW |
| `server/simulator/transition_engine.py` | M5: additive soft-violation degradation, M14: AEs scale with enrollment |
| `server/judge.py` | M7: overconfidence scoped to FDA+budget violations only |
| `tests/test_integration.py` | Updated 2 tests for M1, M3 |
| `tests/test_judge.py` | Updated 1 test for M7 |
| `train_colab_v3.ipynb` | CONTINUATION_STEPS 4→14, MAX_COMPLETION_LEN 256→512, NUM_GENERATIONS 8→4 |

---

## Verification

- **249/249 tests pass** after all changes
- Power check: `calculate_power(0.5, 100, 0.05)` = 0.9424 ✅
- All fix categories verified through existing test coverage + manual checks
