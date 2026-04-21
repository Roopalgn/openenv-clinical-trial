# Dashboard Metrics Table Format

> **Inspired by:** Bio Experiment's `dashboard.html` with live metric tables and trajectory visualizations. KubeSRE's reward curve plots with trend annotations. VRAM's training dashboard showing loss, reward, and capability radar in real time. The dashboard is both a dev tool (debugging training) and a pitch asset (3-minute demo).

## Purpose

The dashboard (`dashboard.html` + `dashboard.py` backend) displays five panels:

1. **Episode Replay** — step-by-step walkthrough of a selected episode
2. **Reward Curves** — training progress over time
3. **Curriculum Progression** — tier advancement timeline
4. **Scenario Breakdown** — per-scenario success rates and weak spots
5. **Agent Capability Radar** — multi-axis skill profile

This document defines the **data format** for each panel so that Suyash (backend) and Roopal (frontend) can build independently.

---

## Panel 1: Episode Replay

### Data Source

Episode transcript JSONL (see `milestone_map.md` reset behavior and reward_spec.md logging).

### Table Format

```json
{
    "panel": "episode_replay",
    "episode_id": 142,
    "scenario_id": "solid_tumor_chemo",
    "tier": 2,
    "tier_name": "intermediate",
    "total_reward": 7.83,
    "success": true,
    "steps": [
        {
            "step": 1,
            "action_type": "run_dose_escalation",
            "action_params": {"dose_mg": 50, "cohort_size": 6},
            "phase_detected": "phase_i_design",
            "reward_breakdown": {
                "r_validity": 0.3,
                "r_ordering": 0.2,
                "r_info_gain": 0.35,
                "r_efficiency": 0.1,
                "r_novelty": 0.1,
                "r_penalty": 0.0,
                "r_shaping": 0.17
            },
            "step_reward": 1.22,
            "milestones_completed": 2,
            "milestones_new": ["scenario_reviewed", "dose_escalation_started"],
            "budget_remaining": 2410000,
            "observation_summary": "Cohort of 6 patients treated at 50mg. No DLTs observed. Mild GI effects in 1 patient."
        },
        {
            "step": 2,
            "action_type": "run_dose_escalation",
            "action_params": {"dose_mg": 100, "cohort_size": 6},
            "phase_detected": "phase_i_design",
            "reward_breakdown": { "...": "..." },
            "step_reward": 1.05,
            "milestones_completed": 2,
            "milestones_new": [],
            "budget_remaining": 2320000,
            "observation_summary": "..."
        }
    ],
    "terminal_reward_breakdown": {
        "r_terminal_success": 6.2,
        "r_terminal_calibration": 3.0,
        "r_terminal_power": 2.0,
        "r_terminal_fda": 2.0,
        "r_terminal_budget": 1.0,
        "r_terminal_futility": 0.0,
        "r_terminal_overconf": 0.0
    },
    "agent_conclusion": {
        "effect_estimate": 0.55,
        "responder_population": "EGFR+",
        "mechanism": "EGFR tyrosine kinase inhibition",
        "sample_size": 120,
        "power": 0.92,
        "p_value": 0.003
    }
}
```

### Display

- Step-by-step table with expandable rows
- Phase-colored timeline bar (color-coded by detected phase)
- Milestone progress indicator (filled dots for completed milestones)
- Reward waterfall chart showing per-step reward components

---

## Panel 2: Reward Curves

### Data Source

Reward CSV logged by `train.py` per-episode.

### CSV Schema

```csv
episode,tier,scenario_id,total_reward,r_validity,r_ordering,r_info_gain,r_efficiency,r_novelty,r_penalty,r_shaping,r_terminal_success,r_terminal_calibration,r_terminal_power,r_terminal_fda,r_terminal_budget,r_terminal_futility,r_terminal_overconf,success,steps,milestones,timestamp
1,0,solid_tumor_chemo,-1.82,0.45,0.10,-0.05,0.02,0.30,-0.45,-0.19,-1.0,0.0,0.0,-1.0,1.0,0.0,0.0,false,100,3,2026-04-25T10:01:00Z
2,0,autoimmune_biologic,2.15,1.20,1.80,0.95,0.08,0.50,-0.30,0.42,5.0,1.5,1.5,1.0,1.0,0.0,-0.5,true,22,15,2026-04-25T10:01:30Z
```

### Metrics Table (Aggregated per Window)

```json
{
    "panel": "reward_curves",
    "window_size": 20,
    "data_points": [
        {
            "window_start": 1,
            "window_end": 20,
            "avg_reward": -0.85,
            "avg_reward_trend_slope": 0.12,
            "success_rate": 0.15,
            "component_avgs": {
                "r_validity": 0.52,
                "r_ordering": 0.35,
                "r_info_gain": 0.18,
                "r_efficiency": 0.06,
                "r_novelty": 0.25,
                "r_penalty": -0.38,
                "r_shaping": 0.10
            },
            "tier_distribution": {"warmup": 20}
        },
        {
            "window_start": 21,
            "window_end": 40,
            "avg_reward": 1.25,
            "avg_reward_trend_slope": 0.18,
            "success_rate": 0.30,
            "component_avgs": { "..." : "..." },
            "tier_distribution": {"warmup": 12, "beginner": 8}
        }
    ],
    "trend_line": {
        "slope": 0.15,
        "r_squared": 0.72,
        "p_value": 0.001
    },
    "best_episode": {"episode": 187, "reward": 13.5},
    "final_rolling_avg": 8.2,
    "initial_rolling_avg": -1.1
}
```

### Charts

1. **Total reward scatter** + rolling average line (window=20)
2. **Component trend lines** — separate line per reward component (stacked area optional)
3. **Success rate bar** — windowed success rate with tier annotations
4. **Best/Mean/Final** stats annotation in top-right corner

---

## Panel 3: Curriculum Progression

### Data Source

Curriculum log from `CurriculumController` (see `curriculum_policy.md`).

### Metrics Table

```json
{
    "panel": "curriculum_progression",
    "tier_history": [
        {"episode": 1,   "tier": 0, "tier_name": "warmup"},
        {"episode": 35,  "tier": 1, "tier_name": "beginner",     "trigger": "all_scenarios_60pct"},
        {"episode": 78,  "tier": 2, "tier_name": "intermediate",  "trigger": "all_scenarios_55pct"},
        {"episode": 165, "tier": 3, "tier_name": "advanced",      "trigger": "all_scenarios_45pct"},
        {"episode": 380, "tier": 4, "tier_name": "expert",        "trigger": "all_scenarios_35pct"}
    ],
    "current_tier": 4,
    "episodes_in_tier": 42,
    "per_scenario_mastery": {
        "solid_tumor_chemo": {"success_rate": 0.48, "episodes_in_window": 25},
        "autoimmune_biologic": {"success_rate": 0.36, "episodes_in_window": 25},
        "cns_depression": {"success_rate": 0.32, "episodes_in_window": 25},
        "rare_disease_orphan": {"success_rate": 0.28, "episodes_in_window": 25}
    },
    "weak_spot": "rare_disease_orphan",
    "fast_track_eligible": false
}
```

### Charts

1. **Tier timeline** — horizontal bar showing episodes spent at each tier (color-coded)
2. **Per-scenario success heatmap** — rows=scenarios, columns=episode windows, cells=success rate (green→red)
3. **Advancement event markers** — vertical lines on the reward chart marking tier transitions

---

## Panel 4: Scenario Breakdown

### Data Source

Per-scenario aggregation from reward CSV + curriculum log.

### Metrics Table

```json
{
    "panel": "scenario_breakdown",
    "scenarios": [
        {
            "scenario_id": "solid_tumor_chemo",
            "disease": "NSCLC",
            "difficulty": "Easy→Expert",
            "episodes_total": 120,
            "success_rate_overall": 0.52,
            "success_rate_by_tier": {
                "warmup": 0.78,
                "beginner": 0.55,
                "intermediate": 0.42,
                "advanced": 0.28,
                "expert": 0.15
            },
            "avg_reward_by_tier": {
                "warmup": 6.2,
                "beginner": 4.1,
                "intermediate": 2.3,
                "advanced": 0.8,
                "expert": -0.2
            },
            "common_failure_modes": [
                "missed_EGFR_subgroup",
                "underpowered_sample_size",
                "wrong_dose_selected"
            ],
            "biomarker_usage_rate": 0.65,
            "avg_milestones": 14.2
        },
        { "...": "..." }
    ]
}
```

### Charts

1. **Success rate grouped bar chart** — scenarios × tiers
2. **Failure mode pie chart** per scenario (from terminal reward breakdown)
3. **Selection frequency** — how often the weak-spot targeting selects each scenario

---

## Panel 5: Agent Capability Radar

> **Inspired by:** VRAM's capability radar chart with 6+ axes showing multi-dimensional agent skill.

### Radar Axes (6 Dimensions)

```json
{
    "panel": "capability_radar",
    "axes": [
        {
            "name": "Phase Compliance",
            "description": "Correct clinical workflow ordering",
            "metric": "phase_compliance_rate",
            "value": 0.88,
            "max": 1.0,
            "source": "r_ordering average (normalized)"
        },
        {
            "name": "FDA Compliance",
            "description": "Rule engine pass rate",
            "metric": "fda_pass_rate",
            "value": 0.82,
            "max": 1.0,
            "source": "r_terminal_fda (normalized)"
        },
        {
            "name": "Information Gathering",
            "description": "Quality of Phase I experiments",
            "metric": "info_gain_score",
            "value": 0.71,
            "max": 1.0,
            "source": "r_info_gain average (normalized)"
        },
        {
            "name": "Calibration",
            "description": "Accuracy of claims vs ground truth",
            "metric": "calibration_score",
            "value": 0.64,
            "max": 1.0,
            "source": "r_terminal_calibration / 5.0"
        },
        {
            "name": "Efficiency",
            "description": "Budget and time management",
            "metric": "efficiency_score",
            "value": 0.73,
            "max": 1.0,
            "source": "r_efficiency + r_terminal_budget (normalized)"
        },
        {
            "name": "Adaptability",
            "description": "Use of adaptive strategies (biomarkers, futility, amendments)",
            "metric": "adaptability_score",
            "value": 0.55,
            "max": 1.0,
            "source": "biomarker_usage + futility_detection + amendment_recovery (normalized)"
        }
    ],
    "comparison": {
        "random": [0.12, 0.08, 0.05, 0.02, 0.15, 0.00],
        "scripted": [0.95, 0.80, 0.40, 0.20, 0.60, 0.00],
        "trained": [0.88, 0.82, 0.71, 0.64, 0.73, 0.55]
    }
}
```

### Display

- Radar/spider chart with 3 overlaid polygons (random, scripted, trained)
- Color coding: random=red, scripted=orange, trained=green
- This is the **headline visual** for the pitch — shows RL agent surpasses scripted on 4/6 axes, especially calibration and adaptability (which scripted can never do)

---

## Data Flow Summary

```
train.py
  ├─► reward CSV  ────────────► Panel 2: Reward Curves
  ├─► curriculum log ──────────► Panel 3: Curriculum Progression  
  └─► episode JSONL ───────────► Panel 1: Episode Replay

eval_compare.py
  ├─► per-scenario JSON ───────► Panel 4: Scenario Breakdown
  └─► capability metrics ──────► Panel 5: Capability Radar

dashboard.py (backend)
  ├─► reads all above files
  ├─► serves dashboard.html via /dashboard
  └─► streams live updates via SSE during training
```

---

## File Format Requirements for Backend

All data files use UTF-8 encoding:

| File | Format | Location | Writer |
|------|--------|----------|--------|
| Reward CSV | CSV with header row | `results/rewards.csv` | `train.py` |
| Curriculum log | JSONL (one JSON object per line) | `results/curriculum.jsonl` | `CurriculumController` |
| Episode transcripts | JSONL (one JSON object per episode) | `results/transcripts.jsonl` | Episode Logger |
| Evaluation results | JSON (single object) | `results/<policy>_<tier>.json` | `eval_compare.py` |
| Comparison report | Markdown table | `results/comparison_report.md` | `eval_compare.py` |

All timestamps in ISO 8601 format. All floating-point values to 4 decimal places.
