#!/usr/bin/env python3
"""
Check HF Space endpoints: /ping, /reset, /step, /state

Usage:
    python scripts/check_hf_space.py [--base-url URL]

Exits 0 if all endpoints respond correctly, non-zero otherwise.
"""

import argparse
import json
import sys

import requests

BASE_URL = "https://roopalgn-openenv-clinical-trial.hf.space"

SAMPLE_ACTION = {
    "action_type": "set_primary_endpoint",
    "parameters": {"endpoint": "overall_survival", "timepoint_months": 12},
    "justification": "Overall survival is the gold standard primary endpoint for oncology trials.",
    "confidence": 0.85,
}


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return ok


def run_checks(base_url: str) -> bool:
    results = []

    # ------------------------------------------------------------------
    # /ping  (GET)
    # ------------------------------------------------------------------
    try:
        r = requests.get(f"{base_url}/ping", timeout=15)
        ok = r.status_code == 200 and r.json().get("status") == "ok"
        results.append(check("/ping", ok, f"status={r.status_code} body={r.text[:120]}"))
    except Exception as exc:
        results.append(check("/ping", False, str(exc)))

    # ------------------------------------------------------------------
    # /reset  (POST)
    # ------------------------------------------------------------------
    try:
        r = requests.post(f"{base_url}/reset", json={"seed": 42}, timeout=30)
        ok = r.status_code == 200
        if ok:
            body = r.json()
            # Expect a TrialObservation — must have these keys
            required = {"scenario_description", "available_actions", "steps_taken", "done"}
            missing = required - body.keys()
            ok = len(missing) == 0
            detail = f"status={r.status_code}" + (f" missing_keys={missing}" if missing else " keys OK")
        else:
            detail = f"status={r.status_code} body={r.text[:120]}"
        results.append(check("/reset", ok, detail))
    except Exception as exc:
        results.append(check("/reset", False, str(exc)))

    # ------------------------------------------------------------------
    # /step  (POST)  — requires a prior /reset to have an active episode
    # ------------------------------------------------------------------
    try:
        r = requests.post(f"{base_url}/step", json=SAMPLE_ACTION, timeout=30)
        ok = r.status_code == 200
        if ok:
            body = r.json()
            required = {"observation", "reward", "done", "info"}
            missing = required - body.keys()
            ok = len(missing) == 0
            detail = f"status={r.status_code}" + (f" missing_keys={missing}" if missing else " keys OK")
        else:
            detail = f"status={r.status_code} body={r.text[:120]}"
        results.append(check("/step", ok, detail))
    except Exception as exc:
        results.append(check("/step", False, str(exc)))

    # ------------------------------------------------------------------
    # /state  (GET)
    # ------------------------------------------------------------------
    try:
        r = requests.get(f"{base_url}/state", timeout=15)
        ok = r.status_code == 200
        if ok:
            body = r.json()
            required = {"episode_id", "step_count", "difficulty", "scenario_id"}
            missing = required - body.keys()
            ok = len(missing) == 0
            detail = f"status={r.status_code}" + (f" missing_keys={missing}" if missing else " keys OK")
        else:
            detail = f"status={r.status_code} body={r.text[:120]}"
        results.append(check("/state", ok, detail))
    except Exception as exc:
        results.append(check("/state", False, str(exc)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Result: {passed}/{total} endpoints passed")
    return passed == total


def main() -> None:
    parser = argparse.ArgumentParser(description="Check HF Space endpoints")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL of the HF Space")
    args = parser.parse_args()

    print(f"Checking HF Space at: {args.base_url}\n")
    all_passed = run_checks(args.base_url)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
