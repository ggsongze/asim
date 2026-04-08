#!/usr/bin/env python3
"""Analyze per-block setpoint outputs and resulting PMV for each mode.

Reads training phase_trace + block_reward_trace to check:
1. What setpoints each mode produced
2. Whether resulting PMV stayed within mode target range
3. Per-block mode comparison

Usage: .venv_qwen/bin/python analyze_block_pmv.py <training_dir>
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/home/AD/user/lab/asim")


def analyze(training_dir: str):
    training_path = Path(training_dir)

    # Load phase trace
    traces = []
    with open(training_path / "phase_trace.jsonl") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    # Mode PMV targets (new ranges)
    pmv_targets = {
        "cooling": (-0.5, 0.0),
        "balanced": (-0.1, 0.2),
        "energy_saving": (0.2, 0.5),
        # Legacy names
        "comfort": (-0.2, 0.1),
    }

    block_labels = ["06:30-08:30", "08:30-10:30", "10:30-12:30",
                    "12:30-14:30", "14:30-16:30", "16:30-19:00"]

    # Collect block candidate data
    candidates = [t for t in traces if t["phase"] == "block_candidate_done"]

    # Group by step
    by_step = defaultdict(list)
    for c in candidates:
        by_step[c["step_index"]].append(c)

    print(f"Training dir: {training_path}")
    print(f"Total steps: {len(by_step)}")
    print(f"Total candidates: {len(candidates)}")
    print()

    # Per-block, per-mode reward analysis
    print("=== Per-Block Mode Reward + PMV + Energy Summary ===")
    has_pmv = any("pmv_violation" in c for c in candidates)
    if has_pmv:
        print(f'{"Block":<14s} {"Mode":<15s} {"Count":>5s} {"Reward":>8s} {"PMV viol":>9s} {"HVAC kWh":>9s} {"NetGrid":>9s}')
    else:
        print(f'{"Block":<14s} {"Mode":<15s} {"Count":>5s} {"Mean":>8s} {"Zeros":>5s} {"Min":>8s} {"Max":>8s}')
    print("-" * 75)

    block_mode_rewards = defaultdict(lambda: defaultdict(list))
    block_mode_pmv = defaultdict(lambda: defaultdict(list))
    block_mode_hvac = defaultdict(lambda: defaultdict(list))
    block_mode_netgrid = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        block_mode_rewards[c["block_index"]][c["mode"]].append(c["relative_block_reward"])
        if "pmv_violation" in c:
            block_mode_pmv[c["block_index"]][c["mode"]].append(c["pmv_violation"])
        if "hvac_kwh" in c:
            block_mode_hvac[c["block_index"]][c["mode"]].append(c["hvac_kwh"])
        if "net_grid_kwh" in c:
            block_mode_netgrid[c["block_index"]][c["mode"]].append(c["net_grid_kwh"])

    for bi in sorted(block_mode_rewards.keys()):
        bl = block_labels[bi] if bi < len(block_labels) else f"block_{bi}"
        for mode in ["cooling", "comfort", "balanced", "energy_saving"]:
            if mode not in block_mode_rewards[bi]:
                continue
            vals = block_mode_rewards[bi][mode]
            mean = sum(vals) / len(vals)
            if has_pmv and mode in block_mode_pmv.get(bi, {}):
                pmv_vals = block_mode_pmv[bi][mode]
                hvac_vals = block_mode_hvac[bi].get(mode, [0])
                ng_vals = block_mode_netgrid[bi].get(mode, [0])
                avg_pmv = sum(pmv_vals) / len(pmv_vals) if pmv_vals else 0
                avg_hvac = sum(hvac_vals) / len(hvac_vals) if hvac_vals else 0
                avg_ng = sum(ng_vals) / len(ng_vals) if ng_vals else 0
                print(f'{bl:<14s} {mode:<15s} {len(vals):>5d} {mean:>+8.3f} {avg_pmv:>9.3f} {avg_hvac:>9.1f} {avg_ng:>+9.1f}')
            else:
                zeros = sum(1 for v in vals if abs(v) < 0.001)
                print(f'{bl:<14s} {mode:<15s} {len(vals):>5d} {mean:>+8.3f} {zeros:>5d} {min(vals):>+8.3f} {max(vals):>+8.3f}')
        print()

    # Per-step block detail
    print("=== Per-Step Block Winners ===")
    block_dones = [t for t in traces if t["phase"] == "block_done"]
    by_step_done = defaultdict(list)
    for d in block_dones:
        by_step_done[d["step_index"]].append(d)

    # Show last episode
    max_step = max(by_step_done.keys()) if by_step_done else 0
    last_ep_start = max(1, max_step - 14)

    print(f"\nLast episode (steps {last_ep_start}-{max_step}):")
    print(f'{"Step":>5s}', end='')
    for bi in range(6):
        bl = block_labels[bi] if bi < len(block_labels) else f"b{bi}"
        print(f' {bl:>14s}', end='')
    print()
    print("-" * 95)

    for step_idx in range(last_ep_start, max_step + 1):
        if step_idx not in by_step_done:
            continue
        blocks = sorted(by_step_done[step_idx], key=lambda x: x["block_index"])
        print(f'{step_idx:>5d}', end='')
        for b in blocks:
            mode_short = {"cooling": "C", "comfort": "C", "balanced": "B", "energy_saving": "E"}.get(b["winner_mode"], "?")
            r = b["winner_reward"]
            zero_flag = "·" if abs(r) < 0.001 else ""
            print(f' {mode_short}{r:>+7.2f}{zero_flag:>5s}', end='')
        print()

    # Summary: how many zero-reward blocks
    total_blocks = len(block_dones)
    zero_blocks = sum(1 for d in block_dones if abs(d["winner_reward"]) < 0.001)
    print(f"\nZero-reward blocks: {zero_blocks}/{total_blocks} ({zero_blocks/total_blocks*100:.0f}%)")

    # Per-block zero rate
    print("\nPer-block zero rate:")
    for bi in range(6):
        bl = block_labels[bi] if bi < len(block_labels) else f"b{bi}"
        bi_blocks = [d for d in block_dones if d["block_index"] == bi]
        bi_zeros = sum(1 for d in bi_blocks if abs(d["winner_reward"]) < 0.001)
        print(f"  {bl}: {bi_zeros}/{len(bi_blocks)} ({bi_zeros/len(bi_blocks)*100:.0f}%)" if bi_blocks else f"  {bl}: no data")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to latest training dir
        dirs = sorted(PROJECT_ROOT.glob("result/gspo/miami_grpo_*"), key=lambda p: p.stat().st_mtime)
        if dirs:
            analyze(str(dirs[-1]))
        else:
            print("Usage: python analyze_block_pmv.py <training_dir>")
    else:
        analyze(sys.argv[1])
