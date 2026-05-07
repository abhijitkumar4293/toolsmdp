---
name: plan
description: Review project plan and results to recommend next steps. Use when planning work, checking progress, validating completed work, or understanding the project goal. The plan derives from the paper draft.
---

# Project Planner

Review the ToolSMDP project status and help plan next steps.

The research plan derives from the paper draft at `paper_draft/toolsmdp/main.tex`.
The implementation plan is tracked in `plan.md` and results in `results.md`.

## Current Plan

!`cat plan.md`

## Current Results

!`cat results.md`

## Analysis Instructions

Based on the plan and results above:

1. **Identify current milestone and step** — what is the active work?
2. **Check blockers** — are there incomplete prerequisites, running jobs, or failed validations?
3. **Review recent results** — what changed since the last completed step?
4. **Recommend next action** — what should be done next, and why?

## Decision Framework

- **Before moving to next step:** Check that the current step's deliverables are all checked off
- **If results don't match expectations:** Flag for investigation BEFORE moving forward
- **If a job is running:** Check status with `/aml`, report ETA, suggest what can be done in parallel
- **If multiple paths exist:** Recommend the one with clearest success criteria
- **For compute-heavy steps:** Verify AML environment has the right dependencies

## Output Format

Provide a concise assessment:
- **Current status**: milestone + step + what's happening
- **Blockers**: anything preventing progress
- **Recommended next step**: what to do and why
- **Success criteria**: how to verify it worked
