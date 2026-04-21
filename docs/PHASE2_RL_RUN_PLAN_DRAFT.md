# Phase 2 RL Run Plan Draft

Status: draft for review only. This is a planning document, not an instruction to start training immediately.

## 1. Goal

The goal of the first RL phase is narrow:

- start from the current confirmed Phase 2 mainline
- keep the executable `slot_turn_correction v3` interface frozen
- optimize the **forecast adapter behavior** under that frozen interface
- judge success only by downstream forecast metrics, not by standalone diagnostic proxy scores

This plan is for the first RL entry only. It is deliberately conservative.

## 2. Current Decision Boundary

Current confirmed mainline:

- diagnostic branch: `runs/phase2_diagnostic_slot_turn_correction_v3_20260421_010206/sft/final_adapter`
- confirm run root: `runs/phase2_diagnostic_slot_turn_correction_v3_confirm_20260421_090002`
- forecast-side confirmed variant:
  - `predicted_slot_turn_track_plus_baseline_intensity_scale_1p20_v3`
  - held-out `track_error_km = 145.58`
  - held-out `mean_track_diff_vs_official_km = 125.26`

Retained stability controls:

- `predicted_slot_turn_track_plus_baseline_intensity_scale_1p20_v1`
- `predicted_slot_correction v1 + baseline intensity + scale 1.20`
- `baseline_forecast_grpo_physical_v2`
- `baseline_forecast_grpo_reward_v2`

Post-confirm semantic audit verdict:

- current confirmed gain is mainly driven by `lon semantics`
- `lat-only targeted SFT cleanup` is **not** the next mandatory blocker
- we can move to RL preparation

## 3. Core Technical Decision

The current confirmed mainline is **not** a single adapter.

It is a composed system:

- frozen diagnostic adapter `slot_turn_correction v3`
- deterministic prompt override / renderer path
- forecast adapter currently backed by `baseline_forecast_sft_v2`

Therefore, the first RL run should train:

- **forecast adapter only**

And it should keep frozen:

- diagnostic adapter `v3`
- prompt override schema
- slot-locked deterministic renderer
- baseline intensity integration path

This means the correct RL target is:

- **integrated forecast behavior under frozen `v3` prompt injection**

not:

- raw diagnostic JSON generation
- free-form end-to-end mixed diagnostic + forecast generation

## 4. Frozen Assets For RL-v0

Base forecast SFT init:

- `runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter`

Frozen diagnostic adapter:

- `runs/phase2_diagnostic_slot_turn_correction_v3_20260421_010206/sft/final_adapter`

Historical retained control:

- `runs/phase2_diagnostic_slot_turn_correction_v1_20260420_095215/sft/final_adapter`

Existing GRPO controls:

- physical-error-led:
  - `runs/phase1_baseline_v2_formal_20260415_013403/grpo/adapter_step-000050`
- reward-led:
  - `runs/phase1_baseline_v2_formal_20260415_013403/grpo/adapter_reward-0.7544_step-000003`

Data root:

- `data/training_rebuilt_v2_20260414_guidancefix`

Base RL splits:

- `data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/rl_train.jsonl`
- `data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/rl_val.jsonl`
- `data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/rl_test.jsonl`

## 5. Required Prep Before First RL Run

These are required prep items. The plan assumes they will be completed before training starts.

### 5.1 Materialize Frozen-Override RL Splits

Current GRPO data loading does **not** support online `prompt_overrides`.

`cyclone_training.datasets.load_grpo_datasets()` reads `messages` directly from RL JSONL and has no injection hook.

So the first RL run should use offline-materialized RL datasets with the `v3` override already inserted into the user prompt.

Recommended derived view:

- `data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only_slot_turn_v3_rl/rl_train.jsonl`
- `data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only_slot_turn_v3_rl/rl_val.jsonl`
- `data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only_slot_turn_v3_rl/rl_test.jsonl`

Optional retained control view:

- `data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only_slot_turn_v1_rl/...`

Recommended implementation shape:

1. build `sample_id -> overridden_prompt` using the frozen diagnostic adapter
2. rewrite the RL JSONL `messages` field offline
3. preserve original `verification`
4. add provenance fields to `train_metadata`, such as:
   - `diagnostic_override_variant = slot_turn_correction_v3`
   - `diagnostic_override_source_run`
   - `diagnostic_override_mode = predicted`

### 5.2 Extend RL Verification To Support Dual-Track Reward

Current GRPO reward implementation in [cyclone_training/rewards.py](/root/Cyclone_next/cyclone_training/rewards.py:432) only scores against truth-side track and intensity.

It does **not** directly score:

- `mean_track_diff_vs_official_km`
- `mean_intensity_diff_vs_official_kt`

And current RL verification only stores forecast slot times:

- `verification.forecast_slots = [{valid_day, valid_hhmmz}, ...]`

It does not yet contain official forecast lat/lon/vmax per slot.

Therefore, the preferred RL-v0 plan requires a small dataset/reward extension:

Add a new verification field, for example:

- `verification.official_forecast_slots`

Each entry should include:

- `valid_time_utc`
- `valid_day`
- `valid_hhmmz`
- `lat`
- `lon`
- `vmax_kt`

Then extend the reward function to expose a dual-track objective:

- truth-side track accuracy term
- official-gap closing term

### 5.3 Freeze RL Objective And Guardrails

Preferred first-run RL objective:

- main objective:
  - truth-side `track_error_km`
  - official-gap `mean_track_diff_vs_official_km`

Guardrails:

- `intensity_error_kt`
- `mean_intensity_diff_vs_official_kt`
- `strict_parseable_rate`
- `slot_time_match_rate_vs_official`

Explicitly **not** acceptable as primary reward:

- `reward_mean` by itself
- `coverage` by itself
- standalone diagnostic `macro-F1`

### 5.4 Add One Dedicated Check Script

Before first launch, add one dedicated preflight validator for the RL run.

Planned file:

- `scripts/check_phase2_rl_slot_turn_v3.py`

It should verify at least:

- dataset root is the frozen rebuilt root
- RL train/eval files point to the materialized `slot_turn_v3` override view
- GRPO config uses `baseline_forecast_sft_v2` as `adapter_init_path`
- reference adapter matches the intended init path
- reward config is the new Phase 2 dual-track reward config
- prompt override provenance is frozen and points to `slot_turn_correction v3`

## 6. RL-v0 Training Object

Trainable object:

- one forecast adapter

Initialization:

- `adapter_init_path = runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter`

Reference model:

- same adapter path for the first run

Reason:

- current confirmed mainline is a composed system, not a single SFT checkpoint
- the trainable part is still the forecast adapter
- the `v3` diagnostic branch should be treated as frozen environment context, not as RL train target

## 7. Draft Config / Script Set

Planned config files:

- `configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase2_slot_turn_v3_dualtrack_smoke.yaml`
- `configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase2_slot_turn_v3_dualtrack_pilot.yaml`
- `configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase2_slot_turn_v3_dualtrack_formal.yaml`

Planned helper scripts:

- `scripts/check_phase2_rl_slot_turn_v3.py`
- `scripts/run_phase2_rl_slot_turn_v3_smoke.sh`
- `scripts/run_phase2_rl_slot_turn_v3_background.sh`
- optional:
  - `scripts/run_phase2_rl_slot_turn_v3_pilot.sh`
  - `scripts/run_phase2_rl_slot_turn_v3_formal.sh`

## 8. Recommended Training Schedule

### 8.1 Smoke

Purpose:

- verify data format
- verify reward function shape
- verify training launch and checkpoint save path

Draft settings:

- `max_train_samples = 256`
- `max_steps = 8`
- `num_generations = 2`
- `max_prompt_length = 896`
- `max_completion_length = 256`
- `reward_save_threshold = 0.30`
- `stop_after_reward_checkpoint = true`

Pass conditions:

- training runs cleanly
- reward is non-degenerate
- formatted outputs stay parseable
- no evidence of slot-time collapse

### 8.2 Pilot

Purpose:

- test whether RL helps at all before formal compute
- choose whether to keep the dual-track reward weights or rebalance them

Draft settings:

- full `rl_train`
- `max_steps = 25` to `40`
- `save_steps = 5`
- `num_generations = 4`
- keep learning rate in the same order as existing GRPO baseline: `2e-6`

Checkpoint evaluation:

- evaluate every saved adapter on the frozen `slot_turn_v3` overridden held-out split
- compare against:
  - SFT init under frozen `v3` override
  - `v1` retained stability control
  - existing GRPO reward-led / physical-error-led controls

Pilot go condition:

- at least one checkpoint beats the current `v3` no-RL control on one primary track metric
- and does not regress the other primary track metric materially

### 8.3 Formal

Purpose:

- run one single formal RL confirmation pass after pilot settings are frozen

Draft settings:

- full `rl_train`
- `max_steps = 100`
- `save_steps = 10`
- `num_generations = 8`
- `generation_batch_size = 8`
- same LoRA shape and optimizer family as the current reward-led baseline unless pilot clearly justifies deviation

Formal evaluation:

- same held-out sample policy as the current confirmatory compare
- primary report should be at least:
  - `track_error_km`
  - `mean_track_diff_vs_official_km`
  - `intensity_error_kt`
  - `mean_intensity_diff_vs_official_kt`
  - `strict_parseable_rate`
  - `slot_time_match_rate_vs_official`

## 9. Evaluation Protocol

Main benchmark:

- evaluate the RL adapter on the **materialized `slot_turn_v3` overridden held-out RL test split**

Primary comparison target:

- frozen no-RL current mainline behavior under the same `v3` override view

Recommended report structure:

1. RL candidate vs frozen no-RL `v3` control
2. RL candidate vs `v1` retained control
3. RL candidate vs historical GRPO controls
4. RL candidate vs expert official upper bound

Important:

- do not compare an RL adapter evaluated on overridden prompts against a control evaluated on the original non-overridden prompt distribution and call that a clean result
- keep the prompt regime matched within each comparison block

## 10. Draft Go / No-Go Criteria

### 10.1 Smoke

Go if:

- launch works
- reward is numerically stable
- outputs remain parseable

No-go if:

- malformed completions dominate
- slot-time alignment breaks
- reward saturates without meaningful forecast quality

### 10.2 Pilot

Go if at least one checkpoint:

- improves `track_error_km` by `>= 1 km` relative to frozen `v3` no-RL control
- and does not worsen `mean_track_diff_vs_official_km` by more than `1 km`
- and does not worsen `intensity_error_kt` by more than `0.3 kt`
- and keeps `strict_parseable_rate >= 0.995`
- and keeps `slot_time_match_rate_vs_official = 1.0000`

Stop if:

- reward rises but both primary track metrics stay flat or regress
- the best checkpoint only improves secondary metrics
- gains appear only through format artifacts or slot-time drift

### 10.3 Formal

Declare RL-v0 positive only if held-out full confirm:

- is non-regressive on both primary track metrics
- improves at least one primary track metric by a practically meaningful margin
- preserves parseability and slot-time alignment
- does not materially worsen intensity

Suggested first-pass practical threshold:

- one of:
  - `track_error_km` improves by `>= 2 km`
  - `mean_track_diff_vs_official_km` improves by `>= 2 km`
- and the other primary metric does not regress by more than `1 km`

## 11. Preferred Reward Strategy

Preferred default:

- **dual-track reward**

Why:

- current project gate is dual-track
- truth-only reward can improve physical skill while missing expert-gap closing
- official-gap closing is part of what current Phase 2 is explicitly trying to recover

Fallback only if prep time is limited:

- truth-side reward as RL-v0
- but keep dual-track offline evaluation as the hard gate

This fallback is acceptable as a contingency plan, not as the preferred default.

## 12. Controls

Retained controls to keep in all reporting:

- frozen no-RL `v3` mainline
- frozen no-RL `v1` stability control
- `baseline_forecast_grpo_physical_v2`
- `baseline_forecast_grpo_reward_v2`

Optional ablation:

- same RL plan but with truth-only reward

Purpose:

- isolate whether any gain comes from:
  - the frozen `v3` interface itself
  - RL in general
  - the new dual-track reward specifically

## 13. Main Risks

1. The reward implementation is not yet dual-track.

Current code must be extended before the preferred RL run can start.

2. The current mainline is composed, not monolithic.

If this fact is ignored, we may accidentally train the wrong object.

3. `v3` gain is mainly driven by `lon semantics`.

If reward shaping is too loose, RL may amplify that shortcut without actually improving the weakest semantic axis.

4. The prompt regime must stay matched.

Offline-materialized override views are required for a clean first run.

## 14. Recommended Next Action

Before any RL preparation script is written, freeze these four decisions:

1. Use frozen `slot_turn_correction v3` prompt-injected RL views.
2. Train forecast adapter only.
3. Implement dual-track reward by extending RL verification with official forecast slots.
4. Treat `v1` as retained stability control, not as the training init.

Once those four decisions are accepted, the next concrete implementation step is:

- write the smoke config
- write the preflight checker
- materialize the `slot_turn_v3` RL train/val/test views

Only after that should we start RL prep scripts or launch training.
