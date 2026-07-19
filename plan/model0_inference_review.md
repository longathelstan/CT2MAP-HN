# CT2MAP-HN — Model 0 Baseline Review: Root Cause of the Uniform Heatmap

**Reviewer:** Principal AI Research Engineer (internal)
**Scope:** Stage 1 / Model 0 baseline only. No Stage 2+ recommendations.
**Date:** 2026-07-19
**Method:** Full read of docs (source of truth) → source code → empirical checkpoint-load test.

---

## Executive summary

The clinically-unsatisfactory heatmap (uniform ~0.57 flooding the entire volume
including air, checkerboard/edge texture — see `model_quality_analysis.md`,
CHUM-012) is **not** primarily an under-training or loss problem. It is an
**inference-time reconstruction bug**, empirically proven:

> Loading the real trained checkpoint `checkpoints/baseline_p40/best_model.pth`
> (epoch 184, val_loss 0.299) into the model that `infer_case._build_model()`
> constructs matches **0 of 102 parameters**. Inference runs on a
> **randomly-initialized network.**

The documentation's leading hypothesis (HU preprocessing mismatch) is real but
**secondary** — it cannot be the primary cause because *no* trained weights are
in the network at inference time. Four compounding defects are documented below,
ordered by severity. Fixing C1 and C2 is necessary and likely sufficient to
restore a non-uniform, lesion-focused heatmap; H1 and H2 must also be fixed
before the baseline is a trustworthy benchmark.

| ID | Severity | One-line |
|----|----------|----------|
| **C1** | Critical | Inference rebuilds the wrong architecture → 0/102 trained weights load (random net). |
| **C2** | Critical | Double sigmoid on the heatmap path → output structurally floored into [0.5, 0.73]. |
| **H1** | High | Train crops a tight ROI around the GT lesion mask; inference runs on the full uncropped volume → severe distribution shift + background flooding. |
| **H2** | High | Inference HU-normalization default (−1024/1024) diverges from training (−200/300) when the config lacks a `preprocessing` block. |
| **M1** | Medium | The `UNIFORM_OUTPUT_FAILURE` guardrail exists but is never called during inference. |
| **M2** | Medium | Heads hardcode `BatchNorm3d` while the backbone uses `InstanceNorm` (contradicts the small-batch decision in `plan/task.md`). |

---

## C1 — Inference reconstructs the wrong architecture; zero trained weights load

### Problem
The trained checkpoint is a `BaselineUNet` (a MONAI `BasicUNet` backbone wrapped
under `self.backbone`, plus `heatmap_head`, `lesion_head`, `triage_head`).
`CaseInferencer._build_model()` has no branch that builds `BaselineUNet`; its
registry is only `{swin_unetr, basic_unet, unet}`. For a `basicunet` config it
constructs a **bare** MONAI `BasicUNet`. The parameter names therefore differ by
the `backbone.` prefix (and the heads are absent), so `load_state_dict(..., strict=False)`
loads nothing and silently succeeds.

### Root Cause
Two independent code paths build the model — `src/models/build_model()` for
training vs. `CaseInferencer._build_model()` for inference — and they have
drifted. `strict=False` (chosen to tolerate the optional uncertainty head) masks
a total key mismatch instead of surfacing it.

### Evidence
**From code:**
- `src/models/baseline_unet.py`: backbone assigned to `self.backbone = BasicUNet(...)`; heads `self.heatmap_head/lesion_head/triage_head`. → checkpoint keys are `backbone.*`, `heatmap_head.*`, `lesion_head.*`, `triage_head.*`.
- `src/inference/infer_case.py` `_build_model()` (lines ~147–190): branches only for `swinunetr | basicunet | unet`. The `basicunet` branch builds `BasicUNet(... out_channels=1, features=(32,32,64,128,256,32))` with **no `backbone.` prefix and no heads**.
- `_load_model()`: `model.load_state_dict(state_dict, strict=False)` — mismatch swallowed silently.

**Empirical (this review, env `ct2map`, CPU):**
```
checkpoint keys: 102   prefixes: ['backbone','heatmap_head','lesion_head','triage_head']
  epoch: 184   best_val_loss: 0.2993
  stored model cfg: {'name':'baseline_unet','features':[32,64,128,256,512,32],'norm':'instance', ...}
inference model (bare BasicUNet): 82 params
KEYS THAT MATCH (would load): 0
  missing_keys: 82   unexpected_keys: 102
  inference key sample:  'conv_0.conv_0.adn.N.bias'
  checkpoint key sample: 'backbone.conv_0.conv_0.adn.N.bias'
```
The checkpoint even used `features=[32,64,128,256,512,32]`, so tensor shapes
would not match either — but the `backbone.` prefix alone already guarantees
0 matches.

**From documentation:** `model_quality_analysis.md` reports output "nearly
uniform ~0.5, unimodal not bimodal" with high-frequency texture. A random 3-D
conv net fed a normalized CT produces exactly this: spatially incoherent,
edge-like activations centered near the sigmoid's 0.5 midpoint.

**From ML theory:** An untrained conv stack is a fixed random filter bank; its
response to any input is high-frequency noise with no learned semantics. After
sigmoid it concentrates near 0.5. This is the signature in the ITK-SNAP screenshot.

**From medical imaging:** A metabolic risk map that lights up air and background
with ~0.5 has no anatomical grounding — consistent with a network that never
learned CT→risk, i.e. random weights.

### Impact
Fatal. Every inference output to date is from an untrained network; all
downstream numbers (triage 0.925, 9 candidates, 87%-volume component) are
artifacts of noise, not of the trained model. Training quality cannot be judged
until this is fixed. `best_val_loss=0.299` shows the *trained* model did learn
something — it is simply never loaded at inference.

### Validation Experiment
1. Reproduce the 0/102 match (done above) — confirms the mechanism.
2. Add a `baseline_unet` branch to `_build_model()` (below), reload the same
   checkpoint with `strict=True`, and assert `len(missing)==len(unexpected)==0`.
3. Re-run CHUM-012; confirm the heatmap is no longer uniform and that
   background/air voxels drop toward 0.

### Recommended Fix
In `src/inference/infer_case.py._build_model()`, add a branch that builds the
same wrapper used in training, and switch to `strict=True` (allowing only the
known-optional `uncertainty_head` to be missing):
```python
elif model_type_normalised in ("baselineunet", "baseline"):
    from src.models.baseline_unet import BaselineUNet
    model = BaselineUNet(
        in_channels=in_channels,
        features=tuple(model_cfg.get("features", (32,32,64,128,256,32))),
        dropout=model_cfg.get("dropout", 0.0),
        norm=model_cfg.get("norm", "instance"),
    )
```
The forward returns a dict; `infer()` must read `output["heatmap"]` (see C2).
Prefer reconstructing the model from the **checkpoint's own stored `config`**
(`ck["config"]["model"]`) rather than a separate YAML, so architecture can never
drift from the weights again. Load with `strict=True` and fail loudly on mismatch.

### Files to Modify
- `src/inference/infer_case.py` (`_build_model`, `_load_model`, `infer`/`infer_with_mc_dropout` output-key handling).

### Scientific Justification
A benchmark must be reproducible and the reported inputs must correspond to the
reported outputs. Silent `strict=False` loading violates both. Rebuilding from
the checkpoint's embedded config and asserting a full key match makes the
train→infer contract explicit and prevents this class of bug from recurring —
correctness before performance, as the review principles require.

---

## C2 — Double sigmoid on the heatmap path

### Problem
Even after C1 is fixed, the heatmap is squashed twice through a sigmoid:
`HeatmapHead` ends in `nn.Sigmoid()`, and `CaseInferencer._postprocess()` applies
`torch.sigmoid()` again to that already-[0,1] output. `sigmoid([0,1]) → [0.5, 0.731]`.

### Root Cause
`_postprocess` was written assuming the model emits **logits** (as a bare
`BasicUNet`/`SwinUNETR` would), but the training heads already apply the final
sigmoid. The two paths disagree about where the activation lives.

### Evidence
**From code:**
- `src/models/heads.py` L54/68 (`HeatmapHead`): `self.activation = nn.Sigmoid()`; `forward` returns `self.activation(self.conv_block(x))`.
- `src/inference/infer_case.py` L339 (`_postprocess`): `heatmap = torch.sigmoid(output).squeeze(...)`.

**From documentation:** `model_quality_analysis.md` cursor value 0.5712 and
"everything ~0.5" — the lower bound of `sigmoid([0,1])` is 0.5, and 0.57 sits
squarely inside `[0.5, 0.731]`. This is a near-exact quantitative match to a
double-sigmoid on a low-variance input.

**From ML theory:** Re-applying a saturating nonlinearity compresses dynamic
range and destroys contrast; a target heatmap with true zeros in background can
never be reproduced because the output floor is 0.5. This alone forces the
"floods everything" appearance and makes the fixed-0.5 threshold in postprocess
select the entire volume.

### Impact
Critical and independent of C1. With correct weights but this bug, the heatmap
would still be floored at 0.5 everywhere → `threshold_heatmap(0.5)` flags the
whole ROI → the 87%-volume "candidate" and the inflated triage score persist.

### Validation Experiment
On a single trained case, log `output.min()/max()` before postprocess. If already
in [0,1], remove the second sigmoid and confirm background voxels reach ~0 and
the histogram becomes bimodal (the plan's acceptance criterion).

### Recommended Fix
In `_postprocess`, do **not** re-apply sigmoid when the model already outputs
probabilities. Either drop the `torch.sigmoid` call and just `clip(0,1)`, or make
it conditional on a `model.outputs_logits` flag. Given the heads own the sigmoid,
the correct fix is to remove it from `_postprocess`.

### Files to Modify
- `src/inference/infer_case.py` (`_postprocess`).

### Scientific Justification
The activation must be applied exactly once, at a defined location. Fixing this
restores the model's true output distribution — a precondition for any
meaningful thresholding, connected-component, or triage computation.

---

## H1 — Train/inference ROI mismatch (crop defined by the GT lesion mask)

### Problem
Training data is cropped to a tight bounding box **around the ground-truth GTV
lesion mask** (+10-voxel margin). Inference performs **no crop** and runs
sliding-window over the entire volume (air, shoulders, table). The model is
therefore evaluated on inputs from a distribution it never saw in training, and
on regions (background/air) that were always excluded during training.

### Root Cause
The ROI definition uses information — the lesion segmentation — that exists only
at training time. It is not a reproducible, inference-available preprocessing
step, so the two pipelines cannot match by construction.

### Evidence
**From code:**
- `src/dataio/preprocess.py` `_crop_bbox`: `coords = np.argwhere(mask > 0)`; bbox from `coords.min/max` + margin. `crop_head_neck_roi` returns the **full volume** when `mask is None`.
- `scripts/prepare_data.py` (~L383): builds processed `.npy` with `crop.method='bbox'` using the mask → training tensors are tightly lesion-cropped.
- `src/inference/infer_case.py` `_preprocess`: resample → `np.clip` HU → min-max normalize → tensor. **No crop step.**

**From documentation:** `model_quality_analysis.md` — candidate #1 = 81.5M voxels
(87% of a 375×500×500 = 93.75M volume) and `elapsed 5021s`. Both are symptoms of
sliding-window over an enormous **uncropped** volume, not the ~192³ ROI the model
trained on.

**From ML theory:** Covariate shift — P(input at test) ≠ P(input at train). A
network trained only on lesion-centered crops has undefined behaviour on air and
shoulders; predictions there are unconstrained.

**From medical imaging:** HNSCC GTVs occupy a small, consistent anatomical region.
Training only on GT-centered crops teaches "given you are near a lesion, …", which
cannot generalize to "find the lesion in a whole-body-ish CT."

### Impact
High. Independently guarantees background flooding and the multi-thousand-second
runtimes, and biases the model toward always-positive behaviour (it only ever saw
lesion-containing patches at 0.7 positive-sampling ratio).

### Validation Experiment
Evaluate the (correctly-loaded, C1/C2-fixed) model on the **processed `.npy`**
validation tensors (same ROI crop as training) before touching raw NIfTI. If the
heatmap is lesion-focused on processed data but floods on raw full-volume data,
H1 is confirmed as the residual cause. (This is exactly the "processed `.npy` eval
before raw NIfTI" step already in `plan/task.md`.)

### Recommended Fix
Make inference cropping mask-free and reproducible. Options, in order of
preference for a baseline:
1. **Anatomical foreground crop** — threshold body from air (e.g. HU > −500),
   take the largest connected body component's bbox, and run inference within it.
   Requires no lesion mask and matches "crop to the patient, not the lesion."
2. If keeping fixed-size behaviour, use `crop_head_neck_roi(method="fixed_size")`
   centered on the body centroid, and — critically — **retrain** with the same
   mask-free crop so train and inference agree.

The key principle: the training crop must be reproducible at inference. A
GT-mask-defined crop is not, so either the training crop changes to a mask-free
rule (preferred; requires a retrain, which the plan already sanctions) or the
model must be trained to localize within a mask-free body ROI.

### Files to Modify
- `src/inference/infer_case.py` (`_preprocess` — add mask-free body crop).
- `src/dataio/preprocess.py` / `configs/preprocess.yaml` (align the training crop rule if switching to body-based cropping).

### Scientific Justification
Train/test consistency is the foundation of a valid benchmark. A crop that
depends on the label is a form of test-time leakage in disguise and makes the
reported metrics unattainable in the CT-only deployment the project targets.

---

## H2 — Inference HU normalization default diverges from training

### Problem
Training normalizes HU to `[-200, 300]` (`configs/preprocess.yaml` `hu_clip`,
and the checkpoint's stored `preprocessing: {hu_min:-200, hu_max:300}`).
`CaseInferencer._preprocess` reads `hu_min/hu_max` from `config["preprocessing"]`
with **defaults −1024 / 1024**. Any inference config lacking a `preprocessing`
block (e.g. `configs/demo.yaml`) silently uses the wrong window.

### Root Cause
Inference HU range is config-supplied with permissive defaults, not derived from
the checkpoint's stored preprocessing.

### Evidence
- **Code:** `infer_case.py._preprocess`: `hu_min = preproc_cfg.get("hu_min", -1024); hu_max = preproc_cfg.get("hu_max", 1024)`; then `(x - hu_min)/(hu_max - hu_min)`.
- **Code/config:** `configs/demo.yaml` has no `preprocessing:` block → defaults apply.
- **Checkpoint:** stores `preprocessing: {hu_min:-200, hu_max:300}` (shown in C1 dump).
- **Docs:** `model_quality_analysis.md` names this mismatch as the prime suspect.

### Impact
High but secondary to C1. With −1024/1024, soft-tissue contrast (the −200..300
band the model was trained on) is compressed into a narrow sub-range of [0,1],
so the input barely resembles training inputs. This compounds C1/H1; on its own
it degrades but would not fully uniformize a correctly-loaded model.

### Validation Experiment
After C1/C2, run one case twice: once with hu[−200,300], once with [−1024,1024];
compare heatmap contrast and background suppression. Expect materially better
lesion focus with the matched window.

### Recommended Fix
Read `hu_min/hu_max`, `target_spacing`, and the normalization method from the
**checkpoint's stored `config['preprocessing']`** at inference; fail loudly if
absent rather than defaulting. This guarantees train/infer parity for all cases.

### Files to Modify
- `src/inference/infer_case.py` (`_preprocess`, `_load_model` to expose stored preprocessing).

### Scientific Justification
Intensity normalization is part of the learned function's domain. Reproducibility
requires that the exact training normalization travel with the weights.

---

## M1 — The uniform-output guardrail is defined but never called

**Problem/Evidence:** `postprocess.quality_check_heatmap()` implements the
`UNIFORM_OUTPUT_FAILURE` flag (largest component > 25% of ROI) mandated by
`plan/task.md`, but `grep` shows it is never invoked in `infer_case.py`.
**Impact (Medium):** The exact failure the plan anticipated shipped undetected.
**Fix:** Call `quality_check_heatmap(heatmap, candidates)` inside `infer()` and
`infer_with_mc_dropout()`, attach flags to `summary.json`, and log a warning.
**Files:** `src/inference/infer_case.py`. **Justification:** A benchmark needs
automated failure detection so silent regressions cannot masquerade as results.

---

## M2 — Heads use BatchNorm while the backbone uses InstanceNorm

**Problem/Evidence:** `heads.py` hardcodes `nn.BatchNorm3d` in `HeatmapHead`/
`LesionHead`, while `baseline_unet.py` builds the backbone with `norm="instance"`
(and the checkpoint confirms `norm:'instance'`). `plan/task.md` explicitly chose
instance norm because BatchNorm is unstable at the small 3-D batch sizes used
(`baseline_p40.yaml batch_size=1`). **Impact (Medium):** BatchNorm on batch_size=1
tracks unreliable running statistics; at inference (eval mode) those statistics
are applied, which can distort head outputs — a real but lesser contributor once
C1/C2/H1/H2 are fixed. **Fix:** Make the head norm configurable and default it to
match the backbone (`InstanceNorm3d` or `GroupNorm`); retrain. **Files:**
`src/models/heads.py`, `src/models/baseline_unet.py`. **Justification:**
Consistency with the documented small-batch decision; removes a known source of
eval-time instability.

---

## Ordered remediation (baseline scope only)

1. **C1** — add `baseline_unet` to inference; rebuild from the checkpoint's
   stored config; load `strict=True`. *(code-only, no retrain)*
2. **C2** — remove the second sigmoid in `_postprocess`; read `output["heatmap"]`.
   *(code-only)*
3. Re-run CHUM-012 on the **processed `.npy`** ROI tensors (H1 validation path).
   If lesion-focused → the two critical bugs were the story; proceed to raw NIfTI.
4. **H2** — bind HU/spacing/normalization to the checkpoint. *(code-only)*
5. **H1** — switch to a mask-free body crop for inference and align the training
   crop; **retrain** the baseline (the plan already sanctions a clean reset).
6. **M1/M2** — wire the guardrail; unify head normalization; retrain.

None of the above introduces Stage 2+ machinery (no Swin, no distillation, no
representation learning). Every change either fixes a train/inference contract or
restores a documented baseline decision.

> **Note on stage discipline:** All recommendations here belong to Stage 1
> baseline validation. Architectural ideas (Swin UNETR, PET distillation, domain
> adaptation) remain out of scope and should not be introduced until this
> baseline produces a bimodal, lesion-focused heatmap on the processed
> validation set.
