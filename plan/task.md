# Task Plan Cho Opus 4.6: Cải Thiện Model 0 Baseline

## Mục Tiêu
Làm lại **Model 0 = CT-only BasicUNet/UNet baseline** thành baseline đáng tin: không output uniform quanh `0.5`, không phủ heatmap toàn volume, có eval định lượng và gallery trực quan trước khi chuyển sang Model 1/2.

## Task 1: Sửa Config Schema Cho Baseline
- Chuẩn hóa `configs/baseline_nnunet.yaml` theo đúng key mà code hiện dùng:
  - `training.num_epochs`
  - `training.batch_size`
  - `training.use_amp`
  - `training.grad_accumulation_steps`
  - `training.clip_grad_norm`
  - `training.val_interval`
  - `training.save_interval`
  - `training.early_stop_patience`
  - `optimizer.name`, `optimizer.lr`, `optimizer.weight_decay`
  - `scheduler.name`
  - `loss.w_heatmap`, `loss.w_lesion`, `loss.w_triage`
  - `paths.checkpoint_dir`, `paths.tensorboard_dir`, `paths.metrics_dir`
- Giữ preprocessing parity:
  - `preprocessing.target_spacing: [1.0, 1.0, 1.0]`
  - `preprocessing.hu_min: -200`
  - `preprocessing.hu_max: 300`
- Set baseline v2 output riêng:
  - `checkpoints/model0_unet_v2`
  - `runs/model0_unet_v2`
  - `results/model0_unet_v2`
  - `outputs/model0_unet_v2`

## Task 2: Thêm Config Validation
- Thêm validator khi load config trong `src/train/train_baseline.py`.
- Nếu phát hiện key cũ bị ignore như `training.epochs`, `training.lr`, `training.optimizer`, `training.scheduler`, `training.loss`, thì raise error hoặc log warning rõ ràng.
- In ra resolved config quan trọng trước khi train:
  - epochs, batch size, optimizer, lr, scheduler, loss weights, target type, paths.

## Task 3: Sửa Model 0 Không Train Triage
- Trong config baseline v2:
  - `loss.w_heatmap: 1.0`
  - `loss.w_lesion: 1.0`
  - `loss.w_triage: 0.0`
- Trong `CombinedLoss`, nếu weight bằng `0`, không cần compute loss tương ứng để tránh triage label lệch lớp ảnh hưởng baseline.
- Triage score ở Model 0 chỉ tính rule-based sau inference, không dùng supervised triage head làm tiêu chí chính.

## Task 4: Sửa Loss Để Chống Output Uniform
- Heatmap loss:
  - dùng `base: l1` hoặc `base: mse`
  - bật `use_focal: true`
  - `lesion_weight: 10.0`
  - `focal_gamma: 2.0`
- Lesion loss:
  - Dice + focal BCE
  - tăng positive emphasis, ví dụ `focal_alpha: 0.75`
- Đảm bảo prediction và target đều shape `(B,1,D,H,W)` và trong `[0,1]`.

## Task 5: Cho BaselineUNet Dùng Norm Từ Config
- Hiện `BaselineUNet` hardcode BatchNorm.
- Thêm tham số `norm` vào constructor.
- Map:
  - `instance` -> `("instance", {"affine": True})`
  - `batch` -> `("batch", {"affine": True})`
- Baseline v2 dùng `norm: instance` vì batch/patch 3D nhỏ.

## Task 6: Thêm Data Audit Script
- Tạo script audit không train, ví dụ `scripts/audit_model0_data.py`.
- Report tối thiểu:
  - số case train/val/test
  - shape CT/mask/target
  - CT quantiles
  - binary mask foreground fraction
  - gaussian target quantiles
  - case có target rỗng hoặc shape mismatch
- Output:
  - `results/model0_unet_v2/data_audit.json`
  - `results/model0_unet_v2/data_audit.csv`

## Task 7: Thêm Overfit Smoke Test
- Tạo mode/script train nhỏ, ví dụ `scripts/overfit_model0.py`.
- Dùng 2-4 case cố định, train vài trăm iterations.
- Acceptance:
  - lesion Dice train tăng rõ, mục tiêu `>0.8` nếu dữ liệu/target đúng.
  - heatmap median nền không quanh `0.5` toàn ROI.
  - component lớn nhất không phủ phần lớn ROI.
- Nếu overfit fail thì không chạy full train.

## Task 8: Train Lại Model 0 Từ Đầu
- Không dùng checkpoint hiện tại làm baseline chính.
- Train command:
  - `python -m src.train.train_baseline --config configs/baseline_nnunet.yaml`
- Lưu:
  - best checkpoint
  - final checkpoint
  - train/val loss history
  - per-loss TensorBoard curves: heatmap, lesion, total.

## Task 9: Evaluation Chuẩn Cho Processed Data
- Tạo eval script cho processed `.npy`, ví dụ `scripts/eval_model0_processed.py`.
- Chạy trên val để chọn threshold, không dùng test để chọn threshold.
- Metrics:
  - Dice
  - IoU
  - precision/recall/F1
  - heatmap MAE/MSE
  - foreground/background mean activation
  - largest component fraction
- Lưu threshold tốt nhất vào:
  - `results/model0_unet_v2/threshold.json`

## Task 10: Inference/Gallery Cho Sanity Check
- Sinh gallery 10-20 case val/test:
  - CT slice
  - ground-truth mask
  - predicted heatmap overlay
  - binary prediction ở threshold đã chọn
- Lưu vào:
  - `outputs/model0_unet_v2/gallery`
- Với mỗi case lưu summary:
  - heatmap min/median/max
  - fraction voxels `> threshold`
  - largest component fraction
  - Dice/IoU nếu có mask.

## Task 11: Sửa Postprocess Guardrail
- Trong `compute_triage_score` hoặc eval wrapper, thêm guardrail:
  - nếu largest component fraction `> 0.25`, flag `UNIFORM_OUTPUT_FAILURE`.
  - nếu heatmap median nằm trong khoảng đáng ngờ quanh `0.45-0.55` và foreground quá lớn, flag `LOW_QUALITY_HEATMAP`.
- Khi có flag, không dùng triage recommendation như kết quả hợp lệ.

## Task 12: Raw NIfTI Inference Sau Cùng
- Trước hết eval trên processed `.npy`.
- Sau khi processed inference ổn, mới sửa raw NIfTI path:
  - preprocess raw CT giống train
  - crop ROI
  - predict ROI
  - paste/resample heatmap về CT gốc
  - save NIfTI có đúng geometry để mở trong ITK-SNAP.
- Không dùng raw NIfTI inference để đánh giá baseline chính nếu parity chưa được chứng minh.

## Acceptance Criteria Cuối
- Data audit pass, không shape mismatch.
- Overfit smoke test pass.
- Full train có loss giảm thật, không chỉ quanh `1.07`.
- Val threshold được chọn từ val set và lưu lại.
- Test/gallery không còn heatmap phủ toàn volume.
- `CHUM-012` không còn candidate lớn chiếm gần toàn ảnh như hiện tại.
- Có report cuối:
  - `results/model0_unet_v2/model0_eval.json`
  - `results/model0_unet_v2/model0_eval.csv`
  - `outputs/model0_unet_v2/gallery`

## Assumptions
- Scope chỉ là Model 0.
- Không làm PET distillation, Swin UNETR, MoE, graph KD trong task này.
- Checkpoint cũ chỉ dùng để so sánh, không dùng làm baseline chính.
- Triage supervised tạm bỏ vì label `high_risk` hiện quá lệch lớp.
