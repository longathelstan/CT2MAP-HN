# Plan: Làm Lại Model 0 Thành Baseline Đáng Tin

## Summary
- Mục tiêu là biến **Model 0** thành baseline sạch: `CT -> lesion/gaussian heatmap`, dùng để chứng minh pipeline dữ liệu, preprocessing, target, loss và inference đúng trước khi sang Swin/KD.
- Không dùng checkpoint hiện tại làm baseline chính; chỉ giữ để so sánh. Train lại từ đầu sau khi sửa config/data/inference.
- Trong Model 0, **không train triage head bằng `high_risk` hiện tại** vì label lệch mạnh `720/782=True`; triage chỉ là rule-based từ heatmap sau khi baseline ổn.

## Key Changes
- Chuẩn hóa config Model 0:
  - Sửa `configs/baseline_nnunet.yaml` về schema thật mà code dùng: `training.num_epochs`, `training.batch_size`, `optimizer.name/lr/weight_decay`, `scheduler`, `loss`, `paths`.
  - Thêm config validation: fail fast nếu có key bị ignore như `training.epochs`, `training.optimizer`, `training.loss`.
  - Dùng output dir mới `outputs/model0_unet_v2`, checkpoint dir mới `checkpoints/model0_unet_v2`.

- Sửa pipeline train:
  - Model 0 train với `w_heatmap=1.0`, `w_lesion=1.0`, `w_triage=0.0`.
  - Heatmap loss: weighted/focal L1 hoặc MSE, `lesion_weight=10`, để tránh output uniform quanh `0.5`.
  - Lesion loss: Dice + focal BCE, positive alpha cao hơn hiện tại, ví dụ `focal_alpha=0.75`.
  - Cho `BaselineUNet` thật sự dùng `norm: instance` từ config thay vì hardcode BatchNorm.
  - Thêm overfit smoke test 2-4 case trước full train; không được train full nếu overfit test fail.

- Sửa preprocessing/inference parity:
  - Inference phải dùng đúng pipeline train: orientation, spacing, HU clip `[-200,300]`, normalize, **crop ROI**.
  - Đánh giá Model 0 trước trên `data/processed/*.npy` để loại trừ lỗi raw NIfTI geometry.
  - Sau đó mới hỗ trợ raw NIfTI bằng cách crop ROI, predict ROI, rồi paste/resample heatmap về không gian CT gốc.
  - Không threshold cố định `0.5`; chọn threshold trên validation set, lưu vào eval artifact, dùng lại cho test/demo.

- Thêm audit/eval cho Model 0:
  - Data audit: kiểm shape CT/mask/target, target quantiles, foreground fraction, split counts, label leakage.
  - Prediction audit: histogram heatmap, background median, lesion-region mean, max component fraction.
  - Eval report gồm Dice/IoU/F1 theo threshold, best val threshold, test metrics, gallery good/bad cases.
  - Nếu một connected component chiếm quá lớn ROI, ví dụ `>25%`, flag là `UNIFORM_OUTPUT_FAILURE` và không dùng triage score đó.

## Test Plan
- Unit/static checks:
  - Config validation bắt được key sai/ignored.
  - Dataset sample trả đúng keys: `ct`, `heatmap`, `lesion_mask`, `case_id`.
  - Loss chạy được với batch nhỏ và `w_triage=0`.

- Smoke tests:
  - Overfit 2-4 case trong tối đa vài trăm iterations; yêu cầu lesion Dice train cao rõ rệt, ví dụ `>0.8`, và heatmap không uniform.
  - Inference trên processed `CHUM-012` phải sinh heatmap ROI cục bộ, không phủ toàn ảnh.

- Acceptance gates:
  - Data audit không có shape mismatch hoặc target rỗng bất thường.
  - Validation chọn được threshold hợp lý từ val set, không chọn từ test.
  - Trên gallery 10-20 case, heatmap phải bám quanh lesion/mask hơn nền.
  - Không công nhận Model 0 nếu output vẫn có median quanh `0.5` toàn volume hoặc component lớn phủ phần lớn ROI.

## Assumptions
- Chọn hướng **reset sạch Model 0** làm baseline chính; không cố cứu checkpoint hiện tại.
- Scope chỉ là Model 0; chưa làm Swin UNETR, PET distillation, MoE, graph KD.
- Target chính vẫn là `gaussian_heatmap` từ lesion mask; PET-derived target để dành Model 2.
- `high_risk` hiện tại chưa đủ tốt cho supervised triage, nên triage của Model 0 là postprocess/rule-based.
