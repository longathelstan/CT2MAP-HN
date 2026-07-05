# CT2MAP-HN Baseline Model — Phân tích chất lượng Output

> Báo cáo phân tích chi tiết kết quả inference trên case `CHUM-012` để đưa cho GPT 5.5 Pro nghiên cứu.

---

## 1. Tổng quan kết quả

| Metric | Giá trị | Đánh giá |
|---|---|---|
| **Triage score** | 0.925 | ⚠️ Cao bất thường |
| **Số candidates** | 9 | OK |
| **Candidate #1 volume** | 81,565,920 voxels | 🔴 87% tổng volume |
| **Candidate #1 mean_intensity** | 0.5455 | ⚠️ Chỉ hơi trên threshold |
| **Candidate #1 max_intensity** | 0.9999 | OK |
| **Heatmap shape** | 375 × 500 × 500 = 93.75M voxels | — |
| **Elapsed time** | 5,021s (~1.4 giờ) | 🔴 Quá chậm |

---

## 2. Vấn đề chính: Model Output gần như uniform ~0.5

### 2.1 Phân tích candidate #1

```
Candidate #1:
  volume_voxels: 81,565,920  (87% of 93,750,000 total voxels)
  bounding_box: [0:375, 0:500, 0:500]  ← TOÀN BỘ VOLUME
  mean_intensity: 0.5455
  max_intensity: 0.9999
```

**Diễn giải:** Model predict gần như MỌI voxel có giá trị ≥ 0.5 (threshold). Candidate #1 là một connected component khổng lồ chiếm 87% volume, với mean chỉ 0.5455. Điều này cho thấy:

1. **Model output phân bố quanh 0.5** — hầu hết voxels có giá trị trong khoảng [0.49, 0.55]
2. **Threshold 0.5 quá thấp** cho model này — cần threshold cao hơn để phân biệt vùng thực sự có nguy cơ
3. **Model chưa học tốt** — output lý tưởng phải có phân bố bimodal (0 cho background, cao cho lesion), nhưng thực tế gần như unimodal quanh 0.5

### 2.2 Phân tích các candidates còn lại (#2 → #9)

| Candidate | Volume (voxels) | Mean Intensity | Max Intensity |
|---|---|---|---|
| #2 (label 18957) | 363 | 0.4916 | 0.5475 |
| #3 (label 16053) | 214 | 0.4920 | 0.5445 |
| #4 (label 40621) | 158 | 0.4968 | 0.5145 |
| #5 (label 65878) | 134 | 0.4926 | 0.5257 |
| #6 (label 19197) | 117 | 0.4927 | 0.5465 |
| #7 (label 31445) | 103 | 0.4939 | 0.5148 |
| #8 (label 35630) | 101 | 0.4955 | 0.5150 |
| #9 (label 38090) | 100 | 0.4968 | 0.5137 |

**Nhận xét:**
- Tất cả candidates #2–#9 có `mean_intensity < 0.5` — chúng là "đảo" nhỏ nằm DƯỚI threshold, bị tách ra bởi `extract_connected_components` nhưng thực ra là noise
- `max_intensity` của tất cả đều rất thấp (0.51–0.55) — không có vùng nào model tự tin là lesion
- Volume rất nhỏ (100–363 voxels) so với candidate #1 (81.5M)

### 2.3 Label numbering bất thường

Các label: `1, 16053, 18957, 19197, 31445, 35630, 38090, 40621, 65878`

Label numbers rất lớn cho thấy `scipy.ndimage.label()` đã tìm ra **hàng chục ngàn connected components** trước khi filter bởi `min_size=100`. Đây là vì:
- Volume 93.75M voxels với threshold 0.5
- Hầu hết voxels > 0.5 → tạo 1 component khổng lồ
- Các voxels < 0.5 tạo thành hàng ngàn "lỗ" (holes) nhỏ → mỗi lỗ lại tạo separate labels
- `ndimage.label()` phải label TẤT CẢ components (bao gồm cả trên 0.5 và dưới 0.5)

> [!WARNING]
> Thực ra `ndimage.label()` chỉ label vùng = 1 (trên threshold), nhưng nếu gần như toàn bộ volume là 1, thì vùng = 0 cũng tạo thành nhiều "islands" nhỏ. Label #65878 cho thấy ít nhất có 65,878+ raw components trước khi filter.

---

## 3. Phân tích Triage Score

[Công thức triage](file:///d:/LowngWorkspace/CT2MAP-HN/src/inference/postprocess.py#L162-L244):

```
triage = 0.35 × max_activation     # = 0.35 × 0.9999 ≈ 0.35
       + 0.30 × volume_high_risk   # = 0.30 × min(frac_>0.7 × 100, 1.0)
       + 0.20 × num_suspicious     # = 0.20 × min(9/5, 1.0) = 0.20
       + 0.15 × uncertainty_conf   # = 0.15 × 0.5 = 0.075 (no MC dropout)
```

Triage = 0.925. Breakdown:
- `max_activation = 0.9999` → Có ít nhất 1 voxel cực cao (0.35 contribution)
- `volume_high_risk`: frac voxels > 0.7 × 100, capped at 1.0 → rất có thể ≈ 1.0 (0.30 contribution)
- `num_suspicious = min(9/5, 1)` = 1.0 (0.20 contribution)
- `uncertainty = 0.5` (neutral, no MC dropout) (0.075 contribution)

**Total ≈ 0.35 + 0.30 + 0.20 + 0.075 = 0.925** ✓ khớp

---

## 4. Nguyên nhân gốc rễ

### Giả thuyết 1: Model chưa hội tụ (under-trained)
- BasicUNet với features `[32, 64, 128, 256, 512, 32]` là kiến trúc hợp lý
- Training 60 epochs (có checkpoints 10–60) — có thể cần nhiều epoch hơn (config set 300)
- Checkpoint `best_model.pth` có thể được save quá sớm

### Giả thuyết 2: Target generation không đúng
- Target (ground truth) cho training là "Gaussian-smoothed lesion mask → PET-derived soft target"
- Nếu soft target phân bố quanh 0.5, model sẽ học output ~0.5 cho mọi thứ
- Cần kiểm tra phân bố giá trị của target labels trong training data

### Giả thuyết 3: Loss function vấn đề
- Config dùng `loss.primary: mse` với `ssim: 0.1, gradient: 0.05`
- MSE loss trên soft targets có thể khiến model "an toàn" bằng cách predict mean value
- MSE không phạt heavily khi predict 0.5 cho mọi voxel (nếu target mean ≈ 0.5)

### Giả thuyết 4: Preprocessing mismatch
- Config inference không có `preprocessing` section → defaults: `hu_min=-1024, hu_max=1024`
- Nhưng [preprocess.yaml](file:///d:/LowngWorkspace/CT2MAP-HN/configs/preprocess.yaml) dùng `hu_clip: [-200, 300]` cho training
- **MISMATCH!** Training normalize HU [-200, 300] → [0,1], inference normalize [-1024, 1024] → [0,1]
- Cùng 1 voxel 100 HU: training = (100+200)/500 = 0.6, inference = (100+1024)/2048 = 0.549
- Điều này giải thích tại sao model output ~0.5 cho mọi thứ — input distribution lệch!

> [!CAUTION]  
> **Giả thuyết 4 rất có khả năng là nguyên nhân chính!** Preprocessing mismatch giữa training và inference sẽ khiến model nhận input hoàn toàn khác với lúc train, dẫn đến output gần uniform ~0.5.

---

## 5. Khuyến nghị

### Ưu tiên 1 — Fix Preprocessing mismatch (Critical)
Thêm config `preprocessing` vào [baseline_nnunet.yaml](file:///d:/LowngWorkspace/CT2MAP-HN/configs/baseline_nnunet.yaml):
```yaml
preprocessing:
  target_spacing: [1.0, 1.0, 1.0]
  hu_min: -200
  hu_max: 300
```

### Ưu tiên 2 — Kiểm tra phân bố target training data
- Load vài file `*_pet.npy` hoặc target từ `data/processed/`
- Vẽ histogram giá trị → xác nhận target có bimodal distribution hay uniform

### Ưu tiên 3 — Đánh giá thêm checkpoints
- So sánh `best_model.pth` vs `final_model.pth` vs `checkpoint_epoch_0060.pth`
- Check validation loss trend trong TensorBoard logs

### Ưu tiên 4 — Điều chỉnh threshold
- Nếu sau fix preprocessing mà output vẫn tập trung quanh 0.5 → tăng threshold lên 0.6–0.7
- Hoặc dùng adaptive threshold (Otsu) thay vì fixed threshold

### Ưu tiên 5 — Loss function
- Cân nhắc Focal MSE hoặc thêm Dice loss component để tập trung vào vùng lesion
- SSIM loss weight 0.1 có thể quá thấp

---

## 6. Files liên quan

| File | Mô tả |
|---|---|
| [infer_case.py](file:///d:/LowngWorkspace/CT2MAP-HN/src/inference/infer_case.py) | Inference pipeline (preprocessing ở L217–L278) |
| [postprocess.py](file:///d:/LowngWorkspace/CT2MAP-HN/src/inference/postprocess.py) | Thresholding + CC extraction + triage score |
| [baseline_nnunet.yaml](file:///d:/LowngWorkspace/CT2MAP-HN/configs/baseline_nnunet.yaml) | Model config (THIẾU `preprocessing` section) |
| [preprocess.yaml](file:///d:/LowngWorkspace/CT2MAP-HN/configs/preprocess.yaml) | Training preprocessing (hu_clip: [-200, 300]) |
| [summary.json](file:///d:/LowngWorkspace/CT2MAP-HN/outputs/baseline/eval_case/summary.json) | Kết quả inference CHUM-012 |
