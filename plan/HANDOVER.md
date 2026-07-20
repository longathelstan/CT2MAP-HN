# CT2MAP-HN — Tài liệu Handover (Model 0 Baseline)

> **Cập nhật:** 2026-07-20 · Git HEAD `f055623` · Stage 1 / Model 0 baseline
> **Vai trò người viết:** Technical Project Manager + AI Research Assistant

---

## ⚡ Quick Context (đọc 30 giây)

1. **Dự án:** CT2MAP-HN — ước lượng thông tin chuyển hóa **giống PET từ CT đơn thuần** để phân loại/triage ung thư đầu-cổ (HNSCC). PET chỉ dùng khi train; **inference BẮT BUỘC chỉ dùng CT**.
2. **Dataset:** HECKTOR, **782 ca**, 8 trung tâm (CHUM/CHUP/CHUS/CHUV/HGJ/HMR/MDA/USZ). Split: train 547 / val 78 / test 157.
3. **Model 0 (baseline):** `BaselineUNet` — 1 backbone U-Net + 3 head (`heatmap_head`, `lesion_head`, `triage_head`). features `[32,64,128,256,512,32]`, norm `instance`. Checkpoint: `checkpoints/baseline_p40/best_model.pth` (epoch 184, best_val_loss 0.2993, 102 keys).
4. **Đang ở đâu:** Vừa **tái tạo xong toàn bộ 782 ca dữ liệu processed** với crop mới (CT-only body crop + z-window 360mm). Đây là điều kiện tiên quyết để train lại baseline.
5. **4 bug nghiêm trọng đã sửa:** C1 (load checkpoint 0/102 → sửa), C2 (double sigmoid → sửa), H2 (HU window sai → sửa), H1 (crop dùng GT mask, không tái lập được ở inference → sửa bằng crop CT-only).
6. **Đã xác minh bằng số liệu:** trên CHUM-012 (processed), model localize đúng tổn thương, **Dice 0.747** vs GT gaussian; background bị đẩy về ~0.02 (không còn tràn nền).
7. **Việc kế tiếp:** chạy **overfit smoke test** (cổng Dice > 0.8) trên dữ liệu vừa tái tạo → nếu qua thì train baseline đầy đủ (`configs/model0_v2.yaml`).
8. **Nguyên tắc:** chỉ sửa/đánh giá **Model 0 baseline** — KHÔNG đề xuất Swin UNETR, ViT, Knowledge Distillation, Domain Adaptation, Diffusion, Foundation Model hay redesign lớn (đó là Stage 2+).
9. **Bài học gỡ lỗi:** mỗi cell chạy trong **sandbox namespace riêng** → `pgrep`/`ps` KHÔNG thấy tiến trình job nền. Chỉ tin `host.exec_peek(exec_id)` + mtime của log; đừng suy ra "job chết" từ pgrep.

---

## 1. Mục tiêu dự án

- **Bài toán:** Xây dựng framework **chỉ dùng CT** để ước lượng thông tin chuyển hóa giống PET/CT, hỗ trợ triage lâm sàng ở nơi không có PET/CT. Mục tiêu KHÔNG phải thay thế PET, mà cung cấp hướng dẫn chuyển hóa + ước lượng bất định + hỗ trợ phân loại từ CT.
- **Ưu tiên đầu vào:** (1) CT có thuốc cản quang > (2) planning CT > (3) CT không cản quang. PET chỉ có khi train; inference chỉ CT.
- **Dataset:** HECKTOR head/neck, **782 ca**, 8 trung tâm. Mỗi ca raw: `{ID}__CT.nii.gz`, `{ID}__PT.nii.gz` (PET), `{ID}.nii.gz` (mask GTV). Đặt tại `/data/lowngworkspace/hecktor_raw/`. Split seed 42: train 547 / val 78 / test 157.
- **Kiến trúc (Model 0 baseline):** `BaselineUNet` (`src/models/baseline_unet.py`) — backbone U-Net 3D, features `[32,64,128,256,512,32]`, norm `instance`, dropout 0.1; 3 head trên decoder output:
  - `heatmap_head`: Conv3d→Norm→ReLU→Conv3d→**Sigmoid** → heatmap chuyển hóa [0,1]
  - `lesion_head`: tương tự → xác suất tổn thương
  - `triage_head`: MLP → điểm triage
- **Roadmap:** Stage 1 baseline (BasicUNet/nnUNet — ĐANG LÀM) → Stage 2 Swin UNETR → Stage 3 PET-guided Knowledge Distillation → Stage 4 đóng góp nghiên cứu (metabolic/domain-invariant representation).
- **Đầu ra mong muốn:** Metabolic Heatmap · Lesion Candidate Map · Triage Score · Uncertainty Map (tương lai).
- **Giả thuyết đang kiểm chứng:** CT đơn thuần chứa đủ tín hiệu để định vị vùng tăng chuyển hóa (proxy cho SUV cao) đủ tốt cho triage; một U-Net baseline được train với target gaussian heatmap (dẫn xuất từ GTV) có thể tái tạo bản đồ tập trung ở tổn thương thay vì tràn nền.

---

## 2. Những việc đã hoàn thành (theo trình tự)

| # | Việc đã làm | Lý do | Kết quả | Kết luận |
|---|-------------|-------|---------|----------|
| 1 | **Đọc toàn bộ docs** (`model_quality_analysis.md`, `IMPLEMENTATION_PLAN.md`, `ban_dac_ta.md`, `plan/task.md`, `plan/implement_plan.md`, `plan/review.md`) | Docs = single source of truth | Nắm plan remediation 12 task "Làm lại Model 0 Baseline v2" | `plan/task.md` là kế hoạch chuẩn |
| 2 | **Review inference** theo chuẩn MICCAI reviewer | Heatmap tràn toàn ảnh (ảnh ITK-SNAP: giá trị ~0.57 ở vùng lẽ ra =0) | Báo cáo `plan/model0_inference_review.md` (6 finding C1/C2/H1/H2/M1/M2) | Xác định 4 bug nghiêm trọng |
| 3 | **Load-test checkpoint** | Kiểm tra khớp trọng số | Checkpoint 102 key, inference chỉ build model 82 param, **0/102 khớp** | Xác nhận C1 |
| 4 | **Sửa C1** (`infer_case.py`) | Model dựng sai kiến trúc | Thêm nhánh `baseline_unet`; dựng lại model + preprocessing từ `checkpoint["config"]`; raise nếu key mismatch | 102/102 khớp |
| 5 | **Sửa C2** (`infer_case.py`) | Double sigmoid ép output về [0.5,0.73] | `_postprocess` bỏ sigmoid thứ 2 (head đã có Sigmoid) | Output bimodal đúng |
| 6 | **Sửa H2** (`infer_case.py`) | HU window inference −1024/1024 ≠ train −200/300 | HU lấy từ checkpoint preprocessing | Hết covariate shift HU |
| 7 | **Validate CHUM-012** (processed .npy) | Xác minh fix bằng số liệu | Dice 0.747; bg→0.02, lesion→0.51; triage 0.80; component lớn nhất 1.6% | C1/C2/H2 đúng |
| 8 | **Figure before/after** | Trực quan hóa tác động C2 | `CHUM-012_before_after.png` (artifact) | Minh họa isolcate C2 |
| 9 | **Diagnostic độ sâu GTV** (`_hn_depth_par.py`, 149 ca) | Chọn z-window cho crop CT-only | body-top=đỉnh sọ (145/149); GTV sâu nhất 311mm dưới body-top | Landmark ổn định |
| 10 | **Sửa H1** (`preprocess.py`) | Crop cũ dùng GT mask, không tái lập ở inference | Thêm `compute_body_mask` + `_crop_body` (crop CT-only) + z-window 360mm | Crop tái lập được từ CT |
| 11 | **Validate crop** (`validate_body_crop_zwindow.py`, 4 ca) | Đảm bảo không mất GTV | Tất cả z→361 lát, GTV giữ 100% | Crop an toàn |
| 12 | **Tái tạo toàn bộ dữ liệu** (782 ca) | Áp crop mới cho cả dataset | 782/782 `_ct/_mask` + 782/782 targets (binary+gaussian); z_extent 360mm nhất quán | Dữ liệu sẵn sàng train |

---

## 3. Những lỗi đã phát hiện

### C1 — Checkpoint không được load (0/102 key) · **ĐÃ SỬA**
- **Triệu chứng:** heatmap giống edge map/nhiễu, tràn toàn ảnh kể cả không khí.
- **Root cause:** `infer_case.py::_build_model` chỉ có registry `{swinunetr, basicunet, unet}`, KHÔNG có `baseline_unet` → dựng `BasicUNet` trần (82 param) trong khi checkpoint có 102 key prefix `backbone./heatmap_head./lesion_head./triage_head.`. `load_state_dict(strict=False)` nuốt mismatch → **chạy trên model random**.
- **Xác minh:** load-test in ra `keys_matched=0, missing=82, unexpected=102`.
- **Cách sửa:** thêm nhánh `baselineunet` trong `_build_model`; đọc `checkpoint["config"]`, dựng lại model + merge preprocessing TRƯỚC khi build; raise `RuntimeError` nếu có key thừa/thiếu (chỉ `uncertainty_head` được phép thiếu).

### C2 — Double sigmoid · **ĐÃ SỬA**
- **Triệu chứng:** giá trị heatmap kẹt trong [0.5, 0.731]; con trỏ ITK-SNAP đọc 0.5712 ở vùng nền.
- **Root cause:** head kết thúc bằng `nn.Sigmoid()`; `_postprocess` áp `torch.sigmoid()` LẦN NỮA (L339) → nén [0,1]→[0.5,0.731].
- **Xác minh:** corrected min 0.0 / max 0.98 / mean 0.038 vs buggy min 0.5 / max 0.727. **Lưu ý nguồn số:** mean "buggy" 0.509 là từ **reimplement double-sigmoid độc lập** trên CHUM-012 processed, KHÔNG phải giá trị đo trực tiếp từ lần chạy full-pipeline lỗi ban đầu (giá trị đo trực tiếp đó là **0.5455**, trên raw NIfTI uncropped). Hai con số khác nhau vì khác đầu vào (processed ROI vs raw toàn ảnh) — cả hai đều xác nhận double sigmoid ép mean vào [0.5,0.73], nhưng đừng coi 0.509 là số đo của production run.
- **Cách sửa:** `self.model_outputs_probabilities=True` cho baseline_unet → bỏ sigmoid thứ 2.

### H2 — HU window inference sai · **ĐÃ SỬA**
- **Triệu chứng:** phân phối cường độ đầu vào lệch train.
- **Root cause:** `_preprocess` mặc định HU −1024/1024, train dùng −200/300.
- **Cách sửa:** HU lấy từ `checkpoint["config"]["preprocessing"]` (−200/300).

### H1 — Crop ROI dùng GT lesion mask · **ĐÃ SỬA** (bug sâu nhất)
- **Triệu chứng:** train/inference lệch phân phối; inference không có mask nên không crop được như train.
- **Root cause:** `crop_head_neck_roi` method `bbox` suy bbox từ `np.argwhere(mask>0)` → **không tái lập được ở inference** (vi phạm "Inference MUST use CT only").
- **Xác minh:** đọc code + kiểm tra pipeline; crop cũ phụ thuộc mask.
- **Cách sửa:** `compute_body_mask` (ngưỡng HU −500 → binary_closing → largest CC → fill_holes) + `_crop_body` (bbox thân theo y/x + z-window 360mm tính từ đỉnh thân). Áp cả ở `prepare_data.py` và `infer_case._preprocess` (mask=None khi inference).

### M1 — Guardrail UNIFORM_OUTPUT_FAILURE chưa được gọi · **CHƯA SỬA**
- **Root cause:** `quality_check_heatmap` định nghĩa ở `postprocess.py:181` nhưng `infer_case.py` không gọi.
- **Cách sửa (đề xuất):** gọi trong `infer()`, cờ `UNIFORM_OUTPUT_FAILURE` nếu component lớn nhất > 25% ROI.

### M2 — Head hardcode BatchNorm3d · **CHƯA SỬA**
- **Root cause:** `heads.py` dùng `nn.BatchNorm3d` (dòng 50,98,188) trong khi backbone dùng InstanceNorm → không nhất quán, BN xấu ở batch_size=1.
- **Cách sửa (đề xuất):** cho head nhận norm theo config, thống nhất InstanceNorm.

### Giả thuyết SAI đã bác bỏ (ghi lại để không lặp)
- **"Job nền chết vì OOM"** khi `pgrep` trả 0 tiến trình: SAI. Mỗi cell ở sandbox namespace riêng nên không thấy tiến trình job nền. Thực tế 3–4 job vẫn chạy chồng nhau → mới là nguyên nhân OOM. Chỉ tin `host.exec_peek` + mtime log.
- **"2 ca cắt GTV vì cơ thể dài hơn cửa sổ 360mm"**: SAI. Body span của MDA-146/MDA-210 chỉ 279mm < 360mm → cửa sổ z KHÔNG bind. Crop bị chặn bởi **sàn body-bottom / FOV scan giới hạn**, không phải độ dài cửa sổ.

---

## 4. Các kết luận đã được xác minh (có bằng chứng)

- **Checkpoint load đúng:** sau fix C1, 102/102 key khớp (epoch 184, best_val_loss 0.2993).
- **Double sigmoid là bug:** đo được buggy [0.5,0.727] vs corrected [0.0,0.98] trên cùng trọng số.
- **Model có học:** trên CHUM-012 processed, pred_mean trong lesion 0.513 vs ngoài 0.020; **Dice 0.747** vs GT gaussian@0.5.
- **HU window đúng:** −200/300 lấy từ checkpoint, khớp train.
- **Crop CT-only tái lập được:** không cần mask; validate 4 ca giữ 100% GTV.
- **body-top = đỉnh sọ, landmark ổn định:** 145/149 ca (97%) hẹp dần về phía đầu.
- **z-window 360mm đủ rộng:** GTV sâu nhất 311mm dưới body-top; 360mm chừa ~50mm biên. **KHÔNG bị giới hạn bởi cửa sổ 360mm** ở các ca đã kiểm (kể cả 2 ca clip, vì body span < 360mm).
- **Tái tạo dữ liệu hoàn tất:** 782/782 ca crop mới + 782/782 targets; 780 ca giữ 100% GTV, 2 ca 98.1%/99.65%.

---

## 5. Những vấn đề còn đang điều tra

- **Overfit smoke test trên dữ liệu MỚI:** overfit cũ (MDA-174 dice 0.908, CHUM-056 0.934) chạy trên crop CŨ. Cần chạy lại trên crop mới để xác nhận pipeline train vẫn học được (cổng Dice > 0.8 + histogram bimodal).
- **Hiệu năng full-dataset:** chưa train baseline đầy đủ trên dữ liệu mới → chưa biết Dice/Recall trên val/test.
- **False positive / miss GTV:** chưa đo có hệ thống (cần `eval_model0_processed.py` — hiện THIẾU).
- **Domain shift giữa trung tâm:** 8 site, có thể khác phân phối; chưa phân tích per-center.
- **2 ca clip GTV (MDA-146, MDA-210):** mất <2% thể tích do FOV; cần quyết định giữ nguyên hay xử lý riêng.
- **Parity processed vs raw NIfTI:** Task 12 yêu cầu eval processed .npy trước; chưa chứng minh parity đủ để dùng raw NIfTI cho đánh giá chính.
- **M1, M2:** chưa wire guardrail / chưa thống nhất norm ở head.

---

## 6. Các script quan trọng

| Script | Chức năng | Input | Output |
|--------|-----------|-------|--------|
| `scripts/prepare_data.py` | Preprocess: orient→resample(1mm)→**crop body+zwindow**→clip HU→normalize | raw NIfTI + `configs/preprocess.yaml` + manifest | `{ID}_ct.npy`, `_mask.npy`, `_pet.npy`, `_crop_info.json` |
| `scripts/create_targets.py` | Tạo target từ mask đã crop | `_mask.npy` + `configs/data.yaml` | `targets/{ID}_binary.npy`, `_gaussian_heatmap.npy`, `_label.json` |
| `src/dataio/preprocess.py` | Lõi preprocess: `compute_body_mask`, `crop_head_neck_roi`, `_crop_body`, `preprocess_case` | — | — |
| `src/models/baseline_unet.py` | Định nghĩa `BaselineUNet` (backbone + 3 head) | — | — |
| `src/inference/infer_case.py` | `CaseInferencer`: load checkpoint, preprocess (crop CT-only), sliding-window infer, postprocess | CT NIfTI + checkpoint | heatmap, candidates, triage |
| `src/inference/postprocess.py` | `threshold_heatmap`, `extract_connected_components`, `compute_triage_score`, `quality_check_heatmap` | heatmap | candidates, triage, cờ QC |
| `src/train/train_baseline.py` | Entrypoint train baseline | dataset processed + config | checkpoint |
| `scripts/validate_model0_processed.py` | Validate model trên processed .npy (Dice vs GT) | processed tensor + checkpoint | JSON metrics |
| `scripts/validate_body_crop_zwindow.py` | Kiểm tra crop giữ GTV + z-window | raw + preprocess cfg | thống kê crop |
| `scripts/overfit_model0.py` | Overfit smoke test (2 ca, cổng Dice) | 2 ca processed | `overfit_summary.json` |
| `scripts/_hn_depth_par.py` | Diagnostic độ sâu GTV dưới body-top (song song) | raw NIfTI | `_hn_depth_par.json` |
| `scripts/eval_model0_processed.py` | **THIẾU — cần viết** (Task 9): eval val, chọn threshold trên val không phải test | processed val | metrics + threshold |

---

## 7. Kết quả thực nghiệm

**Validate CHUM-012 (processed, sau fix C1/C2/H2):**
- params_loaded 102/102; ct_shape [74,77,72]
- corrected heatmap: min 0.0 / max 0.98 / mean 0.038; frac≥0.5 = 2.5%; frac≤0.05 = 88%
- buggy (double sigmoid): min 0.5 / max 0.727 / mean 0.509 (số từ reimplement độc lập trên CHUM-012 processed; production run gốc trên raw NIfTI đo mean 0.5455 — khác đầu vào); frac≥0.5 = 100%
- pred_mean trong lesion 0.513 / ngoài 0.020; **Dice vs GT gaussian@0.5 = 0.747**
- Pipeline patched: triage 0.80; component lớn nhất 6647/410256 vox (1.6%); không cờ UNIFORM_OUTPUT_FAILURE

**Overfit smoke test (crop CŨ — cần chạy lại):** passed=true; MDA-174 lesion_dice 0.908; CHUM-056 lesion_dice 0.934; không cờ heatmap.

**Diagnostic độ sâu GTV (149 ca):** top-narrower 145/149 (97%); GTV inferior-most dưới body-top (mm): min 56, p50 202, p90 252, p95 271, p99 288, **max 311**; GTV superior-most 20–216mm.

**Thống kê crop (782 ca):** z_extent 360mm nhất quán; ví dụ CHUM-012 375→361 (55%), CHUP-000 852→361 (30%), CHUV-001 507→361 (21%), MDA-001 1017→361 (29%); **780/782 giữ 100% GTV**.

**Ca đặc biệt / lỗi:**
- MDA-146: GTV retained **98.12%** (body_z_range [2,281], span 279mm; crop z [2,282] bị chặn bởi sàn body-bottom, KHÔNG bởi cửa sổ 360mm).
- MDA-210: GTV retained **99.65%** (cùng nguyên nhân FOV).

---

## 8. Những việc cần làm tiếp (theo ưu tiên)

### Priority 1 — Overfit smoke test trên dữ liệu MỚI
- **Mục tiêu:** xác nhận pipeline train học được trên crop mới trước khi tốn compute train đầy đủ.
- **Cách làm:** `scripts/overfit_model0.py` trên 2 ca (vd MDA-174 + CHUM-056), overfit vài trăm step.
- **Tiêu chí hoàn thành:** lesion Dice > 0.8 **và** histogram heatmap bimodal (bg→~0, lesion→cao); không cờ UNIFORM_OUTPUT_FAILURE.

### Priority 2 — Train baseline đầy đủ
- **Mục tiêu:** có checkpoint Model 0 hợp lệ trên toàn train set với crop tái lập được.
- **Cách làm:** train với `configs/model0_v2.yaml` (w_heatmap 1.0 / w_lesion 1.0 / w_triage 0.0; Focal L1; lesion_weight 10; focal_alpha 0.75; norm instance). **Cần GPU** — Tesla P40 hiện KHÔNG truy cập được từ sandbox; phải chạy trên host hoặc bật GPU passthrough.
- **Tiêu chí hoàn thành:** val loss hội tụ; heatmap val tập trung ở tổn thương (không tràn nền); checkpoint lưu kèm config.

### Priority 3 — Đánh giá + hoàn thiện guardrail
- **Mục tiêu:** đo hiệu năng khách quan + đóng các finding còn lại.
- **Cách làm:** (a) viết `scripts/eval_model0_processed.py` (Task 9) — eval val, **chọn threshold trên val, không phải test**; (b) wire M1 (`quality_check_heatmap` vào `infer()`); (c) sửa M2 (thống nhất norm ở head).
- **Tiêu chí hoàn thành:** báo cáo Dice/Recall/FP trên val; guardrail hoạt động; head dùng InstanceNorm nhất quán backbone.

---

## 9. Những điều AI mới KHÔNG được giả định

- **Không suy đoán khi chưa có bằng chứng số liệu.** Mọi kết luận phải đo được.
- **Không kết luận chỉ từ visualization.** Ảnh heatmap chỉ là gợi ý; luôn kiểm bằng thống kê (mean, frac≥threshold, Dice, kích thước component).
- **Không giả định crop lỗi nếu chưa đo body span.** Ví dụ 2 ca clip GTV KHÔNG do cửa sổ 360mm (body span 279mm < 360mm) mà do FOV/sàn body-bottom.
- **Không tin `pgrep`/`ps` để biết job nền còn sống.** Mỗi cell ở sandbox namespace riêng → không thấy tiến trình job. Chỉ dùng `host.exec_peek(exec_id)` + mtime của log. Lưu ý: note "kernel SIGKILL/restart" trong stderr của một cell kiểm tra là về kernel REPL của cell đó, KHÔNG phải về exec nền đang nhắm tới.
- **Không chạy chồng nhiều job preprocess.** Scan toàn thân khi resample 1mm tạo mảng tạm lớn; nhiều worker/nhiều job cùng lúc → OOM. Chạy MỘT job; giảm worker cho các ca lớn.
- **Luôn phân biệt 3 tầng:** preprocessing (crop/HU/normalize) ≠ model (kiến trúc/trọng số) ≠ visualization (colormap/overlay). Bug có thể ở bất kỳ tầng nào.
- **Chỉ làm việc ở phạm vi Model 0 baseline.** KHÔNG đề xuất Swin UNETR, ViT, Knowledge Distillation, Domain Adaptation, Representation Learning, MoE, Diffusion, Foundation Model, hay redesign lớn — đó là Stage 2+.
- **Ưu tiên khi review:** Scientific correctness > Clinical usefulness > Robustness > Reproducibility > Performance. Đúng đắn trước, tối ưu sau.
- **Giữ reproducibility.** Config-driven; không refactor không cần thiết; giải thích lý do mọi thay đổi.

---

## 10. Trạng thái hiện tại

Chúng ta đang ở **cuối bước chuẩn bị dữ liệu** của pipeline Model 0. **Đã CHẮC CHẮN:** 4 bug chặn việc train lại baseline (C1 load checkpoint, C2 double sigmoid, H2 HU window, H1 crop-by-mask) đều đã sửa và xác minh bằng số liệu; toàn bộ **782/782 ca đã được tái tạo dữ liệu processed** với crop CT-only + z-window 360mm (tái lập được ở inference), targets đã tạo đầy đủ, chỉ 2 ca mất <2% GTV do FOV. Trên CHUM-012 (processed) model localize đúng tổn thương (Dice 0.747), background về ~0.02 — nghĩa là trọng số đã học có ý nghĩa và pipeline inference giờ đúng.

**CHƯA CHẮC CHẮN:** pipeline train còn học tốt trên crop mới hay không (overfit test cũ chạy trên crop cũ); hiệu năng thực trên val/test toàn bộ; mức false positive; domain shift giữa 8 trung tâm.

**Việc ĐẦU TIÊN AI mới cần làm:** chạy **overfit smoke test** (`scripts/overfit_model0.py`) trên 2 ca của dữ liệu vừa tái tạo, với cổng **Dice > 0.8** và histogram bimodal. Nếu qua → tiến hành train baseline đầy đủ (`configs/model0_v2.yaml`, cần GPU). Nếu không qua → debug pipeline train/target trước khi train đầy đủ.

**Lưu ý hạ tầng:** GPU Tesla P40 KHÔNG truy cập được từ sandbox (mọi thứ đã chạy trên CPU). Train đầy đủ cần chạy trên host hoặc bật GPU passthrough. Dữ liệu tại `/data/lowngworkspace/` (grant rw); ổ `/data` từng bị unmount một lần — nếu mất truy cập, re-grant qua host access.
