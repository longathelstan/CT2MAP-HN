# 🏥 Lộ trình triển khai dự án CT2MAP-HN (12 Tuần)

**CT2MAP-HN** là hệ thống AI suy diễn bản đồ nguy cơ chuyển hóa (metabolic heatmap) từ ảnh CT đầu cổ, phục vụ **phân luồng bệnh nhân** (triage) — KHÔNG phải sinh ảnh PET giả. Dự án dành cho cuộc thi VISEF.

> **Framing quan trọng:** Hệ thống này hỗ trợ sàng lọc, KHÔNG thay thế PET/CT. Tuyệt đối tránh gọi output là "PET giả" hay "synthetic PET".

---

## 🔵 Giai đoạn A — Dựng nền (Tuần 1–2) — [HOÀN THÀNH HÔM NAY]
- **Mục tiêu:** Setup codebase, cấu hình, xử lý dữ liệu và tạo data loader.
- **Trạng thái:** Toàn bộ ~60 files source code nền tảng (Foundation) đã được Antigravity generate thành công tại `g:\Clone\CT2MAP-HN`.
- **Việc cần làm tiếp theo của bạn (trên Server):**
  1. Tải dữ liệu HECKTOR raw vào thư mục `data/raw/hecktor`.
  2. Chạy `python scripts/build_manifest.py --config configs/data.yaml`
  3. Chạy `python scripts/prepare_data.py --config configs/preprocess.yaml`
  4. Chạy `python scripts/create_targets.py --config configs/data.yaml`

---

## 🟢 Giai đoạn B — Baseline (Tuần 3–5)
- **Mục tiêu:** Train và đánh giá mô hình Baseline (3D UNet) để có điểm mốc (benchmark).
- **Việc cần làm trên Server:**
  1. Chạy training: `bash scripts/train_baseline.sh`
  2. Xem kết quả trên TensorBoard.
  3. Đánh giá: Chạy inference và tính metrics cơ bản để lấy số so sánh.
- **Tiêu chuẩn vượt qua:** Baseline infer ổn trên test set, có metrics rõ ràng.

---

## 🟡 Giai đoạn C — Proposed Model & Distillation (Tuần 6–7)
- **Mục tiêu:** Train kiến trúc đề xuất (Swin UNETR) và áp dụng Knowledge Distillation từ ảnh PET.
- **Việc cần làm trên Server:**
  1. Train mạng Teacher (nhìn cả CT + PET)
  2. Train mạng Student (chỉ nhìn CT) bằng lệnh: `bash scripts/train_swin.sh --distill`
- **Tiêu chuẩn vượt qua:** Metrics của mô hình Đề xuất phải tốt hơn Baseline hoặc có khả năng giải thích (explainability) tốt hơn nhờ Distillation.

---

## 🟠 Giai đoạn D — Uncertainty + Triage (Tuần 8)
- **Mục tiêu:** Xây dựng cơ chế lọc rủi ro an toàn cho y tế.
- **Việc cần làm:**
  - Áp dụng MC Dropout (đã code sẵn trong `src/inference/infer_case.py`).
  - Phân loại bệnh nhân (Triage): Nguy cơ cao/thấp, Độ tin cậy cao/thấp.
  - Những ca có Uncertainty cao (mô hình không chắc chắn) sẽ được cảnh báo để bác sĩ đọc lại.

---

## 🔴 Giai đoạn E — Demo & Trình diễn (Tuần 9–12)
- **Mục tiêu:** Đóng gói sản phẩm, làm Dashboard tương tác và viết báo cáo/slide nộp VISEF.
- **Việc cần làm:**
  1. Chạy Streamlit: `bash scripts/launch_demo.sh`
  2. Tạo 5-10 "demo cases" đặc trưng (ca thành công, ca khó, ca có uncertainty cao).
  3. Viết Slide và chuẩn bị phần Q&A bảo vệ trước ban giám khảo.

---

## ⚠️ Lưu ý kỹ thuật cho 4x RTX A6000
- Code hiện tại đã được cấu hình tối ưu cho server của bạn: dùng DistributedDataParallel, mixed precision (`torch.cuda.amp`), batch size lớn cho Swin UNETR và Gradient Checkpointing.
- Hãy theo dõi nhiệt độ và RAM của 4 GPU khi chạy Giai đoạn B và C.
