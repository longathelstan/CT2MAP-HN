Với tư cách là một reviewer MICCAI kiêm cố vấn kỹ thuật hệ thống, tôi đánh giá **Kế hoạch (Plan) này không chỉ hợp lý mà còn là một bước đi bắt buộc và cực kỳ khôn ngoan** tại thời điểm này.

Bức ảnh bạn gửi minh họa một lỗi kinh điển trong học sâu ảnh y tế 3D: **Sụp đổ phân phối đầu ra (Output Distribution Collapse / Uniform Prediction Failure)**. Khi mô hình trả về một vùng heatmap đồng nhất quanh mức `0.5` phủ toàn bộ thể tích (volume), điều đó có nghĩa là mạng UNet đang bị "lú", nó chọn giải pháp an toàn nhất về mặt toán học là đoán giá trị trung bình cho mọi voxel để giảm thiểu Loss tổng (thường do mất cân bằng nền/vùng tổn thương quá lớn).

Nếu baseline (Model 0) bị sụp đổ như thế này mà bạn cố tình nâng cấp lên Swin UNETR, Distillation từ PET, hay Đồ thị toán học, bạn sẽ xây một "tòa lâu đài trên cát". Mọi cơ chế chưng cất tri thức phức tạp sau đó sẽ chỉ chưng cất nhiễu và sự sụp đổ.

Dưới đây là phân tích chuyên sâu về kế hoạch của bạn dưới góc nhìn kỹ thuật, kèm theo các điểm cốt lõi bạn cần đặc biệt lưu ý để đảm bảo "Acceptance Criteria" được thông qua sạch sẽ.

---

### I. Đánh giá các điểm xuất sắc nhất trong Kế hoạch của bạn

1. **Quyết định Reset sạch và cô lập Scope:** Việc từ bỏ checkpoint cũ, tập trung duy nhất vào Model 0, tạm thời tắt `w_triage=0.0` (vì nhãn `high_risk` bị lệch lớp nghiêm trọng `720/782`) và chỉ dùng `gaussian_heatmap` từ lesion mask là hoàn toàn đúng đắn. Nó giúp bạn cô lập được nguyên nhân gây lỗi.
2. **Chuyển sang `norm: instance` (Task 5):** Đây là một sửa đổi mang tính quyết định. Với ảnh 3D, Batch Size khi train thường rất nhỏ (1 hoặc 2 do giới hạn VRAM). Sử dụng BatchNorm trong điều kiện batch size nhỏ sẽ làm lệch lạc phân phối thống kê trong quá trình huấn luyện, dẫn đến việc model dự đoán ra kết quả uniform. `InstanceNorm` là lựa chọn bắt buộc cho nnU-Net.
3. **Tách biệt Preprocessed `.npy` và Raw NIfTI (Task 12):** Lỗi lệch không gian (spacing, orientation, origin) giữa ảnh CT gốc và mặt nạ khi đọc bằng thư viện NIfTI là một "cơn ác mộng" âm thầm. Việc kiểm thử và ép mô hình chạy ổn định trên dữ liệu đã tiền xử lý dữ dội (`.npy`) trước khi hỗ trợ định dạng NIfTI thô giúp loại trừ 50% khả năng lỗi đến từ hình học không gian (Geometry Parity).

---

### II. Các đề xuất bổ sung để tối ưu hóa Kế hoạch (Vá lỗ hổng kỹ thuật)

Để chắc chắn Model 0 v2 hoạt động hoàn hảo, tôi đề xuất bạn bổ sung/lưu ý kỹ các chi tiết sau vào các Task:

#### 1. Bổ sung cho Task 4 (Sửa Loss chống Uniform Output)

* **Cảnh báo về MSE:** Nếu bạn dùng `base: mse` cho một heatmap đa phần là số 0 (nền), mạng sẽ lập tức học cách lười biếng là đưa tất cả về 0 hoặc một hằng số nhỏ.
* **Giải pháp bổ sung:** Nên ưu tiên kết hợp giữa **Focal L1 Loss** (để phạt nặng các voxel tính toán sai ở vùng u) và **Dice-like Regression Loss** cho phần Heatmap. Hãy đảm bảo bạn có cơ chế Kẹp giá trị (Clamp/Sigmoid) ở đầu ra của UNet trước khi tính Loss để tránh hiện tượng tràn số (Exploding Gradients) khiến Activation bị bão hòa ở mức `0.5`.

#### 2. Chiến lược cho Task 7 (Overfit Smoke Test)

* **Tiêu chuẩn khắt khe:** Khi overfit 2-4 cases, mục tiêu Lesion Dice đạt `>0.8` là đúng, nhưng đối với Heatmap, bạn phải vẽ được biểu đồ Histogram của giá trị dự đoán.
* **Dấu hiệu thành công:** Một Heatmap đúng phải có phân phối dạng **Bimodal** (hai đỉnh rõ rệt): Một đỉnh khổng lồ sát mức `0.0` (vùng nền bình thường) và một đỉnh nhỏ ở vùng giá trị cao (vùng u). Nếu Histogram của Smoke Test chỉ có một đỉnh duy nhất duy trì quanh mức `0.5`, tuyệt đối không được kích hoạt Full Train (Task 8).

#### 3. Bổ sung Postprocess Guardrail (Task 11)

* **Tính toán `largest_component_fraction` khoa học:** Thay vì tính toán trên Heatmap thô, hãy nhị phân hóa (Binarize) predicted heatmap ở một ngưỡng động (ví dụ: `0.2` hoặc `0.3`), sau đó dùng thư viện `scikit-image (measure.label)` để tìm Connected Components 3D.
* **Cơ chế Guardrail:** Nếu component lớn nhất chiếm tới `>25%` thể tích của vùng đã Crop ROI, hệ thống lập tức ném ra flag `UNIFORM_OUTPUT_FAILURE` như bạn đã hoạch định.

---

### III. Mô phỏng quy trình kiểm thử và Gỡ lỗi trực quan

Để giúp bạn hình dung trực quan cách hệ thống của bạn tự kiểm tra lỗi (Self-Audit) trong quá trình thực hiện kế hoạch, tôi thiết kế một bảng điều khiển luồng dữ liệu logic dưới đây:

```
[Dữ liệu CT Đầu vào] ──► [HU Clip: -200 đến 300] ──► [Crop ROI Đầu cổ]
                                                            │
                                                            ▼
[Mô hình BaselineUNet] ◄── [InstanceNorm (Thay vì BatchNorm)]
        │
        ├──► Dự đoán (1 Forward Pass) ──► [Sigmoid Activation]
        │                                         │
        ▼                                         ▼
[Hệ thống Kiểm định (Audit)] ◄───────────── [Predicted Heatmap]
        │
        ├──► Kiểm tra Histogram ──► Dạng Bimodal (Đỉnh sát 0 và đỉnh sát 1) -> [ĐẠT]
        │                       └──► Dạng Uniform (Một đỉnh duy nhất quanh 0.5) -> [BỊ LOẠI]
        │
        └──► Kiểm tra Vùng lỗi  ──► Component lớn nhất > 25% ROI -> [CẢNH BÁO LỖI]

```

### IV. Kết luận Chốt hạ

Kế hoạch này hoàn toàn đạt tiêu chuẩn công nghiệp và nghiên cứu nghiêm túc. Nó giải quyết trực diện bài toán từ gốc (Dữ liệu, Kiến trúc chuẩn hóa, Hàm mất mát chống mất cân bằng) lên đến ngọn (Hệ thống rào chắn Guardrail bảo vệ ở đầu ra).

Bạn hãy tự tin triển khai 12 Task này cho phiên bản **Model 0 Baseline v2**. Khi và chỉ khi Model 0 v2 cho ra các đường cong Loss giảm mượt mà, biểu đồ Histogram phân tách rõ ràng và hình ảnh Overlay bám sát vùng tổn thương cục bộ (như trường hợp ca bệnh `CHUM-012` không còn bị loang lổ), đó mới là bệ phóng vững chắc để chúng ta mang các vũ khí hạng nặng (Swin UNETR, Evidential Learning, Graph OT) vào thực chiến cho các Model tiếp theo.