---

# BẢN ĐẶC TẢ NGHIÊN CỨU (RESEARCH BLUEPRINT)
**Dự án:** CT2MAP-HN v2.0
**Định hướng công bố:** Hội nghị chuyên ngành (VD: MICCAI)
**Chủ đề:** Học biểu diễn chuyển hóa bất biến theo miền từ CT đa phương thức thông qua chưng cất tri thức có hướng dẫn từ PET (Domain-Invariant Metabolic Representation Learning from Multi-Protocol CT via PET-Guided Distillation).

---

## I. Giả thuyết Khoa học (Research Hypothesis)

Thông tin chuyển hóa không nên và không thể được ánh xạ trực tiếp từ đặc trưng hình thái của CT. Thay vào đó, mô hình cần học một **biểu diễn chuyển hóa tiềm ẩn (Metabolic Representation)** được ràng buộc bởi cấu trúc giải phẫu, bất biến trước các phương pháp thu nhận ảnh (Non-contrast, Contrast-enhanced, Planning CT), và được giám sát bởi tín hiệu PET trong quá trình huấn luyện.

---

## II. Phương pháp Tiếp cận Cốt lõi

Phương pháp khắc phục nhược điểm của việc sinh PET giả (như CPDM) và phương pháp nội suy trực tiếp từ CT sang bản đồ nhiệt (Heatmap).

### 1. Phân tách và Dung hợp Không gian Ẩn (Cross-Representation Fusion)

Dữ liệu đầu vào thực tế (Contrast CT, Planning CT) chứa các thông tin giao thoa giữa giải phẫu và chuyển hóa (ví dụ: mức độ ngấm thuốc, hoại tử, xâm lấn mạch). Do đó, không gian ẩn không được phân tách cứng nhắc thành $Z = [Z_{anatomy} \parallel Z_{metabolism}]$.

* **Cơ chế:** Sử dụng chung một Encoder (Swin UNETR) trích xuất đặc trưng lâm sàng tổng quát, sau đó phân nhánh thành hai "Chuyên gia" (Mixture of Experts):
* *Anatomy Expert:* Học vị trí, hình dạng, ngữ cảnh mô mềm, hạch, u.
* *Metabolism Expert:* Học mức độ tương quan với vùng tăng hấp thu FDG.


* **Dung hợp (Cross-attention):** Hai nhánh trao đổi đặc trưng liên tục. Ví dụ: Sự tồn tại của "hạch" (Anatomy) cung cấp ngữ cảnh không gian để đánh giá "nguy cơ chuyển hóa của hạch đó" (Metabolism).

### 2. Biểu diễn Bất biến theo Miền (Domain-invariant Representation)

Dữ liệu CT có sự sai biệt lớn về phân bố Hounsfield Unit (HU) do giao thức chụp (Planning CT có nhiễu, mask cố định; Contrast CT có thuốc cản quang).

* **Giải pháp:** Xây dựng một không gian ẩn bổ sung (Acquisition/Domain Representation) để hấp thụ các sai biệt về thiết bị, loại CT. Từ đó, *Metabolic Representation* được cách ly khỏi nhiễu miền, chỉ tập trung vào tín hiệu sinh học cốt lõi của khối u.

---

## III. Kiến trúc Hệ thống (System Architecture)

### 1. Sơ đồ Luồng dữ liệu (Data Flow)

Mô hình hoạt động theo cơ chế Teacher-Student trong giai đoạn huấn luyện, và suy luận độc lập trên CT trong thực tế.

```text
[Giai đoạn Huấn luyện]
PET Input ──► PET Encoder ──► Metabolic Teacher ──► Metabolic Latent Space (Target)
                                                         ▲
                                                         │ (Domain-aware Distillation)
[Giai đoạn Suy luận & Huấn luyện Student]                │
Multi-domain CT (Non-contrast / Contrast / Planning)     │
   │                                                     │
   ▼                                                     │
Shared Encoder (Swin UNETR)                              │
   │                                                     │
   ├────────► Domain/Acquisition Branch                  │
   │                                                     │
   ├────────► Anatomy Expert Branch ◄──────┐             │
   │                                       │(Cross       │
   └────────► Metabolism Expert Branch ────┴───(Fusion)──┘
                                           │
   ┌───────────────────────────────────────┴──────────────────────────────────────┐
   ▼                                       ▼                                      ▼
Metabolic Heatmap                      Triage Score                        Uncertainty Score
(Vị trí nguy cơ chuyển hóa cao)        (Phân luồng ưu tiên PET/CT)         (Độ tin cậy của dự đoán)

```

### 2. Tri thức Miền & Graph Contrastive Learning

Khắc phục hạn chế của đối chiếu cấp độ thể tích (Volume-level contrastive) bằng **Graph Contrastive Learning** dựa trên tri thức giải phẫu:

* **Cấu trúc Graph:** Node (Vùng quan tâm - ROI), Edge (Khoảng cách không gian + Mối quan hệ giải phẫu).
* **Mục đích:** Buộc mô hình học mối quan hệ tương quan giữa *u nguyên phát (primary tumor)*, *hạch bạch huyết (lymph node)*, và *mạch máu (vessel)* – bản chất thực sự của sự phân bố tín hiệu PET.

---

## IV. Hàm Mục tiêu Đa nhiệm (Multi-task Objective Functions)

Quá trình tối ưu hóa kết hợp 4 nhóm Loss function để đảm bảo không gian ẩn mang ý nghĩa lâm sàng:

**1. Output Distillation (Heatmap Supervision):**
Bám sát bản đồ nhiệt mục tiêu từ PET hoặc nhãn chuẩn.


$$\mathcal{L}_{out} = |H_s - H_t|$$

**2. Feature-level Knowledge Distillation (KD):**
Học đặc trưng ở nhiều tầng (Layer 1 đến Layer 4) thay vì chỉ học đầu ra. Đáng chú ý, đây là **Domain-aware KD**, không sử dụng Uncertainty làm trọng số (do sự khác biệt sinh học giữa viêm và u độc không thể giải quyết bằng độ bất định). Sự chưng cất được hướng dẫn bởi tri thức giải phẫu (Anatomy-guided).


$$\mathcal{L}_{feat} = \sum_l |F^l_s - F^l_t|$$

**3. Contrastive Loss (Graph-guided):**
Đảm bảo các ca bệnh có kiểu hình chuyển hóa tương đồng sẽ nằm gần nhau trong không gian ẩn, và ngược lại.


$$\mathcal{L}_{contrast}$$

**4. Uncertainty Calibration Loss:**
Tối ưu hóa độ tin cậy của dự đoán, phục vụ trực tiếp cho tính an toàn trong lâm sàng.


$$\mathcal{L}_{uncertainty}$$

**Tổng hợp Hàm Mục tiêu:**


$$\mathcal{L} = \lambda_1 \mathcal{L}_{heatmap} + \lambda_2 \mathcal{L}_{KD} + \lambda_3 \mathcal{L}_{contrast} + \lambda_4 \mathcal{L}_{uncertainty}$$

---

## V. Lộ trình Triển khai và Đánh giá (Ablation Strategy)

Để chứng minh tính hiệu quả của các thành phần (modules) và tránh rủi ro "xếp chồng kỹ thuật" (stacking modules) thiếu cơ sở, hệ thống được thiết kế và đánh giá qua 4 cấp độ mô hình chặt chẽ.

### Bảng 1: Phân cấp Cấp độ Mô hình Nghiên cứu (Nền tảng Đánh giá)

| Cấp độ | Tên Mô hình & Định hướng | Kiến trúc Kỹ thuật Tương ứng | Mục đích Khoa học & Đánh giá Thử nghiệm |
| :--- | :--- | :--- | :--- |
| **Model 0** | **Research Baseline** *(Mốc cơ sở tối thiểu)* | **nnU-Net** ($CT \rightarrow Heatmap$) | Kiểm tra tính đúng đắn của pipeline tiền xử lý dữ liệu, debug hệ thống và thiết lập mức hiệu năng cơ sở tuyệt đối (không có mô hình nào được phép thấp hơn). |
| **Model 1** | **Strong Baseline** *(Đối chuẩn Kiến trúc)* | **Swin UNETR** (CT-only $\rightarrow$ Multi-task) | Chứng minh năng lực của kiến trúc Transformer 3D so với CNN thuần túy (Model 0) trong việc nắm bắt ngữ cảnh không gian xa của giải phẫu đầu cổ. |
| **Model 2** | **Strong Teacher-Student Baseline** *(Mục tiêu VISEF)* | **Swin UNETR + PET Distillation** ($CT + KD_{output/feature}$) | Đánh giá giá trị thực sự của Knowledge Distillation tiêu chuẩn từ PET sang CT. Trả lời câu hỏi: *"Chỉ dùng chưng cất đặc trưng thông thường đã đủ giải quyết bài toán chưa?"* |
| **Model 3** | **CT2MAP-HN v2.0** ***(Proposed Method - Mục tiêu MICCAI)*** | **Swin UNETR + Domain-invariant Metabolic Rep (MoE) + Graph Relational KD** | **Đóng góp khoa học cốt lõi:** Chứng minh tính hiệu quả của không gian ẩn đa miền (Split Latent), cơ chế tương tác chéo (Cross-attention) và chưng cất dựa trên đồ thị quan hệ thay vì ép giá trị voxel thô. |

---

### Bảng 2: Lộ trình Triển khai Kỹ thuật Chi tiết (Phase-based Roadmap)

| Giai đoạn | Nhiệm vụ Kỹ thuật Trọng tâm | Đầu ra Kỳ vọng (Deliverables) |
| :--- | :--- | :--- |
| **Giai đoạn 1** *(Thiết lập Baseline)* | **Tái tạo Model 0 & Model 1:** • Xây dựng pipeline chuẩn hóa dữ liệu đa phương thức (Contrast, Non-contrast, Planning CT). • Huấn luyện nnU-Net và Swin UNETR (CT-only). | • Pipeline dữ liệu hoạt động 100%. • Metrics cơ sở (Dice, AUROC) của mạng CNN và Transformer 3D. |
| **Giai đoạn 2** *(Huấn luyện Teacher & Standard KD)* | **Phát triển Model 2:** • Pretraining Teacher: Huấn luyện PET Teacher để trích xuất biểu diễn chuyển hóa ổn định. • Triển khai chưng cất tri thức (KD) tiêu chuẩn sang CT Student. | • Bộ tệp trọng số (weights) của Teacher chuẩn bị cho bước nâng cao. • **Model 2 hoạt động ổn định (Đóng gói làm bản báo cáo dự thi bước đầu).** |
| **Giai đoạn 3** *(Representation Learning)* | **Tái cấu trúc Encoder cho Model 3:** • Chuyển đổi sang kiến trúc Mixture of Experts (MoE). • Tách không gian ẩn thành các nhánh: Anatomy, Metabolism, Acquisition. • Thiết lập cơ chế Cross-Attention Fusion giữa các nhánh. | • Hoàn thiện bản vẽ kỹ thuật (network architecture) và code module cho Model 3. • Khắc phục nhiễu miền từ Planning/Contrast CT. |
| **Giai đoạn 4** *(Domain-aware Distillation)* | **Tối ưu hóa Learning & Graph Relational KD (Model 3):** • Thiết lập Graph Contrastive Loss (từ ROI Node & Edge). • Ép nhánh Metabolism của CT học cấu trúc quan hệ tương đồng với Teacher PET (thay vì học giá trị tuyệt đối). | • Hoàn tất quá trình hội tụ loss của Model 3. • Xóa bỏ rủi ro chênh lệch sinh học (Iodine vs. FDG). |
| **Giai đoạn 5** *(Multi-task Output & Calibration)* | **Hoàn thiện Bộ giải mã (Decoder):** • Tinh chỉnh các head đầu ra độc lập. • Áp dụng Uncertainty Calibration Loss để chuẩn hóa xác suất. • Tối ưu đồng thời: Heatmap Quality và Triage AUROC. | • Hệ thống đầu ra định hướng quyết định lâm sàng hoàn chỉnh (Heatmap, Triage Score, Uncertainty Score). |
| **Giai đoạn 6** *(Ablation Study & Evaluation)* | **Đánh giá So sánh Chéo (Bắt buộc cho Bài báo Khoa học):** • Thực hiện kiểm chứng bóc tách tuần tự: Model 1 $\rightarrow$ Thêm Dual Latent $\rightarrow$ Thêm Cross Fusion $\rightarrow$ Thêm Graph KD $\rightarrow$ Thêm Uncertainty (Model 3). | • Số liệu định lượng (Tables/Charts) chứng minh sự đóng góp thiết yếu của từng module. • **Bản thảo bài báo khoa học (Paper Draft) hoàn chỉnh.** |

---

## VI. Đóng góp Khoa học Dự kiến (Expected Contributions)

Khung nghiên cứu này thay đổi định vị của dự án từ một "mô hình học PET giả" thành một **hệ thống phân tích nguy cơ chuyển hóa định hướng quyết định lâm sàng**, với 4 đóng góp chính:

1. **Dual-latent Representation & Mixture of Experts:** Phương pháp biểu diễn song song (Anatomy và Metabolism) có sự tương tác qua lại (Cross-attention Fusion), loại bỏ nhiễu miền (Contrast/Planning CT).
2. **Domain-aware Distillation:** Đề xuất kỹ thuật chưng cất tri thức không ép CT giống PET về mặt giá trị tuyệt đối, mà ép CT học **mẫu quan hệ phân bố (metabolic relationship)** tương đồng với PET.
3. **Graph-guided Heatmap Learning:** Ứng dụng đồ thị để mô hình hóa quan hệ không gian giải phẫu giữa khối u nguyên phát, hạch và mô mềm.
4. **Decision-oriented Outputs:** Tích hợp 3 đầu ra độc lập phục vụ toàn diện luồng công việc lâm sàng (Giải thích bằng Heatmap, Phân luồng bằng Triage Score, An toàn bằng Uncertainty Score).

---

## VII. QUẢN TRỊ RỦI RO NGHIÊN CỨU (RISK MANAGEMENT)

| Nhóm rủi ro | Mô tả vấn đề lâm sàng / Kỹ thuật | Giải pháp khắc phục (Mitigation Strategy) |
| :--- | :--- | :--- |
| **Thiếu tính giải thích của Latent Space** | Hội đồng đặt câu hỏi: "Không gian ẩn $Z$ thực chất biểu diễn thông tin gì?" | Sử dụng Attention Map, Visualization t-SNE, Latent Probing; đo lường hệ số tương quan (Correlation) giữa Latent features với chỉ số SUV hoặc Lesion mask thực tế. |
| **Hiệu suất KD thấp (Negative Transfer)** | PET Teacher không thể "dạy" được sinh viên CT do khoảng trống sinh học quá lớn, dẫn đến nhiễu hệ thống. | Ràng buộc chưng cất tri thức bằng Prior Mask (chỉ học ở các vùng Lesion/Metabolic Prior). So sánh chặt chẽ giữa `CT-only` và `CT + Feature-KD` trong Ablation. |
| **Kích thước dữ liệu hạn chế** | Dữ liệu ung thư đầu cổ đa phương thức (Multi-protocol CT) thường có cỡ mẫu nhỏ, khó huấn luyện Transformer 3D từ đầu. | Tích hợp Patch-based training, ROI sampling, Data augmentation chuyên sâu, và Self-supervised pretraining cho khối Shared Encoder. |