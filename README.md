# 🧠 CT2MAP-HN

> **Hệ thống AI suy diễn bản đồ nguy cơ chuyển hóa từ ảnh CT đầu-cổ để phân loại bệnh nhân**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![MONAI 1.3+](https://img.shields.io/badge/MONAI-1.3%2B-3cb371.svg)](https://monai.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu phần cứng](#-yêu-cầu-phần-cứng)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
- [Hướng dẫn tải dữ liệu HECKTOR](#-hướng-dẫn-tải-dữ-liệu-hecktor)
- [Hướng dẫn chạy Pipeline](#-hướng-dẫn-chạy-pipeline)
- [Stack kỹ thuật](#-stack-kỹ-thuật)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 🎯 Giới thiệu

**CT2MAP-HN** (CT-to-Metabolic-Activity-Prediction for Head & Neck) là hệ thống AI
suy diễn **bản đồ nguy cơ chuyển hóa** (metabolic risk heatmap) từ ảnh CT đầu-cổ,
phục vụ **phân loại ưu tiên bệnh nhân** (patient triage) trong ung thư đầu cổ.

### Bài toán

Trong thực tế lâm sàng, không phải cơ sở y tế nào cũng có máy PET/CT. Hệ thống này
giúp **ước lượng vùng nguy cơ chuyển hóa cao** chỉ từ ảnh CT thông thường, hỗ trợ bác sĩ
ra quyết định có nên chuyển bệnh nhân đi chụp PET/CT hay không.

### Phương pháp

| Thành phần | Chi tiết |
|---|---|
| **Baseline** | nnU-Net v2 / MONAI BasicUNet (3D voxel regression) |
| **Đề xuất** | Swin UNETR + Knowledge Distillation (Teacher-Student) |
| **Mục tiêu học** | Gaussian-smoothed lesion mask → PET-derived soft target |
| **Đánh giá độ tin cậy** | MC Dropout tại thời điểm inference |
| **Demo** | Streamlit dashboard tương tác |

> ⚠️ **Lưu ý quan trọng:** Đây **KHÔNG** phải là hệ thống sinh ảnh PET giả (synthetic PET).
> Đầu ra là **bản đồ nguy cơ liên tục** (continuous risk map) phục vụ phân loại,
> không phải ảnh PET để chẩn đoán.

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        CT2MAP-HN Pipeline                       │
├─────────┬───────────┬──────────────┬────────────┬───────────────┤
│  Data   │  Preproc  │   Training   │  Inference │     Demo      │
│  Prep   │           │              │            │               │
│         │           │  ┌────────┐  │            │               │
│ HECKTOR │  Resample │  │Teacher │  │ MC Dropout │  Streamlit    │
│ CT+PET  │  Crop     │  │(nnUNet)│  │ N passes   │  Dashboard    │
│ +Mask   │  Normalize│  └───┬────┘  │            │               │
│         │  Orient   │      │ KD    │ Risk Map   │  Visualization│
│         │           │  ┌───▼────┐  │ + Uncert.  │  + Metrics    │
│         │           │  │Student │  │            │               │
│         │           │  │(Swin)  │  │            │               │
│         │           │  └────────┘  │            │               │
└─────────┴───────────┴──────────────┴────────────┴───────────────┘
```

---

## 📁 Cấu trúc thư mục

```
CT2MAP-HN/
├── configs/                    # Cấu hình YAML
│   ├── data.yaml              # Cấu hình dữ liệu
│   ├── preprocess.yaml        # Cấu hình tiền xử lý
│   ├── baseline_nnunet.yaml   # Cấu hình model baseline
│   ├── swin_unetr.yaml       # Cấu hình Swin UNETR
│   ├── distill.yaml           # Cấu hình Knowledge Distillation
│   ├── uncertainty.yaml       # Cấu hình MC Dropout
│   └── demo.yaml             # Cấu hình Streamlit demo
│
├── ct2map/                    # Package chính
│   ├── __init__.py
│   ├── data/                  # Data loading & transforms
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch Dataset classes
│   │   ├── transforms.py      # MONAI transforms pipeline
│   │   └── manifest.py        # Data manifest generation
│   │
│   ├── models/                # Kiến trúc mô hình
│   │   ├── __init__.py
│   │   ├── baseline.py        # nnU-Net / BasicUNet
│   │   ├── swin_unetr.py     # Swin UNETR wrapper
│   │   └── distillation.py   # Teacher-Student framework
│   │
│   ├── training/              # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Loss functions
│   │   └── schedulers.py     # LR schedulers
│   │
│   ├── inference/             # Inference pipeline
│   │   ├── __init__.py
│   │   ├── predictor.py       # Single-pass prediction
│   │   └── uncertainty.py     # MC Dropout uncertainty
│   │
│   ├── evaluation/            # Metrics & evaluation
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── utils/                 # Tiện ích
│       ├── __init__.py
│       ├── config.py          # YAML config loader
│       ├── logging.py         # Logging setup
│       ├── seed.py            # Reproducibility
│       └── io.py              # File I/O helpers
│
├── scripts/                   # Entry-point scripts
│   ├── preprocess.py          # Tiền xử lý dữ liệu
│   ├── train_baseline.py      # Huấn luyện baseline
│   ├── train_swin.py          # Huấn luyện Swin UNETR
│   ├── distill.py             # Knowledge Distillation
│   ├── predict.py             # Inference
│   └── evaluate.py            # Đánh giá
│
├── demo/                      # Streamlit demo app
│   ├── app.py
│   └── components/
│
├── tests/                     # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_transforms.py
│
├── notebooks/                 # Jupyter notebooks (EDA, visualization)
│
├── data/                      # Dữ liệu (không track trong git)
│   ├── raw/                   # Dữ liệu gốc HECKTOR
│   ├── interim/               # Dữ liệu trung gian
│   └── processed/             # Dữ liệu đã xử lý
│
├── outputs/                   # Kết quả training & inference
│
├── .gitignore
├── Dockerfile
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 💻 Yêu cầu phần cứng

| Thành phần | Khuyến nghị |
|---|---|
| **GPU** | 4× NVIDIA RTX A6000 (48GB VRAM mỗi card) |
| **RAM** | 256GB |
| **CPU** | AMD EPYC 7743P hoặc tương đương |
| **Storage** | ≥500GB SSD (dữ liệu HECKTOR + checkpoints) |
| **OS** | Windows 11 + WSL2 / Ubuntu 22.04 |

> 💡 **Ghi chú:** Có thể chạy trên GPU nhỏ hơn (24GB) bằng cách giảm batch size
> và bật `use_checkpoint: true` trong config Swin UNETR.

---

## 🔧 Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-org/CT2MAP-HN.git
cd CT2MAP-HN
```

### 2. Tạo môi trường Conda

```bash
# Tạo environment mới
conda create -n ct2map python=3.10 -y
conda activate ct2map

# Cài đặt PyTorch với CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Cài đặt dependencies

```bash
# Cài đặt tất cả packages
pip install -r requirements.txt

# Cài đặt project ở chế độ development
pip install -e ".[dev]"
```

### 4. Xác minh cài đặt

```bash
python -c "
import torch
import monai
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'MONAI: {monai.__version__}')
"
```

### 5. Sử dụng Docker (tùy chọn)

```bash
# Build Docker image
docker build -t ct2map-hn .

# Chạy container với GPU
docker run --gpus all -p 8501:8501 -v $(pwd)/data:/app/data ct2map-hn
```

---

## 📦 Hướng dẫn tải dữ liệu HECKTOR

### Bộ dữ liệu HECKTOR (HEad and neCK TumOR)

Dự án sử dụng dữ liệu từ **HECKTOR Challenge 2021/2022**, bao gồm:
- Ảnh **CT** đầu-cổ
- Ảnh **PET** (FDG-PET)
- **Mask phân đoạn** khối u (GTVp, GTVn)

### Cách tải

1. **Đăng ký tài khoản** tại [AICROWD HECKTOR](https://www.aicrowd.com/challenges/miccai-2022-hecktor)
   hoặc tải từ [Grand Challenge](https://hecktor.grand-challenge.org/)

2. **Tải dữ liệu:**
   ```bash
   # Tạo thư mục
   mkdir -p data/raw/hecktor

   # Giải nén dữ liệu vào thư mục
   # Cấu trúc mong đợi:
   # data/raw/hecktor/
   # ├── hecktor2022_training/
   # │   ├── <PatientID>/
   # │   │   ├── <PatientID>__CT.nii.gz
   # │   │   ├── <PatientID>__PT.nii.gz
   # │   │   └── <PatientID>__gtv.nii.gz
   # │   └── ...
   # └── hecktor2022_testing/
   #     └── ...
   ```

3. **Xác minh dữ liệu:**
   ```bash
   python scripts/preprocess.py --config configs/data.yaml --verify-only
   ```

---

## 🚀 Hướng dẫn chạy Pipeline

### Bước 1: Tiền xử lý dữ liệu

```bash
# Tiền xử lý: resample, crop, normalize
python scripts/preprocess.py \
    --config configs/preprocess.yaml \
    --data-config configs/data.yaml
```

### Bước 2: Huấn luyện Baseline (Teacher)

```bash
# Huấn luyện mô hình baseline (nnU-Net / BasicUNet)
python scripts/train_baseline.py \
    --config configs/baseline_nnunet.yaml

# Theo dõi training bằng TensorBoard
tensorboard --logdir outputs/baseline/tb_logs
```

### Bước 3: Huấn luyện Swin UNETR (Student)

```bash
# Huấn luyện Swin UNETR với Knowledge Distillation
python scripts/train_swin.py \
    --config configs/swin_unetr.yaml \
    --distill-config configs/distill.yaml
```

### Bước 4: Inference với ước lượng độ bất định

```bash
# Chạy inference với MC Dropout
python scripts/predict.py \
    --config configs/swin_unetr.yaml \
    --uncertainty-config configs/uncertainty.yaml \
    --input-dir data/processed/test \
    --output-dir outputs/predictions
```

### Bước 5: Đánh giá

```bash
# Tính metrics đánh giá
python scripts/evaluate.py \
    --predictions outputs/predictions \
    --ground-truth data/processed/test
```

### Bước 6: Demo tương tác

```bash
# Khởi chạy Streamlit dashboard
streamlit run demo/app.py -- --config configs/demo.yaml
```

---

## 🛠️ Stack kỹ thuật

| Thành phần | Công nghệ |
|---|---|
| **Framework DL** | PyTorch 2.1+, MONAI 1.3+ |
| **Kiến trúc mô hình** | Swin UNETR, nnU-Net v2, BasicUNet |
| **Xử lý ảnh y tế** | SimpleITK, nibabel |
| **Cấu hình** | PyYAML |
| **Theo dõi thực nghiệm** | TensorBoard |
| **Demo** | Streamlit |
| **Môi trường** | Conda, Docker |
| **GPU** | NVIDIA RTX A6000 × 4 (CUDA 12.1) |
| **Testing** | pytest |

---

## ⚖️ Disclaimer

> **⚠️ CHỈ DÀNH CHO MỤC ĐÍCH NGHIÊN CỨU (Research-Use Only)**
>
> Hệ thống CT2MAP-HN được phát triển **hoàn toàn cho mục đích nghiên cứu học thuật**.
> Sản phẩm này:
>
> - **KHÔNG** phải là thiết bị y tế được cấp phép
> - **KHÔNG** được FDA/Bộ Y tế phê duyệt cho sử dụng lâm sàng
> - **KHÔNG** thay thế chẩn đoán của bác sĩ chuyên khoa
> - **KHÔNG** nên được sử dụng để đưa ra quyết định điều trị
>
> Kết quả từ hệ thống chỉ mang tính tham khảo và cần được xác nhận bởi
> chuyên gia y tế có trình độ. Nhóm phát triển **không chịu trách nhiệm**
> cho bất kỳ hậu quả nào phát sinh từ việc sử dụng sai mục đích.

---

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
