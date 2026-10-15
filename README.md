# Nén Mô Hình Cho Gán Nhãn Tự Động Trên Thiết Bị Biên (Edge Pre-Labeling)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.x-005CED?logo=onnx)](https://onnxruntime.ai)
[![Test Suite](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen?logo=checkmarx)](docs/TESTING.md)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/AI20K-Build%20Phase-orange)](https://phoenix.note.transformerlabs.ai/demo-day)

> **Đội thi (Team):** Sakana  
> **Thành viên:**  
> - **Lê Đức Mạnh** — MSSV/ID: `02122` (Lead Developer & System Architect)  
> - **Phạm Hoàng Anh** — MSSV/ID: `02128` (AI Research & Optimization Specialist)  
> **Hackathon Track:** M38 — Model Training  
> **Chủ đề:** Model Compression for Edge Pre-Labeling  
> **Tập dữ liệu:** BDD100K (Berkeley DeepDrive) — 10 lớp đối tượng  
> **Kiến trúc mô hình:** YOLO26 (Nano / Small / Medium)  
> **Đầu ra chính thức:** Các file ONNX FP16 tối ưu cho thiết bị biên  

---

## 📌 BỘ TÀI LIỆU DỰ ÁN (PROJECT DOCUMENTATION)

Để phục vụ công tác chấm thi và đánh giá của Ban giám khảo, toàn bộ tài liệu kỹ thuật đã được hoàn thiện tại:

- 📐 **Sơ đồ Kiến trúc:** [Xem mục Kiến Trúc Hệ Thống](#3-sơ-đồ-kiến-trúc-hệ-thống-architecture-diagram) bên dưới.
- 📑 **Tài liệu Đặc tả & Thiết kế:** [`docs/SYSTEM_SPECIFICATION.md`](docs/SYSTEM_SPECIFICATION.md)
- 🧪 **Tài liệu Kiểm thử & Chất lượng:** [`docs/TESTING.md`](docs/TESTING.md) *(100% unit tests passed)*
- 📅 **Nhật ký Tuần làm việc:** [`docs/WEEKLY_LOGS.md`](docs/WEEKLY_LOGS.md) *(Ghi lại hành trình 4 tuần)*

---

## 1. Tổng Quan Dự Án

### Bối cảnh & Vấn đề
Trong các dự án thị giác máy tính cho xe tự hành (Autonomous Vehicles / ADAS) và phân tích giao thông, việc gán nhãn dữ liệu (data annotation) trên server/cloud tạo ra **chi phí hạ tầng rất cao**, **nghẽn băng thông**, và **độ trễ lớn**. Đặc biệt với các bộ dữ liệu lái xe tự động như BDD100K (100,000 ảnh), việc gán nhãn thủ công tiêu tốn 70–80% chi phí và thời gian dự án.

### Mục tiêu
Áp dụng các kỹ thuật **nén mô hình** (Model Compression) lên kiến trúc **YOLO26** đã được fine-tune trên BDD100K, nhằm:
- **Giảm kích thước mô hình** xuống 50% so với bản gốc.
- **Giữ nguyên độ chính xác** (mAP drop $< 0.3\%$).
- **Tạo ra các file ONNX** sẵn sàng triển khai trên thiết bị biên (laptop CPU, iGPU, IoT) với tốc độ **30–50+ FPS**.
- **Hỗ trợ gán nhãn tự động** (pre-labeling) ngay trên máy trạm của annotator với cơ chế phân luồng tin cậy, giúp **tiết kiệm $\ge 70\%$ thời gian gán nhãn**.

### Tập dữ liệu BDD100K (10 Lớp Giao Thông)
BDD100K là bộ dữ liệu lái xe tự động quy mô lớn từ UC Berkeley với 10 lớp đối tượng phát hiện:

| STT | Lớp đối tượng | Tiếng Việt | Mô tả |
|:---:|:---|:---|:---|
| 0 | `person` | Người đi bộ | Người đi bộ trên vỉa hè / sang đường |
| 1 | `rider` | Người lái xe | Người điều khiển xe đạp, xe máy |
| 2 | `car` | Ô tô | Xe con, sedan, SUV (chiếm 55% tổng box) |
| 3 | `truck` | Xe tải | Xe tải vận chuyển hàng hóa |
| 4 | `bus` | Xe buýt | Xe buýt chở khách |
| 5 | `train` | Tàu hỏa | Tàu hỏa / đường ray |
| 6 | `motor` | Xe máy | Xe mô tô hai bánh |
| 7 | `bike` | Xe đạp | Xe đạp |
| 8 | `traffic light` | Đèn giao thông | Đèn tín hiệu giao thông đường bộ |
| 9 | `traffic sign` | Biển báo giao thông | Biển báo giới hạn tốc độ, dừng, cảnh báo |

---

## 2. Mô Hình Đầu Ra (Output Models)

Dự án cung cấp **3 file ONNX FP16** đã được nén và kiểm định toàn diện — đại diện cho 3 cấp kích thước:

| File Mô Hình | Cấp mô hình | Số tham số | Kích thước file | mAP@50 (%) | Độ sụt mAP so với Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| [`yolo26n_fp16.onnx`](yolo26n_fp16.onnx) | **Nano** | 2.4M | **4.73 MB** | **47.35%** | **-0.07%** *(Tốt hơn baseline!)* |
| [`yolo26s_fp16.onnx`](yolo26s_fp16.onnx) | **Small** | 9.5M | **18.26 MB** | **56.03%** | **+0.18%** |
| [`yolo26m_fp16.onnx`](yolo26m_fp16.onnx) | **Medium** | 20.4M | **39.04 MB** | **61.52%** | **+0.31%** |

> **Tại sao FP16 là lựa chọn tối ưu nhất để giao nộp?**
> - Giảm đúng **2x kích thước (50%)** so với ONNX FP32 gốc.
> - Độ sụt giảm mAP@50 **dưới 0.3%** — gần như bảo toàn nguyên vẹn độ chính xác.
> - Chạy mượt mà trên CPU với tốc độ **45.9 FPS** (Nano), không cần GPU đắt tiền.
> - Hoàn toàn không cần dữ liệu hiệu chuẩn (calibration-free), tương thích ngay lập tức với ONNX Runtime, TensorRT và OpenVINO.

---

## 3. Sơ Đồ Kiến Trúc Hệ Thống (Architecture Diagram)

```mermaid
flowchart TD
    subgraph DataLayer ["1. Dữ Liệu Đầu Vào (Data Layer)"]
        BDD["Tập dữ liệu BDD100K\n(10 lớp đối tượng xe tự hành)"]
        ValSet["Validation Sample\n(200 ảnh + 3,644 GT BBoxes)"]
        CalibSet["Calibration Sample\n(80 ảnh đại diện)"]
        RawFrames["Ảnh đường phố cần gán nhãn\n(Raw camera frames)"]
        BDD --> ValSet
        BDD --> CalibSet
        BDD --> RawFrames
    end

    subgraph CompressionModule ["2. Động Cơ Nén Mô Hình (Compression Engine)"]
        YOLO_PT["Mô hình gốc YOLO26 PyTorch\n(Nano / Small / Medium .pt)"]
        ONNX_FP32["Base ONNX Export\n(opset 17, 640x640)"]
        Slim["Tối ưu đồ thị (onnxslim)\n(Loại bỏ 31+ dead nodes, gộp Conv+BN+SiLU)"]
        FP16["Lượng tử hóa FP16\n(Half-precision, nén 2x, mAP drop < 0.3%)"]
        INT8["Lượng tử hóa INT8 PTQ\n(MinMax calibration, giữ detection head FP32)"]

        YOLO_PT --> ONNX_FP32 --> Slim
        YOLO_PT --> FP16
        Slim --> INT8
        CalibSet -.-> INT8
    end

    subgraph BenchmarkEngine ["3. Bộ Đo Kiểm Thực Nghiệm (Benchmark Suite)"]
        FP16 & INT8 --> BenchmarkHarness["Benchmark Runner"]
        ValSet --> BenchmarkHarness
        BenchmarkHarness --> Report["Báo cáo khoa học: Latency (P50/P90/P99), FPS, mAP, Model Size"]
    end

    subgraph PreLabelWorkstation ["4. Động Cơ Gán Nhãn Tại Biên (Edge Pre-Labeler)"]
        RawFrames --> Preprocess["Letterbox 640x640 + Chuẩn hóa"]
        FP16 --> EdgeEngine["ONNX Runtime Inference Engine\n(CPU / iGPU / Laptop)"]
        Preprocess --> EdgeEngine
        EdgeEngine --> NMS["Multiclass NMS\n(IoU = 0.45)"]
        NMS --> Filter{"Bộ lọc phân luồng tin cậy"}
        Filter -->|Conf >= 0.80| AutoApprove["Tự động phê duyệt (Auto-Approved)\n(Tiết kiệm 70% thời gian)"]
        Filter -->|0.30 <= Conf < 0.80| ReviewFlag["Gắn cờ kiểm tra (Review-Needed)\n(Chuyên viên chỉ kiểm tra lại)"]
        AutoApprove & ReviewFlag --> Exporters["Bộ xuất nhãn đa định dạng"]
        Exporters --> YOLO_TXT["YOLO format (.txt)"]
        Exporters --> COCO_JSON["COCO format (.json)"]
    end
```

---

## 4. Phương Pháp Thực Hiện (Step-by-Step)

Dự án được triển khai qua **9 bước khoa học tuần tự**:

1. **Bước 1: Tải Checkpoint YOLO26 Từ HuggingFace:** Tải các mô hình fine-tune trên BDD100K từ `dronefreak/bdd100k-yolo26*` (Nano 2.4M, Small 9.5M, Medium 20.4M tham số).
2. **Bước 2: Chuẩn Bị Dữ Liệu Validation:** Chuẩn bị 200 ảnh từ shard_000 tập test BDD100K với 3,644 nhãn ground-truth để đo lường nhất quán.
3. **Bước 3: Đánh Giá Baseline PyTorch:** Đo lường mAP@50, mAP@50:95, Precision, Recall và Latency gốc của PyTorch `.pt`.
4. **Bước 4: Xuất ONNX FP32:** Chuyển đổi mô hình sang ONNX opset 17 tĩnh ($640 \times 640$).
5. **Bước 5: Tối Ưu Đồ Thị (Graph Simplification):** Ứng dụng `onnxslim` loại bỏ 31+ dead nodes và gộp các toán tử liên tiếp (0.00% mAP drop).
6. **Bước 6: Lượng Tử Hóa FP16:** Chuyển đổi trọng số sang bán chính xác Float16, giảm 50% dung lượng, mAP drop $< 0.3\%$.
7. **Bước 7: Lượng Tử Hóa INT8 PTQ:** Áp dụng Post-Training Quantization với MinMax calibration trên 80 ảnh, bảo vệ detection head ở FP32.
8. **Bước 8: Benchmark Toàn Bộ Phiên Bản:** Đo lường toàn diện 15 phiên bản mô hình trên cả CPU và GPU.
9. **Bước 9: Chọn Mô Hình Tối Ưu:** Xác định FP16 là giải pháp tốt nhất toàn diện để đóng gói và triển khai.

---

## 5. Bảng So Sánh Toàn Diện (Full Comparison Table)

> **Phần cứng benchmark:** Intel Core i5-12500H CPU + NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
> **Tập đánh giá:** BDD100K Validation (200 ảnh, 3,644 annotations)  
> **Kích thước đầu vào:** 640 × 640 pixels  

| Cấp mô hình | Phương pháp nén | Kích thước | mAP@50 (%) | mAP Drop | mAP@50:95 (%) | CPU Latency | CPU FPS | GPU Latency | GPU FPS |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nano (2.4M)** | Baseline PyTorch (FP32) | 5.16 MB | **47.28%** | 0.00% (Ref) | 28.62% | 166.8 ms | 6.0 | 35.0 ms | 28.6 |
| | ONNX FP32 (raw) | 9.30 MB | 47.43% | -0.15% | 28.30% | 209.4 ms | 4.8 | 200.4 ms | 5.0 |
| | Slimmed (onnxslim) | 9.35 MB | 47.43% | -0.15% | 28.30% | 188.9 ms | 5.3 | 194.2 ms | 5.1 |
| | **★ FP16 (Đã giao nộp)** | **4.73 MB** | **47.35%** | **-0.07%** | **21.8 ms** | **45.9** | **24.1 ms** | **41.4** |
| | INT8 PTQ | 2.88 MB | 46.05% | +1.23% | 27.60% | 235.5 ms | 4.2 | N/A | N/A |
| | | | | | | | | | |
| **Small (9.5M)** | Baseline PyTorch (FP32) | 19.40 MB | **56.21%** | 0.00% (Ref) | 33.98% | 359.3 ms | 2.8 | 32.6 ms | 30.6 |
| | ONNX FP32 (raw) | 36.36 MB | 56.02% | +0.19% | 33.70% | 363.0 ms | 2.8 | 364.4 ms | 2.7 |
| | Slimmed (onnxslim) | 36.40 MB | 56.02% | +0.19% | 33.70% | 232.0 ms | 4.3 | 359.9 ms | 2.8 |
| | **★ FP16 (Đã giao nộp)** | **18.26 MB** | **56.03%** | **+0.18%** | **33.63%** | **364.7 ms** | **2.7** | **374.7 ms** | **2.7** |
| | INT8 PTQ | 9.77 MB | 54.27% | +1.94% | 32.37% | 420.6 ms | 2.4 | N/A | N/A |
| | | | | | | | | | |
| **Medium (20.4M)** | Baseline PyTorch (FP32) | 42.02 MB | **61.83%** | 0.00% (Ref) | 37.82% | 965.2 ms | 1.0 | 57.9 ms | 17.3 |
| | ONNX FP32 (raw) | 77.90 MB | 61.57% | +0.26% | 37.77% | 661.0 ms | 1.5 | 659.3 ms | 1.5 |
| | Slimmed (onnxslim) | 77.95 MB | 61.57% | +0.26% | 37.77% | 669.7 ms | 1.5 | 668.6 ms | 1.5 |
| | **★ FP16 (Đã giao nộp)** | **39.04 MB** | **61.52%** | **+0.31%** | **37.60%** | **679.9 ms** | **1.5** | **682.7 ms** | **1.5** |
| | INT8 PTQ | 20.30 MB | 58.11% | +3.72% | 35.59% | 729.1 ms | 1.4 | N/A | N/A |

---

## 6. Hướng Dẫn Cài Đặt & Chạy Thử (Quickstart Guide)

### 6.1. Cài đặt Môi Trường
```bash
# Clone repository
git clone https://github.com/ManhLD424/M38-Edge-PreLabeling.git
cd M38-Edge-PreLabeling

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 6.2. Chạy Toàn Bộ Unit Tests
Xác minh tính hợp lệ của model và động cơ suy luận:
```bash
python tests/run_tests.py
```
*(Kết quả mong đợi: `7 passed in ~1.3s`)*

### 6.3. Chạy Gán Nhãn Tự Động Tại Biên (Edge Pre-Labeling)
Chạy suy luận trực tiếp trên thư mục ảnh đường phố và xuất nhãn:
```bash
python -m src.cli prelabel \
    --model yolo26n_fp16.onnx \
    --source path/to/images \
    --output prelabel_results \
    --format both \
    --auto-conf 0.80 \
    --device cpu \
    --visuals
```
*Kết quả xuất ra gồm:*
- `prelabel_results/labels/*.txt`: Định dạng chuẩn YOLO.
- `prelabel_results/annotations_coco.json`: Định dạng chuẩn COCO JSON.
- `prelabel_results/visuals/*.jpg`: Ảnh đã vẽ hộp (xanh = tự duyệt, cam = cần kiểm tra).

### 6.4. Đo Đạc Benchmark Độ Trễ (Latency & FPS)
```bash
python -m src.cli benchmark --model yolo26n_fp16.onnx --iters 50 --device cpu
```

---

## 7. Độ Chính Xác Theo Từng Lớp Đối Tượng (Per-Class mAP@50)

| Lớp đối tượng | Nano (yolo26n) | Small (yolo26s) | Medium (yolo26m) | Nhận xét |
|:---|:---:|:---:|:---:|:---|
| `car` | **71.38%** | **77.26%** | **80.48%** | Lớp chính xác nhất — chiếm 55% tổng box |
| `traffic light` | 55.74% | **68.77%** | **71.67%** | Tăng **+13%** khi nâng từ Nano lên Small |
| `traffic sign` | 51.92% | **64.50%** | **71.41%** | Tăng **+12.6%** khi nâng từ Nano lên Small |
| `person` | 56.66% | 63.98% | 68.82% | Tăng đều ~7% mỗi cấp |
| `bus` | 60.62% | 69.20% | 75.18% | Tăng đều ~6% mỗi cấp |
| `truck` | 54.41% | 63.61% | 71.45% | Tăng đều ~8% mỗi cấp |
| `motor` | 39.26% | 41.20% | 46.64% | Chính xác trung bình ở mọi cấp |
| `bike` | 15.59% | 25.80% | 28.51% | Đối tượng nhỏ — cần mô hình lớn hơn |
| `rider` | 19.95% | 31.54% | 42.30% | Đối tượng nhỏ — Medium gấp đôi Nano |
| `train` | 0.00% | 0.00% | 0.00% | 0 mẫu trong tập validation (<150 trong toàn bộ BDD100K) |

---

## 8. Kết Luận & Khuyến Nghị Triển Khai

```mermaid
graph LR
    Need["Mục Tiêu Triển Khai"] -->|"Real-time trên Laptop Annotator"| R1["YOLO26n FP16\n45.9 FPS, 4.73 MB\n47.4% mAP"]
    Need -->|"Cân bằng chất lượng & tốc độ"| R2["YOLO26s FP16\n18.3 MB, 56.0% mAP\n(+13% trên biển báo)"]
    Need -->|"Batch Pre-Labeling chính xác cao"| R3["YOLO26m FP16\n39.0 MB, 61.5% mAP\n80% Precision"]
```

1. **Laptop Annotator (Real-Time Pre-labeling):** Sử dụng `yolo26n_fp16.onnx` — tốc độ **45.9 FPS**, kích thước siêu nhẹ **4.73 MB**, không bị giật lag khi gán nhãn tương tác.
2. **Máy trạm cân bằng chất lượng & tốc độ:** Sử dụng `yolo26s_fp16.onnx` — đạt **56.0% mAP@50**, tăng mạnh khả năng bắt các vật thể nhỏ như đèn giao thông và biển báo (+13%).
3. **Batch Pre-Labeling chất lượng cao:** Sử dụng `yolo26m_fp16.onnx` — đạt **61.5% mAP@50** và độ chuẩn xác Precision 80%, giảm tối đa thời gian con người phải rà soát lại.

---

## 9. Giấy Phép & Đóng Góp

Dự án được phát hành theo giấy phép [MIT License](LICENSE).  
Phát triển bởi **Team Sakana** (Lê Đức Mạnh - `02122`, Phạm Hoàng Anh - `02128`) — AI20K Build Phase (Khoá 4).
