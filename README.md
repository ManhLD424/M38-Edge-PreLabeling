# M38 — Edge Prelabel Studio

Gợi ý và kiểm nhãn 2D trên CPU: chạy YOLOv8s/ONNX ngay trong browser, chỉnh bounding boxes và xuất **CVAT for images 1.1 XML**. Backend Python hỗ trợ PyTorch, ONNX Runtime và OpenVINO FP16/INT8.

Bản sửa 16/10/2026 thay demo nhãn viết sẵn bằng inference thật và predictions riêng từng ảnh. Xem [audit](docs/AUDIT_2026-10-16.md) và [báo cáo sửa](docs/REPAIR_2026-10-16.md). Các số mAP/drop/năng suất cũ đã bị rút lại vì chưa được đo đúng phương pháp.

## Chạy Studio

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.cli.main serve --host 127.0.0.1 --port 8080
```

Mở http://127.0.0.1:8080/. Browser CPU và replay chạy với artifacts trong docs; không cần weights Python. Backend chỉ bind loopback và dùng cùng UI với Pages. Có thể xem static bằng `python -m http.server 8081 --directory docs`, khi đó không có CPU local API.

Trên Pages, model và WASM được phục vụ cùng site; ảnh riêng không gửi về server. Browser CPU tải model khoảng 43 MiB lần đầu và chạy inference thật trên thiết bị; latency không phải 53ms cố định. Replay ghi rõ kết quả đã chạy và nguồn phép đo.

## Kiểm nhãn và xuất

1. Chọn ảnh mẫu hoặc mở JPEG/PNG/WebP của bạn, chọn Browser CPU rồi Chạy prelabel.
2. Zoom/pan, chọn hộp, sửa tọa độ/nhãn, kéo di chuyển/resize, xóa hoặc vẽ bổ sung. Ctrl+Z hoàn tác. V/B chọn/vẽ, Space+drag dịch ảnh.
3. Confidence và nhóm lớp chi phối cả viewer lẫn export; hộp sửa tay luôn được giữ. Đánh dấu đã kiểm lưu snapshot đúng bộ lọc.
4. Xuất ảnh hiện tại hoặc ảnh đã kiểm thành XML. Trong CVAT chọn **CVAT for images 1.1**. Task cần tên ảnh và ontology tương ứng. JSON Studio là dữ liệu phiên riêng, không phải CVAT JSON.

Phiên chỉnh nhãn nằm trong bộ nhớ tab. Dùng Lưu phiên JSON để giữ dữ liệu trước khi đóng. Browser đã được kiểm thử chạy ONNX, sửa hộp và export. Chưa thực hiện authenticated import/save/reload lên CVAT của người dùng.

## Models và tái tạo

Weights Python không nằm trong Git. Browser ONNX tại docs/models/yolov8s.onnx được version cùng manifest SHA-256; mỗi replay runtime có fingerprints. Upstream: [Ultralytics](https://github.com/ultralytics/ultralytics), [licensing](https://www.ultralytics.com/license). MIT của repo không thay license upstream.

```powershell
python tools/fetch_weights.py
python -m src.cli.main compress --weights models/raw/yolov8s.pt --split-manifest data/split_manifest.json --image-root data/sample_images --output-root models/rebuilt
python -m src.cli.main prelabel-2d --source data/sample_images/BDD06.jpg --model models/rebuilt/openvino_fp16/yolov8s_fp16.xml --out outputs/prelabels-v2 --conf 0.25
python -m src.cli.main info
```

Compress không ghi đè output-root đã tồn tại. Chọn model mới rõ ràng khi prelabel; không tự chuyển runtime đang serve. CPU local dùng đường dẫn trong src/core/inference/factory.py; M38_CPU_THREADS mặc định 4. Model fetch kiểm tra byte SHA hoặc fingerprint tensor tham số; serialization/metadata khác được thông báo rõ và chỉ chấp nhận nếu toàn bộ tham số model trùng. Weights khác sẽ dừng. Split 16 calibration / 10 holdout đóng băng bằng hash chỉ là split ảnh minh họa; chưa có scene provenance hay human GT.

## Đánh giá

```powershell
# Agreement với PyTorch trên 10 ảnh holdout, KHÔNG phải mAP
python -m src.cli.main compare-runtimes --save-json outputs/consistency-v2.json
# Timing ảnh thật, giữ raw từng lượt và cấu hình threads
python -m src.cli.main benchmark-2d --runs 3 --warmup 3 --save-json outputs/latency-v2.json
# AP chuẩn COCO cần file GT do người gán nhãn
python -m src.cli.main eval-accuracy --annotations path/to/human_gt.json --image-root path/to/images --runtime openvino_fp16 --save-json outputs/coco-accuracy.json
```

Raw data mới ở [docs/predictions](docs/predictions/). Không cam kết mAP/drop hoặc tiết kiệm thời gian annotator. COCO pretrained không phủ đầy đủ rider hay các loại traffic signs của BDD. INT8 cần review kỹ hơn vì vị trí hộp có thể lệch FP32.

## 3D experimental

```powershell
python -m src.cli.main infer-3d --pcd data/pointclouds/demo.pcd --score-thresh 0.3 --full-scene
```

Mặc định **geometry heuristic**, không phải learned PointPillars. Score heuristic không phải xác suất đã calibration. --checkpoint yêu cầu weights/dependencies thật, nạp lỗi sẽ báo lỗi. Output là proposal JSON riêng, không đoán label_id CVAT. Adapter REST yêu cầu ontology/version đúng, chưa có live E2E hoặc temporal association.

## Kiểm thử và build dữ liệu

```powershell
python -m pytest -q
node --test tests/web-core.test.mjs
python tools/validate_studio.py
python tools/build_studio_data.py
```

Lệnh cuối cần 4 model local và sinh lại replay từ inference thật. Các kiểm tra không thay thế quality trên GT hoặc pilot annotator. Workflow tùy chỉnh Pages chạy tests và kiểm tra hash/predictions trước publish docs. Cần cấu hình nguồn Pages dùng workflow đó; một luồng Pages build khác đã được quan sát vẫn publish độc lập. Public chỉ thay đổi sau commit/push/deploy được chủ repo cho phép.
