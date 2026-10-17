# Đối chiếu demo M38 với đề

Nguồn yêu cầu: ảnh đề người dùng cung cấp ngày 17/10/2026.

**M38 — Model Compression for Edge Pre-Labeling.** Dataset BDD100K.
Tối ưu model để chạy pre-label trên laptop/edge. Đầu ra: ONNX/TFLite model
và latency benchmark. Chỉ số: latency, throughput, mAP drop.

| Yêu cầu | Bằng chứng hiện có | Đánh giá |
|---|---|---|
| Model chạy trên laptop/CPU | PyTorch, ONNX Runtime, OpenVINO FP16 và INT8 qua Python local; ONNX WASM trong browser | Có inference thật. Pages không chạy Python/OpenVINO native |
| Artifact triển khai | ONNX export, OpenVINO FP16/INT8, fingerprints | Có ONNX. Chưa có TFLite; không tự hiểu dấu slash là bắt buộc cả hai định dạng |
| BDD100K | 26 ảnh mẫu đường phố trong project, split 16 calibration / 10 holdout khác hash | Chưa có bộ annotation GT và provenance/split chính thức đủ để kết luận chất lượng trên BDD100K |
| Latency benchmark | Log 4 threads, 26 ảnh × 3 lượt, warmup, mean/P95 và raw timing | Có. Một lần bấm UI không thay cho benchmark |
| Throughput | Hiện bảng có ảnh/s tuần tự tính từ nghịch đảo mean latency | Là tốc độ suy ra từ latency, chưa phải benchmark throughput streaming độc lập |
| mAP drop | Có kiểm tra agreement với baseline, evaluator COCO AP đã có | Chưa đo GT mAP và ΔmAP. Agreement không thay thế được |

## Điều chỉnh nội dung demo

- Trình bày nén và deployment CPU làm trục chính; thao tác CVAT XML là ví dụ đầu ra pre-label.
- Thêm cảnh chọn và chạy mới cả bốn runtime native trên cùng BDD01, tách khỏi replay.
- Ghi rõ laptop local và browser public là hai môi trường chạy khác nhau.
- Giữ điều kiện đo và giới hạn chất lượng. Không gọi các số agreement là mAP.
- Không đưa nhánh 3D heuristic vào phần chứng minh hoàn thành đề M38 này.
- Bỏ slide/cảnh kết thúc “Tiếp theo…” theo yêu cầu, vẫn giữ các giới hạn gắn trực tiếp với kết quả đang trình bày.
- Voice đọc M38 là “Mờ ba tám”, Prelabel là “Pre lây bờ”; phụ đề giữ cách viết M38 và Prelabel.

## Kết luận

Demo có hướng triển khai phù hợp đề, nhưng trước đây quá thiên về công cụ kiểm nhãn browser
và chưa thể hiện bốn runtime native. Bản sửa bổ sung phần này. Không thể tuyên bố hoàn thành
toàn bộ tiêu chí đánh giá khi chưa có mAP drop trên GT BDD100K và phép đo throughput được
định nghĩa rõ. Web public chạy OpenVINO native cần một backend được triển khai riêng;
không thể bật khả năng này chỉ bằng đổi menu trên GitHub Pages.
