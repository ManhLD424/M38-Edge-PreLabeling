# M38 screencast — phụ đề tiếng Việt

Footage thật; chỉ cắt khoảng chờ, không giả lập chuột/UI. Không có giọng đọc hoặc nhạc; AAC là track im lặng để tương thích trình phát. Cảnh cảnh báo tải của Edge đã được cắt; chưa xác nhận tải hoặc import CVAT.

| Bắt đầu | Kết thúc | Phụ đề |
|---|---|---|
| 00:00:00,000 | 00:00:05,000 | M38: tạo nhãn gợi ý ngay trên laptop.  |
| 00:00:05,000 | 00:00:15,000 | Cùng một ảnh, chạy lần lượt bốn runtime CPU.  |
| 00:00:15,000 | 00:00:25,000 | PyTorch — Chạy Prelabel  |
| 00:00:25,000 | 00:00:30,500 | ONNX Runtime  |
| 00:00:30,500 | 00:00:40,500 | ONNX Runtime  |
| 00:00:40,500 | 00:00:47,500 | OpenVINO FP16  |
| 00:00:47,500 | 00:00:55,500 | OpenVINO FP16  |
| 00:00:55,500 | 00:01:04,500 | OpenVINO INT8 — model lượng tử hóa  |
| 00:01:04,500 | 00:01:14,500 | OpenVINO INT8 — model lượng tử hóa  |
| 00:01:14,500 | 00:01:20,500 | Phóng to để kiểm tra hộp.  |
| 00:01:20,500 | 00:01:25,500 | Chọn hộp cần sửa.  |
| 00:01:25,500 | 00:01:31,500 | Kéo để dời hộp.  |
| 00:01:31,500 | 00:01:37,800 | Kéo góc để đổi kích thước.  |
| 00:01:37,800 | 00:01:43,800 | Hoàn tác thao tác vừa rồi.  |
| 00:01:43,800 | 00:01:47,800 | Hoặc nhập tọa độ trực tiếp.  |
| 00:01:47,800 | 00:01:53,800 | Hoặc nhập tọa độ trực tiếp.  |
| 00:01:53,800 | 00:02:00,300 | Áp dụng tọa độ.  |
| 00:02:00,300 | 00:02:04,300 | Kiểm tra ảnh trước khi xuất.  |
| 00:02:04,300 | 00:02:10,300 | Đánh dấu ảnh đã kiểm.  |
| 00:02:10,300 | 00:02:13,000 | Xuất nhãn CVAT XML.  |
| 00:02:13,000 | 00:02:18,700 | Studio tạo XML cho 1 ảnh, 9 hộp.  |
| 00:02:18,700 | 00:02:23,200 | Benchmark CPU: 4 threads, 26 ảnh × 3 lượt sau warmup.  |
| 00:02:23,200 | 00:02:30,000 | INT8: 38,80 ms — nhanh hơn PyTorch 3,96 lần. Chưa đo mAP trên bộ nhãn chuẩn độc lập. |
