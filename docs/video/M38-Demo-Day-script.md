# M38 Demo Day: kịch bản video

Video 2 phút, không lời, phụ đề tiếng Việt. Cảnh ghép từ ảnh chụp thao tác thật ngày 17/10/2026. Không phải quay màn hình liên tục. Thời gian chờ inference đã cắt.

| Thời gian | Hình ảnh | Lời dẫn đề xuất |
|---|---|---|
| 00:00–00:10 | M38 Prelabel Studio | M38 hỗ trợ tạo nhãn gợi ý trên CPU, rồi để con người kiểm sửa trước khi xuất dữ liệu. |
| 00:10–00:22 | 01  Chọn ảnh và nguồn suy luận | Chọn ảnh mẫu hoặc mở ảnh riêng. Chế độ ONNX Browser xử lý ảnh ngay trên thiết bị, không gửi ảnh lên server. |
| 00:22–00:34 | 02  Chạy prelabel thật | Bấm Chạy prelabel. Trong lần demo này, ONNX WASM CPU trả 9 hộp trên BDD01. Đoạn chờ đã được cắt. |
| 00:34–00:44 | Kiểm tra kết quả theo từng ảnh | Browser báo 1.439,6 ms cho lần chạy này, chưa tính tải model. Tốc độ thay đổi theo thiết bị và runtime. |
| 00:44–00:58 | 03  Sửa hộp bằng tọa độ | Chọn hộp trong danh sách và sửa X trái từ 0,1 thành 8 để minh họa thao tác. Đây chưa phải xác nhận hộp đúng ground truth. |
| 00:58–01:10 | 04  Đánh dấu kiểm và xuất XML | Sau thao tác kiểm sửa, xuất 1 ảnh với 9 hộp theo CVAT for images 1.1. Tên ảnh và nhãn cần khớp task CVAT. |
| 01:10–01:24 | 05  Đánh giá nén trên CPU Python | Benchmark 4 threads, 26 ảnh × 3 lượt: INT8 đạt mean 38,80 ms, tỷ số latency nhanh hơn PyTorch 3,96 lần trong phép đo này. |
| 01:24–01:36 | Dung lượng và phạm vi so sánh | Artifact INT8 là 11,24 MiB, ONNX là 42,79 MiB. Dung lượng file không phải RAM. Benchmark Python khác phép đo browser. |
| 01:36–01:50 | Chất lượng cần ground truth độc lập | INT8 ghép được 94 trên 97 hộp PyTorch ở IoU từ 0,5. Đây là độ đồng thuận giữa model, chưa phải mAP ground truth. |
| 01:50–02:00 | Sản phẩm và dữ liệu gốc công khai | Mở Studio để dùng thử và trang kết quả để kiểm tra log. Bước tiếp theo: mAP thật, RAM đỉnh và pilot thời gian annotator. |