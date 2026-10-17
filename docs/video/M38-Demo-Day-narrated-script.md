# M38: video hướng dẫn có giọng đọc tiếng Việt

Cách đọc: M38 = Mờ ba tám; Prelabel = Pre lây bờ. Phụ đề giữ M38 và Prelabel. Giọng tổng hợp: vi-VN-NamMinhNeural. Cảnh ghép từ ảnh chụp thao tác thật, không phải quay liên tục. Thời gian chờ đã cắt. Các khung cam chỉ vị trí cần chú ý.

| Bắt đầu | Thao tác | Kết quả cần thấy | Lời đọc |
|---|---|---|---|
| 00:00:00,000 | Tạo nhãn gợi ý trên CPU | Người dùng kiểm sửa trước khi xuất | Đây là M38 Prelabel Studio của đội Sakana. |
| 00:00:04,502 | Tạo nhãn gợi ý trên CPU | Người dùng kiểm sửa trước khi xuất | Công cụ tạo nhãn gợi ý trên CPU, để con người kiểm tra và chỉnh sửa trước khi xuất dữ liệu. |
| 00:00:10,877 | Bấm BDD01 trong danh sách ảnh mẫu | Tên ảnh giữa màn hình là BDD01.jpg | Đầu tiên, chọn ảnh B D D không một trong danh sách ảnh mẫu ở bên trái. |
| 00:00:16,294 | Bấm BDD01 trong danh sách ảnh mẫu | Tên ảnh giữa màn hình là BDD01.jpg | Kiểm tra tên ảnh ở đầu vùng xem để chắc chắn đang thao tác đúng ảnh. |
| 00:00:21,294 | Chọn Chạy thật · ONNX Browser CPU | Ảnh chạy trên CPU của thiết bị | Ở Nguồn suy luận, chọn Chạy thật, ONNX Browser CPU. |
| 00:00:26,300 | Chọn Chạy thật · ONNX Browser CPU | Ảnh chạy trên CPU của thiết bị | Với ảnh riêng, trình duyệt xử lý trên thiết bị và không gửi ảnh về server. |
| 00:00:31,550 | Bấm nút Chạy prelabel | Chờ thông báo hoàn tất | Tiếp theo, bấm nút Chạy prelabel. |
| 00:00:35,140 | Bấm nút Chạy prelabel | Chờ thông báo hoàn tất | Lần đầu cần tải model. |
| 00:00:37,932 | Bấm nút Chạy prelabel | Chờ thông báo hoàn tất | Video chuyển thẳng sang kết quả và đã cắt thời gian chờ. |
| 00:00:42,314 | Kiểm từng hộp và nhãn trong danh sách | Ảnh này có 9 hộp gợi ý | Khi hoàn tất, các hộp xuất hiện trên ảnh và trong danh sách bên phải. |
| 00:00:47,248 | Kiểm từng hộp và nhãn trong danh sách | Ảnh này có 9 hộp gợi ý | Lần chạy này có chín hộp. |
| 00:00:50,046 | Kiểm từng hộp và nhãn trong danh sách | Ảnh này có 9 hộp gợi ý | Hãy kiểm tra vật thể bỏ sót và hộp nhận diện sai. |
| 00:00:54,020 | Đọc dòng timing dưới vùng ảnh | Không so trực tiếp với log Python | Dòng dưới ảnh ghi thời gian browser khoảng một phẩy bốn bốn giây, chưa tính tải model. |
| 00:00:59,979 | Đọc dòng timing dưới vùng ảnh | Không so trực tiếp với log Python | Đây là phép đo khác với benchmark Python ở phần sau. |
| 00:01:04,193 | Bấm hàng 1. car ở danh sách bên phải | Hộp được chọn và mở ô tọa độ | Để sửa một hộp, bấm hàng đầu tiên trong danh sách bên phải. |
| 00:01:08,743 | Bấm hàng 1. car ở danh sách bên phải | Hộp được chọn và mở ô tọa độ | Studio chọn hộp tương ứng trên ảnh và mở các trường tọa độ. |
| 00:01:13,461 | Đổi X trái từ 0,1 thành 8 | Ví dụ minh họa thao tác chỉnh hộp | Trong ví dụ này, đổi X trái từ không phẩy một thành tám. |
| 00:01:18,059 | Đổi X trái từ 0,1 thành 8 | Ví dụ minh họa thao tác chỉnh hộp | Đây là thao tác minh họa, chưa phải khẳng định tọa độ mới đúng với nhãn chuẩn. |
| 00:01:23,401 | Bấm Áp dụng dưới các ô tọa độ | Hộp hiển thị trạng thái đã sửa | Sau khi nhập tọa độ, bấm Áp dụng bên dưới các ô. |
| 00:01:27,735 | Bấm Áp dụng dưới các ô tọa độ | Hộp hiển thị trạng thái đã sửa | Kiểm tra lại hộp trên ảnh. |
| 00:01:30,527 | Bấm Áp dụng dưới các ô tọa độ | Hộp hiển thị trạng thái đã sửa | Nếu thao tác sai, dùng nút Hoàn tác ở thanh công cụ. |
| 00:01:35,077 | Kiểm hết ảnh, rồi Đánh dấu đã kiểm | Bộ đếm tăng thành 1 ảnh đã kiểm | Sau khi kiểm toàn bộ hộp và bổ sung vật thể bỏ sót, bấm Đánh dấu đã kiểm. |
| 00:01:40,661 | Kiểm hết ảnh, rồi Đánh dấu đã kiểm | Bộ đếm tăng thành 1 ảnh đã kiểm | Trong cảnh minh họa, bộ đếm đã tăng thành một ảnh. |
| 00:01:44,953 | Bấm Xuất ảnh này · CVAT XML | Thông báo xuất 1 ảnh, 9 hộp | Bấm Xuất ảnh này, CVAT XML. |
| 00:01:49,095 | Bấm Xuất ảnh này · CVAT XML | Thông báo xuất 1 ảnh, 9 hộp | Thông báo xác nhận một ảnh và chín hộp. |
| 00:01:52,445 | Bấm Xuất ảnh này · CVAT XML | Thông báo xuất 1 ảnh, 9 hộp | Khi nhập vào CVAT, chọn CVAT for images một chấm một, với tên ảnh và nhãn khớp task. |
| 00:01:59,659 | Chọn CPU local torch rồi Chạy prelabel | Đọc nhãn Chạy thật · Local torch | Trên laptop, mở Studio qua server Python local. |
| 00:02:04,743 | Chọn CPU local torch rồi Chạy prelabel | Đọc nhãn Chạy thật · Local torch | Chọn PyTorch rồi bấm Chạy Prelabel. |
| 00:02:08,327 | Chọn CPU local torch rồi Chạy prelabel | Đọc nhãn Chạy thật · Local torch | Đây là kết quả inference thật trên cùng ảnh, không phải replay. |
| 00:02:13,261 | Chọn CPU local onnx rồi chạy lại | Đọc nhãn Chạy thật · Local onnx | Tiếp theo, chọn ONNX Runtime và chạy trên cùng ảnh. |
| 00:02:17,931 | Chọn CPU local onnx rồi chạy lại | Đọc nhãn Chạy thật · Local onnx | Lần nạp đầu và từng lần chạy có thể khác nhau, nên số tại đây không thay bảng benchmark nhiều lượt. |
| 00:02:24,113 | Chọn CPU local openvino_fp16 | Bấm chạy để suy luận mới | Với OpenVINO FP mười sáu, cũng chọn runtime rồi bấm chạy. |
| 00:02:29,743 | Chọn CPU local openvino_fp16 | Bấm chạy để suy luận mới | Các hộp được tạo mới trên CPU laptop. |
| 00:02:33,368 | Chọn CPU local openvino_int8 | Kết quả mới từ model lượng tử hóa | Cuối cùng, chọn OpenVINO INT tám và chạy lại. |
| 00:02:38,206 | Chọn CPU local openvino_int8 | Kết quả mới từ model lượng tử hóa | Model lượng tử hóa vẫn cần người kiểm nhãn. |
| 00:02:42,123 | Chọn CPU local openvino_int8 | Kết quả mới từ model lượng tử hóa | Trang public GitHub chỉ chạy ONNX Browser, vì Pages không có server Python. |
| 00:02:48,185 | Mở Kết quả thực nghiệm trên Studio | Đọc điều kiện đo trước khi so sánh | Trang Kết quả thực nghiệm so sánh các runtime Python CPU. |
| 00:02:52,783 | Mở Kết quả thực nghiệm trên Studio | Đọc điều kiện đo trước khi so sánh | Phép đo dùng bốn threads, hai mươi sáu ảnh và ba lượt đo sau warmup. |
| 00:02:57,992 | So cùng workload và thread budget | 38,80 ms; tỷ số latency 3,96× | INT tám đạt mean latency ba mươi tám phẩy tám mili giây. |
| 00:03:02,926 | So cùng workload và thread budget | 38,80 ms; tỷ số latency 3,96× | Tỷ số latency nhanh hơn PyTorch ba phẩy chín sáu lần trong phép đo này. |
| 00:03:08,135 | So cùng workload và thread budget | 38,80 ms; tỷ số latency 3,96× | Dung lượng artifact INT tám là mười một phẩy hai bốn MiB, không phải RAM. |
| 00:03:13,693 | Phân biệt agreement và mAP | Cần ground truth độc lập | INT tám ghép được chín mươi bốn trên chín mươi bảy hộp PyTorch tại IoU từ không phẩy năm. |
| 00:03:19,995 | Phân biệt agreement và mAP | Cần ground truth độc lập | Đây là độ đồng thuận giữa model. |
| 00:03:23,120 | Phân biệt agreement và mAP | Cần ground truth độc lập | Dự án chưa có mAP trên ground truth độc lập. |

## Link sản phẩm
https://manhld424.github.io/M38-Edge-PreLabeling/
https://manhld424.github.io/M38-Edge-PreLabeling/results.html