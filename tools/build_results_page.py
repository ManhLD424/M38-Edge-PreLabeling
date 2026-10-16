"""Render the coach-facing report from retained measurements (no ML execution)."""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
NAMES = {'torch': 'PyTorch', 'onnx': 'ONNX Runtime',
         'openvino_fp16': 'OpenVINO FP16', 'openvino_int8': 'OpenVINO INT8'}


def build():
    latency = json.loads((DOCS / 'predictions/latency.json').read_text())
    consistency = json.loads((DOCS / 'predictions/consistency.json').read_text())
    manifest = json.loads((DOCS / 'predictions/manifest.json').read_text())
    baseline = latency['evaluations']['torch']['mean_ms']
    rows, bars, agreements = [], [], []
    for key, name in NAMES.items():
        result = latency['evaluations'][key]
        model = manifest['models'][key]
        assert result['model_sha256'] == model['sha256'], key
        assert len(result['raw']) == len(latency['images']) * latency['runs_per_image']
        mean = result['mean_ms']
        rows.append(f'<tr><th scope="row">{name}</th><td>{mean:.2f}</td>'
                    f'<td>{result["p95_ms"]:.2f}</td><td>{1000 / mean:.2f}</td>'
                    f'<td>{baseline / mean:.2f}×</td><td>{model["size_bytes"] / 2**20:.2f}</td></tr>')
        bars.append(f'<div class="bar-row"><span>{name}</span><div class="bar-track">'
                    f'<div class="bar-fill" style="width:{mean / baseline * 100:.2f}%"></div>'
                    f'</div><strong>{mean:.2f} ms</strong></div>')
        if key != 'torch':
            agreement = consistency['evaluations'][key]
            agreements.append(f'<tr><th scope="row">{name}</th>'
                              f'<td>{agreement["matched_at_50"]} / {agreement["ref_total_boxes"]}</td>'
                              f'<td>{agreement["retention_at_50_percent"]:.2f}%</td>'
                              f'<td>{agreement["mean_matched_iou"]:.4f}</td></tr>')
    speedup = baseline / latency['evaluations']['openvino_int8']['mean_ms']
    ratio = manifest['models']['onnx']['size_bytes'] / manifest['models']['openvino_int8']['size_bytes']
    versions = html.escape(', '.join(f'{k} {v}' for k, v in latency['versions'].items()))
    page = f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M38 · Kết quả thực nghiệm</title><meta name="description" content="Số đo CPU, kích thước model và bằng chứng thực nghiệm M38 dành cho lab coach.">
<link rel="stylesheet" href="assets/studio.css"><link rel="stylesheet" href="assets/results.css"></head>
<body><a class="skip" href="#results">Đến kết quả</a>
<header class="topbar"><div class="brand"><span class="mark" aria-hidden="true">M38</span><div><h1>Kết quả thực nghiệm</h1><p class="muted">Model Compression / Edge Pre-Labeling</p></div></div>
<nav class="top-links" aria-label="Liên kết"><a href="index.html">Mở Prelabel Studio</a><a href="#sources">Dữ liệu gốc</a></nav></header>
<main id="results" class="report">
<section class="report-intro"><p class="eyebrow">BÁO CÁO CHO LAB COACH · 16/10/2026</p><h2>Nén model: nhanh hơn bao nhiêu,<br>và đã kiểm chứng đến đâu?</h2>
<p>YOLOv8s pretrained COCO, suy luận CPU trên ảnh đường phố. Các số dưới đây là kết quả benchmark Python đã lưu; thời gian ONNX chạy trực tiếp trên thiết bị được hiển thị riêng trong Studio.</p>
<div class="report-note">Đã đo latency, kích thước artifact và độ đồng thuận giữa runtime. Chưa có ground truth để kết luận mAP hoặc mức giảm độ chính xác sau nén.</div></section>
<section class="metric-grid" aria-label="Kết quả nổi bật">
<article class="panel metric"><span>INT8 so với PyTorch</span><strong>{speedup:.2f}×</strong><p>Tỷ số mean latency, cùng workload và 4 threads.</p></article>
<article class="panel metric"><span>Artifact INT8 so với ONNX</span><strong>{ratio:.2f}× nhỏ hơn</strong><p>So tổng dung lượng file model; không phải mức giảm RAM.</p></article>
<article class="panel metric"><span>Quy mô đo latency</span><strong>26 ảnh × 3 lượt</strong><p>3 warmup; 78 lần predict cho mỗi runtime.</p></article></section>
<section class="panel report-section"><div class="report-heading"><p class="eyebrow">01 / HIỆU NĂNG CPU</p><h2>Latency và kích thước model</h2><p>Thấp hơn là nhanh hơn. Bao gồm preprocessing và NMS; không tính đọc ảnh từ ổ đĩa hay nạp model.</p></div>
<div class="bars" aria-label="So sánh mean latency">{''.join(bars)}</div>
<div class="table-wrap" tabindex="0" role="region" aria-label="Bảng hiệu năng, cuộn ngang trên màn hình nhỏ"><table><caption>Số đo Python CPU · 4 inference threads</caption><thead><tr><th scope="col">Runtime</th><th scope="col">Mean (ms)</th><th scope="col">P95 (ms)</th><th scope="col">Ảnh/s tuần tự*</th><th scope="col">Tăng tốc**</th><th scope="col">Model (MiB)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="table-note">* 1000 / mean latency, không phải FPS streaming. ** Mean PyTorch / mean runtime. MiB = 1.048.576 bytes; OpenVINO tính cả XML + BIN. Checkpoint PT có cách lưu khác ONNX: dung lượng file không chứng minh dtype tính toán hoặc tỷ lệ nén trọng số tương đương.</p></section>
<section class="panel report-section"><div class="report-heading"><p class="eyebrow">02 / ĐỘ ĐỒNG THUẬN</p><h2>So với prediction PyTorch</h2><p>10 ảnh holdout, 97 hộp tham chiếu; confidence 0,25, NMS IoU 0,45. Calibration INT8 dùng 16 ảnh có hash khác holdout.</p></div>
<div class="table-wrap" tabindex="0" role="region" aria-label="Bảng độ đồng thuận"><table><caption>Ghép hộp cùng lớp tại IoU ≥ 0,5 · không phải mAP</caption><thead><tr><th scope="col">Runtime</th><th scope="col">Hộp ghép / tham chiếu</th><th scope="col">Giữ hộp tham chiếu</th><th scope="col">Mean IoU hộp ghép</th></tr></thead><tbody>{''.join(agreements)}</tbody></table></div>
<p class="table-note">PyTorch có thể dự đoán sai. Đồng thuận 100% không có nghĩa là chính xác 100%. INT8 sinh 99 hộp, trong đó 94 hộp ghép được với 97 hộp PyTorch. Tập ảnh nhỏ và chưa xác minh độc lập theo scene.</p></section>
<section class="panel report-section"><div class="report-heading"><p class="eyebrow">03 / TRẠNG THÁI BẰNG CHỨNG</p><h2>Những phép đo còn cần hoàn thiện</h2></div>
<dl class="evidence"><div><dt>mAP@50, mAP@50:95 và ΔmAP</dt><dd><b>Chưa đo</b> — cần nhãn chuẩn độc lập và ontology thống nhất cho baseline, FP16, INT8.</dd></div>
<div><dt>RAM đỉnh khi suy luận</dt><dd><b>Chưa có kết quả đủ chuẩn</b> — cần lấy mẫu bộ nhớ trong lúc chạy, tách từng runtime.</dd></div>
<div><dt>Tiết kiệm thời gian gán nhãn</dt><dd><b>Chưa đo</b> — cần pilot so sánh gán nhãn tay với sửa prelabel, kèm lỗi còn sót.</dd></div>
<div><dt>CVAT import → save → reload</dt><dd><b>Chưa xác minh trên CVAT thật</b> — XML đã kiểm bằng parser độc lập; chưa có vòng kiểm tra xác thực trên server.</dd></div>
<div><dt>Nhánh 3D</dt><dd><b>Thử nghiệm heuristic</b> — chưa có bằng chứng nén hoặc chất lượng learned PointPillars.</dd></div></dl></section>
<section class="panel report-section" id="sources"><div class="report-heading"><p class="eyebrow">04 / TÁI LẬP & NGUỒN</p><h2>Điều kiện đo và dữ liệu gốc</h2></div>
<div class="protocol"><p><b>Thời điểm log (UTC):</b> {html.escape(latency['created_at'])}</p><p><b>Máy đo:</b> {html.escape(latency['hardware'])}; {html.escape(latency['cpu'])}; {latency['hardware_threads']} luồng phần cứng, giới hạn suy luận {latency['inference_threads']} threads.</p><p><b>Phiên bản:</b> {versions}.</p><p><b>Giới hạn:</b> chưa kiểm soát nhiệt độ/nguồn điện như phòng benchmark. Không suy rộng kết quả CPU Python sang browser WASM hoặc thiết bị edge khác.</p>
<div class="source-links"><a href="predictions/latency.json">Log latency từng ảnh + model hashes</a><a href="predictions/consistency.json">Log độ đồng thuận</a><a href="predictions/manifest.json">Artifact sizes + image hashes</a><a href="BENCHMARK_REPORT.md">Báo cáo benchmark</a><a href="REPAIR_2026-10-16.md">Báo cáo sửa lỗi</a></div>
<details><summary>Lệnh tái lập trên môi trường đã cài model</summary><pre>python -m src.cli.main benchmark-2d --runs 3 --warmup 3 --save-json outputs/latency-v2.json
python -m src.cli.main compare-runtimes --save-json outputs/consistency-v2.json
python tools/build_results_page.py</pre></details></div></section>
</main><footer class="session-footer"><span>M38 · Số liệu có nguồn, chỉ số chưa đo được ghi rõ.</span><a href="index.html">Thử inference và kiểm nhãn trong Studio</a></footer></body></html>
'''
    target = DOCS / 'results.html'
    if '--check' in sys.argv:
        assert target.read_text(encoding='utf-8') == page, 'Run python tools/build_results_page.py to refresh the report'
        print('Results page matches measured logs and model hashes')
    else:
        target.write_text(page, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    build()
