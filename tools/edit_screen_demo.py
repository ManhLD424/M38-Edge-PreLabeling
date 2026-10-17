"""Cut genuine screen footage and add captions; never synthesize UI/cursor frames."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT/'outputs/demo-final-2026-10-18/raw-screen.mp4')
parser.add_argument('--build', type=Path, default=ROOT/'outputs/screencast-captioned-2026-10-18')
parser.add_argument('--ffmpeg', type=Path, default=ROOT/'outputs/video-deps/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe')
args = parser.parse_args()
B = args.build.resolve()
B.mkdir(parents=True, exist_ok=True)
FF, SOURCE = str(args.ffmpeg.resolve()), args.source.resolve()
# Each segment retains real-time playback. Gaps remove inactivity only.
EDITS = [(13,35),(36,48),(49,56),(68,90),(91,115.5),(119,148),(152,169),
         (170,179),(180,188),(188.5,199),(200,206),(211,221),(224,233),
         (236,244),(256,264),(278.5,291),(300.5,317.3)]
CUES = [
 (13,19,'M38: tối ưu mô hình để tạo nhãn gợi ý trên laptop/edge.','Model Compression for Edge Pre-Labeling'),
 (19,23,'Footage thao tác thật, đã cắt các khoảng chờ.','Không thêm giọng đọc hoặc chuyển động chuột giả.'),
 (23,31,'Studio local có bốn runtime CPU chạy thật.','Các mục “Kết quả đã chạy” bên dưới chỉ là replay.'),
 (31,35,'Chọn PyTorch · CPU local trên ảnh BDD01.','Giữ cùng ảnh để minh họa các runtime.'),
 (36,39,'Bấm Chạy Prelabel và chờ inference hoàn tất.','Nút chạy tạm khóa trong khi xử lý.'),
 (39,48,'Các hộp xuất hiện cùng nhãn “Local torch”.','Timing lần bấm này không thay benchmark sau warmup.'),
 (49,56,'Mở menu runtime để chuyển sang ONNX Runtime.','Đây vẫn là backend CPU local.'),
 (68,76,'Chọn ONNX Runtime · CPU local.','Ảnh vẫn là BDD01; chưa dùng kết quả replay.'),
 (76,81,'Bấm chạy mới và chờ các hộp xuất hiện.','Mỗi lần chạy được gắn với đúng ảnh và runtime.'),
 (81,90,'Kiểm tra dòng “Local onnx” dưới vùng ảnh.','Thời gian nạp model và latency từng lần có thể khác nhau.'),
 (91,101,'Chọn OpenVINO FP16 · CPU local.','Tiếp tục dùng cùng ảnh đầu vào.'),
 (101,110,'Bấm Chạy Prelabel để thực hiện inference FP16 mới.','Chờ trạng thái xử lý kết thúc.'),
 (110,115.5,'Kết quả hiện nhãn “Local openvino_fp16”.','Các hộp là gợi ý và cần người kiểm tra.'),
 (119,127,'Tiếp theo là model lượng tử hóa OpenVINO INT8.','INT8 hiện được triển khai ở định dạng OpenVINO IR.'),
 (127,138,'Chọn INT8 rồi bấm Chạy Prelabel.','Giữ lại thao tác chọn runtime và click chạy thật.'),
 (138,148,'Kết quả mới hiển thị “Local openvino_int8”.','Không suy ra chất lượng mAP từ hình minh họa này.'),
 (152,160,'Phóng to ảnh lên 125% để quan sát hộp rõ hơn.','Đây là thao tác zoom thật trong Studio.'),
 (160,169,'Bấm chọn hộp chiếc xe ở giữa ảnh.','Hộp được tô cam và các trường tọa độ được mở.'),
 (170,179,'Kéo bên trong hộp để di chuyển.','Tọa độ thay đổi theo vị trí mới của hộp.'),
 (180,188,'Kéo góc dưới bên phải để đổi kích thước.','Đây là thao tác minh họa, chưa phải nhãn ground truth.'),
 (188.5,199,'Bấm Hoàn tác một lần.','Hộp trở về kích thước trước lần resize vừa rồi.'),
 (200,206,'Có thể chỉnh hộp bằng các trường tọa độ.','Nhập lại tọa độ trước thao tác kéo để minh họa.'),
 (211,221,'Điền các cạnh trái, trên, phải và dưới.','Chưa áp dụng thì vị trí hộp vẫn giữ nguyên.'),
 (224,233,'Bấm Áp dụng để cập nhật hộp trên ảnh.','Kiểm tra vị trí hiển thị sau khi nhập tọa độ.'),
 (236,244,'Cuộn đến phần kiểm ảnh và xuất nhãn.','Khi sử dụng thực tế, cần kiểm toàn bộ ảnh trước khi đánh dấu.'),
 (256,264,'Bấm Đánh dấu đã kiểm trong workflow minh họa.','Bộ đếm tăng lên 1 ảnh; không phải chứng nhận nhãn đúng GT.'),
 (278.5,282,'Bấm Xuất ảnh này · CVAT XML.','Studio báo xuất 1 ảnh và 9 hộp đang hiển thị.'),
 (282,291,'Edge yêu cầu người dùng xác nhận tải tệp XML.','Chưa xác nhận tải; chưa thực hiện import/save/reload CVAT.'),
 (300.5,307.5,'Mở trang Kết quả thực nghiệm của dự án.','Số trên trang đến từ benchmark nhiều lượt đã lưu.'),
 (307.5,312,'Benchmark CPU: 4 threads, 26 ảnh × 3 lượt sau warmup.','Kết quả này được đo riêng với latency từng lần bấm UI.'),
 (312,317.3,'Agreement với PyTorch không phải mAP trên ground truth.','Chưa có mAP drop BDD100K hoặc throughput streaming độc lập.')]


def run(command, log):
    with (B/log).open('w', encoding='utf-8') as handle:
        subprocess.run(command, cwd=B, stdout=handle, stderr=subprocess.STDOUT, check=True)


def stamp(value, ass=False):
    unit = 100 if ass else 1000
    n = round(value*unit)
    h, n = divmod(n, 3600*unit)
    m, n = divmod(n, 60*unit)
    s, n = divmod(n, unit)
    return f'{h}:{m:02}:{s:02}.{n:02}' if ass else f'{h:02}:{m:02}:{s:02},{n:03}'


offset, timeline, mapped, files = 0, [], [], []
for i, (start, end) in enumerate(EDITS):
    duration = end-start
    clip = B/f'clip-{i:02}.mp4'
    files.append(clip)
    timeline.append({'source_in':start,'source_out':end,'output_in':offset,'output_out':offset+duration,'speed':1})
    for a, z, line1, line2 in CUES:
        left, right = max(a,start), min(z,end)
        if right > left:
            mapped.append((offset+left-start,offset+right-start,line1,line2))
    run([FF,'-hide_banner','-y','-ss',str(start),'-i',str(SOURCE),'-t',str(duration),
         '-vf','scale=-2:972,pad=1920:1080:(ow-iw)/2:0:color=0x101820,fps=24',
         '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(clip)],f'clip-{i:02}.log')
    offset += duration
    print(f'Clip {i+1}/{len(EDITS)} complete', flush=True)
mapped.sort()
srt = '\n\n'.join(f'{i+1}\n{stamp(a)} --> {stamp(z)}\n{l1}\n{l2}' for i,(a,z,l1,l2) in enumerate(mapped))+'\n'
(B/'M38-Screencast-vi.srt').write_text(srt, encoding='utf-8')
ass = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,34,&H00FFFFFF,&H00FFFFFF,&H00101820,&H00101820,0,0,0,0,100,100,0,0,1,0,0,2,45,45,16,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
for a,z,l1,l2 in mapped:
    ass += f'Dialogue: 0,{stamp(a,True)},{stamp(z,True)},Default,,0,0,0,,{l1}\\N{l2}\n'
(B/'captions.ass').write_text(ass, encoding='utf-8')
(B/'concat.txt').write_text(''.join("file '"+p.name+"'\n" for p in files), encoding='utf-8')
(B/'edit-decisions.json').write_text(json.dumps({'source':str(SOURCE),
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'cut_only':True,'synthetic_UI_or_cursor':False,'duration_seconds':offset,
    'timeline':timeline,'cues':mapped}, ensure_ascii=False, indent=2), encoding='utf-8')
run([FF,'-hide_banner','-y','-f','concat','-safe','0','-i',str(B/'concat.txt'),
     '-f','lavfi','-i','anullsrc=channel_layout=stereo:sample_rate=48000',
     '-vf',"ass=filename='captions.ass'",'-map','0:v:0','-map','1:a:0','-t',str(offset),
     '-c:v','libx264','-preset','slow','-crf','20','-pix_fmt','yuv420p',
     '-c:a','aac','-b:a','64k','-movflags','+faststart',str(B/'M38-Screencast-captioned.mp4')],'final-encode.log')
(B/'M38-Screencast-timecodes.md').write_text('# M38 screencast — phụ đề tiếng Việt\n\n'
    'Footage thật; chỉ cắt khoảng chờ, không giả lập chuột/UI. Không có giọng đọc hoặc nhạc; '
    'AAC là track im lặng để tương thích trình phát.\n\n| Bắt đầu | Kết thúc | Phụ đề |\n|---|---|---|\n'+
    ''.join(f'| {stamp(a)} | {stamp(z)} | {l1} {l2} |\n' for a,z,l1,l2 in mapped), encoding='utf-8')
print('Final duration', offset, flush=True)
