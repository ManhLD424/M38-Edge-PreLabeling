"""Cut genuine screen footage and add captions; never synthesize UI/cursor frames."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT/'outputs/demo-final-2026-10-18/raw-screen.mp4')
parser.add_argument('--build', type=Path, default=ROOT/'outputs/screencast-short-2026-10-18')
parser.add_argument('--ffmpeg', type=Path, default=ROOT/'outputs/video-deps/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe')
args = parser.parse_args()
B = args.build.resolve()
B.mkdir(parents=True, exist_ok=True)
FF, SOURCE = str(args.ffmpeg.resolve()), args.source.resolve()
# Each segment retains real-time playback. Gaps remove inactivity only.
EDITS = [(13,18),(23,33),(36,46),(69,74.5),(75,85),(94,101),
         (107,115),(124,133),(135,145),(152,158),(161,166),(171,177),
         (179.7,186),(188,194),(200,204),(212,218),(224.5,231),
         (236,240),(256,262),(279.3,282),(300.3,306),(306,317.3)]
CUES = [
 (13,18,'M38: tạo nhãn gợi ý ngay trên laptop.',''),
 (23,33,'Cùng một ảnh, chạy lần lượt bốn runtime CPU.',''),
 (36,46,'PyTorch — Chạy Prelabel',''),
 (69,85,'ONNX Runtime',''),
 (94,115,'OpenVINO FP16',''),
 (124,145,'OpenVINO INT8 — model lượng tử hóa',''),
 (152,158,'Phóng to để kiểm tra hộp.',''),
 (161,166,'Chọn hộp cần sửa.',''),
 (171,177,'Kéo để dời hộp.',''),
 (179.7,186,'Kéo góc để đổi kích thước.',''),
 (188,194,'Hoàn tác thao tác vừa rồi.',''),
 (200,218,'Hoặc nhập tọa độ trực tiếp.',''),
 (224.5,231,'Áp dụng tọa độ.',''),
 (236,240,'Kiểm tra ảnh trước khi xuất.',''),
 (256,262,'Đánh dấu ảnh đã kiểm.',''),
 (279.3,282,'Xuất nhãn CVAT XML.',''),
 (300.3,306,'Studio tạo XML cho 1 ảnh, 9 hộp.',''),
 (306,310.5,'Benchmark CPU: 4 threads, 26 ảnh × 3 lượt sau warmup.',''),
 (310.5,317.3,'INT8: 38,80 ms — nhanh hơn PyTorch 3,96 lần.',
              'Chưa đo mAP trên bộ nhãn chuẩn độc lập.')]



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
srt = '\n\n'.join(f'{i+1}\n{stamp(a)} --> {stamp(z)}\n'+ '\n'.join(filter(None,(l1,l2))) for i,(a,z,l1,l2) in enumerate(mapped))+'\n'
(B/'M38-Screencast-short-vi.srt').write_text(srt, encoding='utf-8')
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
    text = r'\N'.join(filter(None,(l1,l2)))
    ass += f'Dialogue: 0,{stamp(a,True)},{stamp(z,True)},Default,,0,0,0,,{text}\n'
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
     '-c:a','aac','-b:a','64k','-movflags','+faststart',str(B/'M38-Screencast-short.mp4')],'final-encode.log')
(B/'M38-Screencast-short-timecodes.md').write_text('# M38 screencast — phụ đề tiếng Việt\n\n'
    'Footage thật; chỉ cắt khoảng chờ, không giả lập chuột/UI. Không có giọng đọc hoặc nhạc; '
    'AAC là track im lặng để tương thích trình phát. Cảnh cảnh báo tải của Edge đã được cắt; chưa xác nhận tải hoặc import CVAT.\n\n| Bắt đầu | Kết thúc | Phụ đề |\n|---|---|---|\n'+
    ''.join(f'| {stamp(a)} | {stamp(z)} | {l1} {l2} |\n' for a,z,l1,l2 in mapped), encoding='utf-8')
print('Final duration', offset, flush=True)
