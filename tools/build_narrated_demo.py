"""Build a Vietnamese narrated walkthrough from retained, genuine UI captures.

Install edge-tts, Pillow, imageio-ffmpeg in an isolated directory and pass it via
--deps. Requires the original captures under --assets; never fabricates UI state.
"""
import argparse
import asyncio
import hashlib
import json
import subprocess
import struct
import sys
import wave
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--deps', required=True)
parser.add_argument('--assets', required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.deps).resolve()))
import edge_tts
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(args.assets).resolve()
BUILD = ROOT / 'outputs/narrated-demo'
OUT = ROOT / 'docs/video'
BUILD.mkdir(parents=True, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
VOICE = 'vi-VN-NamMinhNeural'
# Regions refer to the original 1265 x 712 browser captures.
SCENES = [
 ('02-inferred.png', 'M38 Prelabel Studio', 'Tạo nhãn gợi ý trên CPU', 'Người dùng kiểm sửa trước khi xuất', None,
  ['Đây là M ba tám Prelabel Studio của đội Sakana. Công cụ tạo nhãn gợi ý trên CPU, để con người kiểm tra và chỉnh sửa trước khi xuất dữ liệu.']),
 ('01-ready.png', '01 / Chọn ảnh', 'Bấm BDD01 trong danh sách ảnh mẫu', 'Tên ảnh giữa màn hình là BDD01.jpg', (37,662,130,708),
  ['Đầu tiên, chọn ảnh B D D không một trong danh sách ảnh mẫu ở bên trái. Kiểm tra tên ảnh ở đầu vùng xem để chắc chắn đang thao tác đúng ảnh.']),
 ('01-ready.png', '02 / Chọn nguồn suy luận', 'Chọn Chạy thật · ONNX Browser CPU', 'Ảnh chạy trên CPU của thiết bị', (36,190,239,235),
  ['Ở Nguồn suy luận, chọn Chạy thật, ONNX Browser CPU. Với ảnh riêng, trình duyệt xử lý trên thiết bị và không gửi ảnh về server.']),
 ('01-ready.png', '03 / Bắt đầu suy luận', 'Bấm nút Chạy prelabel', 'Chờ thông báo hoàn tất', (36,315,239,363),
  ['Tiếp theo, bấm nút Chạy prelabel. Lần đầu cần tải model. Video chuyển thẳng sang kết quả và đã cắt thời gian chờ.']),
 ('02-inferred.png', '04 / Đọc kết quả', 'Kiểm từng hộp và nhãn trong danh sách', 'Ảnh này có 9 hộp gợi ý', (976,107,1225,200),
  ['Khi hoàn tất, các hộp xuất hiện trên ảnh và trong danh sách bên phải. Lần chạy này có chín hộp. Hãy kiểm tra vật thể bỏ sót và hộp nhận diện sai.']),
 ('02-inferred.png', 'Thời gian browser được đo riêng', 'Đọc dòng timing dưới vùng ảnh', 'Không so trực tiếp với log Python', (650,600,944,634),
  ['Dòng dưới ảnh ghi thời gian browser khoảng một phẩy bốn bốn giây, chưa tính tải model. Đây là phép đo khác với benchmark Python ở phần sau.']),
 ('02-inferred.png', '05 / Chọn hộp cần sửa', 'Bấm hàng 1. car ở danh sách bên phải', 'Hộp được chọn và mở ô tọa độ', (976,151,1225,199),
  ['Để sửa một hộp, bấm hàng đầu tiên trong danh sách bên phải. Studio chọn hộp tương ứng trên ảnh và mở các trường tọa độ.']),
 ('03-edit.png', '06 / Nhập tọa độ mới', 'Đổi X trái từ 0,1 thành 8', 'Ví dụ minh họa thao tác chỉnh hộp', (976,585,1107,632),
  ['Trong ví dụ này, đổi X trái từ không phẩy một thành tám. Đây là thao tác minh họa, chưa phải khẳng định tọa độ mới đúng với nhãn chuẩn.']),
 ('04-export.png', '07 / Áp dụng thay đổi', 'Bấm Áp dụng dưới các ô tọa độ', 'Hộp hiển thị trạng thái đã sửa', (980,224,1093,263),
  ['Sau khi nhập tọa độ, bấm Áp dụng bên dưới các ô. Kiểm tra lại hộp trên ảnh. Nếu thao tác sai, dùng nút Hoàn tác ở thanh công cụ.']),
 ('04-export.png', '08 / Kiểm và xuất nhãn', 'Kiểm hết ảnh, rồi Đánh dấu đã kiểm', 'Bộ đếm tăng thành 1 ảnh đã kiểm', (980,343,1211,395),
  ['Sau khi kiểm toàn bộ hộp và bổ sung vật thể bỏ sót, bấm Đánh dấu đã kiểm. Trong cảnh minh họa, bộ đếm đã tăng thành một ảnh.']),
 ('04-export.png', '09 / Tải CVAT XML', 'Bấm Xuất ảnh này · CVAT XML', 'Thông báo xuất 1 ảnh, 9 hộp', (980,397,1211,440),
  ['Bấm Xuất ảnh này, CVAT XML. Thông báo xác nhận một ảnh và chín hộp. Khi nhập vào CVAT, chọn CVAT for images một chấm một, với tên ảnh và nhãn khớp task.']),
 ('06-native-torch.png', 'Chạy thật trên laptop / PyTorch', 'Chọn CPU local torch rồi Chạy prelabel', 'Đọc nhãn Chạy thật · Local torch', (30,184,244,238),
  ['Trên laptop, mở Studio qua server Python local. Chọn PyTorch rồi bấm Chạy Prelabel. Đây là kết quả inference thật trên cùng ảnh, không phải replay.']),
 ('07-native-onnx.png', 'Chạy thật trên laptop / ONNX Runtime', 'Chọn CPU local onnx rồi chạy lại', 'Đọc nhãn Chạy thật · Local onnx', (30,184,244,238),
  ['Tiếp theo, chọn ONNX Runtime và chạy trên cùng ảnh. Lần nạp đầu và từng lần chạy có thể khác nhau, nên số tại đây không thay bảng benchmark nhiều lượt.']),
 ('08-native-fp16.png', 'Chạy thật trên laptop / OpenVINO FP16', 'Chọn CPU local openvino_fp16', 'Bấm chạy để suy luận mới', (30,184,244,238),
  ['Với OpenVINO FP mười sáu, cũng chọn runtime rồi bấm chạy. Các hộp được tạo mới trên CPU laptop.']),
 ('09-native-int8.png', 'Chạy thật trên laptop / OpenVINO INT8', 'Chọn CPU local openvino_int8', 'Kết quả mới từ model lượng tử hóa', (30,184,244,238),
  ['Cuối cùng, chọn OpenVINO INT tám và chạy lại. Model lượng tử hóa vẫn cần người kiểm nhãn. Trang public GitHub chỉ chạy ONNX Browser, vì Pages không có server Python.']),
 ('05-benchmark.png', '10 / Xem kết quả thực nghiệm', 'Mở Kết quả thực nghiệm trên Studio', 'Đọc điều kiện đo trước khi so sánh', (80,204,1170,244),
  ['Trang Kết quả thực nghiệm so sánh các runtime Python CPU. Phép đo dùng bốn threads, hai mươi sáu ảnh và ba lượt đo sau warmup.']),
 ('05-benchmark.png', 'INT8 trong phép đo CPU Python', 'So cùng workload và thread budget', '38,80 ms; tỷ số latency 3,96×', (80,449,1170,504),
  ['INT tám đạt mean latency ba mươi tám phẩy tám mili giây. Tỷ số latency nhanh hơn PyTorch ba phẩy chín sáu lần trong phép đo này. Dung lượng artifact INT tám là mười một phẩy hai bốn MiB, không phải RAM.']),
 ('../demo-day-build/slide-6.png', 'Giới hạn kết luận chất lượng', 'Phân biệt agreement và mAP', 'Cần ground truth độc lập', None,
  ['INT tám ghép được chín mươi bốn trên chín mươi bảy hộp PyTorch tại IoU từ không phẩy năm. Đây là độ đồng thuận giữa model. Dự án chưa có mAP trên ground truth độc lập.']),

]


def run(command):
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-4000:])
    return result


def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf', size)


def wrapped(draw, value, face, width):
    lines, line = [], ''
    for word in value.split():
        trial = (line + ' ' + word).strip()
        if line and draw.textlength(trial, font=face) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def draw_text(draw, value, xy, size, width, color='white', bold=False):
    face = font(size, bold)
    lines = wrapped(draw, value, face, width)
    for index, line in enumerate(lines):
        draw.text((xy[0], xy[1] + index * (size + 8)), line, font=face, fill=color)
    return len(lines) * (size + 8)


def render(scene, narration, destination):
    file, title, action, expected, region, _ = scene
    original = Image.open(ASSETS / file).convert('RGB')
    frame = Image.new('RGB', (1600, 900), '#101B28')
    draw = ImageDraw.Draw(frame)
    draw_text(draw, title, (28, 22), 36, 1540, bold=True)
    original.thumbnail((1180, 664), Image.Resampling.LANCZOS)
    frame.paste(original, (20, 92))
    if region:
        sx, sy = original.width / 1265, original.height / 712
        x1, y1, x2, y2 = region
        box = (20 + x1*sx, 92 + y1*sy, 20 + x2*sx, 92 + y2*sy)
        draw.rectangle(box, outline='#FF5E27', width=5)
    draw_text(draw, 'THAO TÁC', (1230, 105), 23, 340, '#FFAC83', True)
    end = draw_text(draw, action, (1230, 153), 29, 335, bold=True)
    y = max(315, 153 + end + 35)
    draw_text(draw, 'KẾT QUẢ CẦN THẤY', (1230, y), 21, 335, '#FFAC83', True)
    draw_text(draw, expected, (1230, y+48), 27, 335)
    draw_text(draw, 'Giọng đọc tổng hợp\nTiếng Việt', (1230, 640), 19, 335, '#BBC8D6')
    height = draw_text(draw, narration.replace('M ba tám','M38'), (28, 779), 26, 1540)
    assert height <= 102, f'Subtitle does not fit: {narration}'
    draw_text(draw, 'SAKANA · Cảnh ghép từ thao tác thật · Thời gian chờ đã cắt', (28, 873), 16, 1540, '#BBC8D6')
    frame.save(destination)


async def make_voice(index, narration):
    target = BUILD / f'voice-{index:02}.mp3'
    key = hashlib.sha256((VOICE+'pronunciation-v2'+narration).encode()).hexdigest()
    keyfile = target.with_suffix('.sha256')
    if not target.exists() or not keyfile.exists() or keyfile.read_text() != key:
        await edge_tts.Communicate(narration.replace('M ba tám', 'Mờ ba tám').replace('Prelabel', 'Pre lây bờ').replace('prelabel', 'Pre lây bờ'), VOICE, rate='+0%').save(str(target))
        keyfile.write_text(key)
    return target


def timestamp(seconds):
    ms = round(seconds * 1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


async def main():
    clips, subtitles, storyboard = [], [], []
    elapsed = 0.0
    voice_jobs = []
    for scene in SCENES:
        for piece in scene[-1][0].split('. '):
            voice_jobs.append((piece.strip()+'.').replace('..', '.'))
    semaphore = asyncio.Semaphore(3)
    async def generate(index, piece):
        async with semaphore:
            return await make_voice(index, piece)
    await asyncio.gather(*(generate(i, piece) for i, piece in enumerate(voice_jobs)))
    for index, scene in enumerate(SCENES):
        narration = scene[-1][0]
        # Split long captions into separate spoken sentences while preserving order.
        pieces = [part.strip()+'.' for part in narration.split('. ') if part.strip()]
        pieces = [part.replace('..', '.') for part in pieces]
        for part_index, piece in enumerate(pieces):
            number = len(clips)
            audio = await make_voice(number, piece)
            wav = BUILD / f'voice-{number:02}.wav'
            run([FF, '-y', '-i', str(audio), '-ar', '48000', '-ac', '1', str(wav)])
            with wave.open(str(wav)) as reader:
                speech_duration = reader.getnframes()/reader.getframerate()
            duration = speech_duration + .35
            frame = BUILD / f'frame-{number:02}.png'
            render(scene, piece, frame)
            clip = BUILD / f'clip-{number:02}.mp4'
            run([FF, '-y', '-loop', '1', '-framerate', '24', '-i', str(frame), '-i', str(wav),
                 '-t', str(duration), '-af', 'apad=pad_dur=0.35', '-c:v', 'libx264',
                 '-preset', 'ultrafast', '-crf', '20', '-pix_fmt', 'yuv420p',
                 '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', str(clip)])
            # Align caption starts to the actual encoded container duration.
            data = clip.read_bytes()
            pos = data.index(b'mvhd') + 4
            if data[pos] == 0:
                timescale, ticks = struct.unpack('>II', data[pos+12:pos+20])
            else:
                timescale = struct.unpack('>I', data[pos+20:pos+24])[0]
                ticks = struct.unpack('>Q', data[pos+24:pos+32])[0]
            duration = ticks / timescale
            clips.append(clip)
            subtitles.append(f'{number+1}\n{timestamp(elapsed)} --> {timestamp(elapsed+speech_duration)}\n{piece.replace('M ba tám','M38')}\n')
            storyboard.append({'start':round(elapsed,3),'duration':round(duration,3),'title':scene[1],
                               'action':scene[2],'expected':scene[3],'narration':piece,
                               'source':scene[0],'highlight':scene[4]})
            elapsed += duration
            print(f'Built clip {number+1}: {elapsed:.1f}s', flush=True)
    concat = BUILD / 'concat.txt'
    concat.write_text('\n'.join(f"file '{path.as_posix()}'" for path in clips))
    output = OUT / 'M38-Demo-Day-narrated.mp4'
    run([FF, '-y', '-f', 'concat', '-safe', '0', '-i', str(concat),
         '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
         '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-ar', '48000', '-c:a', 'aac', '-b:a', '160k',
         '-movflags', '+faststart', str(output)])
    run([FF, '-v', 'error', '-i', str(output), '-f', 'null', '-'])
    (OUT/'M38-Demo-Day-narrated.srt').write_text('\n'.join(subtitles), encoding='utf-8')
    script = ['# M38: video hướng dẫn có giọng đọc tiếng Việt\n',
              'Cách đọc: M38 = Mờ ba tám; Prelabel = Pre lây bờ. Phụ đề giữ M38 và Prelabel. Giọng tổng hợp: vi-VN-NamMinhNeural. Cảnh ghép từ ảnh chụp thao tác thật, không phải quay liên tục. Thời gian chờ đã cắt. Các khung cam chỉ vị trí cần chú ý.\n',
              '| Bắt đầu | Thao tác | Kết quả cần thấy | Lời đọc |', '|---|---|---|---|']
    for item in storyboard:
        script.append(f'| {timestamp(item["start"])} | {item["action"]} | {item["expected"]} | {item["narration"].replace("M ba tám", "M38")} |')
    script.extend(['\n## Link sản phẩm', 'https://manhld424.github.io/M38-Edge-PreLabeling/',
                   'https://manhld424.github.io/M38-Edge-PreLabeling/results.html'])
    (OUT/'M38-Demo-Day-narrated-script.md').write_text('\n'.join(script), encoding='utf-8')
    evidence = {'voice':VOICE,'seconds':round(elapsed,3),'resolution':[1600,900],
                'scenes':len(SCENES),'clips':len(clips),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                'bytes':output.stat().st_size,'decode_check':'passed','storyboard':storyboard}
    (BUILD/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k!='storyboard'}),flush=True)


asyncio.run(main())
