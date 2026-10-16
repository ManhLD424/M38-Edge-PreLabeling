"""Loopback inference server sharing the exact same UI as GitHub Pages."""
from __future__ import annotations
import json
import math
import mimetypes
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
import cv2
from src.core.inference.factory import MODELS as MODEL_PATHS, make_engine

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / 'docs'
SAMPLE_ROOT = ROOT / 'data/sample_images'
CACHED_ENGINES = {}
ENGINE_LOCK = threading.Lock()


def get_engine(runtime):
    if runtime not in MODEL_PATHS:
        raise ValueError('Unknown runtime')
    if runtime not in CACHED_ENGINES:
        CACHED_ENGINES[runtime] = make_engine(runtime)
    return CACHED_ENGINES[runtime]


def sample_path(name):
    if not isinstance(name, str) or not name or '/' in name or '\\' in name or Path(name).name != name:
        raise ValueError('Invalid sample name')
    path = (SAMPLE_ROOT / name).resolve()
    if not path.is_relative_to(SAMPLE_ROOT.resolve()) or path.suffix.lower() not in ('.jpg','.jpeg','.png'):
        raise ValueError('Invalid sample name')
    if not path.is_file():
        raise FileNotFoundError('Sample not found')
    return path


class WebMVPHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, status=200):
        self._send(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8', status)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/api/samples':
                self._send_json({'samples':sorted(p.name for p in SAMPLE_ROOT.glob('*.jpg'))})
            elif parsed.path == '/api/capabilities':
                self._send_json({'runtimes':[key for key,path in MODEL_PATHS.items() if (ROOT/path).is_file()]})
            elif parsed.path == '/api/image':
                path = sample_path(parse_qs(parsed.query).get('name',[''])[0])
                self._send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'image/jpeg')
            elif parsed.path.startswith('/api/'):
                self._send_json({'error':'Endpoint not found'},404)
            else:
                self._serve_static(parsed.path)
        except ValueError as exc:
            self._send_json({'error':str(exc)},400)
        except (FileNotFoundError, IsADirectoryError):
            self._send_json({'error':'File not found'},404)

    def _serve_static(self, request_path):
        relative = unquote(request_path).lstrip('/') or 'index.html'
        if relative in ('mvp/','mvp/index.html'):
            relative = 'index.html'
        path = (WEB_ROOT / relative).resolve()
        if not path.is_relative_to(WEB_ROOT.resolve()):
            raise ValueError('Invalid path')
        if not path.is_file():
            raise FileNotFoundError(path)
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix == '.mjs':
            mime = 'text/javascript'
        self._send(path.read_bytes(), mime)

    def do_POST(self):
        if urlparse(self.path).path != '/api/predict':
            self._send_json({'error':'Endpoint not found'},404)
            return
        try:
            length = int(self.headers.get('Content-Length','0'))
            if length <= 0 or length > 4096:
                self._send_json({'error':'Request body must be 1..4096 bytes'},413)
                self.close_connection = True
                return
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            if not isinstance(data,dict):
                raise ValueError('Expected JSON object')
            runtime = data.get('runtime','openvino_fp16')
            if runtime not in MODEL_PATHS:
                raise ValueError('Unknown runtime')
            conf = float(data.get('conf',.25))
            if not math.isfinite(conf) or not 0 <= conf <= 1:
                raise ValueError('Confidence must be between 0 and 1')
            path = sample_path(data.get('image_name','BDD01.jpg'))
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Unreadable image')
            # Serialize runtime mutation and inference: thresholds cannot leak across concurrent requests.
            with ENGINE_LOCK:
                engine = get_engine(runtime)
                engine.conf_thresh = conf
                start = time.perf_counter()
                detections = engine.predict(image)
                elapsed = (time.perf_counter()-start)*1000
            from src.core.benchmark.memory import MemoryBenchmark
            import hashlib
            self._send_json({'image_name':path.name,'width':image.shape[1],'height':image.shape[0],
                'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'runtime':runtime,'conf':conf,'latency_ms':round(elapsed,2),
                'model_size_mb':MemoryBenchmark.get_model_size_mb(engine.model_path),
                'detections':[{'class_name':d.class_name,'class_id':d.class_id,
                    'confidence':d.confidence,'box_xyxy':list(d.box_xyxy)} for d in detections]})
        except (ValueError, TypeError, UnicodeError) as exc:
            self._send_json({'error':str(exc)},400)
        except FileNotFoundError:
            self._send_json({'error':'Requested sample/model is missing'},404)
        except Exception:
            self._send_json({'error':'Inference failed; check the local model and server logs'},500)


def run_server(port=8080, host='127.0.0.1'):
    server = ThreadingHTTPServer((host,port),WebMVPHandler)
    print(f'M38 server: http://{host}:{server.server_port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    import sys
    run_server(int(sys.argv[1]) if len(sys.argv)>1 else 8080)
