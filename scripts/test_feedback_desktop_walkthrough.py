"""CLI-only, headless desktop walkthrough for offline feedback.

Generates a synthetic tone and a valid input/provenance pair, uses Chromium's
DevTools protocol across two separate browser processes to prove playback,
marking/export, and localStorage draft restoration, then saves through the
desktop host CLI. It never opens a visible browser window.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feedback_page import generate, input_digest


def _recv_exact(sock, n):
    chunks = []
    while n:
        part = sock.recv(n)
        if not part:
            raise RuntimeError("Chrome DevTools websocket closed")
        chunks.append(part)
        n -= len(part)
    return b"".join(chunks)


def _ws_send(sock, payload):
    data = json.dumps(payload).encode("utf-8")
    mask = os.urandom(4)
    n = len(data)
    if n < 126:
        header = bytes([0x81, 0x80 | n])
    elif n < 65536:
        header = bytes([0x81, 0x80 | 126]) + struct.pack("!H", n)
    else:
        header = bytes([0x81, 0x80 | 127]) + struct.pack("!Q", n)
    masked = bytes(v ^ mask[i % 4] for i, v in enumerate(data))
    sock.sendall(header + mask + masked)


def _ws_read(sock):
    first, second = _recv_exact(sock, 2)
    opcode = first & 0x0f
    n = second & 0x7f
    if n == 126:
        n = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4) if second & 0x80 else None
    data = _recv_exact(sock, n)
    if mask:
        data = bytes(v ^ mask[i % 4] for i, v in enumerate(data))
    if opcode == 9:
        # Chrome may ping while media decodes.
        mask = os.urandom(4)
        masked = bytes(v ^ mask[i % 4] for i, v in enumerate(data))
        sock.sendall(bytes([0x8a, 0x80 | len(data)]) + mask + masked)
        return None
    if opcode == 8:
        raise RuntimeError("Chrome DevTools websocket closed")
    return json.loads(data.decode("utf-8")) if opcode == 1 else None


class DevTools:
    def __init__(self, url):
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        self.sock = socket.create_connection((parts.hostname, parts.port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {parts.path} HTTP/1.1\r\nHost: {parts.hostname}:{parts.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if not response.startswith(b"HTTP/1.1 101"):
            raise RuntimeError("Chrome DevTools websocket handshake failed: " + response[:200].decode("ascii", "replace"))
        self.next_id = 0

    def call(self, method, params=None):
        self.next_id += 1
        ident = self.next_id
        _ws_send(self.sock, {"id": ident, "method": method, "params": params or {}})
        while True:
            message = _ws_read(self.sock)
            if message and message.get("id") == ident:
                if "error" in message:
                    raise RuntimeError(f"DevTools {method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression, timeout=20):
        result = self.call("Runtime.evaluate", {
            "expression": expression, "awaitPromise": True, "returnByValue": True, "userGesture": True,
        })
        if result.get("exceptionDetails"):
            raise RuntimeError("browser JavaScript failed: " + json.dumps(result["exceptionDetails"]))
        return result.get("result", {}).get("value")

    def close(self):
        self.sock.close()


def _terminate(proc):
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        proc.terminate()

def launch(chrome, profile, page):
    active = profile / "DevToolsActivePort"
    try:
        active.unlink()
    except FileNotFoundError:
        pass
    proc = subprocess.Popen([
        chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-background-networking", "--disable-sync", "--no-first-run",
        "--autoplay-policy=no-user-gesture-required", "--remote-debugging-port=0",
        f"--user-data-dir={profile}", page.as_uri(),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    port = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Chrome exited early with status {proc.returncode}")
        try:
            lines = active.read_text().splitlines()
            port = int(lines[0])
            break
        except (OSError, ValueError, IndexError):
            time.sleep(0.1)
    if port is None:
        _terminate(proc)
        raise RuntimeError("Chrome did not publish its DevTools port")
    deadline = time.monotonic() + 20
    target = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as response:
                tabs = json.load(response)
            target = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
            if target:
                break
        except Exception:
            time.sleep(0.1)
    if not target:
        _terminate(proc)
        raise RuntimeError("Chrome did not publish a page target")
    return proc, DevTools(target["webSocketDebuggerUrl"])


def stop(proc, client, profile):
    client.close()
    try:
        lines = (profile / "DevToolsActivePort").read_text().splitlines()
        port = int(lines[0])
        browser = DevTools(f"ws://127.0.0.1:{port}{lines[1]}")
        try:
            browser.call("Browser.close")
        finally:
            browser.close()
    except Exception:
        _terminate(proc)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


FIRST = r"""(async()=>{
 const a=document.querySelector('audio');
 if(!a) throw Error('audio element missing');
 a.preload='metadata';a.load();
 await new Promise((resolve,reject)=>{
   if(a.readyState>=1)return resolve();
   a.addEventListener('loadedmetadata',resolve,{once:true});
   a.addEventListener('error',()=>reject(Error('audio decode error '+(a.error&&a.error.code))),{once:true});
   setTimeout(()=>reject(Error('metadata timeout')),5000);
 });
 let started=false, ended=false;
 a.addEventListener('playing',()=>started=true,{once:true});
 a.addEventListener('ended',()=>ended=true,{once:true});
 await a.play();
 await new Promise((resolve,reject)=>{
   if(a.ended){ended=true;return resolve();}
   a.addEventListener('ended',()=>{ended=true;resolve();},{once:true});
   a.addEventListener('error',()=>reject(Error('audio error '+(a.error&&a.error.code))),{once:true});
   setTimeout(()=>reject(Error('audio did not end; duration='+a.duration+' currentTime='+a.currentTime)),5000);
 });
 const chars=[...document.querySelectorAll('button.character')];
 if(chars.length!==3) throw Error('expected 3 CJK controls, got '+chars.length);
 chars[2].click();
 if(chars[2].getAttribute('aria-pressed')!=='true') throw Error('mark click failed');
 const rating=document.querySelector('select[aria-label^="Rating for"]');
 rating.value='accept';rating.dispatchEvent(new Event('change',{bubbles:true}));
 const itemNote=document.querySelector('section.item textarea');
 itemNote.value='synthetic item note';itemNote.dispatchEvent(new Event('input',{bubbles:true}));
 const note=document.getElementById('read-note');
 note.value='synthetic read note';note.dispatchEvent(new Event('input',{bubbles:true}));
 const external=performance.getEntriesByType('resource').map(x=>x.name)
   .filter(x=>/^https?:/.test(x));
 if(external.length) throw Error('external network access: '+external.join(','));
 const csp=document.querySelector('meta[http-equiv="Content-Security-Policy"]').content;
 if(!csp.includes("default-src 'none'")||!csp.includes("connect-src 'none'")) throw Error('offline CSP missing');
 window.__savedBlob=null;
 URL.createObjectURL=b=>{window.__savedBlob=b;return 'blob:desktop-walkthrough';};
 HTMLAnchorElement.prototype.click=function(){};
 document.getElementById('export').click();
 for(let i=0;i<50&&!window.__savedBlob;i++) await new Promise(r=>setTimeout(r,20));
 if(!window.__savedBlob) throw Error('export did not create a file');
 return {result:JSON.parse(await window.__savedBlob.text()),played:{started,ended,duration:a.duration,currentTime:a.currentTime},
         offline:external.length===0};
})()"""

SECOND = r"""(async()=>{
 const chars=[...document.querySelectorAll('button.character')];
 const rating=document.querySelector('select[aria-label^="Rating for"]');
 const itemNote=document.querySelector('section.item textarea');
 const note=document.getElementById('read-note');
 const restored={mark:chars[2]?.getAttribute('aria-pressed'),rating:rating?.value,
   itemNote:itemNote?.value,readNote:note?.value};
 if(restored.mark!=='true'||restored.rating!=='accept'||restored.itemNote!=='synthetic item note'||
    restored.readNote!=='synthetic read note') throw Error('draft did not survive browser restart: '+JSON.stringify(restored));
 window.__savedBlob=null;
 URL.createObjectURL=b=>{window.__savedBlob=b;return 'blob:desktop-walkthrough';};
 HTMLAnchorElement.prototype.click=function(){};
 document.getElementById('export').click();
 for(let i=0;i<50&&!window.__savedBlob;i++) await new Promise(r=>setTimeout(r,20));
 if(!window.__savedBlob) throw Error('re-export failed');
 return {result:JSON.parse(await window.__savedBlob.text()),restored};
})()"""


def main():
    chrome = os.environ.get("CHROME_BIN") or shutil.which("google-chrome") or shutil.which("chromium")
    ffmpeg = shutil.which("ffmpeg")
    if not chrome or not ffmpeg:
        raise SystemExit("SKIP: requires headless Chromium and ffmpeg")
    with tempfile.TemporaryDirectory(prefix="ggtts-desktop-walkthrough-") as tmp:
        directory = Path(tmp)
        audio = directory / "synthetic-read.flac"
        page = directory / "trusted-feedback.html"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:sample_rate=24000:duration=0.25", "-c:a", "flac",
                        "-f", "flac", str(audio)], check=True)
        raw_audio = audio.read_bytes()
        packed = int.from_bytes(raw_audio[18:26], "big")
        rate, samples = packed >> 44, packed & ((1 << 36) - 1)
        item_id = "clip-1"
        text_value = chr(65) + chr(0x4e2d) + chr(0x20000) + chr(0x6587)
        audio_hash = hashlib.sha256(raw_audio).hexdigest()
        source = {"format":"ggtts-feedback-input/1","id":"desktop-walkthrough","items":[
            {"id":item_id,"text":text_value,"audio":"data:audio/flac;base64,"+base64.b64encode(raw_audio).decode("ascii")}]}
        page.write_text(generate(source), encoding="utf-8")
        companion = {
            "format":"ggtts-feedback-provenance/1","id":source["id"],
            "input_sha256":input_digest(source),"output_sha256":audio_hash,
            "sample_rate":rate,"output_samples":samples,
            "render":{"audio":{"sha256":audio_hash}},
            "chunks":[{"id":item_id,"chunk":0,"draw":0,"seed":20261116,"start":0,"end":samples,
                       "text_sha256":hashlib.sha256(text_value.encode("utf-8")).hexdigest(),
                       "audio_sha256":audio_hash,"stop":"eos"}],
        }
        Path(str(page)+".provenance.json").write_text(json.dumps(companion), encoding="utf-8")
        profile = directory / "chrome-profile"
        proc, browser = launch(chrome, profile, page)
        try:
            browser.call("Page.enable")
            first = browser.evaluate(FIRST)
        finally:
            stop(proc, browser, profile)
        proc, browser = launch(chrome, profile, page)
        try:
            browser.call("Page.enable")
            second = browser.evaluate(SECOND)
        finally:
            stop(proc, browser, profile)
        if first["result"] != second["result"]:
            raise AssertionError("browser restart changed exported result")
        if not first["played"]["ended"] or not first["offline"]:
            raise AssertionError("headless play/offline evidence missing")
        result_file = directory / "exported-feedback.json"
        result_file.write_text(json.dumps(second["result"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        saved = subprocess.run([
            sys.executable, str(ROOT / "scripts/feedback_desktop.py"), "save",
            str(page), str(result_file), "--audio", str(audio),
        ], cwd=ROOT, capture_output=True, text=True)
        destination = Path(str(audio) + ".feedback.json")
        if saved.returncode or not destination.is_file():
            raise RuntimeError("desktop save failed: " + saved.stderr)
        stored = json.loads(destination.read_text(encoding="utf-8"))
        if stored != second["result"]:
            raise AssertionError("saved feedback differs from browser export")
        again = subprocess.run([
            sys.executable, str(ROOT / "scripts/feedback_desktop.py"), "save",
            str(page), str(result_file), "--audio", str(audio),
        ], cwd=ROOT, capture_output=True, text=True)
        if again.returncode == 0:
            raise AssertionError("host overwrote an existing result")
        print("PASS: real-time headless Chromium played synthetic FLAC to ended, marked CJK index 2, exported rating and notes offline; a separate browser process restored the draft; host save wrote the export beside audio and refused overwrite.")

if __name__ == "__main__":
    main()
