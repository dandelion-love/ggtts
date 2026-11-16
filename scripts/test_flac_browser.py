#!/usr/bin/env python3
"""Real-time Chrome FLAC regression; requires cargo, ffmpeg, ffprobe and Chromium."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def crc(data, width, poly):
    value = 0
    for byte in data:
        value ^= byte << (width - 8)
        for _ in range(8):
            value = ((value << 1) ^ poly if value & (1 << (width - 1)) else value << 1) & ((1 << width) - 1)
    return value

def plant_fixed(data):
    """Rewrite variable-block frame headers to the earlier defective fixed-block scheme."""
    total = int.from_bytes(data[18:26], "big") & ((1 << 36) - 1)
    result = bytearray(data[:42])
    pos = 42
    frame_number = 0
    offset = 0
    while offset < total:
        n = min(4096, total - offset)
        head = data[pos:pos+4]
        if len(head) != 4 or head[0] != 255 or not head[1] & 1:
            raise AssertionError("expected generated variable-block frame")
        lead = data[pos+4]
        width = 1
        # Count leading one bits directly for the FLAC UTF-8-like number.
        if lead >= 128:
            width = 0
            while lead & (0x80 >> width):
                width += 1
        old_header = 4 + width + (2 if n != 4096 else 0) + 1
        frame_len = old_header + 1 + 2*n + 2
        frame = data[pos:pos+frame_len]
        if len(frame) != frame_len or frame_number >= 128:
            raise AssertionError("unexpected fixture size")
        header = bytearray(head)
        header[1] &= 0xfe
        header.append(frame_number)
        if n != 4096:
            header.extend((n-1).to_bytes(2, "big"))
        header.append(crc(header, 8, 0x07))
        planted = header + frame[old_header:-2]
        planted.extend(crc(planted, 16, 0x8005).to_bytes(2, "big"))
        result.extend(planted)
        pos += frame_len
        offset += n
        frame_number += 1
    if pos != len(data) or total % 4096 == 0:
        raise AssertionError("fixture must end with a short final block")
    return bytes(result)

class CDP:
    def __init__(self, url):
        from urllib.parse import urlsplit
        u = urlsplit(url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=12)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if not response.startswith(b"HTTP/1.1 101"):
            raise RuntimeError("Chrome websocket refused")
    def _read(self, n):
        data = b""
        while len(data) < n:
            part = self.sock.recv(n-len(data))
            if not part:
                raise RuntimeError("Chrome websocket closed")
            data += part
        return data
    def call(self, method, params):
        payload = json.dumps({"id":1,"method":method,"params":params}).encode()
        mask = os.urandom(4)
        length = len(payload)
        header = bytes([0x81, 0x80 | (length if length < 126 else 126)])
        if length >= 126:
            header += struct.pack("!H", length)
        self.sock.sendall(header + mask + bytes(v ^ mask[i % 4] for i,v in enumerate(payload)))
        while True:
            first, second = self._read(2)
            n = second & 127
            if n == 126:
                n = struct.unpack("!H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._read(8))[0]
            data = self._read(n)
            if first & 15 != 1:
                continue
            msg = json.loads(data)
            if msg.get("id") == 1:
                if "error" in msg:
                    raise RuntimeError(str(msg["error"]))
                return msg["result"]

def browser_result(chrome, html, profile):
    proc = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
        "--mute-audio", "--autoplay-policy=no-user-gesture-required",
        "--disable-background-networking", "--no-first-run", "--remote-debugging-port=0",
        "--remote-allow-origins=*", f"--user-data-dir={profile}", html.as_uri()],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        active = profile / "DevToolsActivePort"
        for _ in range(100):
            if active.exists():
                break
            if proc.poll() is not None:
                raise RuntimeError("Chrome exited before debugging endpoint")
            time.sleep(0.05)
        port = int(active.read_text().splitlines()[0])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=5) as r:
            targets = json.load(r)
        target = next(t for t in targets if t["type"] == "page")
        client = CDP(target["webSocketDebuggerUrl"])
        for _ in range(100):
            ready = client.call("Runtime.evaluate", {"expression":"!!document.querySelector('audio')", "returnByValue":True})
            if ready.get("result", {}).get("value") is True:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("audio page did not load")
        expression = """(async()=>{const a=document.querySelector('audio');
          return await new Promise(resolve=>{
            const timer=setTimeout(()=>resolve({outcome:'timeout',time:a.currentTime,error:a.error?.message}),8000);
            a.onended=()=>{clearTimeout(timer);resolve({outcome:'ended',time:a.currentTime});};
            a.onerror=()=>{clearTimeout(timer);resolve({outcome:'error',code:a.error.code,message:a.error.message});};
            a.play().catch(e=>{clearTimeout(timer);resolve({outcome:'rejected',name:e.name,message:a.error?.message});});
          });})()"""
        response = client.call("Runtime.evaluate", {"expression":expression,"awaitPromise":True,"returnByValue":True})
        if "exceptionDetails" in response:
            raise RuntimeError(str(response["exceptionDetails"]))
        return response["result"]["value"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

def main():
    chrome = os.environ.get("CHROME_BIN") or shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("requires Chrome/Chromium, ffmpeg and ffprobe")
    cargo = str(Path.home()/".cargo/bin/cargo")
    subprocess.run([cargo, "build", "--quiet", "-p", "ggtts-cli", "--bin", "ggtts-render"], cwd=ROOT, check=True)
    binary = ROOT/"target/debug/ggtts-render"
    with tempfile.TemporaryDirectory(prefix="ggtts-flac-browser-") as tmp:
        d=Path(tmp); story=d/"story.txt"; story.write_text("Synthetic read\nA中𠀀文", encoding="utf-8")
        original=d/"fixed.flac"
        subprocess.run([str(binary),"--engine","fake","--story",str(story),"--out",str(original),"--log-level","error"],check=True)
        good=original.read_bytes(); bad=plant_fixed(good); planted=d/"planted.flac"; planted.write_bytes(bad)
        def pcm(path):
            return subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","s16le","-"],capture_output=True,check=True).stdout
        good_pcm, bad_pcm = pcm(original), pcm(planted)
        if not good_pcm or good_pcm != bad_pcm:
            raise AssertionError("plant must preserve decoded PCM")
        print("PCM sha256", hashlib.sha256(good_pcm).hexdigest(), flush=True)
        def pts(path):
            p=subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries","packet=pts","-of","csv=p=0",str(path)],capture_output=True,text=True,check=True)
            return p.stdout.splitlines()
        if "N/A" in pts(original) or "N/A" not in pts(planted):
            raise AssertionError("PTS control did not distinguish the planted final frame")
        for name, data in (("good",good),("plant",bad)):
            page=d/(name+".html")
            uri="data:audio/flac;base64,"+base64.b64encode(data).decode("ascii")
            page.write_text("<!doctype html><audio src='"+uri+"'></audio>",encoding="utf-8")
            observed=browser_result(chrome,page,d/(name+"-profile"))
            print(name,observed,flush=True)
            if name=="good" and observed.get("outcome")!="ended":
                raise AssertionError("fixed FLAC did not finish browser playback")
            if name=="plant" and (observed.get("outcome") not in ("error","rejected") or "PTS is not defined" not in (observed.get("message") or "")):
                raise AssertionError("planted defect did not trigger the expected demux error")

if __name__=="__main__":
    main()
