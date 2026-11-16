#!/usr/bin/env python3
"""Run the page in real-time headless Chromium on Linux (POSIX process-group cleanup)."""
import base64, json, os, pathlib, signal, socket, struct, subprocess, sys, tempfile, time, urllib.request
from urllib.parse import urlsplit

def send(sock, value):
    data = json.dumps(value).encode()
    mask = os.urandom(4)
    length = len(data)
    header = bytes([0x81, 0x80 | length]) if length < 126 else (
        bytes([0x81, 0x80 | 126]) + struct.pack("!H", length) if length < 65536 else
        bytes([0x81, 0x80 | 127]) + struct.pack("!Q", length))
    sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

def exact(sock, n):
    data = b""
    while len(data) < n:
        part = sock.recv(n - len(data))
        if not part:
            raise RuntimeError("DevTools socket closed")
        data += part
    return data

def receive(sock):
    header = exact(sock, 2)
    length = header[1] & 127
    if length == 126:
        length = struct.unpack("!H", exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", exact(sock, 8))[0]
    mask = exact(sock, 4) if header[1] & 128 else None
    data = exact(sock, length)
    if mask:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return json.loads(data)

def main():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("browser driver supports Linux only")
    def terminate(_signum, _frame):
        raise RuntimeError("browser driver terminated")
    signal.signal(signal.SIGTERM, terminate)
    chrome, page, pgid_file = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="ggtts-chrome-profile-") as profile:
        process = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--mute-audio",
            "--disable-background-networking", "--disable-sync", "--no-first-run",
            "--autoplay-policy=no-user-gesture-required", "--host-resolver-rules=MAP * ~NOTFOUND",
            "--remote-debugging-port=0", "--user-data-dir=" + profile, pathlib.Path(page).resolve().as_uri()],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"))
        try:
            pathlib.Path(pgid_file).write_text(str(process.pid), encoding="ascii")
            port_file = pathlib.Path(profile) / "DevToolsActivePort"
            deadline = time.monotonic() + 20
            while not port_file.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Chromium failed to start")
                time.sleep(0.05)
            port = int(port_file.read_text().splitlines()[0])
            with urllib.request.urlopen("http://127.0.0.1:%d/json" % port, timeout=5) as r:
                targets = json.load(r)
            url = next(x["webSocketDebuggerUrl"] for x in targets if x["type"] == "page")
            parsed = urlsplit(url)
            sock = socket.create_connection((parsed.hostname, parsed.port), timeout=5)
            key = base64.b64encode(os.urandom(16)).decode()
            request = ("GET %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
                       "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" %
                       (parsed.path, parsed.port, key))
            sock.sendall(request.encode())
            response = b""
            while not response.endswith(b"\r\n\r\n"):
                response += exact(sock, 1)
            if b" 101 " not in response:
                raise RuntimeError("DevTools websocket handshake failed")
            deadline = time.monotonic() + 20
            ident = 0
            while time.monotonic() < deadline:
                ident += 1
                send(sock, {"id": ident, "method": "Runtime.evaluate",
                    "params": {"expression": "JSON.stringify({state:document.body?.dataset.hostTest, error:document.body?.dataset.hostErrors, result:document.body?.dataset.hostResult})", "returnByValue": True}})
                while True:
                    response = receive(sock)
                    if response.get("id") == ident:
                        break
                value = response.get("result", {}).get("result", {}).get("value")
                result = json.loads(value) if value else {}
                if result.get("state") in ("PASS", "FAIL"):
                    print(json.dumps(result))
                    return 0 if result["state"] == "PASS" else 1
                time.sleep(0.05)
            raise RuntimeError("host page did not finish")
        finally:
            # Chrome has subprocesses: terminate its dedicated process group, not just
            # the browser leader. SIGTERM from the Dart deadline enters this finally.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            pathlib.Path(pgid_file).unlink(missing_ok=True)

if __name__ == "__main__":
    sys.exit(main())
