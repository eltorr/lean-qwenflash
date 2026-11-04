"""Minimal HTTP/1.1 plumbing on asyncio streams (stdlib only; one backend connection per request)."""
import json

HOP = {"connection", "keep-alive", "proxy-connection", "host", "content-length", "transfer-encoding", "upgrade", "te",
       "expect"}
REASONS = {200: "OK", 302: "Found", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 409: "Conflict",
           502: "Bad Gateway", 503: "Service Unavailable"}


async def read_request(reader, writer=None):
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, target, _ver = lines[0].split(" ", 2)
    headers = []
    for line in lines[1:]:
        if line:
            k, _, v = line.partition(":")
            headers.append((k.strip(), v.strip()))
    hmap = {k.lower(): v for k, v in headers}
    if writer is not None and hmap.get("expect", "").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")      # curl & co. wait for this before sending large bodies
        await writer.drain()
    body = b""
    if "content-length" in hmap:
        body = await reader.readexactly(int(hmap["content-length"]))
    elif hmap.get("transfer-encoding", "").lower() == "chunked":
        parts = []
        while True:
            size = int((await reader.readuntil(b"\r\n")).split(b";")[0], 16)
            if size == 0:
                await reader.readuntil(b"\r\n")
                break
            parts.append(await reader.readexactly(size))
            await reader.readexactly(2)
        body = b"".join(parts)
    return method, target, headers, body


def backend_request(method, target, headers, body, port):
    out = [f"{method} {target} HTTP/1.1"]
    out += [f"{k}: {v}" for k, v in headers if k.lower() not in HOP]
    out += [f"Host: 127.0.0.1:{port}", "Connection: close"]
    if body or method in ("POST", "PUT", "PATCH"):
        out.append(f"Content-Length: {len(body)}")
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body


async def respond(writer, status, body, ctype="application/json", extra=()):
    if not isinstance(body, (bytes, bytearray)):
        body = json.dumps(body).encode()
    head = [f"HTTP/1.1 {status} {REASONS.get(status, 'OK')}", f"Content-Type: {ctype}",
            f"Content-Length: {len(body)}", "Cache-Control: no-store", "Connection: close", *extra]
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + body)
    try:
        await writer.drain()
    except ConnectionError:
        pass


def close(*writers):
    for w in writers:
        if w:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass


def oai_error(msg, kind="server_busy", code=503):
    return {"error": {"message": msg, "type": kind, "code": code}}


def http_get_json(url, timeout=2.0):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def http_ok(url, timeout=2.0):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


class StreamStats:
    """Watches a proxied completion response to time it: first token, end, and the usage block."""

    def __init__(self):
        self.t_first = None
        self.tail = b""
        self.head = b""
        self.headers_done = False
        self.events = 0
        self.status = None

    def feed(self, chunk, now):
        if not self.headers_done:
            self.head += chunk
            i = self.head.find(b"\r\n\r\n")
            if i < 0:
                return
            self.headers_done = True
            try:
                self.status = int(self.head.split(b" ", 2)[1])
            except (IndexError, ValueError):
                pass
            chunk = self.head[i + 4:]
        if self.t_first is None:
            # SSE: the first event only announces the role; the second carries the first generated token
            self.events += chunk.count(b"data:")
            if self.events >= 2:
                self.t_first = now
        self.tail = (self.tail + chunk)[-8192:]

    def usage(self):
        i = self.tail.rfind(b'"usage"')
        if i < 0:
            return None
        j = self.tail.find(b"{", i)
        depth = 0
        for k in range(j, len(self.tail)):
            depth += {ord("{"): 1, ord("}"): -1}.get(self.tail[k], 0)
            if depth == 0:
                try:
                    return json.loads(self.tail[j:k + 1])
                except ValueError:
                    return None
        return None


async def pipe(reader, writer, on_chunk=None):
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            break
        if on_chunk:
            on_chunk(chunk)
        writer.write(chunk)
        await writer.drain()
