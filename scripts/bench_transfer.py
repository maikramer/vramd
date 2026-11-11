#!/usr/bin/env python3
"""Benchmark: staging JSON+base64 (0.6) vs frames binários+blobs (0.7).

Sobe um `VramdServer` local num Unix socket e mede upload/download de um
ficheiro incompressível. O caminho antigo usa `stage`+`put` em chunks de 2 MiB
base64 (uma ligação por chunk); o novo usa `blob-put`/`blob-get` numa ligação
persistente, sem base64.

    python scripts/bench_transfer.py --size-mib 64
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from vramd import protocol as P
from vramd import transport
from vramd.registry import Registry
from vramd.server import VramdServer
from vramd.transfer import TransferClient

CHUNK = 2 * 1024 * 1024


def _start(tmp: Path) -> tuple[VramdServer, Path]:
    socket_path = tmp / "vramd.sock"
    server = VramdServer(
        registry=Registry(descriptors={}),
        socket_path=socket_path,
        query_free_mib=lambda: 99999,
        clear_vram=lambda: None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not socket_path.exists():
        raise SystemExit("vramd não arrancou")
    return server, socket_path


def _json_rpc(socket_path: Path, request: dict) -> dict:
    sock, token = transport.connect(socket_path, timeout=300)
    with sock:
        sock.sendall((json.dumps(transport.with_auth(request, token)) + "\n").encode())
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode().strip().split("\n")[-1])


def bench_legacy(socket_path: Path, data: bytes) -> float:
    job = _json_rpc(socket_path, {"cmd": P.CMD_STAGE})["job"]
    sha = hashlib.sha256(data).hexdigest()
    started = time.perf_counter()
    for offset in range(0, len(data), CHUNK):
        part = data[offset : offset + CHUNK]
        last = offset + len(part) >= len(data)
        response = _json_rpc(
            socket_path,
            {
                "cmd": P.CMD_PUT,
                "job": job,
                "path": "in/payload.bin",
                "offset": offset,
                "data": base64.b64encode(part).decode(),
                "eof": last,
                "sha256": sha if last else None,
            },
        )
        if response.get("status") != "ok":
            raise SystemExit(f"put falhou: {response}")
    elapsed = time.perf_counter() - started
    _json_rpc(socket_path, {"cmd": P.CMD_CLEAN, "job": job})
    return elapsed


def bench_binary_upload(socket_path: Path, data: bytes) -> tuple[float, str]:
    sha = hashlib.sha256(data).hexdigest()
    with TransferClient(socket_path=socket_path, timeout=600) as client:
        started = time.perf_counter()
        client.put_blob_bytes(sha, data)
        elapsed = time.perf_counter() - started
    return elapsed, sha


def bench_binary_download(socket_path: Path, sha: str, size: int) -> float:
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "payload.bin"
        with TransferClient(socket_path=socket_path, timeout=600) as client:
            started = time.perf_counter()
            client.get_blob(sha, target)
            elapsed = time.perf_counter() - started
        if target.stat().st_size != size:
            raise SystemExit("download incompleto")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=64)
    args = parser.parse_args()
    size = args.size_mib * 1024 * 1024
    data = os.urandom(size)
    print(f"payload: {args.size_mib} MiB incompressível")

    with tempfile.TemporaryDirectory(prefix="vramd-bench-") as td:
        server, socket_path = _start(Path(td))
        try:
            legacy = bench_legacy(socket_path, data)
            upload, sha = bench_binary_upload(socket_path, data)
            download = bench_binary_download(socket_path, sha, size)
        finally:
            server._running = False
        mib = size / (1024 * 1024)
        print(f"legacy  upload  {mib / legacy:8.1f} MiB/s  ({legacy:.3f}s, base64, 1 TCP/chunk)")
        print(f"binary  upload  {mib / upload:8.1f} MiB/s  ({upload:.3f}s, 1 conexão, sem base64)")
        print(f"binary  download{mib / download:8.1f} MiB/s  ({download:.3f}s)")
        print(f"speedup upload: {legacy / upload:.2f}x")


if __name__ == "__main__":
    main()
