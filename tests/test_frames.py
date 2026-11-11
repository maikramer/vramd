"""Codec de frames do protocolo v2 (binário, length-prefixed)."""

from __future__ import annotations

import json
import os
import socket
import threading
import zlib

import pytest

from vramd import frames


def _pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


def _send_in_thread(sock: socket.socket, data: bytes, *, chunk: int = 65_536) -> threading.Thread:
    """Envia data em chunks numa thread (socketpair tem buffer pequeno)."""

    def run() -> None:
        try:
            for start in range(0, len(data), chunk):
                sock.sendall(data[start : start + chunk])
        finally:
            sock.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class TestFrameRoundtrip:
    @pytest.mark.parametrize(
        "frame_type",
        [frames.TYPE_JSON, frames.TYPE_DATA, frames.TYPE_END, frames.TYPE_ERROR, frames.TYPE_EVENT],
    )
    def test_types_roundtrip(self, frame_type: int) -> None:
        left, right = _pair()
        try:
            frames.send_frame(left, frame_type, b"payload", request_id=42, flags=frames.FLAG_RAW)
            frame = frames.FrameReader(right).read_frame()
            assert frame == frames.Frame(frame_type, frames.FLAG_RAW, 42, b"payload")
        finally:
            left.close()
            right.close()

    def test_empty_payload(self) -> None:
        left, right = _pair()
        try:
            frames.send_frame(left, frames.TYPE_END, b"", request_id=7)
            frame = frames.FrameReader(right).read_frame()
            assert frame is not None and frame.payload == b""
        finally:
            left.close()
            right.close()

    def test_large_payload(self) -> None:
        payload = bytes(range(256)) * (16 * 1024)  # 4 MiB
        blob = (
            frames.HEADER.pack(frames.MAGIC, frames.VERSION, frames.TYPE_DATA, frames.FLAG_RAW, 0, 1, len(payload))
            + payload
        )
        left, right = _pair()
        try:
            _send_in_thread(left, blob)
            frame = frames.FrameReader(right).read_frame()
            assert frame is not None and frame.payload == payload
        finally:
            right.close()

    def test_partial_header_and_payload(self) -> None:
        payload = b"x" * 100_000
        blob = (
            frames.HEADER.pack(frames.MAGIC, frames.VERSION, frames.TYPE_DATA, frames.FLAG_RAW, 0, 3, len(payload))
            + payload
        )
        left, right = _pair()
        try:
            _send_in_thread(left, blob, chunk=7919)
            frame = frames.FrameReader(right).read_frame()
            assert frame is not None and frame.payload == payload and frame.request_id == 3
        finally:
            right.close()


class TestFrameErrors:
    def test_clean_eof_returns_none(self) -> None:
        left, right = _pair()
        try:
            left.close()
            assert frames.FrameReader(right).read_frame() is None
        finally:
            right.close()

    def test_eof_mid_header(self) -> None:
        left, right = _pair()
        try:
            left.sendall(b"VRM")
            left.shutdown(socket.SHUT_WR)
            with pytest.raises(frames.FrameError):
                frames.FrameReader(right).read_frame()
        finally:
            left.close()
            right.close()

    def test_eof_mid_payload(self) -> None:
        left, right = _pair()
        try:
            left.sendall(frames.HEADER.pack(frames.MAGIC, frames.VERSION, frames.TYPE_DATA, 0, 0, 1, 10) + b"abc")
            left.shutdown(socket.SHUT_WR)
            with pytest.raises(frames.FrameError):
                frames.FrameReader(right).read_frame()
        finally:
            left.close()
            right.close()

    @pytest.mark.parametrize(
        "header",
        [
            b"XXXX" + bytes(12),
            b"VRMD" + bytes([frames.VERSION + 1]) + bytes(11),
            frames.HEADER.pack(frames.MAGIC, frames.VERSION, 99, 0, 0, 1, 0),
        ],
    )
    def test_invalid_header(self, header: bytes) -> None:
        left, right = _pair()
        try:
            left.sendall(header)
            with pytest.raises(frames.FrameError):
                frames.FrameReader(right).read_frame()
        finally:
            left.close()
            right.close()

    def test_payload_over_limit(self) -> None:
        left, right = _pair()
        try:
            left.sendall(frames.HEADER.pack(frames.MAGIC, frames.VERSION, frames.TYPE_DATA, 0, 0, 1, 1024))
            with pytest.raises(frames.FrameError):
                frames.FrameReader(right).read_frame(max_payload=128)
        finally:
            left.close()
            right.close()


class TestJsonFrames:
    def test_json_roundtrip(self) -> None:
        obj = {"cmd": "blob-has", "sha256": "a" * 64, "n": 1, "ok": True}
        assert frames.decode_json(frames.encode_json(obj)) == obj

    def test_json_rejects_non_object(self) -> None:
        with pytest.raises(frames.FrameError):
            frames.decode_json(json.dumps([1, 2, 3]).encode())

    def test_json_rejects_garbage(self) -> None:
        with pytest.raises(frames.FrameError):
            frames.decode_json(b"not json")


class TestPayloadCodec:
    def test_auto_compresses_repetitive(self) -> None:
        data = b"A" * (2 * 1024 * 1024)
        flags, payload = frames.encode_payload(data)
        assert flags == frames.FLAG_ZLIB
        assert len(payload) < len(data)
        assert frames.decode_payload(flags, payload) == data

    def test_auto_keeps_random_raw(self) -> None:
        data = os.urandom(2 * 1024 * 1024)  # incompressível
        flags, payload = frames.encode_payload(data)
        assert flags == frames.FLAG_RAW
        assert payload == data

    def test_small_payload_stays_raw(self) -> None:
        flags, payload = frames.encode_payload(b"hi")
        assert flags == frames.FLAG_RAW and payload == b"hi"

    def test_none_codec(self) -> None:
        data = b"A" * (2 * 1024 * 1024)
        flags, payload = frames.encode_payload(data, codec="none")
        assert flags == frames.FLAG_RAW and payload is data

    def test_zstd_if_available(self) -> None:
        if not frames.zstd_available():
            pytest.skip("extra zstandard não instalado")
        data = b"B" * (2 * 1024 * 1024)
        flags, payload = frames.encode_payload(data, codec="zstd")
        assert flags in (frames.FLAG_ZSTD, frames.FLAG_RAW)
        assert frames.decode_payload(flags, payload) == data

    def test_unknown_flags(self) -> None:
        with pytest.raises(frames.FrameError):
            frames.decode_payload(99, b"x")

    def test_zlib_bomb_is_rejected(self) -> None:
        """16 MiB de zlib não podem expandir para gigabytes (DoS do supervisor)."""
        bomb = zlib.compress(b"\0" * (32 * 1024 * 1024), 1)
        with pytest.raises(frames.FrameError, match="expande acima"):
            frames.decode_payload(frames.FLAG_ZLIB, bomb)

    def test_raw_payload_over_limit_is_rejected(self) -> None:
        with pytest.raises(frames.FrameError, match="excede o máximo"):
            frames.decode_payload(frames.FLAG_RAW, b"x" * 33, max_bytes=32)

    def test_zstd_bomb_is_rejected(self) -> None:
        if not frames.zstd_available():
            pytest.skip("extra zstandard não instalado")
        import zstandard

        bomb = zstandard.ZstdCompressor(level=1).compress(b"\0" * (32 * 1024 * 1024))
        with pytest.raises(frames.FrameError, match="expande acima"):
            frames.decode_payload(frames.FLAG_ZSTD, bomb)


class TestReaderLineMode:
    def test_peek_does_not_consume(self) -> None:
        left, right = _pair()
        try:
            left.sendall(b'{"cmd":"hello"}\n')
            reader = frames.FrameReader(right)
            assert reader.peek(1) == b"{"
            assert reader.read_line() == b'{"cmd":"hello"}'
        finally:
            left.close()
            right.close()

    def test_line_without_newline_at_eof(self) -> None:
        left, right = _pair()
        try:
            left.sendall(b'{"cmd":"status"}')
            left.shutdown(socket.SHUT_WR)
            assert frames.FrameReader(right).read_line() == b'{"cmd":"status"}'
        finally:
            left.close()
            right.close()

    def test_line_over_limit(self) -> None:
        left, right = _pair()
        try:
            left.sendall(b"x" * 100)
            with pytest.raises(frames.FrameError):
                frames.FrameReader(right).read_line(max_bytes=10)
        finally:
            left.close()
            right.close()
