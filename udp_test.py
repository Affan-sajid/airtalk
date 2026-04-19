#!/usr/bin/env python3
"""
AIRTALK UDP smoke test — prints network info first, then listens on the same
port as draw.py / infer.py (default 5005).

Run (from project root):
  python3 udp_test.py
  python3 udp_test.py --seconds 30
  python3 udp_test.py --send 172.20.10.2   # optional: one test datagram to the pen

If bind fails, something else is already using the port (close draw.py/infer.py).
If you see no packets, the ESP32 firmware HOST_IP must match one of your Mac
LAN addresses shown below (not the public internet IP).
"""

from __future__ import annotations

import argparse
import os
import select
import socket
import subprocess
import sys
import time
from pathlib import Path


def _load_project_env_file() -> None:
    path = Path(__file__).resolve().parent / ".env"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def _outbound_ipv4() -> str | None:
    """IP the kernel would use to reach the internet (good 'default' LAN hint)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _ipconfig_addrs() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in ("en0", "en1", "en2", "en3", "bridge0", "awdl0"):
        try:
            ip = subprocess.check_output(
                ["ipconfig", "getifaddr", name],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if ip:
            out.append((name, ip))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="UDP listen test for AIRTALK (port 5005).")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default 5005)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=120.0,
        help="How long to listen (default 120)",
    )
    parser.add_argument(
        "--send",
        metavar="HOST",
        help="After binding, send one test line to this host (e.g. ESP32 IP from .env)",
    )
    args = parser.parse_args()

    _load_project_env_file()
    esp = (os.environ.get("ESP32_IP") or "").strip()

    print("=== AIRTALK UDP test — diagnostics (read this first) ===\n")
    print(f"hostname: {socket.gethostname()}")

    ob = _outbound_ipv4()
    if ob:
        print(f"default-route IPv4 (outbound): {ob}  ← often the IP to put in ESP32 HOST_IP")

    pairs = _ipconfig_addrs()
    if pairs:
        print("interfaces (macOS ipconfig):")
        for name, ip in pairs:
            print(f"  {name}: {ip}")
    else:
        print("(no en/bridge IPs from ipconfig — check System Settings → Network)")

    if esp:
        print(f"ESP32_IP from .env (Python → pen for 'R'): {esp}")
    else:
        print("ESP32_IP not set in .env (optional; used by draw/infer for recalibrate)")

    print(f"\n=== binding UDP {args.port} on 0.0.0.0 (same as draw.py) ===\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", args.port))
    except OSError as e:
        print(f"BIND FAILED: {e}")
        print("Another process may own the port. Quit draw.py / infer.py / duplicate listeners.")
        sys.exit(1)

    if args.send:
        msg = b"AIRTALK udp_test.py hello\n"
        sock.sendto(msg, (args.send, args.port))
        print(f"sent {len(msg)} bytes to {args.send}:{args.port} (pen may ignore this)\n")

    sock.setblocking(False)
    deadline = time.monotonic() + args.seconds
    first = True
    n = 0
    print(f"listening… Ctrl+C to stop (timeout {args.seconds}s)\n")

    try:
        while time.monotonic() < deadline:
            r, _, _ = select.select([sock], [], [], 1.0)
            if not r:
                continue
            try:
                data, addr = sock.recvfrom(65535)
            except OSError:
                continue
            n += 1
            preview = data[:200]
            try:
                text = preview.decode("utf-8", errors="replace")
            except Exception:
                text = repr(preview)
            if first:
                print(f">>> FIRST PACKET <<< from {addr[0]}:{addr[1]} ({len(data)} bytes)")
                first = False
            line = text.replace("\n", "\\n")
            print(f"#{n} {addr[0]}:{addr[1]} len={len(data)} preview={line!r}")
    except KeyboardInterrupt:
        print("\n(interrupted)")

    sock.close()
    if n == 0:
        print(
            f"\nNo UDP received in {args.seconds}s.\n"
            "Check: same Wi‑Fi, firewall allows UDP 5005 inbound, "
            "and firmware HOST_IP matches your Mac LAN IP (see diagnostics above)."
        )
    else:
        print(f"\nReceived {n} datagram(s). UDP path Mac ← pen looks OK.")


if __name__ == "__main__":
    main()
