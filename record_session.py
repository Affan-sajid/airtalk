#!/usr/bin/env python3
"""
Record AIRTALK serial stream to CSV for debugging physics / mapping.

  python3 record_session.py
  python3 record_session.py /dev/cu.usbmodem1101

While recording: type a step number 0–12 and Enter to set the `phase` column
for following rows. Ctrl+C stops and saves the file under logs/.
"""

from __future__ import annotations

import csv
import glob
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Match draw.py defaults
SERIAL_PORT = "/dev/cu.usbmodem1101"
BAUD = 115200

_INSTRUCTIONS = Path(__file__).resolve().parent / "TEST_INSTRUCTIONS.md"
_LOG_DIR = Path(__file__).resolve().parent / "logs"


def _serial_classes():
    try:
        from serial import Serial as S, SerialException as E
        return S, E
    except (ImportError, AttributeError):
        from serial.serialutil import SerialException as E
        if sys.platform.startswith("win"):
            from serial.serialwin32 import Serial as S
        else:
            from serial.serialposix import Serial as S
        return S, E


def find_port() -> str:
    candidates = (
        glob.glob("/dev/cu.usbmodem*")
        + glob.glob("/dev/cu.usbserial*")
        + glob.glob("/dev/tty.usbmodem*")
        + glob.glob("/dev/tty.usbserial*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
        + glob.glob("COM[0-9]*")
    )
    if not candidates:
        raise SystemExit("No USB serial device found.")
    return candidates[0]


def _print_terminal_instructions() -> None:
    if _INSTRUCTIONS.is_file():
        print(_INSTRUCTIONS.read_text(encoding="utf-8", errors="replace"))
    else:
        print("(Missing TEST_INSTRUCTIONS.md — open that file in the repo.)\n")
    print(
        "\n"
        + "=" * 72
        + "\nRECORDING — type step 0–12 + Enter to label moves; Ctrl+C to stop\n"
        + "=" * 72
        + "\n",
        flush=True,
    )


def _classify_line(raw: str) -> tuple[str, list[str], int]:
    s = raw.strip()
    if not s:
        return "empty", [], 0
    parts = [p.strip() for p in s.split(",")]
    n = len(parts)
    if n in (5, 6) and all(_FIELD_FLOAT.match(p) for p in parts[:4]):
        try:
            int(parts[4])
        except ValueError:
            return "text", parts, n
        if n == 6 and not _FIELD_FLOAT.match(parts[5]):
            return "text", parts, n
        return "imu", parts, n
    return "text", parts, n


_FIELD_FLOAT = re.compile(r"^-?\d+(\.\d+)?$")


def main() -> None:
    Serial, SerialException = _serial_classes()

    if len(sys.argv) > 1:
        port = sys.argv[1]
    elif SERIAL_PORT:
        port = SERIAL_PORT
    else:
        port = find_port()

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_path = _LOG_DIR / f"airtalk_session_{stamp}.csv"

    phase_lock = threading.Lock()
    phase = 0
    stop = threading.Event()

    def _stdin_loop() -> None:
        nonlocal phase
        while not stop.is_set():
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                stop.set()
                return
            if not line:
                time.sleep(0.05)
                continue
            t = line.strip()
            if t.isdigit():
                with phase_lock:
                    phase = int(t)
                print(f"[phase -> {phase}]", flush=True)
            elif t.lower() in ("q", "quit", "exit"):
                stop.set()
                return

    threading.Thread(target=_stdin_loop, daemon=True).start()

    _print_terminal_instructions()
    print(f"Log file: {out_path}\nOpening {port} @ {BAUD} …\n", flush=True)

    try:
        ser = Serial(port, BAUD, timeout=0)
    except SerialException as e:
        raise SystemExit(f"Cannot open port: {e}")

    fieldnames = [
        "elapsed_s",
        "phase",
        "rot_x_100",
        "rot_y_100",
        "lin_x_100",
        "lin_y_100",
        "pen",
        "bias_z_100",
        "num_fields",
        "line_kind",
        "raw",
    ]
    t0 = time.perf_counter()
    rows = 0

    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        try:
            while not stop.is_set():
                try:
                    raw_b = ser.readline()
                except SerialException:
                    break
                if not raw_b:
                    time.sleep(0.002)
                    continue
                raw = raw_b.decode("utf-8", errors="replace").rstrip("\r\n")
                elapsed = time.perf_counter() - t0
                with phase_lock:
                    ph = phase
                kind, parts, n = _classify_line(raw)
                row: dict[str, object] = {
                    "elapsed_s": f"{elapsed:.6f}",
                    "phase": ph,
                    "rot_x_100": "",
                    "rot_y_100": "",
                    "lin_x_100": "",
                    "lin_y_100": "",
                    "pen": "",
                    "bias_z_100": "",
                    "num_fields": n,
                    "line_kind": kind,
                    "raw": raw,
                }
                if kind == "imu" and n >= 5:
                    row["rot_x_100"] = parts[0]
                    row["rot_y_100"] = parts[1]
                    row["lin_x_100"] = parts[2]
                    row["lin_y_100"] = parts[3]
                    row["pen"] = parts[4]
                    if n == 6:
                        row["bias_z_100"] = parts[5]
                writer.writerow(row)
                rows += 1
                if rows % 500 == 0:
                    fp.flush()
        except KeyboardInterrupt:
            pass
        finally:
            fp.flush()

    ser.close()
    stop.set()
    print(f"\nSaved {rows} rows to:\n  {out_path}", flush=True)
    print(
        "\nNext: zip or attach that CSV + a short note (cursor vs hand) for diagnosis.",
        flush=True,
    )


if __name__ == "__main__":
    main()
