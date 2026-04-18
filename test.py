#!/usr/bin/env python3
"""
Minimal serial test for AIRTALK — prints everything from the Arduino.
No pygame. Use this to verify USB / port / baud / firmware.

  python3 test.py
  python3 test.py /dev/cu.usbmodem1201

If you see nothing: wrong port, USB cable (charge-only), or Serial Monitor
open in Arduino IDE (only one app can use the port).
"""

import sys
import glob


# Match draw.py — edit here or pass port as argv[1]
SERIAL_PORT = "/dev/cu.usbmodem1101"
BAUD = 115200
READ_TIMEOUT_SEC = 2.0


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
        raise SystemExit("No USB serial device found under /dev or COM*.")
    return candidates[0]


def main() -> None:
    Serial, SerialException = _serial_classes()

    if len(sys.argv) > 1:
        port = sys.argv[1]
    elif SERIAL_PORT:
        port = SERIAL_PORT
    else:
        port = find_port()

    print(f"Opening {port} @ {BAUD} baud (read timeout {READ_TIMEOUT_SEC}s)…")
    try:
        ser = Serial(port, BAUD, timeout=READ_TIMEOUT_SEC)
    except SerialException as e:
        raise SystemExit(f"Cannot open port: {e}")

    print("Reading lines. Ctrl+C to stop.\n")
    empty = 0
    try:
        while True:
            chunk = ser.readline()
            if not chunk:
                empty += 1
                if empty == 1:
                    print(
                        "[hint] Timed out with no line — board silent, wrong port, "
                        "or port in use (close Arduino Serial Monitor).",
                        flush=True,
                    )
                elif empty % 5 == 0:
                    print(f"[hint] Still no data ({empty * READ_TIMEOUT_SEC:.0f}s)…", flush=True)
                continue
            empty = 0
            text = chunk.decode("utf-8", errors="replace").rstrip("\r\n")
            # Show both decoded text and raw hex if line has weird bytes
            if chunk.endswith(b"\n") or chunk.endswith(b"\r\n"):
                print(text, flush=True)
            else:
                print(f"{text!r}  | raw={chunk!r}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
