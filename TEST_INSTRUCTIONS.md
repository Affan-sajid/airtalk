# AIRTALK — motion test + recording

Close **draw.py** and **Arduino Serial Monitor** before recording (only one program may open the USB port).

## 1. Start recording

From the project folder:

```bash
cd /path/to/AIRTALK
python3 record_session.py
```

Optional port override:

```bash
python3 record_session.py /dev/cu.usbmodem1101
```

The script prints this checklist in the terminal, then writes every serial line to:

`logs/airtalk_session_<timestamp>.csv`

Stop with **Ctrl+C** (file is flushed and closed).

## 2. Phase labels (optional but very helpful)

While it is recording, type a **step number** (0–12) and press **Enter**. The next rows in the CSV will include that number in the `phase` column so whoever reads the log knows which move you were doing.

Example: before you do the “nod” test, type `3` and Enter.

## 3. Test moves (do in order)

| Phase | Pen | Action |
|------:|-----|--------|
| 0 | up | Hands relaxed, pen vertical, **10 s** still — watch idle drift |
| 1 | up | Same as 0 if you want a second baseline |
| 2 | up | Optional: note “started moving” |
| 3 | up | **Nod** (tip toward/away from you), small, slow, ~5 reps |
| 4 | up | **Shake head** (tip left/right), small, slow, ~5 reps |
| 5 | up | **Roll** pen around long axis (twist like screwdriver), tip aimed at same point |
| 6 | down | **Nod** again — draw short strokes |
| 7 | down | **Shake head** — draw short strokes |
| 8 | up | Move hand **straight to your right** (minimize rotation) |
| 9 | up | Move hand **straight forward** (away from body) |
| 10 | down | Slow **“L”** (two short perpendicular segments) |
| 11 | down | Small **arc** or “C” in one motion |
| 12 | down | One **fast flick** |

You can skip phases you do not care about; use `phase` when you can.

## 4. After the run — what to send for diagnosis

Send **both**:

1. The CSV file: `logs/airtalk_session_*.csv`  
2. A short note (message or paste in a text file) with:
   - Screen behavior vs your expectation for a few steps (e.g. “phase 3 moved cursor mostly up”)
   - Anything odd (delay, jumps, wrong direction)

Do **not** need to send this whole markdown file unless you want context bundled in a zip.

## 5. CSV columns

| Column | Meaning |
|--------|---------|
| `elapsed_s` | Seconds since recording started |
| `phase` | Last value you typed + Enter (0 if never set) |
| `rot_x_100` … `bias_z_100` | Parsed fields from firmware (*100 scale); blank if line is not IMU CSV |
| `num_fields` | Number of comma-separated fields |
| `line_kind` | `imu` / `text` / `empty` |
| `raw` | Full line as received (quoted CSV) |

Firmware may still print `CALIBRATING...` and `READY` before numbers — those appear as `text` rows.
