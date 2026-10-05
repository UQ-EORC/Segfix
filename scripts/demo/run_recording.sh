#!/usr/bin/env bash
# Record the segfix walkthrough, then extract review frames and verify the
# saved project. segfix opens on the screen for a quarter of an hour, so
# leave the machine alone while it records.
#
# The scripts live in the repo; everything the run produces (a 400 MB demo
# cloud, the project it edits, the video) goes to a work directory outside
# it -- $SEGFIX_DEMO_WORK, or /tmp/segfix-demo. The video lands in
# $WORK/out/segfix_walkthrough.mp4 and the log in $WORK/out/run.log.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$D/../.." && pwd)"
WORK="${SEGFIX_DEMO_WORK:-/tmp/segfix-demo}"
PY="$REPO/.venv/bin/python"
LOG="$WORK/out/run.log"
mkdir -p "$WORK/out"
exec > >(tee "$LOG") 2>&1
cd "$D" || exit 1

$PY -m py_compile recorder.py make_video.py storyboard.py preflight.py \
  || { echo "SYNTAX ERROR"; exit 1; }
# Check the storyboard against the app before tying up the screen for a
# quarter of an hour: a renamed button or a panel attribute that moved
# otherwise surfaces minutes in, with the recording wasted.
$PY preflight.py || { echo "PREFLIGHT FAILED"; exit 1; }
rm -rf "$WORK/projects" "$WORK/config" && mkdir -p "$WORK/projects" "$WORK/config"
if [ ! -s "$WORK/example_utm.las" ]; then
  $PY "$REPO/scripts/make_sample.py" --spacing 0.015 \
      --origin 204300 7223250 12 "$WORK/example_utm.las" | head -1
fi
XDG_CONFIG_HOME="$WORK/config" $PY - <<PY || exit 1
from pathlib import Path
from segfix import registry, workspace
ws = Path("$WORK/projects/example_utm")
workspace.create_workspace("$WORK/example_utm.las", ws)
registry.add_entry(str(ws), kind="workspace")
print("project reset")
PY
free -h | sed -n 2p

start=$(date +%s)
DISPLAY="${DISPLAY:-:0}" QT_QPA_PLATFORM=xcb XDG_CONFIG_HOME="$WORK/config" \
  $PY make_video.py full "$WORK/out/segfix_walkthrough.mp4" > "$WORK/out/full.log" 2>&1
code=$?
echo "exit=$code  wall=$(( $(date +%s) - start ))s"
grep -v "propagateSizeHints\|This plugin does not support" "$WORK/out/full.log" | tail -30
[ $code -eq 0 ] && [ -s "$WORK/out/segfix_walkthrough.mp4" ] || { echo "RECORDING FAILED"; exit 1; }
ffprobe -v error -show_entries format=duration,size -of compact "$WORK/out/segfix_walkthrough.mp4"

cd "$WORK/out" && rm -f sheet_*.png check_*.png
ffmpeg -v error -i segfix_walkthrough.mp4 -vf "fps=1/8,scale=480:-1,tile=4x4" sheet_%02d.png
$PY - <<'PY'
import re, subprocess
cues = []
for block in open("segfix_walkthrough.srt", encoding="utf-8").read().strip().split("\n\n"):
    lines = block.split("\n")
    h, m, s = re.match(r"(\d+):(\d+):([\d,]+)", lines[1]).groups()
    cues.append((int(h) * 3600 + int(m) * 60 + float(s.replace(",", ".")), " ".join(lines[2:])))
numbered = [text for _, text in cues if re.match(r"^\d+\s*·", text) or "CHAPTER" in text.upper()]
print("numbered section titles in the subtitles:", numbered or "none")
for name, prefix, off in [
    ("title_card", "Marking trees done", 1.2),
    ("gap_1_5x", "Click the same spot again: each click", 8.6),
    ("gap_3x", "Click the same spot again: each click", 13.8),
    ("first_done", "Trees 2 and 3 have no errors", 0.8),
    ("all_done", "Every tree in the table is done", 1.5),
    ("end_card", "Install it with", 2.0),
]:
    hit = next((t for t, text in cues if text.startswith(prefix)), None)
    if hit is None:
        print(f"!! no cue starting {prefix!r}")
        continue
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{hit + off:.2f}", "-i", "segfix_walkthrough.mp4",
                    "-frames:v", "1", f"check_{name}.png"], check=True)
    print(f"check_{name}.png @ {hit + off:.1f}s")
print(f"{len(cues)} cues")
gone = [p for p in ("Point size is set", "The File menu opens", "The arrow keys step")
        if any(text.startswith(p) for _, text in cues)]
print("dropped sections still present:", gone or "none")
PY

echo "=== review progress saved beside the project ==="
$PY - <<PY
import json
from segfix.treecatalog import open_catalog
path = "$WORK/projects/example_utm/example_utm.las"
done = set(json.load(open(path + ".segfix.json")).get("done", []))
trees = set(open_catalog(path).records)
print(f"trees in file: {sorted(trees)}")
print(f"marked done:   {sorted(done)}")
print(f"  {'OK ' if trees <= done else 'BAD'} every tree in the table is marked done")
PY

echo "=== saved project vs pristine example (full resolution) ==="
cd /home/tim/Code/segfix && $PY - <<PY
from segfix.treecatalog import open_catalog
b = open_catalog("$WORK/example_utm.las")
a = open_catalog("$WORK/projects/example_utm/example_utm.las")
cb = {k: v.count for k, v in b.records.items()}
ca = {k: v.count for k, v in a.records.items()}
noise_a = int((a.labels == -1).sum())
new = sorted(set(ca) - set(cb))
checks = {
    "same number of points": b.count == a.count,
    "band moved 1 -> 2": ca[1] + ca[2] == cb[1] + cb[2] and ca[2] > cb[2],
    "tree 8 merged away completely": 8 not in ca,
    "tree 7 = old 7 + old 8": ca.get(7) == cb[7] + cb[8],
    "tree 10 split in two": len(new) == 1 and ca.get(10, 0) + ca[new[0]] == cb[10],
    "tree 11 lost its ground patch": ca.get(11, 0) < cb[11],
    "bush (12) is noise, every point": 12 not in ca and noise_a == cb[12],
    "untouched trees 3-6, 9 identical": all(ca.get(t) == cb[t] for t in (3, 4, 5, 6, 9)),
}
for k, v in checks.items():
    print(f"  {'OK ' if v else 'BAD'} {k}")
PY
echo "ALL DONE - tell Claude it finished"
