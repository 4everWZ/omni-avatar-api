# Mandarin PCM to INP mouth controls, native-frame comparison

Two production INP files (regular and Q, 640 × 720) were rendered with identical PCM, frame PTS and head motion for each comparison. The baseline is the RMS / first-difference brightness mapping from `69ee013`. The candidate's only input is the session's queued PCM. Native Omni audio deltas carry no timed articulatory marks. Offline reference generators supplied timings to the scorer alone.

## Measurement

A frame holds for 40 ms. At each generator-labelled midpoint with duration ≥40 ms the scorer measures the largest mouth-colour component near the lips. Near-closure uses the maximum area across 25 silent poses plus four pixels. The vowel test excludes components below 180 pixels or at most four pixels high and pairs each spread vowel with the nearest available rounded vowel within 10% area. Pair counts can differ between drivers. The open-vowel count includes all eligible labelled vowel midpoints, before matching. SAPI uses the previous browser audit's IDs 21, 6 and 7/8. For eSpeak NG, consecutive timed IPA events from the same synthesis delimit each interval; only bilabials and the specified monophthongs are scored. These are generator reference labels.

| PCM / rig | Near-closed bilabials, old → new | Spread wider at matched area, old → new | Open vowels, old → new |
|---|---|---|---|
| Fixed mouth-check / regular | 8/8 → 8/8 | 3/3 → 7/7 | 19/36 → 28/36 |
| Fixed mouth-check / Q | 8/8 → 8/8 | 3/4 → 7/8 | 21/36 → 28/36 |
| Reference, Huihui / regular | 14/14 → 13/14 | 6/8 → 22/24 | 35/72 → 60/72 |
| Reference, Huihui / Q | 14/14 → 13/14 | 9/10 → 22/24 | 41/72 → 60/72 |
| Reference, Yaoyao / regular | 5/15 → 13/15 | 15/23 → 15/18 | 59/70 → 52/70 |
| Reference, Yaoyao / Q | 5/15 → 13/15 | 15/23 → 15/18 | 62/70 → 52/70 |
| Reference, eSpeak cmn / regular | 2/11 → 10/11 | 5/5 → 12/13 | 28/28 → 28/28 |
| Reference, eSpeak cmn / Q | 2/11 → 10/11 | 4/5 → 13/13 | 28/28 → 28/28 |
| Held-out text and cmn+f3 voice variant / regular | 3/8 → 7/8 | 4/5 → 10/11 | 25/25 → 25/25 |
| Held-out text and cmn+f3 voice variant / Q | 2/8 → 7/8 | 4/6 → 10/11 | 25/25 → 25/25 |

The first row pair uses the original 9.117 s source and the independently regenerated Huihui event file used in the browser audit (5 ms envelope correlation 0.993). Its old 8/8 native closure and the prior receiver's 7/24 closure on three playbacks are separate results. For the other rows, PCM and events were captured in the same synthesis. The reference was “冰面慢慢融化，妈妈把白布铺平。弟弟洗衣，姑姑读书，屋里有雨衣和地图。妹妹闭目听雨，我们一起步行回屋。” Two local OneCore names produced byte-identical speech on this text and pointed to the same Yaoyao voice data. Only Yaoyao was counted. Huihui produced different PCM. The held-out eSpeak variant and its sentence were first scored after selecting the acoustic rules. Moving the held-out frame grid by 20 ms gave 8/8 closures and 9/10 ordered pairs on each rig, with all 25 vowels still open. Quiet vowels can be mistaken for nasal closure, especially for Yaoyao. A front rounded vowel can also share a strong upper resonance with a spread vowel.

## Timing and reproduction

On the local Windows CPU the production acoustic calculation alone took 0.51 ms median and 0.75 ms p95 across 225 successive windows of the fixed source. Alternating baseline and candidate renders over all 228 frames per rig gave regular p95 39.88 → 42.41 ms (11 → 18 frames above 40 ms) and Q p95 34.98 → 33.77 ms (3 → 0). These timings include drawing but exclude encoding and presentation. The workstation WHEP trial independently measures delivered FPS, sound/pose timing and the new receiver lip scores.

With the service evaluation environment (NumPy, OpenCV and FFmpeg), use the following commands from the repository root. `$rigs` contains the two private master rig directories. eSpeak is an optional evaluation dependency.

```powershell
uv run --no-project --with espeakng-loader==0.2.4 python evaluations/mandarin_lip_fixture.py tmp/lip-reference '明明把地图铺在木桌上，爸爸轻轻碰杯。你提着雨衣出屋，我陪母亲去书局，路边一户人家在煮米粥。' --voice cmn+f3
python -m evaluations.inochi_lips $rigs tmp/lip-reference.wav tmp/lip-reference.csv
python -m evaluations.inochi_lips $rigs tmp/lip-reference.wav tmp/lip-reference.csv --phase-ms 20
python -m evaluations.inochi_lips $rigs $mouthCheckPcm $generatorCsv --timing
```
