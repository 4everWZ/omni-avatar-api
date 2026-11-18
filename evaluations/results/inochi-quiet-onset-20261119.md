# Inochi2D mouth onset after silence

Two independent 22 s Chromium calls used a local Windows gateway, the two private master rigs and real remote Omni inference. Each call used a different approved voice input, produced one response and acknowledged playback. The baseline service was `d0e70d0`. The 24 kHz mono PCM from the model response matched the bytes passed to the renderer. The capture included the exact PCM window and the MouthOpen and MouthForm passed to drawing on every frame. Raw recordings are held locally.

| Measurement | Regular | Q |
| --- | ---: | ---: |
| Model audio duration | 7.04 s | 3.72 s |
| Transient peak from model audio start | 121.46 ms | 121.25 ms |
| Transient energy between 5th and 95th percentiles | 3.88 ms | 4.38 ms |
| Correlation with two earlier receiver pulse waveforms | .983 / .992 | .892 / .890 |
| Baseline → corrected phase offsets with a false opening (40 tested) | 23 → 0 | 19 → 0 |
| Peak false MouthOpen before correction | .362 | .157 |
| First sustained PCM onset | 365 ms | 280 ms |
| First mouth frame relative to sustained onset in the actual capture | +5 ms | +1 ms |
| Range relative to sustained onset across replay phases, before and after | −8 to +38 ms | −6 to +70 ms |
| Short voiced excerpt, source times | 4595–4685 ms | 1865–1930 ms |
| Paired additional CPU time p95, timeline read and analysis | 0.021 ms | 0.008 ms |
| Paired additional CPU time p95 on weak windows | 0.239 ms | 0.102 ms |

Neither new live capture happened to sample the transient in an open-mouth phase. Replaying the saved inputs reproduced all 551 and 549 baseline poses exactly. The corrected driver also gave the same poses at those actual frame times. In a separate replay, moving the 25 fps grid through 40 offsets at 1 ms intervals reproduced the opening. At a pulse-hitting offset, comparing the rendered image with a closed-mouth image at identical head pose gave 939 and 788 different pixels on the baseline and zero on both corrected rigs.

Sustained onset was the first pair of 5 ms PCM bins above RMS .012. For each model reply we selected the first 45–100 ms continuously voiced interval after onset with at least two quieter bins on each side. Each original extract, including its two-bin margins, was replayed after silence. All 40 offsets retained the original short-sound openings, with identical jaw values and zero additional first-frame delay. On the full phrase, the first frame with MouthOpen above .1 also had zero additional delay across all offsets. The Q phrase already reached +70 ms on some offsets under that criterion.

The driver checks a weak center packet only after ten quiet 20 ms packets preceding a 20 ms onset margin. It holds the mouth closed if seven adjacent 1 ms bins contain at least 90% of the surrounding 60 ms energy. This uses the existing queued lookahead. Timing above alternated baseline and candidate order over four passes on each recorded frame. It excludes drawing, encoding and browser presentation. These results characterize the two source responses and the local renderer. The earlier recordings retain their receiver-side evidence only.
