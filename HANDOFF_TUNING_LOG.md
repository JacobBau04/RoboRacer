# Handoff: adding your session's work to TUNING_LOG.md

For a Claude Code chat that is about to add work from its own session to this project's log. Read this first, then `TUNING_LOG.md` itself (at least the header, the project summary, and the summary table of any section you add to).

Written 2026-11-05 by the session that restructured the log.

## 1. What the log is for

- `~/f1tenth_ws/src/jacob_opt_obstacles/TUNING_LOG.md` is the **source of record** for the user's RoboRacer project (ESE 6150, Group 12). The user will turn it into personal **website pages for robotics recruiters**.
- Recruiters care most about *how* problems were found. Every entry is framed as **symptom → root cause (with the evidence) → fix → measured result**.
- Keep the log as the full engineering record. The website will be a shorter narrative written from it later; don't write website copy into the log.

## 2. Facts the user confirmed (don't contradict them)

- **Race:** 2026-11-04 on race2, obstacles moved every lap, scored on average lap time: **4th of 14 teams**. The race laps were **not recorded**, so there are no race lap times. Don't invent any.
- **Steering gain during the race:** `steering_angle_to_servo_gain` **−0.8** (offset 0.469). `vesc.yaml` has a "CHANGE TO −0.91" comment, but −0.91 was not used.
- **Authorship:** the user implemented all race code. **Only the initial setup of the replacement VESC (entry V0) was done by teammates.** Credit teammates for V0 only.
- **Which code raced:** the copy `~/f1tenth_ws/src/jacob_opt_obstacles_restart`. It's the active `mpc` package; `jacob_opt_obstacles` has `COLCON_IGNORE`. The restart copy adds stop-and-rescan (log entry O7) and race settings: `MAX_STEER` 0.38, `MAX_ACCEL` 3.3, `MAX_SPEED` 3.3.

## 3. Which file to edit

- **Edit only `jacob_opt_obstacles/TUNING_LOG.md`.** The restart copy has its own `TUNING_LOG.md`, an older copy. Leave it alone; the user will copy the main one over if they want.
- Run folders live in **two places**:
  - `jacob_opt_obstacles/mpc/data/<run>/`: everything up to 2026-11-04 13:00.
  - `jacob_opt_obstacles_restart/mpc/data/<run>/`: runs made with the restart copy. As of 13:02 on Nov 4, there are only `2026-11-04_13-01-02` and `2026-11-04_13-02-19`, both with no complete lap.

  Link the first as `mpc/data/<run>/summary.txt` and the second as `../jacob_opt_obstacles_restart/mpc/data/<run>/summary.txt`.
- Each run folder has `meta.json` (the **true** settings: MPC config, node parameters, VESC config, waypoint md5), `summary.txt`, `raw.npz` and usually `bag/`. **Run notes can be wrong**; trust `meta.json` over the note.

## 4. What is already in the log (check before adding, to avoid duplicates)

Gathered from this session and from reading the other chat transcripts on this Jetson (`~/.claude/projects/-home-nvidia/*.jsonl`, Nov 2–5). If your session covered the same thing, **add what's missing or correct what's wrong in the existing entry**. Don't write a second one.

| IDs | Topic | Time window covered |
|---|---|---|
| M0–M14 | Sim tuning on Levine | Oct 30 – Nov 1 |
| R1–R2 | First real-car runs on Levine, delay compensation | Nov 1 |
| V0–V4 | Replacement VESC retune: Ki/Kd, motor detection, ramp 14000, openloop ERPM 800 | Nov 3 evening – Nov 4 00:14 |
| H1 | Old VESC not showing up on USB, replaced | Nov 3 morning |
| H2 | Car crept backwards: motor direction, fixed with `speed_to_erpm_gain` −4363.1 | Nov 3 ~20:30–21:00 |
| H3 | `OPENBLAS_NUM_THREADS=1` (particle filter 16.7 → 27.7 Hz) | Nov 3 |
| T1–T3 | race1 map fix and centerline, race2 setup, race2 raceline settings over race day | Nov 2 – Nov 4 |
| R3–R9 | race2 tracking fixes: gain −0.8, `pose_offset` 0.27, `delay_comp` 0.20, Rdk 30, Rk 5, mu 0.3, DTK 0.05 / TK 16, trim 0.469, wheelbase 0.324 | Nov 4 06:30–08:29 |
| R10 | v_max 4, a_x 4 → 9; takeovers at the C2 exit | Nov 4 08:34–08:45 |
| C1–C4 | Slow-circle steering test, grip μ ≈ 0.40 (from another chat's analysis, not recomputed), top speed ≥ 5.15 m/s, acceleration ceiling 3.2 m/s² (VESC ramp) | Nov 4 |
| O1–O8 | Real-car obstacle avoidance: TF listener fix, boxes that only grow, phantom obstacles (`wall_margin` 0.25, `beam_time_offset` 0.04), `mpc_back` / `v_start_min`, margins, tight/crawl speeds 1.6/0.9, stop-and-rescan, race | Nov 4 00:25–13:00 |

**Likely gaps** (things this session could not see; your session may have them):
- the details of the time-trial session, and why mu 0.35 / v_max 3.3 / a_x 4 were chosen
- anything done after Nov 4 13:00, including the race itself
- the exact changes behind O4 (margins) and O2 (which fixes actually went into the tracker besides `occlusion_extend`)
- whether the restart feature (O7) fired during the race
- changes to `v_gap_min` / `margin_comfort`: the file has `margin_comfort` 0.38, `v_gap_min` 1.5

## 5. Format to follow

**Next free IDs:** R11, V5, H4, C5, T4, O9. Don't renumber existing entries.

| Prefix | Section (heading in the log) |
|---|---|
| M | `## Entries (newest first)`: sim only |
| R | `## Real car (newest first)`: tracking, speed, MPC on the car |
| V | `## VESC swap and speed-controller tuning (newest first)` |
| H | `## Hardware and platform (newest first)` |
| C | `## Car limits, measured (newest first)` |
| T | `## Tracks and racelines`: one table, newest first, no subsections |
| O | `## Obstacle avoidance on the real car (newest first)` |

Each entry needs **three edits**:
1. **A row in the section's summary table**, at the top (newest first): `| [ID](#anchor) | Result | What changed |`.
2. **A subsection**, placed newest first:
   ```markdown
   ### O9 — Short title that says what happened
   2026-11-04 · run [hh-mm-ss](mpc/data/2026-11-04_hh-mm-ss/summary.txt) · compared with [...](...)

   | Symptom | Root cause | Fix / result |
   |---|---|---|
   | What the user saw, with numbers. | Why, with the evidence (data, replay, test). | The change (exact parameter old → new) and the measured result. |
   ```
3. **The project summary** at the top, if the entry changes a headline: the result table, the "what went wrong" list, the final race configuration, or the tools list. Also update `## Open issues (not milestones)` at the bottom: remove issues your work solved, add new ones.

**Anchors:** GitHub style. Lowercase the heading, drop punctuation (`.`, `,`, `:`, `;`, `(`, `)`, `→`), and turn spaces into `-`. "O3 — Phantom obstacles: the map is up to 0.2 m off" becomes `#o3--phantom-obstacles-the-map-is-up-to-02-m-off`.

**Writing style used throughout:**
- Plain, short sentences; no jargon without a gloss.
- Exact parameter names in backticks, with old → new values.
- Units on every number. Run folders linked.
- Mark estimates as estimates, and say where a number came from if it wasn't recomputed from data.
- **Keep wrong hypotheses** as "first thought X, test Y disproved it" (see C1). Don't silently drop them; it's part of the story.
- No invented numbers. If a value isn't in a run folder or your transcript, say it wasn't recorded.

## 6. Numbers and figures

- **Tracking metrics:** use one rule so runs are comparable. That's what [mpc/tools/make_log_figures.py](mpc/tools/make_log_figures.py) does:
  - the rear axle (pose moved 0.27 m back from the LiDAR) against the raceline
  - only while the MPC drives above 1.0 m/s, at least 2 s after it took over, and after the car first came within 15 cm of the line
  - wall clearance: the closest any corner of the car body (rear axle −0.12 … +0.46 m, ±0.155 m) came to a wall on the particle-filter map

  The recorder's own `summary.txt` measures at the LiDAR and includes rejoining the line, so its numbers differ. Say which one you quote.
- **Figures:**
  - Save as `jacob_opt_obstacles/figures/<prefix>_<name>.png`. The new ones use `r_`.
  - Build them in `make_log_figures.py`: reuse `Run`, `clearance`, `draw_map`, `curvature`, `save` and the rcParams style. Add a function, and call it from `main()`.
  - Colours: the dataviz skill's light palette: blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, plus critical red `#d03b3b` for wall contacts only. Node isn't installed, so the palette validator can't run; the first three slots are documented as colourblind-safe.
  - Look at every PNG after rendering (labels overlap easily).
- **Check links after editing:**
  ```bash
  cd ~/f1tenth_ws/src/jacob_opt_obstacles && python3 - <<'EOF'
  import re, os
  s = open('TUNING_LOG.md').read()
  a = lambda h: re.sub(r'[^\w\- ]', '', h.strip().lower()).replace(' ', '-')
  heads = [a(m) for m in re.findall(r'^#{1,6} (.+)$', s, re.M)]
  print('broken anchors:', sorted({l for l in re.findall(r'\]\(#([^)]+)\)', s) if l not in heads}) or 'none')
  print('duplicate headings:', sorted({h for h in heads if heads.count(h) > 1}) or 'none')
  print('missing files:', [f for f in set(re.findall(r'\]\(((?:mpc|figures|\.\./|HANDOFF)[^)#]*)\)', s)) if not os.path.exists(f)] or 'none')
  EOF
  ```

## 7. Things that are easy to get wrong

- **Installed vs source copies.** Some packages were built without `--symlink-install`, so editing `src/` doesn't change what runs: `min_curv_raceline`'s params, and `f1tenth_stack`'s `vesc.yaml` until the Nov 4 rebuild. Check `meta.json` for what a run actually used.
- **Wheel speed is not ground speed.** VESC speed reads ~5 % high (odometry scale ~0.95).
- **Particle-filter path length is inflated by jitter.** Summing raw 26 Hz positions adds 25–35 % on slow arcs. Measure distance on a path sampled every ~0.25 s.
- **The pose is the LiDAR, 0.27 m ahead of the rear axle.** The MPC tracks the rear axle since R4; the obstacle planner plans for the LiDAR point (handled by `mpc_back`, O5).
- **Timestamps:** Claude transcripts are in UTC; the user and the run folders use local time (EST, UTC−5).
- **Don't run state-changing commands on the car** (launches, builds, git) unless the user asks. For "how do I" questions, give the commands. New car features go in a `COLCON_IGNORE`d copy of the package; the user switches builds.

## 8. When you're done

Tell the user which entries you added or changed (by ID), which numbers came from data and which from transcripts, and anything you found that contradicts the existing log.
