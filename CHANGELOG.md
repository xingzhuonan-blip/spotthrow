# Changelog

## 2026-11-13

### Added
- **GP target throw (`g` in `workflow.py`).** Type a target distance; the GP is
  retrained on `twin_data/dataset/throws.csv` and sets swing time and release
  time. After the throw, `target_x` and `error_x` (= landing_x - target_x) are
  saved in the CSV and in the record. New module `twin/gp_inverse.py`, same
  method as `learning_throw_simulation.ipynb` (Mohammadreza).
- **Earliest gripper release** (`twin/cubic_throw.py`, `min_release`):
  release_time >= 0.28 x swing_time up to a 0.9 s swing, rising to 0.36 x at
  1.0 s. Earlier releases send the ball up onto Spot. Manual settings below the
  limit are refused; the GP only searches above it.

### Fixed
- **No ball candidates:** Motive now streams unlabeled markers under asset id 4,
  not 0. Ball candidates are now all markers that are not part of a streamed
  rigid body (`twin/natnet.py`).
- **Landing taken after a bounce:** the ball finder picked the longest parabola,
  often the bounce (landings of 3.5-4 m instead of ~2.2 m). It now takes the
  earliest parabola that falls from high up, i.e. the first flight
  (`twin/mocap.py`).

### Data (`twin_data/dataset/`)
- Real throws 0.6-1.0 s swing, various release times; `throws.csv` now holds
  44 reliable throws. GP leave-one-out error: median ~3 cm.
- Removed from `throws.csv`:
  - 9 rows that recorded the bounce (re-thrown afterwards);
  - old rows 1-11 (before the fixes) -> `throws_old_1_11.csv`;
  - 6 failed throws (ball landed on Spot) -> `throws_failed.csv`.
- Backups of each step: `throws_before_bounce_fix.csv`,
  `throws_before_remove_1_11.csv`, `throws_before_remove_failed.csv`.
- `gp_real_throws.png`: GP prediction and uncertainty over the real throws.

## 2026-11-11 / 12 (Mohammadreza)
- `--fake` throws go to `throws_fake.csv` (176 simulated throws).
- `learning_throw_simulation.ipynb`: GP inverse model trained on the simulated
  dataset.

## 2026-11-06
- Workflow throws a fixed cubic overhead swing; only swing time and release time
  are set. Joint limits 10 rad/s, 100 rad/s^2; `./spot cubic` command.
- Ball found as the marker track that flies on a parabola; landing extended
  along it if the floor is not seen.
- Each throw saved in `twin_data/<ball>/T<swing>_T<release>/`, landing inside
  the npz (key `landing`).

## 2026-11-05
- Parameter-throw dataset workflow: Spot stays standing, landing x saved to
  `throws.csv` with id; fast gripper opening; sh1 limit -2.3 (box on Spot);
  tracker ignores static markers; `q` sits Spot down.
