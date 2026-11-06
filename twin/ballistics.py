"""Ball flight model: detect release, estimate release state, predict landing,
fit the drag coefficient. Works on measured (noisy, gappy) ball positions."""

import numpy as np

G = np.array([0.0, 0.0, -9.81])
RHO_AIR = 1.225


def drag_k(radius, mass, drag_coeff):
    """Drag acceleration = -k * |v| * v."""
    return 0.5 * RHO_AIR * drag_coeff * np.pi * radius**2 / mass


def simulate_flight(pos0, vel0, k, dt=1e-3, t_max=5.0, ground_z=0.0):
    """RK4 integration of gravity + quadratic drag until the ball reaches ground_z.
    Returns (t, pos) arrays; the last point is interpolated onto ground_z."""
    def acc(v):
        return G - k * np.linalg.norm(v) * v

    x, v = np.array(pos0, float), np.array(vel0, float)
    ts, xs = [0.0], [x.copy()]
    t = 0.0
    while t < t_max:
        k1v, k1x = acc(v), v
        k2v, k2x = acc(v + 0.5 * dt * k1v), v + 0.5 * dt * k1v
        k3v, k3x = acc(v + 0.5 * dt * k2v), v + 0.5 * dt * k2v
        k4v, k4x = acc(v + dt * k3v), v + dt * k3v
        x_new = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
        if x_new[2] <= ground_z and v[2] < 0:
            a = (x[2] - ground_z) / (x[2] - x_new[2])
            ts.append(t - dt + a * dt)
            xs.append(x + a * (x_new - x))
            break
        x = x_new
        ts.append(t)
        xs.append(x.copy())
    return np.array(ts), np.array(xs)


def valid(ball_t, ball_pos):
    m = ~np.isnan(ball_pos).any(axis=1)
    return ball_t[m], ball_pos[m]


def measured_landing(ball_t, ball_pos, ground_z):
    """First crossing of ground_z by the measured ball, interpolated."""
    t, x = valid(ball_t, ball_pos)
    below = np.where(x[:, 2] <= ground_z)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    a = (x[i - 1, 2] - ground_z) / (x[i - 1, 2] - x[i, 2])
    return t[i - 1] + a * (t[i] - t[i - 1]), x[i - 1] + a * (x[i] - x[i - 1])


def extrapolated_landing(ball_t, ball_pos, ground_z, g=9.81):
    """Where the seen flight would reach ground_z, for when the ball is lost
    before the floor (cameras do not see the last part). Fits x, y linear and
    z a parabola with gravity over the seen samples. -> (t, pos) or None."""
    t, x = valid(ball_t, ball_pos)
    if len(t) < 8:
        return None
    t0 = t[0]
    t = t - t0
    A = np.c_[np.ones_like(t), t]
    cx = np.linalg.lstsq(A, x[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A, x[:, 1], rcond=None)[0]
    cz = np.linalg.lstsq(A, x[:, 2] + 0.5 * g * t ** 2, rcond=None)[0]
    # z(t) = cz0 + cz1 t - g t^2 / 2 = ground_z, later root
    a, b, c = -0.5 * g, cz[1], cz[0] - ground_z
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    tl = max((-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a))
    if tl < t[-1]:
        return None
    return tl + t0, np.array([cx[0] + cx[1] * tl, cy[0] + cy[1] * tl, ground_z])


def detect_release(ball_t, grip_dist, after_t, sep_m=0.02, noise_m=0.001):
    """Release time from the ball-to-gripper distance (see kinematics.py).

    Ball-only data cannot tell release apart reliably: just before release the
    hand itself moves on a near-parabolic path. Instead, find the first time
    the ball is clearly away from the gripper (> sep_m), then walk back to the
    last sample where it was still within the sensor noise. The result is
    therefore independent of the (unknown) gripper delay.
    """
    m = (ball_t >= after_t) & ~np.isnan(grip_dist)
    t, d = ball_t[m], grip_dist[m]
    away = np.where(d > sep_m)[0]
    if len(away) == 0:
        return None
    base = np.median(d[: max(away[0], 1)])  # grasp offset error, ~0 when calibrated
    close = np.where(d[: away[0]] < base + 3 * noise_m * np.sqrt(3))[0]
    return t[close[-1]] if len(close) else t[away[0]]


def estimate_release_state(ball_t, ball_pos, release_t, window=0.12, skip=0.02):
    """Least-squares fit of pos(t) = p0 + v0*t + g*t^2/2 on the first `window`
    seconds of free flight, evaluated at release_t. The first `skip` seconds
    are left out in case release_t is a few ms early. Drag is negligible over
    such a short window."""
    t, x = valid(ball_t, ball_pos)
    m = (t >= release_t + skip) & (t <= release_t + skip + window)
    if m.sum() < 4:
        raise ValueError("Not enough ball samples right after release.")
    tt = t[m] - release_t
    A = np.stack([np.ones_like(tt), tt], axis=1)
    p0, v0 = np.zeros(3), np.zeros(3)
    for ax in range(3):
        y = x[m, ax] - 0.5 * G[ax] * tt**2
        (p0[ax], v0[ax]), *_ = np.linalg.lstsq(A, y, rcond=None)
    return p0, v0


def fit_flight(ball_t, ball_pos, release_t, p0, v0, radius, mass, landing_t=None, fit_cd=True,
               cd_init=0.5):
    """Jointly fit the release state (p0, v0) and, optionally, the drag
    coefficient to the whole measured flight (release -> first ground contact).

    Fitting Cd alone with p0/v0 fixed from a short window does not work: the
    velocity error from sensor noise moves the landing point more than drag
    does, and Cd absorbs it. Returns (p0, v0, cd, rms)."""
    from scipy.optimize import least_squares

    t, x = valid(ball_t, ball_pos)
    end_t = np.inf if landing_t is None else landing_t
    m = (t >= release_t) & (t < end_t)
    if m.sum() < 8:
        return p0, v0, float("nan"), float("nan")
    tt, xx = t[m] - release_t, x[m]
    t_max = tt[-1] + 0.05

    def residual(z):
        cd = z[6] if fit_cd else 0.0
        ft, fx = simulate_flight(z[:3], z[3:6], drag_k(radius, mass, cd), dt=2e-3, t_max=t_max,
                                 ground_z=-np.inf)
        pred = np.stack([np.interp(tt, ft, fx[:, a]) for a in range(3)], axis=1)
        return (pred - xx).ravel()

    z0 = np.concatenate([p0, v0, [cd_init] if fit_cd else []])
    lo = np.concatenate([np.full(6, -np.inf), [0.0] if fit_cd else []])
    hi = np.concatenate([np.full(6, np.inf), [3.0] if fit_cd else []])
    sol = least_squares(residual, z0, bounds=(lo, hi))
    rms = float(np.sqrt(np.mean(np.sum(sol.fun.reshape(-1, 3) ** 2, axis=1))))
    return sol.x[:3], sol.x[3:6], (float(sol.x[6]) if fit_cd else 0.0), rms


def refine_release(ball_t, grip, detected_t, earliest_t, p0, v0):
    """The distance test fires a little late (the ball separates slowly). Walk
    the fitted free-flight curve backwards and take the time where it passes
    closest to the gripper: that is where the ball actually left the hand."""
    m = (ball_t >= earliest_t) & (ball_t <= detected_t)
    if not m.any():
        return detected_t
    dt = ball_t[m] - detected_t
    model = p0 + v0 * dt[:, None] + 0.5 * G * dt[:, None] ** 2
    gap = np.linalg.norm(model - grip[m], axis=1)
    return float(ball_t[m][np.argmin(gap)])


def analyze(rec, radius, mass, drag_coeff_prior, grip=None, fit_drag=True):
    """Everything the viewer shows about the ball for one record.
    fit_drag=False skips the (slow) drag-coefficient fit."""
    if grip is None:
        from .kinematics import gripper_path
        grip = gripper_path(rec)
    grip_dist = np.linalg.norm(rec.ball_pos - grip, axis=1)
    out = {"radius": radius, "grip_dist": grip_dist, "grip": grip}
    detected_t = detect_release(rec.ball_t, grip_dist, rec.release_cmd_t)
    if detected_t is None:
        out["release_t"] = None
        return out
    p_det, v_det = estimate_release_state(rec.ball_t, rec.ball_pos, detected_t)
    release_t = refine_release(rec.ball_t, grip, detected_t, rec.release_cmd_t, p_det, v_det)
    dt = release_t - detected_t
    p0 = p_det + v_det * dt + 0.5 * G * dt**2
    v0 = v_det + G * dt
    out["release_t"], out["p0"], out["v0"] = release_t, p0, v0
    horiz = np.linalg.norm(v0[:2])
    out["speed"] = float(np.linalg.norm(v0))
    out["elevation_deg"] = float(np.degrees(np.arctan2(v0[2], horiz)))
    out["heading_deg"] = float(np.degrees(np.arctan2(v0[1], v0[0])))
    # Prediction from the release state alone (what the planner would use).
    # The ball centre touches the floor at floor_z + radius.
    ground_z = rec.floor_z + radius
    ft, fx = simulate_flight(p0, v0, drag_k(radius, mass, drag_coeff_prior), ground_z=ground_z)
    out["pred_t"], out["pred_pos"] = ft + release_t, fx
    out["pred_landing"] = fx[-1]
    ml = measured_landing(rec.ball_t, rec.ball_pos, ground_z)
    out["meas_landing"] = None if ml is None else ml[1]
    if ml is not None:
        out["landing_error"] = float(np.linalg.norm(fx[-1][:2] - ml[1][:2]))
    landing_t = None if ml is None else ml[0]
    if not fit_drag:
        out["fitted_cd"], out["fit_rms"], out["cd_identifiable"] = float("nan"), float("nan"), False
        return out
    *_, cd, rms = fit_flight(rec.ball_t, rec.ball_pos, release_t, p0, v0, radius, mass, landing_t,
                             cd_init=drag_coeff_prior)
    *_, rms_no_drag = fit_flight(rec.ball_t, rec.ball_pos, release_t, p0, v0, radius, mass, landing_t,
                                 fit_cd=False)
    out["fitted_cd"], out["fit_rms"] = cd, rms
    # If adding drag barely improves the fit, the flight is too short/slow to identify it.
    out["cd_identifiable"] = bool(rms_no_drag - rms > 0.0005)
    return out
