"""Closed-loop throw optimization: Bayesian optimization (Gaussian process +
expected improvement), numpy only.

Each round: suggest throw parameters -> throw (simulation, fake lab or real
robot) -> measure the landing point -> update the model. The objective is
the miss distance to a target on the floor (robot frame).

The history is saved after every throw (JSON), so a session on the real robot
can be stopped (battery swap, ...) and resumed with --resume.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from throw_trajectory import check_trajectory, cubic_joint_trajectory

NO_LANDING_MISS = 3.0  # objective value when the ball's landing was not measured


# ------------------------------------------------------------ throw space ---

@dataclass
class ThrowSpace:
    """What the optimizer may change, and how that becomes a trajectory.

    Default: shoulder + elbow swing (the forward throw found in simulation),
      duration          swing time (s)
      release_fraction  when the gripper opens, as a fraction of the swing
      sh1_end           shoulder pitch at the end of the swing: range
      sh0               shoulder yaw, the whole swing: aiming left / right
    """
    names: tuple = ("duration", "release_fraction", "sh1_end", "sh0")
    lower: tuple = (0.9, 0.3, -1.6, -0.5)
    upper: tuple = (2.0, 0.8, -0.6, 0.5)
    start_pose: tuple = (0.0, -2.2, 3.0, 0.0, 0.0, 1.57)
    el0_end: float = 1.8
    max_vel: float = 2.5
    max_acc: float = 10.0

    def bounds(self):
        return np.array(self.lower, float), np.array(self.upper, float)

    def to_throw(self, x):
        """Parameter vector -> (duration, release_fraction, trajectory)."""
        p = dict(zip(self.names, x))
        start = list(self.start_pose)
        start[0] = p["sh0"]
        end = [p["sh0"], p["sh1_end"], self.el0_end, start[3], start[4], start[5]]
        traj = cubic_joint_trajectory(start, end, p["duration"])
        return p["duration"], p["release_fraction"], traj

    def feasible(self, x):
        return not check_trajectory(*self.to_throw(x)[2], self.max_vel, self.max_acc)


# ---------------------------------------------------------- GP + suggest ---

class GaussianProcess:
    """GP regression with an RBF kernel on inputs scaled to [0, 1].
    Length scale and noise are picked by marginal likelihood on a small grid."""

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y)) or 1.0
        self.y = (np.asarray(y, float) - self.y_mean) / self.y_std
        best = None
        for ls in (0.1, 0.2, 0.3, 0.5, 0.8):
            for noise in (1e-4, 1e-3, 1e-2, 1e-1):
                K = self._k(self.X, self.X, ls) + noise * np.eye(len(self.X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
                nll = 0.5 * self.y @ alpha + np.sum(np.log(np.diag(L)))
                if best is None or nll < best[0]:
                    best = (nll, ls, noise, L, alpha)
        _, self.ls, self.noise, self.L, self.alpha = best
        return self

    @staticmethod
    def _k(A, B, ls):
        d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-0.5 * np.maximum(d2, 0) / ls**2)

    def predict(self, Xs):
        Ks = self._k(np.asarray(Xs, float), self.X, self.ls)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(1.0 - np.sum(v**2, axis=0), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std


def expected_improvement(mu, sigma, best, xi=0.005):
    from math import erf, sqrt
    z = (best - mu - xi) / sigma
    cdf = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return (best - mu - xi) * cdf + sigma * pdf


@dataclass
class Optimizer:
    space: ThrowSpace = field(default_factory=ThrowSpace)
    target: tuple = (1.0, 0.0)
    n_init: int = 6                      # random throws before the model takes over
    history: list = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed + len(self.history))

    def _scale(self, X):
        lo, hi = self.space.bounds()
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _random_feasible(self, n, around=None, spread=0.1):
        lo, hi = self.space.bounds()
        out, tries = [], 0
        while len(out) < n and tries < 50 * n:
            tries += 1
            if around is None:
                x = lo + self.rng.random(len(lo)) * (hi - lo)
            else:
                x = np.clip(around + self.rng.normal(0, spread, len(lo)) * (hi - lo), lo, hi)
            if self.space.feasible(x):
                out.append(x)
        return np.array(out)

    def suggest(self):
        done = [h for h in self.history if h.get("miss") is not None]
        if len(done) < self.n_init:
            return self._random_feasible(1)[0]
        X = np.array([h["x"] for h in done])
        y = np.array([h["miss"] for h in done])
        gp = GaussianProcess().fit(self._scale(X), y)
        best_x = X[np.argmin(y)]
        cands = np.vstack([self._random_feasible(1500), self._random_feasible(500, around=best_x)])
        mu, sigma = gp.predict(self._scale(cands))
        ei = expected_improvement(mu, sigma, y.min())
        return cands[int(np.argmax(ei))]

    def tell(self, x, landing, record_file=None):
        miss = NO_LANDING_MISS if landing is None else float(np.linalg.norm(np.asarray(landing[:2]) - self.target))
        entry = {"x": [float(v) for v in x], "params": dict(zip(self.space.names, map(float, x))),
                 "landing": None if landing is None else [float(v) for v in landing[:2]],
                 "miss": miss, "record": record_file, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.history.append(entry)
        return entry

    def best(self):
        done = [h for h in self.history if h.get("miss") is not None]
        return min(done, key=lambda h: h["miss"]) if done else None

    # -- persistence --

    def save(self, path):
        Path(path).write_text(json.dumps({"target": list(self.target), "space": {
            "names": self.space.names, "lower": self.space.lower, "upper": self.space.upper,
            "start_pose": self.space.start_pose, "el0_end": self.space.el0_end,
            "max_vel": self.space.max_vel, "max_acc": self.space.max_acc}, "history": self.history}, indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        s = d["space"]
        space = ThrowSpace(tuple(s["names"]), tuple(s["lower"]), tuple(s["upper"]), tuple(s["start_pose"]),
                           s["el0_end"], s["max_vel"], s["max_acc"])
        return cls(space=space, target=tuple(d["target"]), history=d["history"])


# ------------------------------------------------------------------ loop ---

def run(opt, thrower, n_iter, history_path, data_dir, prefix, on_result=None, stop_at=None):
    """thrower(duration, release_fraction, trajectory, target) -> ThrowRecord.
    on_result(record, analysis, entry) is called after every throw."""
    from . import ballistics

    for _ in range(n_iter):
        x = opt.suggest()
        duration, rf, traj = opt.space.to_throw(x)
        k = len(opt.history) + 1
        print(f"\n[round {k}] " + ", ".join(f"{n}={v:.3f}" for n, v in zip(opt.space.names, x)))
        rec = thrower(duration, rf, traj, opt.target)
        rec.params["optimizer_round"] = k
        path = rec.save(Path(data_dir) / f"{prefix}_round{k:03d}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        P = rec.params
        a = ballistics.analyze(rec, P["ball_radius"], P["ball_mass"], P["drag_coeff"], fit_drag=False)
        landing = a.get("meas_landing")
        entry = opt.tell(x, landing, str(path))
        opt.save(history_path)
        best = opt.best()
        print(f"  landing {None if landing is None else np.round(landing[:2], 3)} -> miss "
              f"{entry['miss'] * 100:.1f} cm | best so far {best['miss'] * 100:.1f} cm (round "
              f"{opt.history.index(best) + 1})")
        if on_result is not None:
            on_result(rec, a, entry)
        if stop_at is not None and entry["miss"] <= stop_at:
            print(f"Reached the {stop_at * 100:.0f} cm goal.")
            break
    warn_at_bounds(opt)
    return opt.best()


def warn_at_bounds(opt):
    """A best throw sitting on a parameter limit usually means the target is out
    of reach of this throw space (not an optimizer problem)."""
    best = opt.best()
    if best is None or best["miss"] < 0.05:
        return
    lo, hi = opt.space.bounds()
    x = np.array(best["x"])
    for name, xi, l, h in zip(opt.space.names, x, lo, hi):
        if min(xi - l, h - xi) < 0.02 * (h - l):
            print(f"NOTE: best throw has {name} = {xi:.3f} at its limit [{l}, {h}]. The target may be "
                  f"out of reach: widen that limit, raise the speed limits, or move the target.")


def save_plot(opt, path):
    """Miss per round + landing points around the target (PNG). Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: no plot)")
        return None
    miss = np.array([h["miss"] for h in opt.history]) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    rounds = np.arange(1, len(miss) + 1)
    ax1.plot(rounds, miss, "o-", color="#888888", label="miss")
    ax1.plot(rounds, np.minimum.accumulate(miss), "-", color="#1f6feb", lw=2, label="best so far")
    ax1.set_xlabel("round")
    ax1.set_ylabel("miss (cm)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    pts = np.array([h["landing"] for h in opt.history if h["landing"] is not None])
    if len(pts):
        ax2.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap="viridis", s=30, label="landings")
    ax2.plot(*opt.target, "x", color="#d73a49", ms=12, mew=3, label="target")
    ax2.set_xlabel("x forward (m)")
    ax2.set_ylabel("y left (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
