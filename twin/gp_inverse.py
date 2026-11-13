"""Learned inverse model: which swing time / release time throws to a target x.

Same method as learning_throw_simulation.ipynb (Mohammadreza): a Gaussian
process maps (swing_time, release_time) -> landing_x, trained on the dataset
CSV; for a target distance we search the two times whose predicted landing is
the target, preferring places where the GP is sure (std < sigma_max).

The GP is retrained from the whole CSV every time, so each new throw that
workflow.py adds is used for the next suggestion.
"""

import csv

import numpy as np

SIGMA_MAX = 0.05  # m, wanted prediction std (as in the notebook)


def load_xy(path):
    X, y = [], []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    X.append([float(r["swing_time"]), float(r["release_time"])])
                    y.append(float(r["landing_x"]))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return np.array(X).reshape(-1, 2), np.array(y)


class InverseGP:
    def __init__(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        self.X, self.y = X, y
        self.xs, self.zs = StandardScaler(), StandardScaler()
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2))
                  + WhiteKernel(0.1, (1e-5, 1e1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=False,
                                           random_state=0)
        self.gp.fit(self.xs.fit_transform(X), self.zs.fit_transform(y.reshape(-1, 1)).ravel())

    def predict(self, swing, release):
        mu, std = self.gp.predict(self.xs.transform([[swing, release]]), return_std=True)
        return float(self.zs.inverse_transform(mu.reshape(-1, 1))[0, 0]), float(std[0] * self.zs.scale_[0])

    def solve(self, target, feasible=None, sigma_max=SIGMA_MAX):
        """-> dict(swing_time, release_time, predicted, std, sure) or None.
        Searches only inside the range of the data (no extrapolation) and
        keeps release_time < swing_time; `feasible(swing, release)` can reject
        more (robot limits)."""
        from scipy.optimize import minimize
        lo, hi = self.X.min(axis=0), self.X.max(axis=0)
        bounds = list(zip(lo, hi))
        from .cubic_throw import min_release
        cons = [{"type": "ineq", "fun": lambda p: p[0] - p[1] - 0.02},
                {"type": "ineq", "fun": lambda p: p[1] - min_release(p[0])}]  # earliest release that throws
        sure = cons + [{"type": "ineq", "fun": lambda p: sigma_max - self.predict(*p)[1]}]
        obj = lambda p: (self.predict(*p)[0] - target) ** 2
        # start from the throws that landed closest to the target
        starts = self.X[np.argsort(np.abs(self.y - target))[:5]]
        best = None
        for constraints, is_sure in ((sure, True), (cons, False)):
            for x0 in starts:
                r = minimize(obj, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
                if not r.success:
                    continue
                s, rel = (float(v) for v in r.x)
                if feasible is not None and not feasible(s, rel):
                    continue
                mu, std = self.predict(s, rel)
                if is_sure and std > sigma_max:
                    continue
                if best is None or abs(mu - target) < abs(best["predicted"] - target):
                    best = {"swing_time": s, "release_time": rel, "predicted": mu, "std": std, "sure": is_sure}
            if best is not None:
                return best
        return None


def suggest(csv_path, target, feasible=None, min_rows=5):
    """Train on csv_path and solve for target. -> (result dict or None, message)."""
    X, y = load_xy(csv_path)
    if len(y) < min_rows:
        return None, f"only {len(y)} throws in the dataset; need at least {min_rows}"
    if len(np.unique(X, axis=0)) < 3:
        return None, "the dataset has fewer than 3 different settings; throw a few more settings first"
    gp = InverseGP(X, y)
    res = gp.solve(target, feasible)
    if res is None:
        return None, f"no setting found for {target} m (data covers {y.min():.2f}-{y.max():.2f} m)"
    return res, f"trained on {len(y)} throws"
