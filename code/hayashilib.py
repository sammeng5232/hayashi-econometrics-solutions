"""Shared helpers for the solutions to Hayashi, Econometrics (2000).

Only two things are needed often enough to be worth sharing: a least-squares
routine that reports what a regression printout reports, and a uniform way of
writing figures into ../figures.  Everything else lives in the chapter scripts.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is fixed)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
    "lines.linewidth": 1.4,
    "savefig.dpi": 200,
})

FIG = Path(__file__).resolve().parent.parent / "figures"
DATA = Path(__file__).resolve().parent.parent / "data"


def load_asc(*parts):
    """Read one of Hayashi's ASCII data files from ../data.

    The files are DOS text and end with a ^Z (0x1a) byte, which numpy's loadtxt
    reads as a short final row; it is stripped here.
    """
    path = DATA.joinpath(*parts)
    text = path.read_text(encoding="latin-1").replace("\x1a", "")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    width = max(len(r) for r in rows)
    if min(len(r) for r in rows) != width:      # ragged file: caller handles it
        return rows
    return np.array(rows, dtype=float)


def save(fig, name):
    """Write fig to ../figures/<name>.pdf and close it."""
    FIG.mkdir(exist_ok=True)
    path = FIG / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path.name}")
    return path


class OLS:
    """OLS of y on X (no constant is added: put it in X yourself).

    Attributes: b, e, ssr, s2, ser, vcv (= s2 (X'X)^-1), se, R2, R2uc, n, K.
    """

    def __init__(self, y, X, robust=False):
        y = np.asarray(y, float).ravel()
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] != y.size:
            X = X.T
        self.y, self.X = y, X
        self.n, self.K = X.shape
        XX = X.T @ X
        self.XXinv = np.linalg.inv(XX)
        self.b = self.XXinv @ (X.T @ y)
        self.e = y - X @ self.b
        self.ssr = float(self.e @ self.e)
        self.s2 = self.ssr / (self.n - self.K)
        self.ser = np.sqrt(self.s2)
        if robust:                       # White's heteroskedasticity-robust matrix
            S = (X * self.e[:, None]).T @ (X * self.e[:, None])
            self.vcv = self.XXinv @ S @ self.XXinv * self.n / (self.n - self.K)
        else:
            self.vcv = self.s2 * self.XXinv
        self.se = np.sqrt(np.diag(self.vcv))
        ybar = y.mean()
        self.R2 = 1.0 - self.ssr / float(((y - ybar) ** 2).sum())
        self.R2uc = 1.0 - self.ssr / float(y @ y)

    def t(self, value=None):
        value = np.zeros(self.K) if value is None else np.asarray(value, float)
        return (self.b - value) / self.se

    def wald(self, R, r):
        """F-ratio (1.4.9) for R beta = r and its p-value."""
        from scipy import stats
        R, r = np.atleast_2d(R), np.atleast_1d(r)
        d = R @ self.b - r
        mid = np.linalg.inv(R @ self.vcv @ R.T)
        F = float(d @ mid @ d) / R.shape[0]
        p = float(stats.f.sf(F, R.shape[0], self.n - self.K))
        return F, p

    def summary(self, names=None, title=""):
        names = names or [f"x{k + 1}" for k in range(self.K)]
        out = [title] if title else []
        for k, nm in enumerate(names):
            out.append(f"  {nm:<22}{self.b[k]:>12.4f} ({self.se[k]:.4f})")
        out.append(f"  {'R2':<22}{self.R2:>12.4f}")
        out.append(f"  {'SER':<22}{self.ser:>12.4f}")
        out.append(f"  {'SSR':<22}{self.ssr:>12.4f}")
        out.append(f"  {'n':<22}{self.n:>12d}")
        return "\n".join(out)


class GMM:
    """Single-equation GMM of y on regressors Z using instruments X.

    W is the weighting matrix; the default is Sxx^{-1}, which gives 2SLS.
    Attributes: d (delta-hat), e, Sxz, sxy, g, J, S (estimated from the own
    residuals), avar (asymptotic variance of sqrt(n)(d - delta)), se (robust),
    se_h (conventional, valid under conditional homoskedasticity), sigma2, ssr.
    """

    def __init__(self, y, Z, X, W=None):
        y = np.asarray(y, float).ravel()
        Z = np.atleast_2d(np.asarray(Z, float))
        X = np.atleast_2d(np.asarray(X, float))
        n = y.size
        self.n, self.L, self.K = n, Z.shape[1], X.shape[1]
        self.Sxz = X.T @ Z / n
        self.sxy = X.T @ y / n
        self.Sxx = X.T @ X / n
        self.W = np.linalg.inv(self.Sxx) if W is None else np.asarray(W, float)
        A = self.Sxz.T @ self.W @ self.Sxz
        self.d = np.linalg.solve(A, self.Sxz.T @ self.W @ self.sxy)
        self.e = y - Z @ self.d
        self.ssr = float(self.e @ self.e)
        self.sigma2 = self.ssr / n
        self.g = self.sxy - self.Sxz @ self.d
        self.J = float(n * self.g @ self.W @ self.g)
        self.S = (X * self.e[:, None]).T @ (X * self.e[:, None]) / n
        Ai = np.linalg.inv(A)
        self.avar = Ai @ (self.Sxz.T @ self.W @ self.S @ self.W @ self.Sxz) @ Ai
        self.se = np.sqrt(np.diag(self.avar) / n)
        # conventional standard errors: Avar = sigma^2 (Sxz' Sxx^-1 Sxz)^-1
        self.se_h = np.sqrt(np.diag(
            self.sigma2 * np.linalg.inv(self.Sxz.T @ np.linalg.inv(self.Sxx) @ self.Sxz)) / n)
        self.ser = np.sqrt(self.ssr / (n - self.L))

    def sargan(self):
        """Sargan's statistic (3.8.10') and its p-value, for the 2SLS weighting."""
        from scipy import stats
        stat = float(self.n * self.g @ np.linalg.inv(self.Sxx) @ self.g / self.sigma2)
        df = self.K - self.L
        return stat, df, float(stats.chi2.sf(stat, df))

    def summary(self, names, title="", robust=False):
        se = self.se if robust else self.se_h
        out = [title] if title else []
        for k, nm in enumerate(names):
            out.append(f"  {nm:<12}{self.d[k]:>10.4f} ({se[k]:.4f})")
        return "\n".join(out)


def restricted_ols(y, X, R, r):
    """Restricted least squares of Analytical Exercise 1.5: argmin SSR s.t. R b = r."""
    u = OLS(y, X)
    R, r = np.atleast_2d(R), np.atleast_1d(r)
    A = u.XXinv @ R.T
    lam = np.linalg.solve(R @ A, R @ u.b - r)
    b = u.b - A @ lam
    e = y - X @ b
    return b, float(e @ e), lam
