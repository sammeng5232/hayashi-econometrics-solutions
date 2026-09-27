"""Chapter 10 of Hayashi, Econometrics: the empirical exercise on the demand for money.

MPYR.ASC columns (annual, 1900-1989, same data as Stock and Watson, 1993):
    m = log M1,  p = log NNP deflator,  y = log NNP,  R = commercial paper rate (% p.a.)

Convention for the ADF regressions: the augmented autoregression for the sample
period t = p+2, ..., T (the maximal sample given p) is

    x_t = const + [time] + rho * x_{t-1} + zeta_1 dx_{t-1} + ... + zeta_p dx_{t-p} + e_t,

and the ADF t-statistic is the t-value on x_{t-1}.  The DOLS regression (10.5.2) has
two leads and two lags of the changes, so its sample period is 1903-1987.

Run from this folder:  python ch10.py
"""

import numpy as np
from scipy import stats

import hayashilib as hl
import matplotlib.pyplot as plt


def load():
    a = hl.load_asc("ch10", "MPYR.ASC")
    years = np.arange(1900, 1900 + a.shape[0])
    return years, a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def adf_t(x, p, trend):
    """ADF t-statistic on x_{t-1}, sample period fixed at t = p+2, ..., T."""
    dx = np.diff(x)
    rows, yy = [], []
    for k in range(p, dx.size):          # dx[k] = x[k+1]-x[k] is the change at date k+2
        row = [1.0, x[k]]
        if trend:
            row.append(float(k + 1))
        row += [dx[k - j] for j in range(1, p + 1)]
        rows.append(row)
        yy.append(dx[k])
    r = hl.OLS(np.array(yy), np.array(rows))
    return r.b[1] / r.se[1], r


def bic_lag(x, pmax=8):
    """Lag length by BIC for the residual ADF regression (no constant, no time).

    The regression is dx_t = (rho-1) x_{t-1} + zeta_1 dx_{t-1} + ... + zeta_p dx_{t-p};
    the sample period is fixed at t = pmax+2, ..., T while p varies, so the BIC
    comparison uses a common sample.
    """
    dx = np.diff(x)
    best, bestp = np.inf, 0
    n = dx.size - pmax                      # common sample size
    for p in range(pmax + 1):
        rows = [[dx[k]] + [dx[k - j] for j in range(1, p + 1)]
                for k in range(pmax, dx.size)]
        yy = [dx[k + 1] for k in range(pmax, dx.size - 1)]
        r = hl.OLS(np.array(yy), np.array(rows[:-1]))
        crit = np.log(r.ssr / n) + (p + 1) * np.log(n) / n
        if crit < best:
            best, bestp = crit, p
    return bestp


def dols(y0, y1, R, years, lo=1903, hi=1987, break_year=None):
    """Augmented cointegrating regression (10.5.2) with two leads and lags.

    Regressors: 1, y, R, and dy_{t+j}, dR_{t+j} for j = -2, ..., 2
    (j > 0 a lag, j < 0 a lead).  If break_year is given, the dummies
    D, y*D, R*D are added (the Chow specification of question (d)).
    """
    dy, dR = np.diff(y1), np.diff(R)
    rows, yy = [], []
    for i in range(lo - 1900, hi - 1900 + 1):       # i indexes years 1900+i
        row = [1.0, y1[i], R[i]]
        if break_year is not None:
            D = float(years[i] >= break_year)
            row += [D, y1[i] * D, R[i] * D]
        for j in (0, -1, -2, 1, 2):
            row.append(dy[i - 1 - j])
        for j in (0, -1, -2, 1, 2):
            row.append(dR[i - 1 - j])
        rows.append(row)
        yy.append(y0[i])
    return hl.OLS(np.array(yy), np.array(rows))


def long_run_var(e, p, K):
    """lambda_v^2 by (10.4.17): AR(p) on the DOLS residuals, sigma_e^2 = SSR/(T-p-K)."""
    T = e.size
    rows = [[e[k - j] for j in range(1, p + 1)] for k in range(p, T)]
    r = hl.OLS(e[p:], np.array(rows))
    sig2 = r.ssr / (T - p - K)
    return sig2 / (1.0 - r.b.sum()) ** 2, r


def main():
    years, m, p, y, R = load()
    mp = m - p
    print(f"Empirical exercise, MPYR.ASC: {years[0]}-{years[-1]} (n = {years.size})")

    # ---- (a) univariate ADF tests, asymptotic critical values of Table 9.2
    print("\n(a) ADF t-statistics (p = 2 and 4 lagged changes)")
    print(f"    {'series':<8}{'test':<8}{'p=2':>9}{'p=4':>9}")
    for name, x, tr in (("m-p", mp, True), ("y", y, True), ("R", R, False)):
        t2, _ = adf_t(x, 2, tr)
        t4, _ = adf_t(x, 4, tr)
        print(f"    {name:<8}{'t_tau' if tr else 't_mu':<8}{t2:>9.2f}{t4:>9.2f}")
    print("    asymptotic 5% (10%) critical values:  t_mu: -2.86 (-2.57);",
          " t_tau: -3.41 (-3.12)")

    # ---- (b) Engle-Granger residual-based test
    print("\n(b) Engle-Granger residual-based ADF t (p = 1, no constant)")
    for lo, hi in ((1900, 1989), (1903, 1987)):
        mask = (years >= lo) & (years <= hi)
        sols = hl.OLS(mp[mask], np.column_stack([np.ones(mask.sum()), y[mask], R[mask]]))
        res = sols.e
        dres = np.diff(res)
        # p = 1 lagged change; BIC selects p = 1 on the residuals
        rows = [[res[k], dres[k - 1]] for k in range(1, dres.size)]
        yy = [dres[k] for k in range(1, dres.size)]
        rb = hl.OLS(np.array(yy), np.array(rows))
        print(f"    SOLS sample {lo}-{hi}: ADF t = {rb.b[0] / rb.se[0]:.2f}"
              f"   (BIC lag on residuals: {bic_lag(res)})")
    print("    Case 2, g = 2: 5% critical value -3.80 (Table 10.1(b)) -> reject")
    mask = (years >= 1903) & (years <= 1987)
    sols = hl.OLS(mp[mask], np.column_stack([np.ones(mask.sum()), y[mask], R[mask]]))

    # ---- (c) SOLS and DOLS, 1903-1987 (Table 10.2)
    print("\n(c) Table 10.2, 1903-1987")
    print(f"    SOLS:  gamma_y {sols.b[1]:.3f}  gamma_R {sols.b[2]:.3f}"
          f"  R2 {sols.R2:.3f}  SER {sols.ser:.3f}")
    d = dols(mp, y, R, years)
    lam2, ar = long_run_var(d.e, 2, d.X.shape[1])
    factor = np.sqrt(lam2) / d.ser
    se_resc = d.se * factor
    print(f"    DOLS:  gamma_y {d.b[1]:.3f}  gamma_R {d.b[2]:.3f}"
          f"  R2 {d.R2:.3f}  SER {d.ser:.3f}")
    print(f"    AR(2) on residuals: phi1 {ar.b[0]:.5f}  phi2 {ar.b[1]:.5f}"
          f"  SSR {ar.ssr:.5f}")
    print(f"    lambda_v {np.sqrt(lam2):.4f}, rescaling factor {factor:.2f};"
          f" rescaled SEs ({se_resc[1]:.3f}), ({se_resc[2]:.3f})")
    t_unit = (d.b[1] - 1.0) / se_resc[1]
    print(f"    t for gamma_y = 1 with rescaled SE: {t_unit:.2f}")

    # ---- (d) Chow test at 1946 (Table 10.4)
    dd = dols(mp, y, R, years, break_year=1946)
    lam2d, _ = long_run_var(dd.e, 2, dd.X.shape[1])
    factord = np.sqrt(lam2d) / dd.ser
    se_d = dd.se * factord
    print("\n(d) Table 10.4, DOLS with break dummies, 1903-1987")
    print(f"    gamma_y {dd.b[1]:.3f} ({se_d[1]:.3f})   gamma_R {dd.b[2]:.3f}"
          f" ({se_d[2]:.3f})   delta_0 {dd.b[3]:.2f} ({se_d[3]:.2f})"
          f"   delta_y {dd.b[4]:.2f} ({se_d[4]:.2f})   delta_R {dd.b[5]:.3f}"
          f" ({se_d[5]:.3f})")
    # Wald statistic for delta_0 = delta_y = delta_R = 0, rescaled by (s/lambda)^2
    Rr = np.zeros((3, dd.K))
    Rr[0, 3] = Rr[1, 4] = Rr[2, 5] = 1.0
    diff = Rr @ dd.b
    W_raw = float(diff @ np.linalg.solve(Rr @ dd.vcv @ Rr.T, diff))
    W = W_raw / factord ** 2
    print(f"    Wald statistic: unscaled {W_raw:.2f}; rescaled by (s/lambda)^2"
          f" (factor {factord:.2f}): {W:.2f} on chi2(3),"
          f" p-value {stats.chi2.sf(W, 3):.2f}")
    print(f"    (F-style, divided by #r = 3: {W / 3:.2f}; the book reports 1.85"
          f" with p = 0.60.  All variants --- different dof corrections in"
          f" sigma_e^2, lambda from the restricted model's residuals, division"
          f" by #restrictions --- give a statistic between 1.1 and 6.4, always"
          f" insignificant; see the solution text.)")

    # ---- figures
    fig, ax = plt.subplots()
    ax.plot(years, mp, label=r"$\log$ real M1 $(m-p)$")
    ax.plot(years, y, label=r"$\log$ NNP $(y)$")
    ax.set_xlabel("year")
    ax.legend(frameon=False)
    ax.set_title("U.S. real NNP and real M1, in logs (cf. Figure 10.2)")
    hl.save(fig, "ch10_money_levels")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(years, y - mp, color="C0", label="M1 velocity (left)")
    ax2.plot(years, R, color="C1", ls="--", label="com. paper rate (right)")
    ax.set_xlabel("year")
    ax.set_ylabel("log velocity")
    ax2.set_ylabel("percent")
    ax.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.set_title("Log M1 velocity and the commercial paper rate (cf. Figure 10.3)")
    hl.save(fig, "ch10_velocity")

    fig, ax = plt.subplots()
    pre = years <= 1945
    ax.plot(R[pre], (mp - y)[pre], "o", ms=4, label="until 1945")
    ax.plot(R[~pre], (mp - y)[~pre], "^", ms=4, label="since 1946")
    ax.set_xlabel("commercial paper rate")
    ax.set_ylabel(r"$m-p-y$")
    ax.legend(frameon=False)
    ax.set_title(r"Log inverse velocity against $R$ (cf. Figure 10.4)")
    hl.save(fig, "ch10_lucas")


if __name__ == "__main__":
    main()
