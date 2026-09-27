"""Chapter 6 of Hayashi, Econometrics: numerical parts of the solutions.

Covers
  * Empirical Exercise 6.1   Bekaert-Hodrick data (DM, POUND, YEN):
        (a) largest |forward premium| week; (b) correlogram of the forecast
        error s30 - f; (c) correlogram and Box-Ljung test of s(t+1) - s(t);
        (d) the unconditional test of Table 6.1; (e) the regression test of
        Table 6.2 with the truncated kernel, q = 4; (f) Bartlett/Newey-West
        with the Newey-West automatic bandwidths (12 yen, 8 DM, 16 pound);
        (g) VARHAC with pmax = [n^{1/3}] and BIC lag selection.
  * Empirical Exercise 6.2   Fama's three-month T-bill test on MISHKIN.ASC:
        correlogram of the ex-post real rate; the Fama regression of PAI3 on
        TB3 with S estimated by (6.6.4) with q = 2.

Run from this folder:  python ch6.py
"""

import numpy as np
from scipy import stats

import hayashilib as hl

BW_NW = {"YEN": 12, "DM": 8, "POUND": 16}   # Newey-West (1994) automatic lags


# ------------------------------------------------------ HAC building blocks
def gamma_g(g, j):
    """Sample autocovariance matrix (6.6.2) of the rows of g at lag j."""
    n = g.shape[0]
    return (g[j:].T @ g[:n - j]) / n


def lrv(g, q, kernel="truncated"):
    """Kernel-based estimator (6.6.3)/(6.6.4) of the long-run variance S."""
    S = gamma_g(g, 0).copy()
    for j in range(1, min(q, g.shape[0] - 1) + 1):
        if kernel == "truncated":
            w = 1.0
        else:                                    # Bartlett (6.6.7)
            x = j / (q + 1.0)
            w = 1.0 - x if abs(x) <= 1.0 else 0.0
        G = gamma_g(g, j)
        S = S + w * (G + G.T)
    return S


def hac_ols(y, X, q, kernel="truncated"):
    """OLS with HAC standard errors: Avar(b) = Sxx^{-1} S Sxx^{-1}."""
    fit = hl.OLS(y, X)
    n = fit.n
    Sxxi = np.linalg.inv(X.T @ X / n)
    S = lrv(X * fit.e[:, None], q, kernel)
    avar = Sxxi @ S @ Sxxi
    se = np.sqrt(np.diag(avar) / n)
    return fit, avar, se


def correlogram(x, lags, demean=True):
    """Sample autocorrelations (2.10.1)-(2.10.2), denominator n."""
    d = x - x.mean() if demean else x.copy()
    g0 = float(d @ d) / x.size
    return np.array([float(d[j:] @ d[:x.size - j]) / x.size / g0
                     for j in range(1, lags + 1)])


def ljung_box(x, lags, demean=True):
    rho = correlogram(x, lags, demean)
    n = x.size
    jj = np.arange(1, lags + 1)
    Q = n * (n + 2) * np.sum(rho ** 2 / (n - jj))
    return Q, float(stats.chi2.sf(Q, lags))


def plot_correlogram(rho, n, name, title):
    fig, ax = __import__("matplotlib.pyplot").pyplot.subplots(figsize=(5.2, 2.6))
    j = np.arange(1, rho.size + 1)
    ax.bar(j, rho, width=0.6, color="0.35")
    band = 1.96 / np.sqrt(n)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(band, color="0.45", linestyle="--", linewidth=1.0)
    ax.axhline(-band, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xlabel("lag")
    ax.set_ylabel(r"$\hat\rho_j$")
    ax.set_title(title)
    hl.save(fig, name)


# ------------------------------------------------------------------ EE 6.1
def load_fx(cur):
    a = hl.load_asc("ch6", f"{cur}.ASC")
    return a[:, 0], np.log(a[:, 1]), np.log(a[:, 2]), np.log(a[:, 3])


def bekaert_hodrick():
    print("Empirical Exercise 6.1 (Bekaert-Hodrick data, 1975-1989)")
    for cur in ["YEN", "DM", "POUND"]:
        date, s, f, s30 = load_fx(cur)
        n = s.size
        e = s30 - f
        print(f"\n=== {cur} (n = {n}) ===")

        # (a) week with the largest |forward premium|
        fp = (f - s) * 1200
        k = int(np.argmax(np.abs(fp)))
        print(f"(a) max |f - s| x 1200 = {abs(fp[k]):.2f} in week "
              f"{int(date[k])}")

        # (b) correlogram of the forecast error
        rho_e = correlogram(e, 40)
        print("(b) correlogram of s30 - f, lags 1-8:",
              np.round(rho_e[:8], 3))
        print("    lags 5-12:", np.round(rho_e[4:12], 3),
              f"  (2/sqrt(n) = {2 / np.sqrt(n):.3f})")
        plot_correlogram(rho_e, n, f"ch6_correlogram_eps_{cur.lower()}",
                         f"Correlogram of $s30 - f$, {cur}/\\$")

        # (c) correlogram of s_{t+1} - s_t and Box-Ljung
        ds = np.diff(s)
        rho_ds = correlogram(ds, 40)
        Q40, p40 = ljung_box(ds, 40)
        print(f"(c) correlogram of Delta s, lags 1-6: {np.round(rho_ds[:6], 3)};"
              f"  Ljung-Box(40) = {Q40:.1f} (p = {p40:.4g})")
        plot_correlogram(rho_ds, ds.size, f"ch6_correlogram_ds_{cur.lower()}",
                         f"Correlogram of $s_{{t+1}} - s_t$, {cur}/\\$")

        # (d) unconditional test (Table 6.1)
        v = np.array([(s30 - s).mean(), (f - s).mean(), e.mean()]) * 1200
        sd = np.array([(s30 - s).std(ddof=1), (f - s).std(ddof=1),
                       e.std(ddof=1)]) * 1200
        ge = np.array([float(e[j:] @ e[:n - j]) / n for j in range(5)])
        se_mean = np.sqrt((ge[0] + 2 * ge[1:].sum()) / n) * 1200
        print(f"(d) means x1200:  s30-s {v[0]:6.2f} ({sd[0]:.1f}),  "
              f"f-s {v[1]:6.2f} ({sd[1]:.2f}),  eps {v[2]:6.2f} ({sd[2]:.1f})")
        print(f"    standard error of the mean = {se_mean:.2f},  "
              f"t = {v[2] / se_mean:.2f}")

        # (e) regression test, truncated kernel with q = 4 (Table 6.2)
        y, X = (s30 - s) * 1200, np.column_stack([np.ones(n), fp])
        fit, Avar, se = hac_ols(y, X, 4, "truncated")
        d = fit.b - np.array([0.0, 1.0])
        W = float(n * d @ np.linalg.inv(Avar) @ d)
        print(f"(e) s30-s = {fit.b[0]:7.2f} ({se[0]:.2f}) + "
              f"{fit.b[1]:6.2f} ({se[1]:.2f}) (f-s),  R2 = {fit.R2:.4f}")
        print(f"    Wald for (0,1): {W:.2f} (p = {stats.chi2.sf(W, 2):.4g})")

        # (f) Bartlett kernel, Newey-West automatic bandwidth
        _, Avar_nw, se_nw = hac_ols(y, X, BW_NW[cur], "bartlett")
        W_nw = float(n * d @ np.linalg.inv(Avar_nw) @ d)
        print(f"(f) Bartlett, q = {BW_NW[cur]}:  SEs ({se_nw[0]:.4f}, "
              f"{se_nw[1]:.4f}),  Wald = {W_nw:.2f} "
              f"(p = {stats.chi2.sf(W_nw, 2):.4g})")

        # (g) VARHAC
        S_vh, phat, pmax = varhac(X * fit.e[:, None])
        Avar_vh = np.linalg.inv(X.T @ X / n) @ S_vh @ np.linalg.inv(X.T @ X / n)
        se_vh = np.sqrt(np.diag(Avar_vh) / n)
        W_vh = float(n * d @ np.linalg.inv(Avar_vh) @ d)
        ptxt = tuple(int(p) for p in phat)
        print(f"(g) VARHAC (pmax = {pmax}, p = {ptxt}):  "
              f"SEs ({se_vh[0]:.4f}, {se_vh[1]:.4f}),  Wald = {W_vh:.2f} "
              f"(p = {stats.chi2.sf(W_vh, 2):.4g})")


def varhac(g):
    """VARHAC estimator (6.6.15) of the long-run variance of the rows of g.

    Step 1 picks a lag length per equation by BIC on the fixed sample
    t = pmax+1, ..., n with objective log(SSR/(n-pmax)) + p*K*log(n-pmax)/(n-pmax).
    Step 2 assembles the implied long-run variance [I - sum A_j]^{-1} Omega
    [I - sum A_j]'^{-1}.
    """
    n, K = g.shape
    pmax = int(n ** (1 / 3))
    T = n - pmax
    phat = np.zeros(K, int)
    coefs = []
    for k in range(K):
        dep = g[pmax:, k]
        best, bp = np.inf, 0
        for p in range(pmax + 1):
            if p == 0:
                e = dep
            else:
                Xv = np.column_stack(
                    [np.ones(T)] + [g[pmax - j:n - j, :]
                                    for j in range(1, p + 1)])
                e = hl.OLS(dep, Xv).e
            ssr = float(e @ e)
            bic = np.log(ssr / T) + p * K * np.log(T) / T
            if bic < best:
                best, bp = bic, p
        phat[k] = bp
        if bp > 0:
            Xv = np.column_stack([np.ones(T)] + [g[pmax - j:n - j, :]
                                                 for j in range(1, bp + 1)])
            coefs.append(hl.OLS(dep, Xv).b)
        else:
            coefs.append(np.array([dep.mean()]))
    pbar = int(phat.max())
    A = [np.zeros((K, K)) for _ in range(pbar)]
    Eres = np.zeros((T, K))
    for k in range(K):
        dep = g[pmax:, k]
        fitv = np.full(T, coefs[k][0])
        for j in range(1, phat[k] + 1):
            blk = coefs[k][1 + (j - 1) * K:1 + j * K]
            A[j - 1][k, :] = blk
            fitv = fitv + g[pmax - j:n - j, :] @ blk
        Eres[:, k] = dep - fitv
    Omega = Eres.T @ Eres / T
    M = np.eye(K) - sum(A)
    S = np.linalg.solve(M, Omega) @ np.linalg.inv(M).T
    return S, phat, pmax


# ------------------------------------------------------------------ EE 6.2
def fama3():
    a = hl.load_asc("ch2", "MISHKIN.ASC")
    year, month, pai1, pai3, tb1, tb3, cpi = (a[:, j] for j in range(7))
    date = year + (month - 1) / 12.0
    mask = (date >= 1953 - 1e-9) & (date <= 1971 + 6 / 12 + 1e-9)
    n = int(mask.sum())
    print(f"\nEmpirical Exercise 6.2 (MISHKIN.ASC, 1/53-7/71, n = {n})")

    # (c) correlogram of the ex-post real rate r_{t+3} = TB3_t - PAI3_t
    r = tb3[mask] - pai3[mask]
    rho_r = correlogram(r, 12)
    Q12, p12 = ljung_box(r, 12)
    print("(c) correlogram of the ex-post real rate, lags 1-12:")
    print("   ", np.round(rho_r, 3))
    print(f"    Ljung-Box(12) = {Q12:.1f} (p = {p12:.3g}); "
          f"Ljung-Box(2) = {ljung_box(r, 2)[0]:.1f}")
    plot_correlogram(correlogram(r, 40), n, "ch6_correlogram_realrate",
                     "Correlogram of the ex-post real rate, 1953-1971")

    # (d) the Fama regression with S from (6.6.4), q = 2
    y, X = pai3[mask], np.column_stack([np.ones(n), tb3[mask]])
    fit, Avar, se = hac_ols(y, X, 2, "truncated")
    d = fit.b - np.array([0.0, 1.0])
    W = float(n * d @ np.linalg.inv(Avar) @ d)
    t1 = (fit.b[1] - 1.0) / se[1]
    print(f"(d) PAI3 = {fit.b[0]:.3f} ({se[0]:.3f}) + {fit.b[1]:.3f} "
          f"({se[1]:.3f}) TB3,  R2 = {fit.R2:.3f}")
    print(f"    t(beta1 = 1) = {t1:.2f};  Wald for (0,1) = {W:.2f} "
          f"(p = {stats.chi2.sf(W, 2):.3g})")
    # for comparison: the Bartlett version with q = 3
    _, Avar_b, se_b = hac_ols(y, X, 3, "bartlett")
    print(f"    [Bartlett q = 3 for comparison: SEs ({se_b[0]:.3f}, "
          f"{se_b[1]:.3f}), t(beta1=1) = {(fit.b[1] - 1) / se_b[1]:.2f}]")


if __name__ == "__main__":
    bekaert_hodrick()
    fama3()
