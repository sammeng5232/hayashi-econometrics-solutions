"""Chapter 9 of Hayashi, Econometrics: numerical parts of the solutions.

Covers
  * Monte Carlo Exercise 9.1  power of the DF t^mu and t^tau tests (T = 100)
  * Monte Carlo Exercise 9.2  size distortions of the ADF rho^mu and t^mu
                              tests for Schwert's (1989) I(1) DGP with
                              theta = -0.8 and 4 lagged changes (T = 100)
  * Empirical Exercise 9.1    Lothian and Taylor's (1996) dollar/sterling real
                              exchange rate (LT.ASC), 1791-1990:
                              (a) ADF lag selection (sequential t, AIC, BIC)
                              (b) DF t^mu on the float, 1974-1990
                              (c) DF t^mu on the gold standard, 1870-1913
                              (d) Chow test, 1791-1974 versus 1974-1990
                              (e) ADF-GLS t^mu on the gold standard
                              (f) variance/covariances of time aggregates

Run from this folder:  python ch9.py
"""

import numpy as np

import hayashilib as hl


# ---------------------------------------------------------------------------
# DF / ADF test statistics
# ---------------------------------------------------------------------------
def df_stats(y, const=True, trend=False, p=0):
    """DF/ADF statistics for the regression

        y_t = (const) (+ trend) + rho y_{t-1}
              + zeta_1 Dy_{t-1} + ... + zeta_p Dy_{t-p} + e_t .

    y is the full vector including any initial values; the regression is run
    on the maximal sample t = p + 1, ..., T (0-based: y has T + 1 entries).
    Returns a dict with rhohat, T*(rhohat - 1), the ADF rho statistic
    T*(rhohat-1)/(1-sum zeta), the t-value for rho = 1, the zeta estimates,
    SSR and the number of observations used.
    """
    y = np.asarray(y, float).ravel()
    T = y.size - 1
    dy = np.diff(y)
    # rows t = p+1, ..., T  (t indexes y, so python index t)
    idx = np.arange(p + 1, T + 1)
    yy = y[idx]
    cols = []
    if const:
        cols.append(np.ones(idx.size))
    if trend:
        cols.append(idx.astype(float))
    cols.append(y[idx - 1])
    for j in range(1, p + 1):
        cols.append(dy[idx - 1 - j])          # Delta y_{t-j}
    X = np.column_stack(cols)
    r = hl.OLS(yy, X)
    k = r.K - 1                               # position of the y_{t-1} coef
    rhohat = r.b[k]
    zeta = r.b[k + 1:]
    t_rho = (rhohat - 1.0) / r.se[k]
    adf_rho = (yy.size) * (rhohat - 1.0) / (1.0 - zeta.sum())
    return dict(rhohat=rhohat, se=r.se[k], t=t_rho, Trho=adf_rho,
                zeta=zeta, ssr=r.ssr, n=yy.size, fit=r)


# ------------------------------------------------- Monte Carlo Exercise 9.1
def mc_power():
    """Finite-sample power of the DF t^mu and t^tau tests.

    DGP: y_t = rho y_{t-1} + e_t, rho = 0.95, e_t iid N(0, 1), y_0 = 0,
    T = 100, nominal size 5%; 5% critical values -2.86 (DF t^mu) and
    -3.41 (DF t^tau) from Table 9.2.
    """
    rng = np.random.default_rng(20260921)
    T, R, rho = 100, 200_000, 0.95
    # vectorized generation: y_t = rho^t y_0 + sum_{s<=t} rho^{t-s} e_s
    # r = (rho, rho^2, ..., rho^T)', A lower triangular Toeplitz with rho^k
    r = rho ** np.arange(1, T + 1)
    first_col = rho ** np.arange(0, T)
    A = np.zeros((T, T))
    for i in range(T):
        A[i, :i + 1] = first_col[:i + 1][::-1]
    # A[t-1, s-1] = rho^{t-s}
    cnt_mu = cnt_tau = 0
    crit_mu, crit_tau = -2.86, -3.41
    B = 5000
    tt = np.arange(1, T + 1, dtype=float)
    for rep in range(R // B):
        E = rng.standard_normal((B, T))
        Y = E @ A.T                                   # (B, T): y_1..y_T
        Y0 = np.zeros((B, 1))                         # y_0 = 0
        Yf = np.concatenate([Y0, Y], axis=1)          # y_0..y_T
        yl = Yf[:, :-1]                               # y_{t-1}, t = 1..T
        yc = Yf[:, 1:]
        # t^mu: regression of y_t on (1, y_{t-1})
        xm = yl.mean(axis=1, keepdims=True)
        ym = yc.mean(axis=1, keepdims=True)
        sxy = ((yl - xm) * (yc - ym)).sum(axis=1)
        sxx = ((yl - xm) ** 2).sum(axis=1)
        b = sxy / sxx
        a = ym.ravel() - b * xm.ravel()
        e = yc - a[:, None] - b[:, None] * yl
        s2 = (e ** 2).sum(axis=1) / (T - 2)
        se = np.sqrt(s2 / sxx)
        t_mu = (b - 1.0) / se
        # t^tau: regression of y_t on (1, t, y_{t-1}); by Frisch-Waugh,
        # regress out (1, t) from y_t and y_{t-1} and do a simple regression
        ones = np.ones((B, T))
        ttv = np.broadcast_to(tt, (B, T))
        # regress out (1, t) from yc and yl (Frisch-Waugh)
        Xd = np.stack([ones, ttv], axis=2)            # (B, T, 2)
        XtX = np.einsum('btk,btl->bkl', Xd, Xd)
        XtXinv = np.linalg.inv(XtX)
        P1 = np.einsum('btk,bkl->btl', Xd, XtXinv)    # (B,T,2)
        def resid(v):
            # v: (B, T, 1); returns residuals from projecting v on (1, t)
            coef = np.einsum('btk,btl->bkl', Xd, v)   # X'v, (B, 2, 1)
            fitted = np.einsum('btl,bl->bt', P1, coef[..., 0])
            return v[..., 0] - fitted
        yl_r = resid(yl[..., None])
        yc_r = resid(yc[..., None])
        b2 = (yl_r * yc_r).sum(axis=1) / (yl_r ** 2).sum(axis=1)
        e2 = yc_r - b2[:, None] * yl_r
        s22 = (e2 ** 2).sum(axis=1) / (T - 3)
        se2 = np.sqrt(s22 / (yl_r ** 2).sum(axis=1))
        t_tau = (b2 - 1.0) / se2
        cnt_mu += int((t_mu < crit_mu).sum())
        cnt_tau += int((t_tau < crit_tau).sum())
    print("\nMonte Carlo Exercise 9.1: power of DF t^mu and t^tau, "
          f"T = {T}, rho = 0.95, {R} replications")
    print(f"  power of DF t^mu (cv -2.86): {cnt_mu / R:.4f}  (book: about 0.124)")
    print(f"  power of DF t^tau (cv -3.41): {cnt_tau / R:.4f}  (book: about 0.092)")


# ------------------------------------------------- Monte Carlo Exercise 9.2
def mc_size():
    """Size distortions of the ADF rho^mu and t^mu tests.

    DGP (Schwert 1989): y_t = rho y_{t-1} + e_t + theta e_{t-1},
    rho = 1, theta = -0.8, e_t iid N(0, 1), y_0 = e_0 = 0, T = 100.
    Regression: augmented autoregression with intercept and p = 4 lagged
    changes, estimated for t = 5, ..., T (actual sample size T - 4 = 96).
    5% asymptotic critical values: ADF rho^mu -14.1, ADF t^mu -2.86.
    """
    rng = np.random.default_rng(20260922)
    T, R, theta, p = 100, 100_000, -0.8, 4
    B = 5000
    idx = np.arange(p + 1, T + 1)                     # t = 5,...,100
    n = idx.size
    cnt_rho = cnt_t = 0
    crit_rho, crit_t = -14.1, -2.86
    for rep in range(R // B):
        E = rng.standard_normal((B, T + 1))           # e_0..e_T
        u = E + theta * np.roll(E, 1, axis=1)         # u_t = e_t + theta e_{t-1}
        u[:, 0] = E[:, 0]                             # u_0 unused
        Y = np.cumsum(u, axis=1)                      # y_t, y_0 = e_0 = 0? see below
        Y = Y - Y[:, :1]                              # enforce y_0 = 0
        yl = Y[:, idx - 1]
        yc = Y[:, idx]
        dy = np.diff(Y, axis=1)                       # dy[:, t-1] = y_t - y_{t-1}
        lags = [dy[:, idx - 1 - j] for j in range(1, p + 1)]  # Dy_{t-j}
        Z = np.stack(lags, axis=2)                    # (B, n, p)
        # full OLS of y_t on X = (1, y_{t-1}, Dy_{t-1}, ..., Dy_{t-4})
        X = np.concatenate([np.ones((B, n, 1)), yl[..., None], Z], axis=2)
        XtXf = np.einsum('btk,btl->bkl', X, X)
        Xty = np.einsum('btk,bt->bk', X, yc)[..., None]   # (B, K, 1)
        beta = np.linalg.solve(XtXf, Xty)[..., 0]         # (B, K)
        zeta = beta[:, 2:]
        rhohat = beta[:, 1]
        resid = yc - np.einsum('btk,bk->bt', X, beta)
        s2 = (resid ** 2).sum(axis=1) / (n - (p + 2))
        XtXinv = np.linalg.inv(XtXf)
        se_rho = np.sqrt(s2 * XtXinv[:, 1, 1])
        t_mu = (rhohat - 1.0) / se_rho
        adf_rho = n * (rhohat - 1.0) / (1.0 - zeta.sum(axis=1))
        cnt_rho += int((adf_rho < crit_rho).sum())
        cnt_t += int((t_mu < crit_t).sum())
    print("\nMonte Carlo Exercise 9.2: exact size of ADF rho^mu and t^mu, "
          f"T = {T}, theta = -0.8, p = 4, {R} replications")
    print(f"  size of ADF rho^mu (cv -14.1): {cnt_rho / R:.4f}  (book: about 0.496)")
    print(f"  size of ADF t^mu  (cv -2.86):  {cnt_t / R:.4f}  (book: about 0.290)")


# ------------------------------------------------------------ Empirical 9.1
def lothian_taylor():
    a = hl.load_asc("ch9", "LT.ASC")
    year = a[:, 0].astype(int)
    S, P, Pstar = a[:, 1], a[:, 2], a[:, 3]
    z = np.log(S) + np.log(Pstar) - np.log(P)        # log real exchange rate
    print(f"\nEmpirical Exercise 9.1 (LT.ASC), {year[0]}-{year[-1]}, "
          f"{year.size} observations")

    # ---- (a) lag selection on the fixed sample 1806-1990
    pmax = int(12 * (year.size / 100) ** 0.25)       # = 14
    print(f"(a) pmax = {pmax} by (9.4.31); fixed selection sample 1806-1990")
    sel = (year >= 1806) & (year <= 1990)
    n_sel = sel.sum()
    results = []
    for p in range(0, pmax + 1):
        # regression of z_t on (1, z_{t-1}, Dz_{t-1..t-p}), t in selection window
        idx = np.where(sel)[0]
        yy = z[idx]
        cols = [np.ones(idx.size), z[idx - 1]]
        dz = np.diff(z)
        for j in range(1, p + 1):
            cols.append(dz[idx - 1 - j])
        X = np.column_stack(cols)
        r = hl.OLS(yy, X)
        aic = np.log(r.ssr / r.n) + r.K * 2.0 / r.n
        bic = np.log(r.ssr / r.n) + r.K * np.log(r.n) / r.n
        results.append((p, r, aic, bic))
    aics = np.array([x[2] for x in results])
    bics = np.array([x[3] for x in results])
    p_aic = int(np.argmin(aics))
    p_bic = int(np.argmin(bics))
    # sequential t-rule at 10%: start from pmax, stop at the first p whose
    # last lagged change has |t| > 1.645; if none, p = 0.
    p_seq = 0
    for p, r, _, _ in reversed(results):
        if p >= 1 and abs(r.b[-1] / r.se[-1]) > 1.645:
            p_seq = p
            break
    print(f"  sequential t (10%): p = {p_seq};  AIC: p = {p_aic};  "
          f"BIC: p = {p_bic}   (book: 14, 14, 0)")
    print("    p    AIC      BIC     t on last lag")
    for p, r, aic, bic in results:
        tlast = r.b[-1] / r.se[-1] if p >= 1 else float("nan")
        print(f"  {p:3d}  {aic:8.4f} {bic:8.4f}  {tlast:8.3f}")
    # ADF t^mu with p = 0 on the maximal sample (1792-1990)
    r0 = hl.OLS(z[1:], np.column_stack([np.ones(z.size - 1), z[:-1]]))
    t_mu0 = (r0.b[1] - 1.0) / r0.se[1]
    print(f"  AR(1) with intercept, 1792-1990 (n = {r0.n}):")
    print(f"    const = {r0.b[0]:.4f} ({r0.se[0]:.4f}), "
          f"rho = {r0.b[1]:.4f} ({r0.se[1]:.4f}), R2 = {r0.R2:.4f}")
    print(f"    DF t^mu = {t_mu0:.4f}  (book: -3.47; Lothian-Taylor t_mu)")

    # ---- (b) float 1974-1990: regression z_t on (1, z_{t-1}), t = 1975..1990
    ib = np.where((year >= 1975) & (year <= 1990))[0]
    rb = hl.OLS(z[ib], np.column_stack([np.ones(ib.size), z[ib - 1]]))
    tb = (rb.b[1] - 1.0) / rb.se[1]
    print(f"\n(b) float 1974-1990 (n = {rb.n}): "
          f"rho = {rb.b[1]:.4f} ({rb.se[1]:.4f}), SSR = {rb.ssr:.6f}, "
          f"R2 = {rb.R2:.4f}")
    print(f"    DF t^mu = {tb:.4f}  (book: -1.21; 5% cv -2.86: fail to reject)")

    # ---- (c) gold standard 1870-1913: t = 1871..1913
    ic = np.where((year >= 1871) & (year <= 1913))[0]
    rc = hl.OLS(z[ic], np.column_stack([np.ones(ic.size), z[ic - 1]]))
    tc = (rc.b[1] - 1.0) / rc.se[1]
    print(f"\n(c) gold standard 1870-1913 (n = {rc.n}): "
          f"rho = {rc.b[1]:.4f} ({rc.se[1]:.4f}), SSR = {rc.ssr:.6f}, "
          f"R2 = {rc.R2:.4f}")
    print(f"    DF t^mu = {tc:.4f}  (book: -2.66; 5% cv -2.86: fail to reject)")

    # ---- (d) Chow test: 1792-1974 (183 obs) versus 1975-1990 (16 obs)
    i1 = np.where((year >= 1792) & (year <= 1974))[0]
    i2 = np.where((year >= 1975) & (year <= 1990))[0]
    ia = np.where((year >= 1792) & (year <= 1990))[0]
    def ar1(idx):
        return hl.OLS(z[idx], np.column_stack([np.ones(idx.size), z[idx - 1]]))
    r1, r2, ra = ar1(i1), ar1(i2), ar1(ia)
    K = 2
    F = ((ra.ssr - r1.ssr - r2.ssr) / K) / ((r1.ssr + r2.ssr) / (r1.n + r2.n - 2 * K))
    print(f"\n(d) Chow test: subsample 1792-1974 (n = {r1.n}): "
          f"rho = {r1.b[1]:.4f} ({r1.se[1]:.4f}), SSR = {r1.ssr:.6f}")
    print(f"    pooled SSR = {ra.ssr:.6f};  SSR_1 + SSR_2 = {r1.ssr + r2.ssr:.6f}")
    from scipy import stats
    print(f"    K*F = {K * F:.4f}  (book: about 1.26); "
          f"chi2(2) 5% cv = {stats.chi2.ppf(0.95, 2):.2f}, p-value = "
          f"{stats.chi2.sf(K * F, 2):.3f}")

    # ---- (e) ADF-GLS t^mu on the gold standard 1870-1913
    ig = np.where((year >= 1870) & (year <= 1913))[0]
    yg = z[ig]
    Tg = yg.size
    rho_bar = 1.0 - 7.0 / Tg
    # quasi-difference (first observation unweighted, as in the text)
    yq = np.concatenate([[yg[0]], yg[1:] - rho_bar * yg[:-1]])
    xq = np.concatenate([[1.0], np.full(Tg - 1, 1.0 - rho_bar)])
    b_gls = float(yq @ xq / (xq @ xq))
    ztil = yg - b_gls
    # BIC lag selection on the demeaned series, pmax = 9, fixed sample
    pmax_g = int(12 * (Tg / 100) ** 0.25)            # = 9
    best = (np.inf, 0, None)
    for p in range(0, pmax_g + 1):
        # regression of ztil_t on (ztil_{t-1}, Dztil_{t-1..t-p}),
        # fixed sample t = pmax_g+1..Tg for comparability
        idx = np.arange(pmax_g, Tg)
        yy = ztil[idx]
        cols = [ztil[idx - 1]]
        dzt = np.diff(ztil)
        for j in range(1, p + 1):
            cols.append(dzt[idx - 1 - j])
        X = np.column_stack(cols)
        r = hl.OLS(yy, X)
        bic = np.log(r.ssr / r.n) + r.K * np.log(r.n) / r.n
        if bic < best[0]:
            best = (bic, p, r)
    p_gls = best[1]
    # final ADF regression with p_gls on the maximal sample, t = p+1, ..., T
    idx = np.arange(p_gls + 1, Tg + 1) - 1           # 0-based into ztil
    yy = ztil[idx]
    cols = [ztil[idx - 1]]
    dzt = np.diff(ztil)
    for j in range(1, p_gls + 1):
        cols.append(dzt[idx - 1 - j])
    X = np.column_stack(cols)
    rg = hl.OLS(yy, X)
    t_gls = (rg.b[0] - 1.0) / rg.se[0]
    print(f"\n(e) ADF-GLS t^mu, gold standard 1870-1913 (T = {Tg}): "
          f"rho_bar = 1 - 7/T = {rho_bar:.4f}")
    print(f"    BIC picks p = {p_gls} (pmax = {pmax_g}); "
          f"ADF-GLS t^mu = {t_gls:.4f}  (book: -2.24)")
    print(f"    5% critical value is the DF_t one, -1.95: "
          f"{'reject' if t_gls < -1.95 else 'fail to reject'} the I(1) null")

    # ---- (f) time aggregation: Corr of the change in the m-period average
    # Weights on year-(j-1) innovations in D(ybar_j): (s-1)/m; on year-j: (m-s+1)/m
    # Var(D ybar) = sigma^2 (2m^2+1)/(3m);  Cov(lag 1) = sigma^2 (m^2-1)/(6m) > 0.
    for m in (5, 50, 250):
        v = (2 * m * m + 1) / (3 * m)
        c = (m * m - 1) / (6 * m)
        print(f"\n(f) time aggregation, m = {m} days: "
              f"Var(Dybar)/sigma^2 = {v:.4f}, first autocovariance/sigma^2 = "
              f"{c:.4f}, first autocorrelation = {c / v:.4f} (-> 0.25 as m grows)")

    # ---- figure: log real exchange rate, 1791-1990
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(year, z, lw=0.9, color="black")
    ax.axhline(0, lw=0.6, color="grey")
    ax.axvline(1914, lw=0.6, color="grey", ls=":")
    ax.set_xlim(1791, 1990)
    ax.set_xlabel("year")
    ax.set_ylabel(r"$z_t = s_t + p_t^* - p_t$")
    ax.set_title("Dollar/sterling log real exchange rate, 1791-1990 "
                 "(1914 value set to 0)")
    hl.save(fig, "ch9_realx")


if __name__ == "__main__":
    mc_power()
    mc_size()
    lothian_taylor()
