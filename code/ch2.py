"""Chapter 2 of Hayashi, Econometrics: numerical parts of the solutions.

Covers
  * Review Question 2.6.3        F(10, inf) and chi^2(10) critical values
  * Empirical Exercise 2.1       Fama's test of market efficiency on MISHKIN.ASC
  * Empirical Exercise 2.2       the Nerlove data again: White's test, WLS, White SEs
  * Monte Carlo Exercise 2.1     uniform errors: t(30) versus N(0,1) critical values
  * Monte Carlo Exercise 2.2     Box-Pierce versus Ljung-Box in samples of 50

Run from this folder:  python ch2.py
"""

import numpy as np
from scipy import stats

import hayashilib as hl


# ------------------------------------------------------------ RQ 2.6.3
def f_versus_chi2():
    print("Review Question 2.6.3:  5% critical values")
    F = stats.f.ppf(0.95, 10, 10**7)          # F(10, inf)
    c = stats.chi2.ppf(0.95, 10)
    print(f"  F_0.05(10, inf) = {F:.4f}")
    print(f"  chi2_0.05(10)   = {c:.4f} = 10 x {c / 10:.4f}")


# ------------------------------------------------- autocorrelation tools
def autocov(z, p):
    """Sample autocovariances (2.10.1), denominator n, series demeaned."""
    z = np.asarray(z, float)
    n = z.size
    d = z - z.mean()
    return np.array([float(d[j:] @ d[:n - j]) / n for j in range(p + 1)])


def box_pierce(z, p):
    g = autocov(z, p)
    rho = g[1:] / g[0]
    n = z.size
    j = np.arange(1, p + 1)
    Q_bp = n * np.sum(rho**2)
    Q_lb = n * (n + 2) * np.sum(rho**2 / (n - j))
    return rho, Q_bp, Q_lb


# ------------------------------------------------------------- EE 2.1
def mishkin():
    a = hl.load_asc("ch2", "MISHKIN.ASC")
    year, month, pai1, pai3, tb1, tb3, cpi = (a[:, j] for j in range(7))
    n_all = year.size
    print(f"\nEmpirical Exercise 2.1 (MISHKIN.ASC, {n_all} observations, "
          f"{int(year[0])}:{int(month[0])} to {int(year[-1])}:{int(month[-1])})")

    # inflation matched with TB1_t: pi_{t+1} = [(CPI_t / CPI_{t-1})^12 - 1] x 100
    pi = np.full(n_all, np.nan)
    pi[1:] = ((cpi[1:] / cpi[:-1]) ** 12 - 1.0) * 100.0
    real = tb1 - pi                                   # ex-post real rate

    date = year + (month - 1) / 12.0
    fama = (date >= 1953 - 1e-9) & (date <= 1971 + 6 / 12 + 1e-9)   # 1/53 - 7/71
    n = int(fama.sum())

    # ---- (d) Table 2.1
    r = real[fama]
    rho, _, _ = box_pierce(r, 12)
    print(f"\n(d) Table 2.1, ex-post real rate, 1/53-7/71: mean = {r.mean():.2f}%, "
          f"s.d. = {r.std(ddof=1):.3f}%, n = {n}")
    print("      j:      " + "".join(f"{j:>7d}" for j in range(1, 13)))
    print("    rho_j:    " + "".join(f"{v:>7.3f}" for v in rho))
    print(f"    std.err:  " + "".join(f"{1 / np.sqrt(n):>7.3f}" for _ in rho))
    Qs = [box_pierce(r, p)[2] for p in range(1, 13)]
    print("    LB Q:     " + "".join(f"{q:>7.1f}" for q in Qs))
    print("    p-value:  " + "".join(f"{100 * stats.chi2.sf(q, p):>6.1f}%"
                                     for p, q in enumerate(Qs, start=1)))

    # ---- (e) the Fama regression with robust standard errors
    y, X = pi[fama], np.column_stack([np.ones(n), tb1[fama]])
    rob = hl.OLS(y, X, robust=True)
    rob0 = hl.OLS(y, X)                                   # same b, conventional SEs
    Srob = (X * rob0.e[:, None]).T @ (X * rob0.e[:, None]) / n
    se_white = np.sqrt(np.diag(np.linalg.inv(X.T @ X / n) @ Srob
                               @ np.linalg.inv(X.T @ X / n) / n))
    print(f"\n(e) pi_(t+1) = {rob0.b[0]:.3f} + {rob0.b[1]:.3f} TB1_t   "
          f"(White SEs {se_white[0]:.3f}, {se_white[1]:.3f})")
    print(f"    R2 = {rob0.R2:.2f}, mean of dep. var = {y.mean():.2f}, "
          f"SER = {rob0.ser:.2f}%, n = {n}")
    t1 = (rob0.b[1] - 1) / se_white[1]
    print(f"    t-ratio for H0: coefficient = 1 is {t1:.2f} (p = {2 * stats.norm.sf(abs(t1)):.2f})")

    # ---- (f) Davidson-MacKinnon variants of the robust standard errors
    K = X.shape[1]
    XXi = np.linalg.inv(X.T @ X / n)
    p_lev = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)   # p_i of (2.5.5)
    variants = {
        "HC0 (2.5.1)": rob0.e**2,
        "HC1 (x n/(n-K))": rob0.e**2 * n / (n - K),
        "HC2 (d = 1)": rob0.e**2 / (1 - p_lev),
        "HC3 (d = 2)": rob0.e**2 / (1 - p_lev) ** 2,
    }
    print("\n(f) robust standard errors, four variants:")
    for name, w in variants.items():
        S = (X * w[:, None]).T @ X / n
        se = np.sqrt(np.diag(XXi @ S @ XXi / n))
        print(f"    {name:<18} const {se[0]:.4f}   TB1 {se[1]:.4f}"
              f"   t(coef = 1) {(rob0.b[1] - 1) / se[1]:>6.3f}")

    # ---- (g) conventional standard errors
    print(f"\n(g) conventional SEs: const {rob0.se[0]:.4f}, TB1 {rob0.se[1]:.4f}"
          f"   t(coef = 1) = {(rob0.b[1] - 1) / rob0.se[1]:.3f}"
          "   [point estimates identical to (e)]")

    # ---- (h) Breusch-Godfrey with p = 12
    p = 12
    e = rob0.e
    E = np.zeros((n, p))
    for j in range(1, p + 1):
        E[j:, j - 1] = e[:-j]                 # e_t for t <= 0 set to zero
    aux = hl.OLS(e, np.column_stack([X, E]))
    nR2 = n * (1 - aux.ssr / float(e @ e))     # uncentered R2: e has mean ~ 0
    nR2c = n * aux.R2
    print(f"\n(h) Breusch-Godfrey, p = 12: nR2 = {nR2c:.1f} "
          f"(uncentered {nR2:.1f}), p-value = {stats.chi2.sf(nR2c, p):.4f}")

    # ---- (j) seasonal dummies
    D = np.zeros((n, 12))
    for m in range(12):
        D[:, m] = (month[fama] == m + 1).astype(float)
    seas = hl.OLS(y, np.column_stack([D, tb1[fama]]))
    F_seas, p_seas = seas.wald(np.column_stack([np.eye(11), -np.ones(11), np.zeros(11)]),
                               np.zeros(11))
    print(f"\n(j) with 12 monthly dummies: TB1 coefficient {seas.b[-1]:.3f} "
          f"(SE {seas.se[-1]:.3f});  F for equal monthly intercepts = {F_seas:.2f} "
          f"(p = {p_seas:.2f})")

    # ---- (k) Mishkin's own inflation measure
    ykm = pai1[fama]
    km = hl.OLS(ykm, X)
    Skm = (X * km.e[:, None]).T @ (X * km.e[:, None]) / n
    se_km = np.sqrt(np.diag(XXi @ Skm @ XXi / n))
    print(f"\n(k) with PAI1 as the inflation rate: {km.b[0]:.3f} + {km.b[1]:.3f} TB1_t "
          f"(White SEs {se_km[0]:.3f}, {se_km[1]:.3f}), R2 = {km.R2:.2f}")

    # ---- (l) post-October 1979 (the sample starts in November 1979)
    post = date >= 1979 + 10 / 12 - 1e-9
    npost = int(post.sum())
    Xp = np.column_stack([np.ones(npost), tb1[post]])
    XXip = np.linalg.inv(Xp.T @ Xp / npost)
    print(f"\n(l) 11/79-12/90 (n = {npost}):")
    for nm, dep in (("CPI inflation", pi), ("PAI1", pai1)):
        m = hl.OLS(dep[post], Xp)
        S = (Xp * m.e[:, None]).T @ (Xp * m.e[:, None]) / npost
        se = np.sqrt(np.diag(XXip @ S @ XXip / npost))
        print(f"    {nm:<14}{m.b[0]:>8.3f} ({se[0]:.3f}) + {m.b[1]:.3f} ({se[1]:.3f}) TB1_t"
              f"   R2 = {m.R2:.2f}")


# ------------------------------------------------------------- EE 2.2
def nerlove_ch2():
    a = hl.load_asc("ch1", "NERLOVE.ASC")
    TC, Q, PL, PF, PK = (a[:, j] for j in range(5))
    n = TC.size
    one, lQ = np.ones(n), np.log(Q)
    y = np.log(TC / PF)
    X4 = np.column_stack([one, lQ, lQ**2, np.log(PL / PF), np.log(PK / PF)])

    print("\nEmpirical Exercise 2.2 (Nerlove's Model 4)")

    # (j) Step 1: OLS; Step 2: regress e^2 on a constant and 1/Q; Step 3: WLS
    ols = hl.OLS(y, X4)
    Z = np.column_stack([one, 1.0 / Q])
    step2 = hl.OLS(ols.e**2, Z)
    print(f"(j) Step 2:  e^2 = {step2.b[0]:.4f} + {step2.b[1]:.4f} (1/Q)   "
          f"(SEs {step2.se[0]:.4f}, {step2.se[1]:.4f})")
    print("    the book's Model 4 variance function is sigma^2 (0.0565 + 2.1377/Q):"
          f" the ratio of the estimates is {step2.b[1] / step2.b[0]:.2f} against "
          f"{2.1377 / 0.0565:.2f}")
    w = 1.0 / np.sqrt(np.maximum(Z @ step2.b, 1e-12))
    wls = hl.OLS(y * w, X4 * w[:, None])
    w_book = 1.0 / np.sqrt(0.0565 + 2.1377 / Q)
    wls_book = hl.OLS(y * w_book, X4 * w_book[:, None])
    print("    WLS with the estimated weights:", np.round(wls.b, 4))
    print("    WLS with the book's weights:   ", np.round(wls_book.b, 4))

    # (i) White's nR2 test for Model 4, on the OLS residuals
    a1, b1, c1 = lQ, np.log(PL / PF), np.log(PK / PF)
    psi = np.column_stack([a1, a1**2, a1**3, a1**4, b1, c1,
                           a1 * b1, a1 * c1, a1**2 * b1, a1**2 * c1,
                           b1**2, b1 * c1, c1**2])
    white = hl.OLS(ols.e**2, np.column_stack([one, psi]))
    nR2 = n * white.R2
    print(f"\n(i) White's test on Model 4 (OLS residuals): nR2 = {nR2:.3f} on "
          f"chi2({psi.shape[1]}), p = {stats.chi2.sf(nR2, psi.shape[1]):.4f}")

    # (k) White standard errors for Model 4
    rob = hl.OLS(y, X4, robust=False)
    S = (X4 * ols.e[:, None]).T @ (X4 * ols.e[:, None]) / n
    XXi = np.linalg.inv(X4.T @ X4 / n)
    se_w = np.sqrt(np.diag(XXi @ S @ XXi / n))
    names = ["const", "log Q", "(log Q)^2", "log(p1/p3)", "log(p2/p3)"]
    print("\n(k) Model 4 by OLS: coefficient, conventional SE, White SE")
    for k, nm in enumerate(names):
        print(f"    {nm:<12}{ols.b[k]:>10.4f}{rob.se[k]:>10.4f}{se_w[k]:>10.4f}")


# ------------------------------------------------------- Monte Carlo 2.1
def mc_uniform(reps=1_000_000, batch=50_000, seed=20260921):
    """Chapter 1's second simulation with uniformly distributed errors."""
    import ch1
    rng = np.random.default_rng(seed)
    t_t, t_n = stats.t.ppf(0.975, ch1.N - 2), stats.norm.ppf(0.975)
    rej_t = rej_n = 0
    done = 0
    while done < reps:
        m = min(batch, reps - done)
        x = ch1.draw_x(m, rng)
        eps = rng.uniform(-0.5, 0.5, size=(ch1.N, m))       # variance 1/12
        y = ch1.BETA[0] + ch1.BETA[1] * x + eps
        _, _, t = ch1.simple_regression(x, y)
        rej_t += int((np.abs(t) > t_t).sum())
        rej_n += int((np.abs(t) > t_n).sum())
        done += m
    print(f"\nMonte Carlo Exercise 2.1 ({reps:,} replications, uniform errors)")
    print(f"  rejection frequency with t_0.025(30) = {t_t:.3f}: {rej_t / reps:.5f}")
    print(f"  rejection frequency with z_0.025     = {t_n:.3f}: {rej_n / reps:.5f}")


# ------------------------------------------------------- Monte Carlo 2.2
def mc_box_pierce(reps=200_000, n=50, p=4, seed=20260922):
    rng = np.random.default_rng(seed)
    crit = stats.chi2.ppf(0.95, p)
    rej_bp = rej_lb = 0
    j = np.arange(1, p + 1)
    for start in range(0, reps, 10_000):
        m = min(10_000, reps - start)
        z = rng.standard_normal((n, m))
        d = z - z.mean(0)
        g0 = (d * d).sum(0) / n
        rho = np.array([(d[k:] * d[:n - k]).sum(0) / n / g0 for k in j])
        rej_bp += int((n * (rho**2).sum(0) > crit).sum())
        rej_lb += int((n * (n + 2) * (rho**2 / (n - j)[:, None]).sum(0) > crit).sum())
    print(f"\nMonte Carlo Exercise 2.2 ({reps:,} replications, n = {n}, p = {p}, "
          f"chi2_0.05({p}) = {crit:.3f})")
    print(f"  Box-Pierce  rejection frequency: {rej_bp / reps:.4f}")
    print(f"  Ljung-Box   rejection frequency: {rej_lb / reps:.4f}")


if __name__ == "__main__":
    f_versus_chi2()
    mishkin()
    nerlove_ch2()
    mc_uniform()
    mc_box_pierce()
