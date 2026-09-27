"""Chapter 5 of Hayashi, Econometrics: numerical parts of the solutions.

Covers
  * Review Question 5.1.1  Q annihilates l_M
  * Review Question 5.3.1  the unbalanced-panel Q for d = (1,0,1)
  * Empirical Exercise 5.1  the speed of convergence on SUM_HES.ASC:
      (a) the Mankiw-Romer-Weil specification, one long difference 1960-1985
      (b) the fixed-effects estimator on the 25-equation system
      (c) GMM on the 24 first-differenced equations, saving rate as instrument

Run from this folder:  python ch5.py
"""

import numpy as np
from scipy import stats

import hayashilib as hl

G_PLUS_D = 0.05           # g + delta, as in Mankiw, Romer and Weil (1992)


# ------------------------------------------------------------- data input
def summerest():
    """Parse SUM_HES.ASC: 125 countries x (1 header + 26 yearly records)."""
    path = hl.DATA / "ch5" / "SUM_HES.ASC"
    text = path.read_text(encoding="latin-1").replace("\x1a", "")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    assert len(rows) == 125 * 27, f"expected 3375 rows, got {len(rows)}"
    ids, com, opec, year, pop, gdp, sav = [], [], [], [], [], [], []
    for c in range(125):
        h = rows[c * 27]
        ids.append(int(h[0])); com.append(int(h[1])); opec.append(int(h[2]))
        for m in range(26):
            y, p, g, s = rows[c * 27 + 1 + m]
            year.append(int(y)); pop.append(float(p))
            gdp.append(float(g)); sav.append(float(s))
    return dict(ids=np.array(ids), com=np.array(com), opec=np.array(opec),
                year=np.array(year), pop=np.array(pop),
                gdp=np.array(gdp), sav=np.array(sav))


def build_panel(d):
    """Reshape the long vectors into country-by-year matrices (n x T)."""
    n, T = 125, 26
    assert d["year"].size == n * T
    def mat(v):
        return np.asarray(v).reshape(n, T)     # country-major, 26 rows each
    return (mat(d["year"]), mat(d["pop"]), mat(d["gdp"]), mat(d["sav"]))


def avar_ols(y, X):
    """(b, e, robust Avar of sqrt(n)(b-beta), conventional Avar)."""
    n, K = X.shape
    XXi = np.linalg.inv(X.T @ X / n)
    b = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    return b, e, XXi @ S @ XXi, (e @ e / (n - K)) * XXi


# --------------------------------------------------------- RQ 5.1.1 check
def check_Q():
    print("Review Question 5.1.1: Q l_M = 0")
    for M in (2, 3, 5):
        Q = np.eye(M) - np.ones((M, M)) / M
        print(f"  M = {M}: max |Q 1_M| = {np.abs(Q @ np.ones(M)).max():.2e}")


# --------------------------------------------------------- RQ 5.3.1 check
def check_Q_unbalanced():
    d = np.array([1, 0, 1.0])
    M = d.size
    Mi = d.sum()
    Q = np.eye(M) - np.outer(d, d) / Mi
    print("\nReview Question 5.3.1: Q for d = (1,0,1)'")
    print(np.array2string(Q, precision=4, suppress_small=True))
    print(f"  max |Q d| = {np.abs(Q @ d).max():.2e}")


# ----------------------------------------------------------- EE 5.1 (a)
def ee_a(d, P):
    com, opec = d["com"], d["opec"]
    _, pop, gdp, _ = P
    n = 125
    y60, y85 = np.log(gdp[:, 0]), np.log(gdp[:, 25])
    growth = y85 - y60
    s_bar = d["sav"].reshape(n, 26).mean(1)
    n_i = (pop[:, 25] / pop[:, 0]) ** (1 / 25) - 1.0     # avg annual growth rate

    print("\nEmpirical Exercise 5.1(a): one long difference, 1960-1985")
    print(f"  mean of s_i = {s_bar.mean():.3f}, mean of n_i = {100*n_i.mean():.2f}%")
    print(f"  corr(growth, y_1960) = {np.corrcoef(growth, y60)[0, 1]:.3f}")

    # unconditional convergence
    X0 = np.column_stack([np.ones(n), y60])
    b0, e0, A0, _ = avar_ols(growth, X0)
    S0 = X0.T @ X0 / n
    mid0 = (X0 * e0[:, None]).T @ (X0 * e0[:, None]) / n
    A0r = np.linalg.inv(S0) @ mid0 @ np.linalg.inv(S0)
    se0 = np.sqrt(np.diag(A0r) / n)
    lam0 = -np.log(1 + b0[1]) / 25
    se_lam0 = se0[1] / (25 * (1 + b0[1]))
    print(f"  unconditional: growth = {b0[0]:.4f} + {b0[1]:.4f} y_60 "
          f"(SE {se0[1]:.4f}), R2 = {1 - e0 @ e0 / ((growth - growth.mean())**2).sum():.3f}, "
          f"lambda = {100*lam0:.3f}% (SE {100*se_lam0:.3f}%)")

    # equation (2'): growth on y_60, log s, log(n+g+d), COM, OPEC
    X = np.column_stack([np.ones(n), y60, np.log(s_bar),
                         np.log(n_i + G_PLUS_D), com, opec])
    b, e, A, Ac = avar_ols(growth, X)
    se = np.sqrt(np.diag(A) / n)
    se_c = np.sqrt(np.diag(Ac) / n)
    names = ["const", "y_1960", "log(s)", "log(n+g+d)", "COM", "OPEC"]
    print("  equation (2'), robust SEs (conventional SEs):")
    for k, nm in enumerate(names):
        print(f"    {nm:<11}{b[k]:>9.4f}  ({se[k]:.4f})  [{se_c[k]:.4f}]")
    R2 = 1 - e @ e / ((growth - growth.mean())**2).sum()
    print(f"    R2 = {R2:.3f}, SER = {np.sqrt(e @ e / (n - X.shape[1])):.4f}")

    rho_hat = 1 + b[1]
    lam = -np.log(rho_hat) / 25
    se_lam = se[1] / (25 * rho_hat)         # delta method: se(b)/(25 rho)
    print(f"  rho = 1 + b = {rho_hat:.4f}, lambda = -log(rho)/25 = {lam:.5f}"
          f" ({100*lam:.3f}% per year), SE = {se_lam:.5f} ({100*se_lam:.3f}%)")
    print(f"  95% CI for lambda: [{lam - 1.96*se_lam:.5f}, {lam + 1.96*se_lam:.5f}]")
    print(f"  implied half-life: {np.log(2)/lam:.1f} years")

    # check the identity y_85 = y_60 + growth: regressing y_85 on the same
    # right-hand side with y_60 in place of growth must return coefficient 1
    Xi = np.column_stack([np.ones(n), y60, growth, np.log(s_bar),
                          np.log(n_i + G_PLUS_D), com, opec])
    bi, ei, Ai, _ = avar_ols(y85, Xi)
    print(f"  check (identity y_85 = y_60 + growth): coefs on y_60 and growth "
          f"= {bi[1]:.6f}, {bi[2]:.6f} (both should be 1)")

    fig, ax = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(1, 2, figsize=(8.6, 3.6))
    ax[0].scatter(y60, growth, s=14, c="0.25")
    ax[0].set_xlabel(r"$\log(\mathrm{GDP}_{1960})$")
    ax[0].set_ylabel(r"growth 1960--1985, $\log(\mathrm{GDP}_{1985})-\log(\mathrm{GDP}_{1960})$")
    ax[0].set_title("unconditional")
    fit = np.polyfit(y60, growth, 1)
    xs = np.linspace(y60.min(), y60.max(), 50)
    ax[0].plot(xs, np.polyval(fit, xs), lw=1)
    # conditional: partial relation -- growth net of covariates vs y60 net of covariates
    Z = np.column_stack([np.ones(n), np.log(s_bar), np.log(n_i + G_PLUS_D), com, opec])
    ry = growth - Z @ np.linalg.solve(Z.T @ Z, Z.T @ growth)
    rx = y60 - Z @ np.linalg.solve(Z.T @ Z, Z.T @ y60)
    ax[1].scatter(rx, ry, s=14, c="0.25")
    bb = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 50)
    ax[1].plot(xs, np.polyval(bb, xs), lw=1)
    ax[1].set_xlabel(r"$y_{1960}$, net of $\log s$, $\log(n+g+\delta)$, COM, OPEC")
    ax[1].set_ylabel("growth, net of the same")
    ax[1].set_title("conditional")
    hl.save(fig, "ch5_convergence")
    return dict(b=b, se=se, lam=lam, se_lam=se_lam, R2=R2, e=e)


# ----------------------------------------------------------- EE 5.1 (b)
def ee_b(d, P):
    com, opec = d["com"], d["opec"]
    _, pop, gdp, sav = P
    n, T = 125, 26
    M = T - 1                                    # 25 equations, m = 1..25
    y = np.log(gdp)                              # n x T, columns 1960..1985
    s_bar = sav.mean(1)
    n_i = (pop[:, 25] / pop[:, 0]) ** (1 / 25) - 1.0

    # transformed data: deviations from country means, stacked (n*M, ...)
    Yd = (y[:, 1:] - y[:, 1:].mean(1, keepdims=True)).reshape(-1)
    Xl = (y[:, :-1] - y[:, :-1].mean(1, keepdims=True)).reshape(-1)
    # 24 year dummies (years 1962..1985, leaving out 1961), demeaned
    Dm = np.zeros((n * M, M - 1))
    for m in range(1, M):
        col = np.zeros((n, M)); col[:, m] = 1.0
        Dm[:, m - 1] = (col - col.mean(1, keepdims=True)).reshape(-1)
    X = np.column_stack([Dm, Xl])
    Nobs = n * M
    K = X.shape[1]

    b = np.linalg.solve(X.T @ X, X.T @ Yd)
    e = Yd - X @ b
    SSR = float(e @ e)
    s2_correct = SSR / (Nobs - n - K)            # book's (5.2.17)
    s2_wrong = SSR / (Nobs - K)                  # book's (5.2.16)
    XXi = np.linalg.inv(X.T @ X)
    se_correct = np.sqrt(s2_correct * np.diag(XXi))
    se_wrong = np.sqrt(s2_wrong * np.diag(XXi))

    rho_b = b[-1]
    lam = -np.log(rho_b) / 1.0                   # t_m - t_{m-1} = 1 year
    se_lam = se_correct[-1] / rho_b
    print("\nEmpirical Exercise 5.1(b): fixed effects, M = 25")
    print(f"  rho = {rho_b:.4f} (SE {se_correct[-1]:.4f}), "
          f"lambda = {lam:.5f} ({100*lam:.2f}% per year), SE = {se_lam:.5f}")
    print(f"  wrong df: SE(rho) = {se_wrong[-1]:.4f}, "
          f"SE ratio = {se_wrong[-1]/se_correct[-1]:.4f}")
    print(f"  SSR = {SSR:.4f}, s^2 (correct df) = {s2_correct:.6f}, "
          f"s^2 (wrong df) = {s2_wrong:.6f}")

    # robust (Proposition 5.3) standard errors
    F_mat = X.reshape(n, M, K)
    E_mat = e.reshape(n, M)
    mid = np.zeros((K, K))
    for i in range(n):
        Fi = F_mat[i]
        mid += (Fi * E_mat[i][:, None]).T @ (Fi * E_mat[i][:, None])
    Arob = np.linalg.inv(F_mat.reshape(-1, K).T @ F_mat.reshape(-1, K) / n) \
        @ (mid / n) @ np.linalg.inv(F_mat.reshape(-1, K).T @ F_mat.reshape(-1, K) / n)
    se_rob = np.sqrt(np.diag(Arob) / n)
    print(f"  robust SE(rho) = {se_rob[-1]:.4f}, "
          f"robust SE(lambda) = {se_rob[-1]/rho_b:.5f}")

    # LSDV cross-check: pooled OLS with 125 country dummies
    Ylv = y[:, 1:].reshape(-1)
    Xl_lv = y[:, :-1].reshape(-1)
    Dc = np.kron(np.eye(n), np.ones((M, 1)))
    Dy = np.tile(np.eye(M), (n, 1))[:, 1:]       # 24 year dummies
    W = np.column_stack([Dc, Dy, Xl_lv])
    b_lsdv = np.linalg.solve(W.T @ W, W.T @ Ylv)
    print(f"  LSDV check: max |b_FE - b_LSDV| = "
          f"{np.abs(b_lsdv[-1] - rho_b):.2e} (rho), SSR_LSDV = "
          f"{float(((Ylv - W @ b_lsdv)**2).sum()):.4f}")
    return dict(rho=rho_b, lam=lam, se=se_correct[-1], se_lam=se_lam,
                se_rob=se_rob[-1], SSR=SSR)


# ----------------------------------------------------------- EE 5.1 (c)
def ee_c(d, P):
    """System GMM on the 24 first-differenced equations, as in (4.6.6).

    Equation m (m = 1..24) is  dy_{m+1} = xi_m + rho dy_m + err, with
    instruments x_im = (1, s_{i,m-1}).  The moment vector g_i is the
    48-vector stacking inst_m * err_m over m; Sxz is block-diagonal.
    """
    _, _, gdp, sav = P
    n = 125
    M = 24
    y = np.log(gdp)
    s = sav
    L, K = 2, M + 1                              # 2 moments per eq; M xis + rho

    # per-country blocks: Zi (M x K), Xi (M*L,) moments
    def blocks(i):
        dy = np.diff(y[i])                       # dy[m-1] = y_m - y_{m-1}
        Zi = np.zeros((M, K))
        Zi[:, M] = dy[:M]                        # regressor y_m - y_{m-1}
        Zi[np.arange(M), np.arange(M)] = 1.0     # equation-specific intercept
        Xi = np.zeros((M, L))
        Xi[:, 1] = s[i, :M]                      # instrument s_{i,m-1}
        Xi[:, 0] = 1.0
        dep = dy[1:M + 1]                        # y_{m+1} - y_m
        return Zi, Xi, dep

    Sxz = np.zeros((M * L, K))
    sxy = np.zeros(M * L)
    store = [blocks(i) for i in range(n)]
    for Zi, Xi, dep in store:
        for m in range(M):
            r = slice(m * L, (m + 1) * L)
            Sxz[r, :] += np.outer(Xi[m], Zi[m]) / n
            sxy[r] += Xi[m] * dep[m] / n

    def solve(W):
        A = Sxz.T @ W @ Sxz
        return np.linalg.solve(A, Sxz.T @ W @ sxy), A

    def moments(delta):
        """n x (M*L) matrix of g_i = e_i (x) x_i and sample mean."""
        G = np.zeros((n, M * L))
        for i, (Zi, Xi, dep) in enumerate(store):
            e = dep - Zi @ delta
            for m in range(M):
                G[i, m * L:(m + 1) * L] = Xi[m] * e[m]
        return G

    def report(delta, tag):
        G = moments(delta)
        S = G.T @ G / n
        return S

    # first step: W = (block-diag of S_xx^m)^{-1}  (the 2SLS weighting)
    W1 = np.zeros((M * L, M * L))
    for m in range(M):
        xm = np.array([[1.0, s[i, m]] for i in range(n)])
        W1[m * L:(m + 1) * L, m * L:(m + 1) * L] = np.linalg.inv(xm.T @ xm / n)
    d1, _ = solve(W1)
    S1 = report(d1, "2SLS")

    # efficient second step
    W2 = np.linalg.inv(S1)
    d2, A2 = solve(W2)
    S2 = report(d2, "efficient GMM")
    gbar = Sxz @ d2 - sxy
    J2 = float(n * gbar @ np.linalg.solve(S2, gbar))
    dfJ = M * L - K
    avar = np.linalg.inv(A2) @ (Sxz.T @ W2 @ S2 @ W2 @ Sxz) @ np.linalg.inv(A2)
    se = np.sqrt(np.diag(avar) / n)

    rho1, rho2 = d1[M], d2[M]
    lam1, lam2 = -np.log(rho1), -np.log(rho2)
    se_lam2 = se[M] / rho2
    print("\nEmpirical Exercise 5.1(c): GMM on 24 first-differenced equations")
    print(f"  2SLS (first step): rho = {rho1:.4f}, lambda = {lam1:.4f}")
    print(f"  efficient GMM: rho = {rho2:.4f} (SE {se[M]:.4f}), "
          f"lambda = {lam2:.4f} ({100*lam2:.1f}% per year), SE = {se_lam2:.4f}")
    print(f"  Sargan/Hansen J = {J2:.2f} on chi2({dfJ}), "
          f"p = {stats.chi2.sf(J2, dfJ):.3f}")
    return dict(rho=rho2, se=se[M], lam=lam2, se_lam=se_lam2, J=J2, df=dfJ,
                rho1=rho1)


if __name__ == "__main__":
    check_Q()
    check_Q_unbalanced()
    d = summerest()
    P = build_panel(d)
    ee_a(d, P)
    ee_b(d, P)
    ee_c(d, P)
