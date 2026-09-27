"""Chapter 1 of Hayashi, Econometrics: numerical parts of the solutions.

Covers
  * Review Question 1.4.5   t and F critical values,  F_a(1,m) = [t_{a/2}(m)]^2
  * Analytical Exercise 1.4(f)(iv)  a numerical example with SSR4 != e'e
  * the Monte Carlo exercise of Chapter 1 (both simulations), and figure S1.1

Run from this folder:  python ch1.py
"""

import numpy as np
from scipy import stats

import hayashilib as hl

RNG = np.random.default_rng(20260920)


# ---------------------------------------------------------------- RQ 1.4.5
def t_vs_F():
    print("Review Question 1.4.5:  F_a(1,m) versus [t_{a/2}(m)]^2")
    print(f"{'m':>5}{'alpha':>8}{'t_{a/2}':>12}{'t^2':>12}{'F_a(1,m)':>12}")
    for m, a in ((30, 0.05), (141, 0.05), (10, 0.01)):
        t = stats.t.ppf(1 - a / 2, m)
        F = stats.f.ppf(1 - a, 1, m)
        print(f"{m:>5}{a:>8}{t:>12.5f}{t * t:>12.5f}{F:>12.5f}")


# ------------------------------------------------------- AE 1.4(f)(iv)
def frisch_waugh_example():
    """Four regressions of Analytical Exercise 1.4(f); SSR4 is the odd one out."""
    X1 = np.ones((4, 1))
    X2 = np.array([[1.0], [2.0], [3.0], [5.0]])
    y = np.array([2.0, 1.0, 4.0, 6.0])

    M1 = np.eye(4) - X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
    yt, X2t = M1 @ y, M1 @ X2

    ssr = lambda X, z: float(z @ z - z @ X @ np.linalg.solve(X.T @ X, X.T @ z))
    e_e = ssr(np.hstack([X1, X2]), y)

    print("\nAnalytical Exercise 1.4(f):  SSR of the four regressions")
    print(f"  SSR1 (y~ on X1)      = {ssr(X1, yt):.4f}   (= y~'y~)")
    print(f"  SSR2 (y~ on X2~)     = {ssr(X2t, yt):.4f}")
    print(f"  SSR3 (y~ on X1, X2)  = {ssr(np.hstack([X1, X2]), yt):.4f}")
    print(f"  SSR4 (y~ on X2)      = {ssr(X2, yt):.4f}   <- differs")
    print(f"  e'e  (y  on X1, X2)  = {e_e:.4f}")


# -------------------------------------------------------- Monte Carlo
BETA = np.array([1.0, 0.5])
SIG = 1.0
N = 32
C, PHI = 2.0, 0.6


def x_machinery(n=N, c=C, phi=PHI):
    """r, d and A of the book's expression (***) for the AR(1) regressor."""
    i = np.arange(1, n + 1)
    r = phi ** i
    d = c * (1.0 - phi ** i) / (1.0 - phi)
    A = np.tril(phi ** (i[:, None] - i[None, :]))
    return r, d, A


def draw_x(reps, rng, n=N, c=C, phi=PHI):
    """n x reps matrix of AR(1) paths started from the stationary distribution."""
    r, d, A = x_machinery(n, c, phi)
    x0 = rng.normal(c / (1 - phi), np.sqrt(1 / (1 - phi**2)), size=reps)
    eta = rng.normal(size=(n, reps))
    return np.outer(r, x0) + d[:, None] + A @ eta


def simple_regression(x, y):
    """Vectorised OLS of y on (1, x), column by column.  Returns b1, b2, t(b2=.5)."""
    n = x.shape[0]
    xbar, ybar = x.mean(0), y.mean(0)
    xd, yd = x - xbar, y - ybar
    Sxx = (xd * xd).sum(0)
    b2 = (xd * yd).sum(0) / Sxx
    b1 = ybar - b2 * xbar
    ssr = (yd * yd).sum(0) - b2**2 * Sxx
    s2 = ssr / (n - 2)
    t = (b2 - BETA[1]) / np.sqrt(s2 / Sxx)
    return b1, b2, t


def monte_carlo(reps=1_000_000, batch=50_000, seed=20260920):
    rng = np.random.default_rng(seed)
    tcrit = stats.t.ppf(0.975, N - 2)

    # --- simulation 1: x drawn once, so this is the distribution given x
    x_fixed = draw_x(1, rng)[:, 0]
    bsum = np.zeros(2)
    rej1 = 0
    t_pool = []
    done = 0
    while done < reps:
        m = min(batch, reps - done)
        y = BETA[0] + BETA[1] * x_fixed[:, None] + SIG * rng.normal(size=(N, m))
        b1, b2, t = simple_regression(np.repeat(x_fixed[:, None], m, 1), y)
        bsum += np.array([b1.sum(), b2.sum()])
        rej1 += int((np.abs(t) > tcrit).sum())
        if len(t_pool) < 4:
            t_pool.append(t)
        done += m
    bmean = bsum / reps

    # --- simulation 2: a fresh x in every replication (unconditional)
    rej2 = 0
    t_uncond = []
    done = 0
    while done < reps:
        m = min(batch, reps - done)
        x = draw_x(m, rng)
        y = BETA[0] + BETA[1] * x + SIG * rng.normal(size=(N, m))
        _, _, t = simple_regression(x, y)
        rej2 += int((np.abs(t) > tcrit).sum())
        if len(t_uncond) < 4:
            t_uncond.append(t)
        done += m

    print(f"\nMonte Carlo exercise ({reps:,} replications, n = {N}, "
          f"t_0.025({N - 2}) = {tcrit:.4f})")
    print(f"  simulation 1: E(b | x) estimated as ({bmean[0]:.5f}, {bmean[1]:.5f})"
          f"   [true (1, 0.5)]")
    print(f"  simulation 1: frequency of |t| > t_0.025 = {rej1 / reps:.5f}")
    print(f"  simulation 2: frequency of |t| > t_0.025 = {rej2 / reps:.5f}")
    se = np.sqrt(0.05 * 0.95 / reps)
    print(f"  (a rejection frequency has standard error {se:.5f} at the true 0.05)")
    return np.concatenate(t_pool), np.concatenate(t_uncond)


def figure_t(t_cond, t_uncond):
    plt = hl.plt
    grid = np.linspace(-4.5, 4.5, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(t_cond, bins=120, range=(-4.5, 4.5), density=True, histtype="step",
            label="simulation 1 (x fixed)")
    ax.hist(t_uncond, bins=120, range=(-4.5, 4.5), density=True, histtype="step",
            label="simulation 2 (x redrawn)")
    ax.plot(grid, stats.t.pdf(grid, N - 2), "k--", lw=1.2, label=f"$t({N - 2})$ density")
    ax.set_xlabel("$t$-ratio for $H_0:\\beta_2 = 0.5$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    hl.save(fig, "ch1_t_distribution")


# ------------------------------------------------------------ Nerlove
def nerlove():
    """Empirical Exercise 1.1: Nerlove's (1963) 145 electric utilities.

    NERLOVE.ASC columns: TC, Q, PL, PF, PK.  In the notation of Section 1.7,
    p1 = PL (labour), p2 = PK (capital), p3 = PF (fuel).
    """
    from scipy import stats

    a = hl.load_asc("ch1", "NERLOVE.ASC")
    TC, Q, PL, PF, PK = (a[:, j] for j in range(5))
    n = TC.size
    one = np.ones(n)
    lQ, lTC = np.log(Q), np.log(TC)

    print(f"\nEmpirical Exercise 1.1 (Nerlove, n = {n})")

    # (b) unrestricted model (1.7.4)
    Xu = np.column_stack([one, lQ, np.log(PL), np.log(PK), np.log(PF)])
    u = hl.OLS(lTC, Xu)
    print(u.summary(["const", "log Q", "log p1 (PL)", "log p2 (PK)", "log p3 (PF)"],
                    "(b) unrestricted (1.7.4): dependent variable log(TC)"))
    print(f"  mean of dep. var.     {lTC.mean():>12.4f}")

    # (c) restricted model (1.7.6)
    y_r = np.log(TC / PF)
    Xr = np.column_stack([one, lQ, np.log(PL / PF), np.log(PK / PF)])
    r = hl.OLS(y_r, Xr)
    print(r.summary(["const", "log Q", "log(p1/p3)", "log(p2/p3)"],
                    "(c) restricted (1.7.6): dependent variable log(TC/PF)"))
    print(f"  mean of dep. var.     {y_r.mean():>12.4f}")
    b5 = 1 - r.b[2] - r.b[3]
    c = np.array([0.0, 0.0, -1.0, -1.0])
    print(f"  implied b5 = 1-b3-b4  {b5:>12.4f} ({np.sqrt(c @ r.vcv @ c):.4f})"
          "   [Review Question 1.7.4]")
    print(f"  returns to scale 1/b2 {1 / r.b[1]:>12.4f}")
    # homogeneity test, computed from the two SSRs as in the text
    F_hom = (r.ssr - u.ssr) / 1 / (u.ssr / (n - 5))
    print(f"  F for homogeneity     {F_hom:>12.4f}   p = {stats.f.sf(F_hom, 1, n - 5):.4f}")

    # (d) Model 1: five groups of 29, separate regressions
    g = np.repeat(np.arange(5), 29)
    ssr1, rts, sig2 = [], [], []
    for j in range(5):
        m = g == j
        fit = hl.OLS(y_r[m], Xr[m])
        ssr1.append(fit.ssr)
        rts.append(1 / fit.b[1])
        sig2.append(fit.s2)
    print("\n(d) Model 1, group by group (29 firms each):")
    print("     group   b2       returns to scale 1/b2    s^2       SSR_j")
    for j in range(5):
        print(f"     {j + 1:>5}  {1 / rts[j]:>7.4f}  {rts[j]:>16.3f}  {sig2[j]:>12.5f}"
              f"  {ssr1[j]:>9.4f}")

    # (e) Model 2: the block-diagonal stacking
    X2 = np.zeros((n, 20))
    for j in range(5):
        X2[g == j, 4 * j:4 * (j + 1)] = Xr[g == j]
    m2 = hl.OLS(y_r, X2)
    same = max(abs(m2.b[4 * j + k] - hl.OLS(y_r[g == j], Xr[g == j]).b[k])
               for j in range(5) for k in range(4))
    print(f"\n(e) Model 2: SSR = {m2.ssr:.6f},  sum_j SSR_j = {sum(ssr1):.6f},"
          f"  max |coefficient difference| = {same:.2e}")

    # (f) Chow test: Model 2 against the common-coefficient model (1.7.6)
    q = 16
    F_chow = (r.ssr - m2.ssr) / q / (m2.ssr / (n - 20))
    print(f"(f) Chow test: F({q},{n - 20}) = {F_chow:.4f},"
          f"  p = {stats.f.sf(F_chow, q, n - 20):.3e}")

    # (g) Model 3: common price elasticities, group-specific intercept and log Q
    X3 = np.zeros((n, 12))
    for j in range(5):
        X3[g == j, 2 * j] = 1.0
        X3[g == j, 2 * j + 1] = lQ[g == j]
    X3[:, 10] = np.log(PL / PF)
    X3[:, 11] = np.log(PK / PF)
    m3 = hl.OLS(y_r, X3)
    F_3 = (m3.ssr - m2.ssr) / 8 / (m2.ssr / (n - 20))
    print(f"(g) Model 3: SSR = {m3.ssr:.4f};  F(8,{n - 20}) = {F_3:.4f},"
          f"  p = {stats.f.sf(F_3, 8, n - 20):.4f}")
    print("     returns to scale by group:",
          "  ".join(f"{1 / m3.b[2 * j + 1]:.3f}" for j in range(5)))

    # (h) Model 4: WLS with the stated variance function
    X4 = np.column_stack([one, lQ, lQ**2, np.log(PL / PF), np.log(PK / PF)])
    w = 1.0 / np.sqrt(0.0565 + 2.1377 / Q)
    m4 = hl.OLS(y_r * w, X4 * w[:, None])
    print(m4.summary(["const", "log Q", "(log Q)^2", "log(p1/p3)", "log(p2/p3)"],
                     "(h) Model 4 by weighted least squares"))
    dlogc = m4.b[1] + 2 * m4.b[2] * lQ          # elasticity of cost w.r.t. output
    print("     returns to scale at the group means:",
          "  ".join(f"{1 / dlogc[g == j].mean():.3f}" for j in range(5)))

    figure_nerlove(lQ, r.e, m4.e, g)
    return r, m4


def figure_nerlove(lQ, e_restricted, e_wls, g):
    plt = hl.plt
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharex=True)
    for ax, e, ttl in ((axes[0], e_restricted, "restricted model (1.7.6)"),
                       (axes[1], e_wls, "Model 4, weighted residuals")):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(lQ, e, "o", ms=3, mfc="none")
        ax.set_title(ttl)
        ax.set_xlabel("$\\log(Q_i)$")
    axes[0].set_ylabel("residual")
    hl.save(fig, "ch1_nerlove_residuals")


if __name__ == "__main__":
    t_vs_F()
    frisch_waugh_example()
    nerlove()
    tc, tu = monte_carlo()
    figure_t(tc, tu)
