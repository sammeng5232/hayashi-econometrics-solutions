"""Chapter 4 of Hayashi, Econometrics: numerical parts of the solutions.

Covers the Empirical Exercise of Chapter 4 (Christensen and Greene's 1970 data
on 99 U.S. electric utilities, GREENE.ASC):

  (a)  replication of the simple statistics in Table 4.3
  (b)  random-effects estimation of the constrained two-equation share system
       (capital-share equation dropped, prices normalized by p_K), with
       Sigma-hat from equation-by-equation OLS; recovery of the remaining
       eight share-equation parameters by the adding-up, homogeneity and
       symmetry restrictions, with delta-method standard errors; numerical
       invariance check against dropping the fuel-share equation instead
  (c)  Sargan's statistic for the constrained system (target: 0.63313,
       p-value 0.42621)
  (d)  chi-squared test of symmetry from the difference of Sargan statistics
  (e)  Wald test of symmetry in the unconstrained system (numerically equal
       to Sargan's statistic for the constrained system)
  (f)  substitution elasticities (4.7.9) averaged over the 99 firms, the
       concavity count and the monotonicity check in the "relevant range"

Figures: figures/ch4_fitted_shares.pdf and figures/ch4_substitution.pdf.

Run from this folder:  python ch4.py
"""

import numpy as np
from scipy import stats

import hayashilib as hl

GAMMA_NAMES = [
    r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$",
    r"$\gamma_{11}$", r"$\gamma_{12}$", r"$\gamma_{13}$",
    r"$\gamma_{21}$", r"$\gamma_{22}$", r"$\gamma_{23}$",
    r"$\gamma_{31}$", r"$\gamma_{32}$", r"$\gamma_{33}$",
    r"$\gamma_{1Q}$", r"$\gamma_{2Q}$", r"$\gamma_{3Q}$",
]


def load():
    """GREENE.ASC: id, TC ($m), Q (mkWh), pL, pK, pF, s1 (labor), s2 (capital)."""
    a = hl.load_asc("ch4", "GREENE.ASC")
    fid, tc, q, pl, pk, pf, s1, s2 = (a[:, j] for j in range(8))
    s3 = 1.0 - s1 - s2
    return fid, tc, q, pl, pk, pf, s1, s2, s3


# ------------------------------------------------------------------ part (a)
def simple_statistics():
    fid, tc, q, pl, pk, pf, s1, s2, s3 = load()
    print("\nEmpirical Exercise (a): Table 4.3 simple statistics (n = 99)")
    print(f"{'':>12}{'mean':>10}{'std.dev.':>12}")
    print(f"{'output':>12}{q.mean() / 1000:>10.2f}{q.std(ddof=1) / 1000:>12.2f}"
          "   (10^9 kWh; data are in millions of kWh)")
    for nm, s in (("labor share", s1), ("capital share", s2), ("fuel share", s3)):
        print(f"{nm:>12}{s.mean():>10.3f}{s.std(ddof=1):>12.3f}")
    print(f"{'cost ($m)':>12}{tc.mean():>10.3f}{tc.std(ddof=1):>12.3f}")


# --------------------------------------------------------------- the systems
def design(pl, pk, pf, q, norm):
    """Regressors of the share system with prices normalized by `norm`.

    norm == 2:  x1 = log(pL/pK), x3 = log(pF/pK)   (capital equation dropped)
    norm == 3:  x1 = log(pL/pF), x2 = log(pK/pF)   (fuel equation dropped)
    Returns the n x 4 matrix [1, xa, xb, log Q].
    """
    base = {2: pk, 3: pf}[norm]
    xa = np.log(pl / base)
    xb = np.log({2: pf, 3: pk}[norm] / base)
    return np.column_stack([np.ones(q.size), xa, xb, np.log(q)])


def system_matrices(X, kept):
    """Zi (M x L) and yi (M x 1) of the common-coefficient format.

    kept = (1, 3): constrained system with capital equation dropped;
           free coefficients d = (a1, g11, g13, g1Q | a3, g33, g3Q)
           and the symmetry restriction g13 = g31 is built in.
    kept = (1, 2): text's system (fuel dropped);
           free coefficients d = (a1, g11, g12, g1Q | a2, g22, g2Q)
           with symmetry g12 = g21 built in.
    """
    n = X.shape[0]
    Zi = np.zeros((n, 2, 7))
    yi = np.zeros((n, 2))
    if kept == (1, 3):
        # eq 1: s1 = a1 + g11 x1 + g13 x3 + g1Q q
        # eq 3: s3 = a3 + g31 x1 + g33 x3 + g3Q q, g31 = g13
        Zi[:, 0, 0:4] = X                       # a1, g11, g13, g1Q
        Zi[:, 1, 4] = X[:, 0]                   # a3
        Zi[:, 1, 2] = X[:, 1]                   # g13 (= g31) multiplies x1
        Zi[:, 1, 5] = X[:, 2]                   # g33
        Zi[:, 1, 6] = X[:, 3]                   # g3Q
    else:
        # eq 1: s1 = a1 + g11 x1 + g12 x2 + g1Q q
        # eq 2: s2 = a2 + g21 x1 + g22 x2 + g2Q q, g21 = g12
        Zi[:, 0, 0:4] = X                       # a1, g11, g12, g1Q
        Zi[:, 1, 4] = X[:, 0]                   # a2
        Zi[:, 1, 2] = X[:, 1]                   # g12 (= g21) multiplies x1
        Zi[:, 1, 5] = X[:, 2]                   # g22
        Zi[:, 1, 6] = X[:, 3]                   # g2Q
    return Zi, yi


def ols_by_equation(y, X):
    """Equation-by-equation OLS of each column of y (n x M) on X."""
    return [hl.OLS(y[:, m], X) for m in range(y.shape[1])]


def sigma_hat(resids):
    """Sigma-hat of Proposition 4.1: sigma_mh = (1/n) sum_i e_im e_ih."""
    return resids.T @ resids / resids.shape[0]


def random_effects(Zi, yi, Sig):
    """RE estimator (4.6.8'), its estimated Avar (4.6.10') and s.e.'s."""
    n = Zi.shape[0]
    Si = np.linalg.inv(Sig)
    A = np.einsum("iml,mh,ihk->lk", Zi, Si, Zi)          # sum_i Zi' Sig^-1 Zi
    B = np.einsum("iml,mh,ih->l", Zi, Si, yi)            # sum_i Zi' Sig^-1 yi
    d = np.linalg.solve(A, B)
    avar = np.linalg.inv(A / n)                           # estimated Avar
    se = np.sqrt(np.diag(avar) / n)
    return d, avar, se


# ------------------------------------------------------- the 15 parameters
def recover_all(d, norm):
    """Map the 7 free RE coefficients onto the 15 share-equation parameters.

    Ordering of the returned 15-vector:
      a1, a2, a3, g11, g12, g13, g21, g22, g23, g31, g32, g33, g1Q, g2Q, g3Q.
    Also returns the 15 x 7 Jacobian D = d(params)/d(d') for the delta method.
    """
    D = np.zeros((15, 7))
    if norm == 2:
        # free: d = (a1, g11, g13, g1Q, a3, g33, g3Q)
        a1, g11, g13, g1Q, a3, g33, g3Q = d
        a2 = 1.0 - a1 - a3
        g12 = -g11 - g13
        g21 = g12
        g31 = g13
        g32 = -g31 - g33
        g23 = g32
        g22 = -g21 - g23          # = g11 + 2 g13 + g33
        g2Q = -g1Q - g3Q
        # Jacobian rows in the ordering above
        D[0, 0] = 1                                   # a1
        D[1, [0, 4]] = -1                             # a2 = 1 - a1 - a3
        D[2, 4] = 1                                   # a3
        D[3, 1] = 1                                   # g11
        D[4, [1, 2]] = -1                             # g12 = -g11 - g13
        D[5, 2] = 1                                   # g13
        D[6, [1, 2]] = -1                             # g21 = g12
        D[7, [1, 2, 5]] = (1, 2, 1)                   # g22 = g11+2g13+g33
        D[8, [2, 5]] = -1                             # g23 = -g13 - g33
        D[9, 2] = 1                                   # g31 = g13
        D[10, [2, 5]] = -1                            # g32 = -g13 - g33
        D[11, 5] = 1                                  # g33
        D[12, 3] = 1                                  # g1Q
        D[13, [3, 6]] = -1                            # g2Q = -g1Q - g3Q
        D[14, 6] = 1                                  # g3Q
        return (np.array([a1, a2, a3, g11, g12, g13, g21, g22, g23,
                          g31, g32, g33, g1Q, g2Q, g3Q]), D)
    else:
        # free: d = (a1, g11, g12, g1Q, a2, g22, g2Q)  (text's normalization)
        a1, g11, g12, g1Q, a2, g22, g2Q = d
        a3 = 1.0 - a1 - a2
        g13 = -g11 - g12
        g21 = g12
        g31 = g13
        g23 = -g21 - g22
        g32 = g23
        g33 = -g31 - g32          # = g11 + 2 g12 + g22
        g3Q = -g1Q - g2Q
        D[0, 0] = 1
        D[1, 4] = 1                                   # a2
        D[2, [0, 4]] = -1                             # a3 = 1 - a1 - a2
        D[3, 1] = 1
        D[4, 2] = 1                                   # g12
        D[5, [1, 2]] = -1                             # g13 = -g11 - g12
        D[6, 2] = 1                                   # g21 = g12
        D[7, 5] = 1                                   # g22
        D[8, [2, 5]] = -1                             # g23 = -g12 - g22
        D[9, [1, 2]] = -1                             # g31 = g13
        D[10, [2, 5]] = -1                            # g32 = g23
        D[11, [1, 2, 5]] = (1, 2, 1)                  # g33 = g11+2g12+g22
        D[12, 3] = 1
        D[13, 6] = 1                                  # g2Q
        D[14, [3, 6]] = -1                            # g3Q = -g1Q - g2Q
        return (np.array([a1, a2, a3, g11, g12, g13, g21, g22, g23,
                          g31, g32, g33, g1Q, g2Q, g3Q]), D)


# ------------------------------------------------------------------ part (b)
def constrained_estimation():
    fid, tc, q, pl, pk, pf, s1, s2, s3 = load()

    print("\nEmpirical Exercise (b): random-effects estimates")
    # --- unconstrained system, capital equation dropped ---------------------
    X = design(pl, pk, pf, q, norm=2)
    yu = np.column_stack([s1, s3])
    ols_u = ols_by_equation(yu, X)
    Sig = sigma_hat(np.column_stack([o.e for o in ols_u]))
    print("  Sigma-hat* from equation-by-equation OLS (unconstrained system):")
    print(f"    [[{Sig[0, 0]: .6f}, {Sig[0, 1]: .6f}],")
    print(f"     [{Sig[1, 0]: .6f}, {Sig[1, 1]: .6f}]]")

    # --- constrained RE, capital equation dropped ---------------------------
    Zi, _ = system_matrices(X, kept=(1, 3))
    yi = yu
    d, avar, se = random_effects(Zi, yi, Sig)
    labs13 = ["a1", "g11", "g13=g31", "g1Q", "a3", "g33", "g3Q"]
    print("  constrained RE estimates (capital share equation dropped):")
    for lb, di, se_i in zip(labs13, d, se):
        print(f"    {lb:>8}{di:>12.6f}   ({se_i:.6f})   t = {di / se_i:8.2f}")

    theta, D = recover_all(d, norm=2)
    V = D @ (avar / 99.0) @ D.T
    se_all = np.sqrt(np.diag(V))
    print("\n  all fifteen share-equation parameters:")
    for nm, th, se_i in zip(GAMMA_NAMES, theta, se_all):
        print(f"    {nm:>14}{th:>12.6f}   ({se_i:.6f})   t = {th / se_i:8.2f}")

    # --- numerical invariance: drop the fuel equation instead ---------------
    X3 = design(pl, pk, pf, q, norm=3)
    yu3 = np.column_stack([s1, s2])
    ols_u3 = ols_by_equation(yu3, X3)
    Sig3 = sigma_hat(np.column_stack([o.e for o in ols_u3]))
    Zi3, _ = system_matrices(X3, kept=(1, 2))
    d3, avar3, se3 = random_effects(Zi3, yu3, Sig3)
    theta3, _ = recover_all(d3, norm=3)
    print("\n  numerical invariance check (fuel share equation dropped):")
    print(f"    max |theta - theta3| over the 15 parameters = "
          f"{np.abs(theta - theta3).max():.3e}")

    # --- Sigma-hat from the full three-equation system (Table 4.4 note) -----
    ols3 = ols_by_equation(np.column_stack([s1, s2, s3]), X3)
    Sig_full = sigma_hat(np.column_stack([o.e for o in ols3]))
    print("\n  3 x 3 Sigma-hat from equation-by-equation OLS of (4.7.15):")
    for row in Sig_full:
        print("    " + "".join(f"{v: .6f}" for v in row))
    print(f"    row sums: {Sig_full.sum(1)}  (singular, as it must be)")

    # --- pooled OLS Sigma-hat, for comparison -------------------------------
    Zp = Zi3.reshape(-1, 7)
    yp = yu3.reshape(-1)
    pooled = hl.OLS(yp, Zp)
    e_pool = pooled.e.reshape(99, 2)
    Sig_pool = sigma_hat(e_pool)
    print("\n  Sigma-hat* from pooled-OLS residuals (common-coefficient format):")
    print(f"    [[{Sig_pool[0, 0]: .6f}, {Sig_pool[0, 1]: .6f}],")
    print(f"     [{Sig_pool[1, 0]: .6f}, {Sig_pool[1, 1]: .6f}]]")

    return dict(X=X, d=d, avar=avar, se=se, theta=theta, se_all=se_all,
                theta3=theta3, Sig=Sig, ols_u=ols_u, q=q,
                s1=s1, s2=s2, s3=s3)


# ------------------------------------------------------------------ part (c)
def sargan(res):
    X, d, Sig = res["X"], res["d"], res["Sig"]
    n = X.shape[0]
    Zi, _ = system_matrices(X, kept=(1, 3))
    yi = np.column_stack([res["s1"], res["s3"]])
    # gn(d) = (1/n) sum_i yi (x) xi  -  (1/n) sum_i (Zi (x) xi) d
    sxy = np.einsum("im,ik->mk", yi, X).reshape(-1) / n
    Sxz = np.einsum("iml,ik->mkl", Zi, X).reshape(8, 7) / n
    g = sxy - Sxz @ d
    S_hat = np.kron(Sig, X.T @ X / n)
    J = float(n * g @ np.linalg.solve(S_hat, g))
    p = float(stats.chi2.sf(J, 1))
    print("\nEmpirical Exercise (c): Sargan's statistic for the constrained system")
    print(f"  J = {J:.5f}   (target 0.63313)")
    print(f"  p-value = {p:.5f}   (target 0.42621),  d.f. = 1")
    res["sargan"] = (J, p)
    return J, p


# --------------------------------------------------------------- parts (d,e)
def tests_of_symmetry(res):
    ols_u, Sig = res["ols_u"], res["Sig"]
    X = res["X"]
    n = X.shape[0]
    # (d) the unconstrained system is just identified: Sargan = 0
    print("\nEmpirical Exercise (d): chi-squared (C-type) test of symmetry")
    J_c, p_c = res["sargan"]
    print(f"  Sargan (unconstrained, just identified) = 0;  "
          f"C = {J_c:.5f} ~ chi2(1), p = {p_c:.5f}")

    # (e) Wald test of g13 = g31 in the unconstrained (multivariate) system
    # multivariate regression: Avar(d-hat) = Sigma (x) [E(xx')]^{-1}
    U = np.linalg.inv(X.T @ X / n)
    g13, g31 = ols_u[0].b[2], ols_u[1].b[1]
    var = (Sig[0, 0] * U[2, 2] - 2 * Sig[0, 1] * U[2, 1]
           + Sig[1, 1] * U[1, 1]) / n
    W = float((g13 - g31) ** 2 / var)
    p_W = float(stats.chi2.sf(W, 1))
    print("\nEmpirical Exercise (e): Wald test of symmetry in the unconstrained system")
    print(f"  g13 = {g13:.6f}, g31 = {g31:.6f}")
    print(f"  W = {W:.5f}   p-value = {p_W:.5f}")
    print(f"  |W - Sargan(constrained)| = {abs(W - res['sargan'][0]):.2e}")
    res["wald"] = (W, p_W)


# ------------------------------------------------------------------ part (f)
def substitution_elasticities(res):
    theta, X = res["theta"], res["X"]
    (a1, a2, a3, g11, g12, g13, g21, g22, g23,
     g31, g32, g33, g1Q, g2Q, g3Q) = theta
    G = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    n = X.shape[0]
    # fitted shares from the constrained system (adding up holds by construction)
    shat = np.empty((n, 3))
    shat[:, 0] = a1 + g11 * X[:, 1] + g12 * (-X[:, 1] - X[:, 2]) * 0  # placeholder
    # rebuild fitted shares directly from the p2-normalized regressors:
    x1, x3, lq = X[:, 1], X[:, 2], X[:, 3]
    shat[:, 0] = a1 + g11 * x1 + g13 * x3 + g1Q * lq
    shat[:, 2] = a3 + g31 * x1 + g33 * x3 + g3Q * lq
    shat[:, 1] = 1.0 - shat[:, 0] - shat[:, 2]

    def eta_matrix(s):
        s1_, s2_, s3_ = s
        ss = np.array([s1_, s2_, s3_])
        Eta = np.empty((3, 3))
        for j in range(3):
            for k in range(3):
                if j == k:
                    Eta[j, j] = (G[j, j] + ss[j] ** 2 - ss[j]) / ss[j] ** 2
                else:
                    Eta[j, k] = (G[j, k] + ss[j] * ss[k]) / (ss[j] * ss[k])
        return Eta

    etas = np.array([eta_matrix(shat[i]) for i in range(n)])
    print("\nEmpirical Exercise (f): substitution elasticities (fitted shares)")
    for lab, (j, k) in (("labor-capital", (0, 1)),
                        ("capital-fuel", (1, 2)),
                        ("labor-fuel", (0, 2))):
        print(f"  eta_{lab}:  mean over 99 firms = {etas[:, j, k].mean(): .4f}")
    # concavity in the relevant range
    bad = sum(1 for i in range(n) if np.linalg.eigvalsh(etas[i]).max() > 1e-12)
    print(f"  concavity violated (eta-matrix not n.s.d.) for {bad} of {n} firms")
    # monotonicity in the relevant range
    print(f"  monotonicity: min fitted share over firms and inputs = "
          f"{shat.min():.5f}  (>= 0 for all firms: {bool((shat >= 0).all())})")
    # concavity of the gamma matrix itself
    print(f"  eigenvalues of (gamma_jk): {np.linalg.eigvalsh(G).round(5)}"
          "  (positive diagonal entries violate concavity)")
    res.update(shat=shat, etas=etas, G=G)


# -------------------------------------------------------------------- figures
def figures(res):
    import matplotlib.pyplot as plt
    shat, etas = res["shat"], res["etas"]
    s1, s2, s3 = res["s1"], res["s2"], res["s3"]

    fig, ax = plt.subplots(1, 3, figsize=(9, 2.9))
    for a, actual, fitted, lab in zip(ax, (s1, s2, s3), shat.T,
                                      ("labor", "capital", "fuel")):
        a.scatter(actual, fitted, s=10, c="0.25")
        lo = min(actual.min(), fitted.min())
        hi = max(actual.max(), fitted.max())
        a.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        a.set_xlabel(f"actual {lab} share")
        a.set_ylabel("fitted share")
        a.set_title(f"{lab} share")
    hl.save(fig, "ch4_fitted_shares")

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(np.sort(etas[:, 0, 1]), np.arange(1, 100) / 99, "k-", lw=1.2)
    ax.axvline(1.0, color="0.45", lw=0.9, ls="--")
    ax.set_xlabel(r"substitution elasticity $\eta_{12}$ (labor--capital)")
    ax.set_ylabel("cumulative share of firms")
    ax.text(1.02, 0.06, "Cobb--Douglas value ($\\eta=1$)", fontsize=8)
    hl.save(fig, "ch4_substitution")


if __name__ == "__main__":
    simple_statistics()
    res = constrained_estimation()
    sargan(res)
    tests_of_symmetry(res)
    substitution_elasticities(res)
    figures(res)
