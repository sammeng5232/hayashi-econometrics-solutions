"""Chapter 3 of Hayashi, Econometrics: the empirical exercise (Griliches's wage equation).

GRILIC.ASC columns (758 individuals from the Blackburn-Neumark extract of the NLS-Y):
    RNS RNS80 MRT MRT80 SMSA SMSA80 MED IQ KWW YEAR AGE AGE80 S S80
    EXPR EXPR80 TENURE TENURE80 LW LW80

Run from this folder:  python ch3.py
"""

import numpy as np
from scipy import stats

import hayashilib as hl

COLS = ("RNS RNS80 MRT MRT80 SMSA SMSA80 MED IQ KWW YEAR AGE AGE80 S S80 "
        "EXPR EXPR80 TENURE TENURE80 LW LW80").split()


def load():
    a = hl.load_asc("ch3", "GRILIC.ASC")
    d = {name: a[:, j] for j, name in enumerate(COLS)}
    d["n"] = a.shape[0]
    return d


def year_dummies(year):
    """Eight dummies for YEAR = 66, ..., 73 (there is no 1972)."""
    years = [66, 67, 68, 69, 70, 71, 73]
    D = np.column_stack([(year == y).astype(float) for y in years])
    assert D.sum() == year.size, "year dummies do not span the sample"
    return D, [f"Y{y}" for y in years]


def main():
    d = load()
    n = d["n"]
    D, dnames = year_dummies(d["YEAR"])

    print(f"Empirical Exercise 3.1 (GRILIC.ASC, n = {n})")

    # ---- (a) descriptive statistics
    print("\n(a) means and standard deviations")
    for nm in COLS:
        v = d[nm]
        print(f"    {nm:<8}{v.mean():>10.3f}{v.std(ddof=1):>10.3f}")
    print(f"    correlation between IQ and S: {np.corrcoef(d['IQ'], d['S'])[0, 1]:.4f}")

    # ---- variables of the wage equation (no constant: the year dummies span it)
    h = np.column_stack([d["EXPR"], d["TENURE"], d["RNS"], d["SMSA"], D])
    hnames = ["EXPR", "TENURE", "RNS", "SMSA"] + dnames
    y = d["LW"]
    Z_noIQ = np.column_stack([d["S"], h])
    Z = np.column_stack([d["S"], d["IQ"], h])
    znames = ["S", "IQ"] + hnames
    excluded = np.column_stack([d["MED"], d["KWW"], d["MRT"], d["AGE"]])

    # ---- (b) three estimates of the schooling coefficient
    ols1 = hl.OLS(y, Z_noIQ)
    ols2 = hl.OLS(y, Z)
    X_b = np.column_stack([d["S"], h, excluded])        # S predetermined
    tsls_b = hl.GMM(y, Z, X_b)
    print("\n(b) Table 3.3")
    print(f"    line 1  OLS            S {ols1.b[0]:.4f} ({ols1.se[0]:.4f})"
          f"  EXPR {ols1.b[1]:.3f}  TENURE {ols1.b[2]:.3f}"
          f"  SEE {ols1.ser:.3f}  R2 {ols1.R2:.3f}")
    print(f"    line 2  OLS with IQ    S {ols2.b[0]:.4f} ({ols2.se[0]:.4f})"
          f"  IQ {ols2.b[1]:.4f} ({ols2.se[1]:.4f})  EXPR {ols2.b[2]:.3f}"
          f"  TENURE {ols2.b[3]:.3f}  SEE {ols2.ser:.3f}  R2 {ols2.R2:.3f}")
    print(f"    line 3  2SLS (IQ endog) S {tsls_b.d[0]:.4f} ({tsls_b.se_h[0]:.4f})"
          f"  IQ {tsls_b.d[1]:.4f} ({tsls_b.se_h[1]:.4f})  EXPR {tsls_b.d[2]:.3f}"
          f"  TENURE {tsls_b.d[3]:.3f}  SEE {tsls_b.ser:.3f}")

    # ---- (c) Sargan's statistic
    stat, df, p = tsls_b.sargan()
    print(f"\n(c) Sargan = {stat:.3f} on chi2({df}), p = {p:.6f}")

    # ---- (d) 2SLS by two explicit regressions
    fitted = np.column_stack([hl.OLS(Z[:, k], X_b).X @ hl.OLS(Z[:, k], X_b).b
                              for k in range(Z.shape[1])])
    second = hl.OLS(y, fitted)
    print("\n(d) two explicit stages: coefficients agree to "
          f"{np.max(np.abs(second.b - tsls_b.d)):.2e}; the second-stage SE of S is "
          f"{second.se[0]:.4f} against the correct {tsls_b.se_h[0]:.4f}")

    # ---- (e) schooling endogenous too
    X_e = np.column_stack([h, excluded])
    tsls_e = hl.GMM(y, Z, X_e)
    stat_e, df_e, p_e = tsls_e.sargan()
    print(f"\n(e) 2SLS with S and IQ endogenous: S {tsls_e.d[0]:.4f} "
          f"({tsls_e.se_h[0]:.4f})  IQ {tsls_e.d[1]:.4f} ({tsls_e.se_h[1]:.4f})"
          f"  EXPR {tsls_e.d[2]:.3f}  TENURE {tsls_e.d[3]:.3f}  SEE {tsls_e.ser:.3f}")
    print(f"    Sargan = {stat_e:.3f} on chi2({df_e}), p = {p_e:.5f}")

    # ---- (f) GMM and the C statistic for the predeterminedness of schooling
    Sh = (X_b * tsls_b.e[:, None]).T @ (X_b * tsls_b.e[:, None]) / n   # from (b)'s residuals
    gmm_full = hl.GMM(y, Z, X_b, W=np.linalg.inv(Sh))
    keep = [k for k in range(X_b.shape[1]) if k != 0]                  # drop S as instrument
    S11 = Sh[np.ix_(keep, keep)]
    gmm_red = hl.GMM(y, Z, X_b[:, keep], W=np.linalg.inv(S11))
    C = gmm_full.J - gmm_red.J
    print(f"\n(f) efficient GMM, S predetermined: S {gmm_full.d[0]:.4f} "
          f"({gmm_full.se[0]:.4f})  IQ {gmm_full.d[1]:.4f} ({gmm_full.se[1]:.4f})")
    print(f"    J = {gmm_full.J:.3f} (df {gmm_full.K - gmm_full.L}),  "
          f"J1 = {gmm_red.J:.3f} (df {gmm_red.K - gmm_red.L}),  "
          f"C = {C:.3f} on chi2(1), p = {stats.chi2.sf(C, 1):.2e}")
    print("    2SLS versus GMM standard errors (S, IQ):"
          f"  2SLS {tsls_b.se_h[0]:.4f}, {tsls_b.se_h[1]:.4f}"
          f"   GMM {gmm_full.se[0]:.4f}, {gmm_full.se[1]:.4f}")
    # efficient GMM with S endogenous too (line 5 of Table 3.3)
    Se = (X_e * tsls_e.e[:, None]).T @ (X_e * tsls_e.e[:, None]) / n
    gmm_e = hl.GMM(y, Z, X_e, W=np.linalg.inv(Se))
    print(f"    line 5 (GMM, S and IQ endogenous): S {gmm_e.d[0]:.4f} ({gmm_e.se[0]:.4f})"
          f"  IQ {gmm_e.d[1]:.4f} ({gmm_e.se[1]:.4f})  SEE {gmm_e.ser:.3f}"
          f"  J = {gmm_e.J:.2f} (p = {stats.chi2.sf(gmm_e.J, gmm_e.K - gmm_e.L):.5f})")

    # ---- (g) dropping MED and KWW from the instruments
    X_g = np.column_stack([h, d["MRT"], d["AGE"]])
    tsls_g = hl.GMM(y, Z, X_g)
    print(f"\n(g) instruments without MED and KWW (K = {X_g.shape[1]}, L = {Z.shape[1]}):"
          f" S {tsls_g.d[0]:.3f}  IQ {tsls_g.d[1]:.3f}")
    for k, nm in ((0, "S"), (1, "IQ")):
        first = hl.OLS(Z[:, k], X_g)
        t = first.b / first.se
        print(f"    first stage for {nm:<3}: t(MRT) = {t[-2]:>6.2f}, t(AGE) = {t[-1]:>6.2f}"
              f",  R2 = {first.R2:.3f}")


if __name__ == "__main__":
    main()
