# Solutions to Fumio Hayashi, *Econometrics* (Princeton University Press, 2000)

Authors: Zijun Meng, Claude Opus 5, and Kimi K3

## Contents

Worked solutions to **all** the exercises of Hayashi's *Econometrics*, Chapters 1–10:
281 review questions, 71 analytical exercises, 10 empirical exercises and 5 Monte
Carlo exercises (367 in all), in one document, `HayashiSolutions.pdf` (230 pages).

| Chapter | Review | Analytical | Empirical | Monte Carlo | Pages |
|---|---:|---:|---:|---:|---|
| 1. Finite-Sample Properties of OLS | 46 | 7 | 1 | 1 | 2–24 |
| 2. Large-Sample Theory | 47 | 12 | 2 | 2 | 25–55 |
| 3. Single-Equation GMM | 45 | 10 | 1 | — | 56–89 |
| 4. Multiple-Equation GMM | 22 | 11 | 1 | — | 90–118 |
| 5. Panel Data | 14 | 7 | 1 | — | 119–138 |
| 6. Serial Correlation | 35 | 10 | 2 | — | 139–163 |
| 7. Extremum Estimators | 28 | 3 | — | — | 164–185 |
| 8. Examples of Maximum Likelihood | 14 | 4 | — | — | 186–197 |
| 9. Unit-Root Econometrics | 19 | 7 | 1 | 2 | 198–217 |
| 10. Cointegration | 11 | — | 1 | — | 218–229 |

**The exercise statements are not reproduced.** Each solution is headed only by the
number of its exercise in the book ("Review Question 2.8.2", "Analytical Exercise
3.10", "Empirical Exercise 9.1", ...), so read the statement in the book first.
Notation, the part letters (a), (b), ... and labels such as (\*) or (2) follow the
book's statements, and numbered results cited as "(2.4.1)" or "Lemma 2.4" refer to
Hayashi (2000). Remarks inside the solutions point out misprints and ambiguities
in the book where we found them.

| File | Contents |
|---|---|
| `HayashiSolutions.tex` / `.pdf` | all the solutions in one self-contained document (230 pages; solutions only, without the exercise statements): title page, linked table of contents, PDF bookmarks for every chapter and exercise |
| `figures/` | figures included by the solutions (produced by the code) |
| `code/` | Python code for the numerical parts (see below) |
| `data/` | place the textbook's data sets here (not redistributed) |

## Building

Compile from this directory with

```bash
pdflatex HayashiSolutions.tex
```

Run it two or three times so the contents page numbers and cross-references
settle. The document needs no local style file; it includes the PDF figures from
`figures/`.

## Empirical exercises

The empirical exercises use the data sets distributed with the textbook. The data
are **not** redistributed here; place them under `data/` in the layout the scripts
expect:

```
data/ch1/NERLOVE.ASC    data/ch4/GREENE.ASC     data/ch6/YEN.ASC
data/ch2/MISHKIN.ASC    data/ch5/SUM_HES.ASC    data/ch9/LT.ASC
data/ch3/GRILIC.ASC     data/ch6/DM.ASC         data/ch10/MPYR.ASC
                        data/ch6/POUND.ASC
```

The scripts in `code/` (one `chN.py` per chapter with numerical work, plus the shared
`hayashilib.py`) depend only on `numpy`, `scipy` and `matplotlib`. Run one with, for
example, `python code/ch4.py`; it prints the numbers quoted in the solutions and
writes its figures to `figures/`. Chapters 7 and 8 have no empirical or Monte Carlo
exercises and hence no script.
