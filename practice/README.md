# Practice folder

You write the code. Eight exercises, each small enough for one sitting (20 to 60 minutes).
Each file starts with a docstring that explains the task, then functions that raise `NotImplementedError`.

## Setup

```bash
pip install pandas numpy lightgbm mlforecast pytest
cd practice
pytest -q                 # at first everything fails with NotImplementedError. That is correct.
```

## How to work

1. Read the exercise docstring. Write on paper what the function must return for a tiny input.
2. Write your own version. Run `pytest -k ex01 -q` (change the number).
3. Read the failure message. It names the check that failed. Fix, re-run.
4. Stuck for more than 20 minutes? Use the hint ladder in Process.md (Module 15), then a precise prompt to an AI tutor.
5. Only after your tests pass: open `solutions/` and compare. Note one thing the solution does differently.
6. Write a two-line entry in your error log: what failed, what fixed it.

To check that the solutions themselves pass: `USE_SOLUTIONS=1 pytest -q`.

## Map to the course

| Exercise | Course module |
|---|---|
| ex01 pandas warm-up | Module 2 |
| ex02 promo counters | Module 4 |
| ex03 baselines | Module 5 |
| ex04 metrics | Module 6 |
| ex05 backtest | Module 7 |
| ex06 LightGBM model | Module 8 |
| ex07 uplift and stock decision | Modules 11 and 12 |
| ex08 feed contract | Module 14 |
