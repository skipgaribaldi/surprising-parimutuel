"""
Compute average payouts per fiscal year, give confidence intervals

Code in support of paper "Surprising payouts in parimutuel numbers games"
Provided under a GPL v3.0 license
"""
import pandas as pd
import numpy as np
import math
import random

SEED = 20260716
STEADY_STATE = range(1986, 2016)      # fiscal years 1986..2015, steady state for NJ


def get_year(yr, state, result=None):
    """From the full list of draw results, get only those in a particular fiscal year"""
    col = state.columns[0]  # 'date' for NJ, 'Date' for MA
    yrdf = state[state[col] >= f"{yr-1}-07-01"]
    yrdf = yrdf[yrdf[col] < f"{yr}-07-01"]

    if result is None:
        result = state.columns[3]  # 'straight' for NJ, 'Exact 4' for MA
        if result == 'straight':  # NJ
            factor = 2
        elif result == 'Exact 4':  # MA
            factor = 0.1
    elif result == 'Exact 3':  # also MA
        factor = 1
    else: 
        print(f"DEBUG: requested {result=}")
        factor = 1

    return np.array(yrdf[result]) * factor
  

def year_report(payouts, label, reps=10_000, seed=SEED):
    n = len(payouts)
    m = np.mean(payouts)
    s = np.std(payouts)
    se = s / math.sqrt(n)
    # t critical value ~1.97 for n around 300; use 2.0 for display honesty,
    # or swap in scipy.stats.t.ppf(0.975, n-1) if scipy is available.
    tcrit = 2.0
    rng = random.Random(seed)
    boot = sorted(
        np.mean(rng.choices(payouts, k=n)) for _ in range(reps)
    )
    lo, hi = boot[int(0.025 * reps)], boot[int(0.975 * reps)]
    print(f"{label}: n={n} drawings")
    print(f"  mean = {m:.1f}   SD = {s:.1f}   SE = {se:.1f}")
    print(f"  normal 95% CI    = ({m - tcrit*se:.1f}, {m + tcrit*se:.1f})")
    print(f"  bootstrap 95% CI = ({lo:.1f}, {hi:.1f})")
    return m, se


def print_header(msg):
    """Print out a nice box"""
    bigmsg = f"*** {msg} ***"
    border = len(bigmsg) * "*"
    print(border)
    print(bigmsg)
    print(border)


if __name__ == "__main__":
    # calculate the average payouts for NJ
    nj = pd.read_csv('../NJ data set/NJ-pick-3.csv')  # results of NJ drawings

    #
    # need to drop EXCLUDE_DATES = {date(1978, 3, 24)}   # the $0-prizes recording error
    #

    print_header("NJ: Statistics for payouts for each year of NJ Pick-It/Pick 3 from inception to 2016")
    for yr in range(1975, 2016):
        year_report(list(get_year(yr, nj)), f"{yr}")  
        
    print_header("NJ: Statistics for the steady state payouts")
    yrdf = nj[nj['date'] >= f"{min(STEADY_STATE)-1}-07-01"]
    yrdf = yrdf[yrdf['date'] < f"{max(STEADY_STATE)}-07-01"]
    year_report(np.array(yrdf['straight']), f"Steady state ({min(STEADY_STATE)}-{max(STEADY_STATE)}")

    # next we do the same for MA
    ma = pd.read_csv('../MA data set/MA-numbers.csv')   # results of MA drawings

    print_header("MA: Statistics for 4-digit straight payouts for MA Numbers Game from inception to 2016")
    for yr in range(1976, 2016):
        year_report(list(get_year(yr, ma)), f"{yr}")  

    print_header("MA: Statistics for 3-digit straight payouts for MA Numbers Game from inception to 2016")
    for yr in range(1976, 2016):
        year_report(list(get_year(yr, ma, result='Exact 3')), f"{yr}")  
