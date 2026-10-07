"""
Bisect the Kentucky data and compare the chi^2 divergence for the first and second halves

Code in support of the paper "Surprising payouts in parimutuel numbers games"
Provided under a GPL v3.0 license
"""
import pandas as pd

df = pd.read_csv('../KY Pick 3/KY-P3.csv', index_col=0, parse_dates=["When"])

# the drawings range from 21395 to 21421
# so we divide into two parts:
#    21395 -- 21407
#    21409 -- 21421

def StrToNum(numstr: str) -> int:
    return int(numstr[0])*100 + int(numstr[2])*10 + int(numstr[4])


def chi_mq(probs):
    """Return the chi^2 divergence chi^2(m||q)"""

    return np.sum((1/possible_tix - probs)**2 / probs)

if __name__ == "__main__":
    print("Splitting KY data in half and computing chi^2(m||q) on both parts")
    first = df[df['First Draw #'] <= 21402] # 415,190 rows
    second = df[df['First Draw #'] >= 21403] # 376,324 rows

    first_wagers = np.zeros(1000)
    second_wagers = np.zeros(1000)
    for df, wagers in [(first, first_wagers), (second, second_wagers)]:
        for _, row in df.iterrows():
            if len(row['Numbers Selected']) < 5:
                print(f"Found weird drawing {row}")
            else:
                wagers[StrToNum(row['Numbers Selected'])] += row['Board Amount']

    first_probs = first_wagers / np.sum(first_wagers)
    second_probs = second_wagers / np.sum(second_wagers)

    print(f"chi^2(m||q) for first half = {chi_mq(first_probs)}")
    print(f"chi^2(m||q) for second half = {chi_mq(second_probs)}")
    reldiff = (chi_mq(first_probs) - chi_mq(second_probs)) / chi_mq(first_probs)
    print(f"Relative difference: {reldiff}")

