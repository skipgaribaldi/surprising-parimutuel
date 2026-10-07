"""
Simulate numbers games drawings

Code in support of the paper "Surprising payouts in parimutuel numbers games"
Provided under a GPL v3.0 license
"""
import numpy as np
import pandas as pd

# --- Simulation Setup (Initialization) ---
KY = pd.read_csv("../KY data set/KY-wagers.csv")  # summarizes the numbers chosen by Kentucky players 
straight_KY = np.array(KY['Probability'])  # the base density (length 1000, 0-999)

CA = pd.read_csv("../CA data set/CA-wagers-straight.csv")
straight_CA = np.array(CA['Probability'])

straight_uniform = np.ones(1000)/1000

possible_tix = 10**3

def test_straight_probs_from_data():
    assert np.isclose(np.mean(straight_KY), 1/possible_tix)
    assert np.isclose(np.mean(straight_CA), 1/possible_tix)

def collision(arr):
    """
    Return the collision probability of a probability density function whose values are represented by arr
    """
    return np.sum([x**2 for x in arr]) / (np.sum(arr)**2)


def test_collision():
    assert np.isclose(collision(straight_uniform)*possible_tix, 1)  # uniform has c*t = 1
    assert np.isclose(collision(straight_KY), 1/93.92488)  # Kentucky has collision = 1/93


def chi_qm(probs):
    """Return the chi^2 divergence chi^2(q||m)"""

    return possible_tix * collision(probs) - 1


def chi_mq(probs):
    """Return the chi^2 divergence chi^2(m||q)"""

    return np.sum((1/possible_tix - probs)**2 / probs)


def create_straight_probs(m:float=1):
    """
    Return a convex combination of the uniform and KY probabilities.
    m=1 (default) gives KY
    m=0 gives uniform
    """
    global possible_tix, straight_KY
    
    straight_uniform = (1/possible_tix)*np.ones(possible_tix)
    
    return (1-m)*straight_uniform + m*straight_KY 


def create_probs(m:float=1, possible_tix:int=10**3):
    straight_probs = create_straight_probs(m)
    assert np.isclose(np.mean(straight_probs), 1/possible_tix)

    # --- Box Density Calculation ---
    # impute probabilities for box bets from KY straight bet bata 
    box_probs = straight_probs.copy()  # start using the probabilities for straight bets
    for i in range(10):
        box_probs[111*i] = 0  # can't box bet on 000, 111, 222...
    box_probs = box_probs/np.sum(box_probs)  # normalize density to have total prob. = 1

    return straight_probs, box_probs


def test_create_probs():
    assert np.all(straight_KY == create_probs(1)[0])
    assert np.all(straight_uniform == create_probs(0)[0])

rados_convex = 0.31307  # the convex combination parameter that gives the same collision probability as Rados's data set
def test_rados():
    rados_probs, box_probs = create_probs(rados_convex)
    assert np.isclose(collision(rados_probs), 1/514)
    assert np.isclose(chi_mq(rados_probs), 0.13884139)


def test_rados_range():
    min_probs, _ = create_probs(0.26)
    max_probs, _ = create_probs(0.36)
    assert np.isclose(chi_mq(min_probs), 0.105696405992)
    assert np.isclose(chi_mq(max_probs), 0.1716480966)

def test_chi_mq_straight():
    assert np.isclose(chi_mq(straight_CA), 0.1158906)
    assert np.isclose(chi_mq(straight_KY), 1.7565)

CA_convex = 0.27701  # the convex combination parameter that gives the same chi_mq as the California data set
fake_CA, fake_CA_box = create_probs(CA_convex) # from matching chi_mq
# np.savetxt("../CA data set/CA-fake.csv", fake_CA)

def test_chi_mq_CA():
    assert np.isclose(chi_mq(straight_CA), chi_mq(fake_CA))

# Normalized versions of the measures have value 1 for the uniform distribution
def norm_collision(probs):
    '''Normalized collision prob, i.e., c*t'''
    return collision(probs)*possible_tix

assert np.isclose(norm_collision(straight_uniform), 1)

def inv_harmonic(arr):
    """
    Return one over the Harmonic mean of the array arr
    """
    return np.sum(1/arr) / len(arr)

def test_inv_harmonic():
    assert inv_harmonic(straight_uniform) == possible_tix  # t = inv_harmonic(uniform)
    assert np.isclose(inv_harmonic(straight_KY), 2756.485555)  # check inv_harmonic on Kentucky

def norm_harmonic(probs):
    '''Normalized inverse harmonic, is 1 for uniform'''
    return inv_harmonic(probs)/possible_tix

def test_norm_harmonic():
    assert np.isclose(norm_harmonic(straight_uniform), 1)


# --- Core Simulation Function ---
def single_drawing(straight_probs, box_probs, mm=229587, pi=[0.85, 0.10, 0.05], debug=False, d=0.5):
    """
    Simulates a single lottery draw, returns zz to represent payout for straight bet as zz:1

    mm: mean total tickets sold
    pi: proportions of Straight (p1), Box (p2), Pair (p3) bets
    d: fraction of revenue paid out in prizes
    straight_probs: probability distribution for straight and pair bets
    box_probs: probability distribution for box bets
    """
    # 1. Draw Winning Number (x, y, z are digits 0-9)
    # The R code uses sample(seq(0,9), 1) three times, which are independent draws
    # and results in a 3-digit number.
    x, y, z = np.random.choice(10, size=3)
    index = int(x * 100 + y * 10 + z) # 0 to 999

    # 2. Winning Probabilities based on Bet Type

    # **Straight (bas)**:
    # Probability of the winning number (x,y,z) occurring.
    straight = straight_probs[index]

    # **Box (box)**:
    # Sum of probabilities for all permutations of (x,y,z).
    # Since the R code *commented out* the step that zeroes out probabilities
    # for non-unique numbers in the box density (pb), this calculation assumes
    # all permutations are still possible in pb, which is simply normalized pr.
    
    # Generate all 6 permutations of the winning digits (x,y,z)
    digits = [x, y, z]
    from itertools import permutations
    
    # Use set to ensure unique permutations are considered once
    num_perms = len(set(permutations(digits)))
    box = 0
    if num_perms == 1:  # winner is 111, 222, etc
        pass
    elif num_perms in [3, 6]:
        for p in set(permutations(digits)):
            r_index = int(p[0] * 100 + p[1] * 10 + p[2])
            # Note: The R script assumes the Box density 'pb' is indexed by the permutation
            # r = sum(c(x,y,z)*10^c(2,1,0)) + 1 -> index
            box += box_probs[r_index]
    else:
        raise ValueError(f"Only {num_perms} permutations of alleged winning number {index}")
    
    # **Pair (par)**:
    # Sum of probabilities where one digit is allowed to change.
    # The R code calculates this as:
    # 1/2 * ( sum_a pr(a,y,z) + sum_a pr(x,y,a) )
    
    par = 0
    # First sum: changing 'x' to 'a' (back pair y,z)
    for a in range(10):
        r_index = int(a * 100 + y * 10 + z)
        par += straight_probs[r_index]

    # Second sum: changing 'z' to 'a' (front pair x,y)
    for a in range(10):
        r_index = int(x * 100 + y * 10 + a)
        par += straight_probs[r_index]
    
    par = par / 2 # have to choose front or back

    if debug:
        print(f'Winning number is {x}{y}{z} with {num_perms} permutations')

    # 3. Simulate Tickets Sold (n1, n2, n3)
    # R functions: rpois(1, lambda) -> numpy.random.poisson(lambda, 1)
    # R functions: round() -> numpy.round() (standard Python round might behave differently on .5)
    lambdas = [mm * pii for pii in pi]

    # Draw from Poisson 
    ns = [ np.round( np.random.poisson(lam) ) for lam in lambdas ]
    nn = sum(ns)

    # 4. Simulate Winning Tickets (x1, x2, x3)
    # R functions: rbinom(1, size, prob) -> numpy.random.binomial(size, prob, 1)

    x1 = np.random.binomial(n=ns[0], p=straight) # Straight winners
    x2 = np.random.binomial(n=ns[1], p=box) # Box winners
    x3 = np.random.binomial(n=ns[2], p=par) # Pair winners

    # 5. Calculate Payout (zz)
    # Payout is based on the formula: nn / (2 * sum_j r_j * X_j)
    # where:
    #   r = 1/c(1, 6, 10) (Straight=1, Box=1/6, Pair=1/10)
    #   X_j = number of winning tickets of type j
    #   d = 1/2 (since tickets are 50 cents, but the R code uses 1/2 implicitly)

    zz = x1 / 1 + x2 / num_perms + x3 / 10
    if debug:
        print(f'   Total tix = {nn} = {ns}')
        print(f'   Prize pool {nn*d} for {x1}, {x2}, {x3} winners')
    
    if zz == 0:
         # Handle division by zero: if no wins, the payout rate is effectively infinite
         # For simplicity, we'll return NaN.
        return {'straight': np.nan, 'pair': np.nan, 'box': np.nan}, {}

    # division by 2 because we simulate a lottery that returns 50% of revenue as prizes
    payouts = {'straight': d*nn/zz, 'pair': d*nn/(zz*10)}
    probs = {'straight': straight, 'pair': par, 'box': box}
    if num_perms == 1:
        payouts['box3'] = 0
        payouts['box6'] = 0
    elif num_perms == 3:
        payouts['box3'] = d*nn/(zz*3)
        payouts['box6'] = 0
    elif num_perms == 6:
        payouts['box3'] = 0
        payouts['box6'] = d*nn/(zz*6)

    for key in payouts:
        payouts[key] = np.round(payouts[key])

    return payouts, probs

# testing code
test_straight_probs, test_box_probs = create_probs(CA_convex)
outs = [single_drawing(test_straight_probs, test_box_probs) for i in range(100000)]
straights = [x[1]['straight'] for x in outs]
assert np.isclose(np.mean(straights), 10**-3, rtol=0.03)

def average_dicts(results: list):
    """
    Passed a list of dicts, where each dict has values that are numbers,
    return a dict with the same keys where each value is the average of the corresponding numbers
    """
    rval = {}
    for key in results[0]:
        rval[key] = np.mean([result[key] for result in results])

    return rval
    

# --- Simulate many drawings ---
def many_drawings(straight_probs, box_probs, num_drawings=309, num_tix=1310143, pi=[0.85, 0.10, 0.05]) -> dict:
    """
    Simulate num_drawings drawings and return the average payout
    num_tix is average number of tickets in each drawing
    straight_probs and box_probs are the ticket-selection distributions
    """
    results = []
    for j in range(num_drawings): # number of draws in the year
        # single_drawing() returns a dictionary with key for the simulated payout
        payout, _ = single_drawing(straight_probs, box_probs, mm=num_tix, pi=pi)
        results.append(payout)
    
    return average_dicts(results)
    

# --- Run Monte Carlo trials ---
def MC_drawings(straight_probs, box_probs, num_drawings:int=309, num_tix:int =1310143, pi=[0.85, 0.10, 0.05], trials:int=10000) -> list[dict]:
    """
    Do trials simulations of a collection of drawings, each averaging num_drawings drawings
    report an array of length trials, each representing the mean payout for num_drawings drawings
    
    num_tix: average number of tickets in each drawing
    trials: number of independent trials to perform
    straight_probs and box_probs are the ticket-selection distributions
    """
    rz = []
    for _ in range(trials): # Monte Carlo runs
        rz.append(many_drawings(straight_probs, box_probs, num_drawings=num_drawings, num_tix=num_tix, pi=pi))

    return rz


def average_MC_drawings(straight_probs, box_probs, num_drawings:int=309, num_tix:int=1310143, pi=[0.85, 0.10, 0.05], trials:int=10000) -> dict:
    """Run Monte Carlo trials and return their average payouts."""
    results = MC_drawings(
        straight_probs,
        box_probs,
        num_drawings=num_drawings,
        num_tix=num_tix,
        pi=pi,
        trials=trials,
    )
    return average_dicts(results)


# --- Now we simulate for fake_CA for a range of sales amounts
def compare_CA_results():
    """
    Simulate many drawings with actual CA numbers chosen (both for straight bets and box bets) and
    simulate many drawings with the "fake_CA" distribution, i.e., the convex combination with the same chi_mq

    Print the relative differences
    """
    straight_probs, box_probs = create_probs(CA_convex)
    fake_results = []
    for i in range(2, 20):
        print(f"--- {i=}")
        tix = i * 10**5
        fake_results.append(
            [tix, many_drawings(straight_probs, box_probs, num_drawings=5*10**3, num_tix=tix, pi=[0.9,0.1,0])['straight']]
        )

    straight_probs = straight_CA  # includes straight and combo bets
    boxdf = pd.read_csv('~/Documents/early-overleaf/CA data set/CA-boxes.csv')
    box_probs = list(boxdf['Probability'])
    real_results = []
    for i in range(2, 20):
        print(f"--- {i=}")
        tix = i * 10**5
        real_results.append(
            [tix, many_drawings(straight_probs, box_probs, num_drawings=5*10**3, num_tix=tix, pi=[0.9,0.1,0])['straight']]
        )

    for i in range(len(fake_results)):
        print(i, f"{(fake_results[i][-1] - real_results[i][-1])/real_results[i][-1]}")


# --- Now we actually simulate for a range of convex combination parameters m
# The aim here is to generate the data for Figure 5

# First we can do it using the functions above
def verify_companion_result():
    """
    Computing average payouts for a range of convex combination parameters

    This allows one to empirically confirm that the main result of the companion paper, 
    which computes the expected straight payout up to an O(n^-2) error term,
    gives a very good approximation to the expected straight payout, by taking the 
    formula with the error term dropped and comparing to the simulation results
    """
    nd = 10**4  # 10,000 drawings
    mm = 10**6  # 1,000,000 tickets per drawing (on average)

    payouts = {}
    results = []

    # these are the convex combinations we will try
    vals = list(np.arange(0.0, 0.2 + 0.01, 0.01)) + list(np.arange(0.0, 1 + 0.01, 0.05))

    for m in vals:
        straight_probs, box_probs = create_probs(float(m))
        ct = collision(straight_probs)*possible_tix
        print(f"--- {m=} {ct=}")
        payouts[m] = many_drawings(straight_probs, box_probs, num_drawings=nd, num_tix=mm, pi=[1,0,0])  # just straight tix
        print(f"   {payouts[m]}")
        print(f"   {500*ct}")
        results.append([ct, payouts[m]])

    out = pd.DataFrame(results)
    out.to_csv('simulation-one-prize-10000-10-6.csv', index=False, header=False)

# alternatively, we can really simplify the simulation function since we are talking about games with only straight bets
# --- Core Simulation Function ---
def straight_drawings(num_tix: int=10**6, num_drawings: int=10**3, d: float=0.5, m: float=0) -> float:
    """
    Simulates num_drawings drawings, each selling on average num_tix tickets
    These are 3-digit, straight only bets
    m specifies the betting distribution: m=0 is uniform, m=1 is Kentucky
    """
    # generate a collection of winning numbers
    rands = np.random.choice(10, size=(3, num_drawings))
    win_nums = rands[0]*100 + rands[1]*10 + rands[0]

    straight_probs, _ = create_probs(m)
    
    sales = np.random.poisson(lam=num_tix, size=num_drawings)

    # while we could pick the winning numbers (by sampling from range(1000) with replacement using np.random.choice,
    # we don't care about the actual winning number, we just care about the probability that someone buys a ticket
    # with that winning number, so we sample from those probabilities instead
    win_probs = np.random.choice(straight_probs, num_drawings, replace=True)

    winners = np.random.binomial(n=sales, p=win_probs) # Straight winners

    if np.prod(winners) != 0:
        prizes = np.round(d*sales / winners)
    else:
        prizes = np.round([d*sales[i] / winners[i] for i in range(num_drawings) if winners[i] != 0])

    return float(np.mean(prizes))


def verify_companion_result2(vals=None, num_drawings: int=10**3, num_tix: int=10**6, d: float=0.5, num_samples=10**3):
    """
    For a range of convex combination parameters (governing the distribution q of tickets chosen by gamblers), 
    simulate many straight-ticket-only drawings.  Create a CSV file where each row lists
        convex combination parameter, average payout from simulations, estimated payout from the formula
    """
    if vals is None:
        vals = np.arange(0, 1+0.0001, 0.025)

    lst = []
    for val in vals:
        if np.isclose(np.round(val*10),val*10):
            print(val)
        val = float(val)
        probs, _ = create_probs(val)
        samples = [straight_drawings(m=val, num_drawings=num_drawings, d=d) for x in range(num_samples)]
        payout = np.mean(samples)
        lst.append({'m': val, 
                    'payout': payout,
                    'estimate': (d/possible_tix)*np.sum(1/probs + (1 - probs)/(num_tix * (probs**2)))
                    })
    
    df = pd.DataFrame(lst)
    df.to_csv('simulation-straight.csv', index=False)


def figure2():
    """
    For a range of sales values, simulate a straight-only game using uniform ticket purchases and 
    report the average payout

    Saves the results in a CSV file that can be used to generate Figure 2
    """
    exps = np.arange(3.1, 8, 0.1)

    lst = []
    for exp in exps:
        if np.isclose(exp, round(exp)):
            print(f"--- {exp=}")
        num_tix = 10**exp
        pay = straight_drawings(
                        num_tix=round(num_tix),
                        num_drawings=10**7,
                        m=0  # tickets are purchased uniformly at random
                    )
        if pay != 0:
            lst.append({'n': num_tix, 
                        'payout': pay})
        
    df = pd.DataFrame(lst)
    df.to_csv('figure2.csv', index=False)


# --- Simulate to generate the contour plot!
# Figure 6
def generate_contour():
    """
    Vary the amount bet on box and pair bets and for each coordinate do a simulation to get the average payout
    We use the convex combination that approximates the Rados distribution
    """
    contour = []
    straight_probs, box_probs = create_probs(rados_convex)
    for pair in list(np.arange(0.0, 0.15, 0.02)) + [0.15]:
        for box in list(np.arange(0.0, 0.15, 0.02)) + [0.15]:
            print(f"--- {box}, {pair}")
            straight = 1 - pair - box
            contour.append([
                box, pair, average_MC_drawings(straight_probs, box_probs, num_drawings=500, num_tix=10**6, pi=[straight, box, pair], trials=5 * 10**3)['straight']
            ])

    np.savetxt("contour.csv", contour, delimiter=",")


# --- Test the combination of increasing sales and gamblers shifting to box and pair bets
def figure6_corners():
    """
    Generate the values just in the lower left and upper right corners of Figure 6
    """
    straight_probs, box_probs = create_probs(0.315)  # parameter chosen to produce chi_mq = 0.14
    assert np.isclose(chi_mq(straight_probs), 0.14, 0.001)  # check that we got the right chi_mq

    print("Simulating FY1976")
    payout76 = average_MC_drawings(straight_probs, box_probs, num_drawings=310, num_tix=343_409, pi=[1,0,0], trials=5 * 10**3)['straight']
    print(f"... {payout76}")

    print("Simulating FY1983")
    payout83 = average_MC_drawings(straight_probs, box_probs, num_drawings=311, num_tix=2_515_755, pi=[0.7,0.15,0.15], trials=5 * 10**3)['straight']
    print(f"... {payout83}")


def generate_data_sets(seed: int=202608) -> None:
    """Generate all simulation data sets reproducibly using the given random seed."""
    np.random.seed(seed)
    figure2()
    verify_companion_result2()
    generate_contour()
    figure6_corners()
