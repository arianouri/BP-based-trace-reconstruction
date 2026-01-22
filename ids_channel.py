import numpy as np

def ids_channel(x, alph_card, p_ins, p_del, p_sub, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    y = []
    t = 0
    N = len(x)

    # Process all symbols
    while t < N:
        u = rng.random()
        
        if u < p_ins:
            # Insertion: Add symbol, do NOT increment t
            y.append(rng.integers(0, alph_card))
        elif u < p_ins + p_del:
            # Deletion: Skip symbol
            t += 1
        elif u < p_ins + p_del + p_sub:
            # Substitution: Replace symbol
            xt = x[t]
            sub = rng.integers(0, alph_card - 1)
            y.append(sub if sub < xt else sub + 1)
            t += 1
        else:
            # Correct transmission
            y.append(x[t])
            t += 1
            
    # Optional: Final insertions after the last symbol
    # In DNA storage, errors can happen at the very end of the strand.
    while rng.random() < p_ins:
        y.append(rng.integers(0, alph_card))

    return np.array(y, dtype=int)