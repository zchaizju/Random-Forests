from collections import Counter
import numpy as np


# shuffle data and label together
def shuffle_together(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


# calculate entropy of set Y
def entropy(Y):
    s = 0.0
    num_all = len(Y)
    distribution = Counter(Y)
    for y, num_y in distribution.items():
        probability_y = (num_y/num_all)
        s += (probability_y)*np.log(probability_y)
    return -s


# calculate IG of y and the two branches of y
def information_gain(y, y_true, y_false):
    return entropy(y)-(entropy(y_true)*len(y_true)+entropy(y_false)*len(y_false))/len(y)

