import numpy as np
import matplotlib.pyplot as plt

N = 200


def is_equal(x, y):
    """ Check that x is equal to y up to uncertainty of +- epsilon """

    # e = 0.0001
    e = 0.1 * 1/N
    if y - e < x < y + e:
        return True
    else:
        return False


def calc_prob(player, a, b, c, d):
    """ Calculate probability for player to win given a, b, c, d. Probability is
            computed by finding nearest neighbords to player. """

    points = [0, a, b, c, d, 1]
    points.remove(player)

    upper_bound = min(filter(lambda x: x >= player, points))
    lower_bound = max(filter(lambda x: x <= player, points))

    if upper_bound == 1:
        prob = (1 - player) + (player - lower_bound)/2
    elif lower_bound == 0:
        prob = player + (upper_bound - player)/2
    else:
        prob = (upper_bound - player)/2 + (player - lower_bound)/2

    return prob


def calc_best_d(a, b, c):
    """ Compute the optimal choice of d given a, b, c. Each of the four regions
			are considered. If D is found to have infinite optimal choices across both
			middle regions, -1 flag is returned. """
			
    epsilon = 1/N
    lower, mid, upper = sorted([a, b, c])
    prob_lower = lower - epsilon/2
    prob_upper = (1 - upper) - epsilon/2
    prob_upper_mid = (upper - mid)/2
    prob_lower_mid = (mid - lower)/2
    best_prob = max(prob_lower, prob_upper, prob_upper_mid, prob_lower_mid)
    if best_prob == prob_lower:
        return lower - epsilon
    elif best_prob == prob_upper:
        return upper + epsilon
    elif is_equal(prob_lower_mid, best_prob) and is_equal(prob_lower_mid, prob_upper_mid):
        return -1
    elif best_prob == prob_upper_mid:
        return (upper + mid)/2
    elif best_prob == prob_lower_mid:
        return (mid + lower)/2


def correct_ranging_prob(player, a, b, c):
    """ If D was found to have infinitely many optimal moves within both middle regions,
			additional calcualtions need to be done when computing the probabilities
			for nearest neighbors. """

    d = calc_best_d(a, b, c)

    lower, mid, upper = sorted([a, b, c])

    if d == -1:
        if player == lower:
            prob = calc_prob(player, a, b, c, mid - 0.25*(mid - lower))
        elif player == upper:
            prob = calc_prob(player, a, b, c, mid + 0.25*(upper - mid))
        elif player == mid:
            prob = calc_prob(player, a, b, c, mid + 0.5*(upper - mid))
    else:
        prob = calc_prob(player, a, b, c, d)

    return prob


def plot(y):
    plt.scatter(np.linspace(0, 1, N+1), y, s=20)
    plt.xlabel(r'$a$')
    plt.ylabel(r'$P(A)$')
    plt.plot([1/6, 1/6], [-0.1, 0.4], '--', color='red')
    plt.plot([5/6, 5/6], [-0.1, 0.4], '--', color='red')
    plt.grid(linestyle='-')
    plt.show()


def main():
    a_probs = []
    best_bs = []
    best_cs = []
    best_cs_given_b = []

    for a in np.linspace(0, 1, N+1):
        print(a)

        best_b_prob = 0
        for b in np.linspace(0, 1, N+1):
            if b == a:
                continue

            best_c_prob = 0
            for c in np.linspace(0, 1, N+1):
                if c == a or c == b:
                    continue

                c_prob = correct_ranging_prob(c, a, b, c)

                if is_equal(c_prob, best_c_prob):
                    # if c_prob == best_c_prob:
                    best_cs.append(c)
                elif c_prob > best_c_prob:
                    best_cs = [c]
                    best_c_prob = c_prob

            b_prob_sum = 0
            for best_c in best_cs:
                b_prob = correct_ranging_prob(b, a, b, best_c)
                b_prob_sum += b_prob
            b_prob = b_prob_sum/len(best_cs)

            if is_equal(b_prob, best_b_prob):
                # if b_prob == best_b_prob:
                best_bs.append(b)
                best_cs_given_b.append(best_cs)
            elif b_prob > best_b_prob:
                best_bs = [b]
                best_cs_given_b = [best_cs]
                best_b_prob = b_prob

        aprob_sum = 0
        for index, best_b in enumerate(best_bs):
            for best_c in best_cs_given_b[index]:
                aprob = correct_ranging_prob(a, a, best_b, best_c)
                aprob_sum += aprob
        aprob = aprob_sum/sum([len(row) for row in best_cs_given_b])
        a_probs.append(aprob)

    plot(a_probs)


if __name__ == "__main__":
    main()
