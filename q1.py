import numpy as np
import matplotlib.pyplot as plt

a = 0.0

def calcbprob(a, b, c):
	prob = 0
	if c < b:
		prob += 1 - b
		prob += (b - c) / 2
	else:
		prob += (b - a) / 2
		prob += (c - b) / 2
	return prob

def calcbestc(a, b):
	epsilon = 0.001
	upper = max(a, b)
	lower = min(a, b)
	prob_lower = lower - epsilon/2
	prob_upper = (1 - upper) - epsilon/2
	prob_between = (upper - lower)/2
	best_prob = max(prob_lower, prob_upper, prob_between)
	if best_prob == prob_lower:
		return lower - epsilon
	elif best_prob == prob_upper:
		return upper + epsilon
	else:
		return (upper - lower)/2 + lower

x = []
probs = []
bestcs = []
for b in np.linspace(0,1,10001):
	if b == a:
		continue
	x.append(b)

	bestc = calcbestc(a, b)
	bestcs.append(bestc)

	bprob = calcbprob(a, b, bestc)
	probs.append(bprob)

plt.grid(linestyle='-')

plt.scatter(x, probs, s=20)
plt.xlabel(r'$b$')
plt.ylabel(r'$P(B)$')
plt.plot([2/3, 2/3], [-1, 1], '--', color='red')
plt.show()
