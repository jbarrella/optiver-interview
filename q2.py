import numpy as np
import matplotlib.pyplot as plt

def calcaprob(a, b, c):
	prob = 0
	if b < a < c:
		prob += (a - b)/2
		prob += (c - a)/2
	elif c < a < b:
		prob += (a - c)/2
		prob += (b - a)/2
	elif c < b < a:
		prob += 1 - a
		prob += (a - b)/2
	elif b < c < a:
		prob += 1 - a
		prob += (a - c)/2
	elif a < b < c:
		prob += a
		prob += (b - a)/2	
	elif a < c < b:
		prob += a
		prob += (c - a)/2
	return prob

def calcbprob(a, b, c):
	prob = 0
	if b < a < c:
		prob += b
		prob += (a - b)/2
	elif c < a < b:
		prob += 1 - b
		prob += (b - a)/2
	elif c < b < a:
		prob += (b - c)/2
		prob += (a - b)/2
	elif b < c < a:
		prob += b
		prob += (c - b)/2
	elif a < b < c:
		prob += (c - b)/2
		prob += (b - a)/2	
	elif a < c < b:
		prob += 1 - b
		prob += (b - c)/2
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
bs = []
bestbs = []
bestcs = []

bprobs = []
cprobs = []
for a in np.linspace(0,1,1001):
	x.append(a)
	bestbprob = 0
	for b in np.linspace(0,1,1001):
		if b == a:
			continue

		bestc = calcbestc(a, b)

		bprob = calcbprob(a, b, bestc)

		if bprob == bestbprob:
			bs.append(b)
		if bprob > bestbprob:
			bs = [b]
			bestbprob = bprob
			# bestb = b

	bestb = np.mean(bs)

	bestbs.append(bestb)
	bestcs.append(calcbestc(a, bestb))
	aprob = calcaprob(a, bestb, calcbestc(a, bestb))
	probs.append(aprob)

plt.scatter(x, probs, s=20)
plt.xlabel(r'$a$')
plt.ylabel(r'$P(A)$')
plt.plot([1/4, 1/4], [-1, 1], '--', color='red')
plt.plot([3/4, 3/4], [-1, 1], '--', color='red')
plt.grid(linestyle='-')
plt.show()
