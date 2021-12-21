import random
import numpy as np
import Reporter


class TravelingSalespersonProblem:
	def __init__(self, distance_matrix):
		self.cost = distance_matrix
		self.dimension = distance_matrix.shape

	def length(self, ind):
		length = 0
		for i in range(self.dimension[0] - 1):
			length += self.cost[ind.path[i]][ind.path[i + 1]]
		length += self.cost[ind.path[self.dimension[0] - 1]][ind.path[0]]
		return length


# Candidate solution representation
class Individual:
	def __init__(self, tsp):
		self.path = np.random.permutation(range(1, tsp.dimension[0]))
		self.path = np.concatenate(([0], self.path))
		self.alpha = 0.2

	def print(self):
		print("Ind Path: ", self.path)


class Solver:
	def __init__(self, tsp):
		self.lambdaa = 1000			# Population size
		self.mu = 500				# Offspring size
		self.k = 3					# Tournament selection
		self.k2 = 3					# K-top elimination
		self.intMax = 500			# Boundary of the domain, not intended to be changed.
		self.nbIterations = 100			# Maximum number of iterations

		self.population = np.empty(self.lambdaa, Individual)
		for i in range(self.lambdaa):
			self.population[i] = Individual(tsp)

	# k-Tournament selection
	def select(self, tsp):
		selected = np.random.choice(self.population, self.k, False)

		values = list(map(tsp.length, selected))

		minIndex = values.index(min(values))

		return selected[minIndex]

	# PMX crossover
	def recombine(self, tsp, p1, p2):
		# https://www.youtube.com/watch?v=ZtaHg1C25Kk
		a = random.randint(1, tsp.dimension[0] - 1)
		b = random.randint(1, tsp.dimension[0] - 1)

		if a > b:
			temp = a
			a = b
			b = temp

		childPath = np.zeros(tsp.dimension[0], dtype=int)

		# 1. Choose a random segment and copy it from p1 to child
		childPath[a:b] = p1.path[a:b]

		i = []
		j = []

		# 2. Starting from the first crossover point, look for elements in that swgment that have not been copied
		for x in range(a, b):
			if p2.path[x] not in p1.path[a:b]:
				i.append(p2.path[x])

		# 3. For each of these i, look in the offspring to see whtat element j has been copied in its place from p1
		for x in range(len(i)):
			index = np.where(p2.path == i[x])[0][0]
			j.append(p1.path[index])

			# 4. Place i into the position occuped by j in p2, since we will not be putting j there (as j is already in the offspring)
			index = np.where(p2.path == j[x])[0][0]

			if childPath[index] == 0:
				childPath[index] = i[x]
			else:
				# 5. If the place occuped by j in p2 has already been filled in the offspring by k, put i in the positioin occuped by k in p2
				kIndex = index
				while(True):
					k = p1.path[kIndex]
					kIndex = np.where(p2.path == k)[0][0]
					if childPath[kIndex] == 0:
						childPath[kIndex] = i[x]
						break

		# 6. Having dealt with the elements from the crossover segment, the rest of the offspring can be filled from p2
		for x in range(1, tsp.dimension[0]):
			if childPath[x] == 0:
				childPath[x] = p2.path[x]

		ind = Individual(tsp)
		ind.path = childPath

		return ind

	def mutate(self, ind):
		if random.random() < ind.alpha:
			# Randomly swap two elements
			for l in range(2):
				i = random.randrange(1, len(ind.path))
				j = random.randrange(1, len(ind.path))

				temp = ind.path[i]
				ind.path[i] = ind.path[j]
				ind.path[j] = temp
		return

	# lambda + mu k-tournament elimination
	def eliminate(self, tsp, population, offspring):
		combined = np.concatenate([population, offspring])

		while combined.size > self.lambdaa:

			selected = np.random.choice(combined, self.k2, False)

			values = list(map(lambda a: tsp.length(a), selected))

			maxIndex = values.index(max(values))

			combined = np.delete(combined, maxIndex)

		return combined


class r0722871:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		tsp = TravelingSalespersonProblem(distanceMatrix)
		solver = Solver(tsp)

		numberOfIterations = 0
		while(True):
			# Your code here.
			# Recombination
			offspring = np.empty(solver.mu, Individual)
			for j in range(solver.mu):
				p1 = solver.select(tsp)
				p2 = solver.select(tsp)

				offspring[j] = solver.recombine(tsp, p1, p2)
				solver.mutate(offspring[j])

			# Mutation
			for j in range(solver.lambdaa):
				solver.mutate(solver.population[j])

			# Elimination
			solver.population = solver.eliminate(tsp, solver.population, offspring)

			# Calculate statistics
			objectives = []

			for j in range(len(solver.population)):
				objectives.append(tsp.length(solver.population[j]))

			meanObjective = np.mean(objectives)
			bestObjective = np.min(objectives)
			bestIndex = objectives.index(bestObjective)
			bestSolution = solver.population[bestIndex].path

			print(numberOfIterations, "Mean length: ", meanObjective, " Best fitnesses: ", bestObjective)
			numberOfIterations += 1
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

		# Your code here.
		return 0


test = r0722871()

test.optimize('tour29.csv')
