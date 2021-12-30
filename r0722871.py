import copy
import math
import random
from statistics import mean
from scipy.stats import expon
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

	def pathLength(self, path):
		length = 0
		for i in range(self.dimension[0] - 1):
			length += self.cost[path[i]][path[i + 1]]
		length += self.cost[path[self.dimension[0] - 1]][path[0]]
		return length

	def subPathLength(self, path):
		length = 0
		for i in range(len(path) - 1):
			length += self.cost[path[i]][path[i + 1]]
		return length


# Candidate solution representation
class Individual:
	def __init__(self, tsp, alpha=np.random.normal(0.2, 0.05), beta=np.random.normal(0.2, 0.05), k2=3):
		# Initialize path
		self.path = np.random.permutation(range(1, tsp.dimension[0]))
		self.path = np.concatenate(([0], self.path))

		# Initialize parameters
		self.k2 = k2
		self.k2 = max(self.k2, 2)
		self.k2 = min(self.k2, 10)

		self.alpha = alpha
		self.alpha = max(self.alpha, 0.05)
		self.alpha = min(self.alpha, 0.5)
		# print(alpha)

		self.beta = beta
		self.beta = max(self.beta, 0.05)
		self.beta = min(self.beta, 0.5)

		# self.mutationStep = mutation_step
		# self.mutationStep = max(self.mutationStep, 1)
		# self.mutationStep = min(self.mutationStep, tsp.maxMutationStep)

	def print(self):
		print("Ind Path: ", self.path)


def nearest_neighbor(ind, tsp):
	path = np.empty(tsp.dimension[0], dtype=int)
	path[0] = 0
	availableCities = list(range(1, tsp.dimension[0]))
	for i in range(1, len(path)):
		best = tsp.cost[path[i-1]][availableCities[0]]
		nextCity = availableCities[0]
		for j in availableCities:
			if tsp.cost[path[i-1]][j] < best:
				best = tsp.cost[path[i-1]][j]
				nextCity = j
		path[i] = nextCity
		availableCities.remove(nextCity)
	ind.path = path


def two_opt(ind, tsp):
	path = list(ind.path)
	best = path
	for i in range(1, tsp.dimension[0] - 2):
		# print('-----')
		currentSubPathLength = tsp.subPathLength(best[i:i+2])
		currentNewSubPathLength = tsp.subPathLength(path[(i+2) - 1:i - 1:-1])
		for j in range(i + 2, tsp.dimension[0]):
			newPath = path[:]
			newPath[i:j] = path[j - 1:i - 1:-1]
			currentSubPathLength += tsp.cost[path[j-1]][path[j]]
			currentNewSubPathLength += tsp.cost[newPath[i]][newPath[i+1]]
			# print(currentSubPathLength)
			if currentNewSubPathLength < currentSubPathLength:
				# print('a')
				best = newPath
			# else:
			# 	print('b')
	return np.array(best)


def lso_insert(ind, tsp):
	path = list(ind.path)
	best = path.copy()
	index = random.randrange(1, tsp.dimension[0])
	city = path.pop(index)
	for i in range(1, tsp.dimension[0]):
		copyPath = path.copy()
		copyPath.insert(i, city)
		if tsp.pathLength(copyPath) < tsp.pathLength(best):
			best = copyPath
	ind.path = np.array(best)


def hamming_distance(ind1, ind2):
	path1 = ind1.path
	path2 = ind2.path
	distance = 0
	for i in range(len(path1)):
		if path1[i] != path2[i]:
			distance += 1
	return distance


class Solver:
	def __init__(self, tsp):
		self.lambdaa = 200			# Population size
		self.mu = 200				# Offspring size
		# self.k = 14					# Tournament selection
		self.k2 = 3					# K-top elimination
		self.eliteSize = 5			# Number of elite candidate solutions that goes to next iteration
		self.intMax = 500			# Boundary of the domain, not intended to be changed.
		self.nbIterations = 100			# Maximum number of iterations

		self.defaultMutationStep = int(math.log(tsp.dimension[0], 5)**2)
		# tsp.maxMutationStep = self.defaultMutationStep * 2

		# Initialize population
		self.population = np.empty(self.lambdaa, Individual)
		for i in range(self.lambdaa):
			alpha = np.random.normal(0.2, 0.05)
			beta = np.random.normal(0.2, 0.05)
			self.population[i] = Individual(tsp, alpha, beta)
			if random.random() < 0.05:
				lso_insert(self.population[i], tsp)
			# 	nearest_neighbor(self.population[i], tsp)
				# self.inversionMutation(self.population[i], 10)
		if tsp.dimension[0] > 50:
			nearest_neighbor(self.population[0], tsp)

		# for i in range(200):
		# 	print(i)
		# 	self.population[i].path = two_opt(self.population[i], tsp)

	# Rank candidates
	def rankCandidates(self, tsp):
		candidates = self.population
		# lengths = {}
		# for i in range(0, len(candidates)):
		# 	lengths[i] = tsp.length(candidates[i])

		l = list(candidates)
		l.sort(key=lambda x: tsp.length(x))
		combinedSorted = np.array(l)

		return combinedSorted

	# k-Tournament selection
	def select(self, ranked_population, iteration, tsp):
		result = []

		# Set up probability distribution function
		pdf = []
		a = math.log(0.99**iteration)/(len(ranked_population) - 1)
		for i in range(1, len(ranked_population)):
			pdf.append(math.exp(a*(i - 1)))

		# Elitism
		# print(ranked_population[0].alpha)
		# print(ranked_population[0].beta)
		for i in range(0, self.eliteSize):
			if random.random() < ranked_population[i].beta or tsp.dimension[0] < 50:
				two_opt(ranked_population[i], tsp)
			lso_insert(ranked_population[i], tsp)
			result.append(ranked_population[i])

		# Select other candidates
		ranked_population = list(ranked_population)
		for i in range(0, len(ranked_population) - self.eliteSize):
			myRandom = random.random()
			for index in range(0, len(ranked_population)):
				# print(pdf[index+20])
				if myRandom < pdf[index]:
					result.append(ranked_population.pop(index))
					break

		return result

	# def selectOld(self, tsp):
	# 	selected = np.random.choice(self.population, self.k, False)
	#
	# 	values = list(map(tsp.length, selected))
	#
	# 	minIndex = values.index(min(values))
	#
	# 	return selected[minIndex]

	# OX crossover
	def recombine(self, tsp, p1, p2):
		childP1Path = []

		# Choose crossover points
		geneA = 1 + int(random.random() * len(p1.path))
		geneB = 1 + int(random.random() * len(p1.path))
		startGene = min(geneA, geneB)
		endGene = max(geneA, geneB)
		# print(geneA)

		for i in range(startGene, endGene):
			childP1Path.append(p1.path[i])

		childP2Path = [item for item in p2.path if item not in childP1Path]

		childPath = childP1Path + childP2Path

		childInd = Individual(tsp, mean([p1.alpha, p2.alpha]), mean([p1.beta, p2.beta]), round(mean([p1.k2, p2.k2])))
		# print(mean([p1.alpha, p2.alpha]))

		childPath = np.array(childPath)
		# Shift path so city 0 is in front
		# childPath = np.roll(childPath, len(childPath) - np.where(childPath == 0)[0])
		childInd.path = childPath

		# print(childInd.mutationStep)
		# print(childPath)
		# print(childInd.k2)
		return childInd

	# # NWOX crossover
	# def recombine(self, tsp, p1, p2):
	# 	n = tsp.dimension[0]
	# 	a = random.randint(0, n)
	# 	b = random.randint(a, n)
	# 	x1, x2 = list(p1.path), list(p2.path)
	# 	y1 = []
	# 	for i in range(0, n):
	# 		if x1[i] not in x2[a:b]:
	# 			y1.append(x1[i])
	# 	y1 = y1[:a] + x2[a:b] + y1[a:]
	#
	# 	childInd = Individual(tsp, mean([p1.alpha, p2.alpha]), round(mean([p1.k2, p2.k2])))
	#
	# 	childPath = np.array(y1)
	# 	# Shift path so city 0 is in front
	# 	# childPath = np.roll(childPath, len(childPath) - np.where(childPath == 0)[0])
	# 	childInd.path = childPath
	#
	# 	return childInd

	# PMX crossover
	# def recombineOld(self, tsp, p1, p2):
	# 	# https://www.youtube.com/watch?v=ZtaHg1C25Kk
	# 	a = random.randint(1, tsp.dimension[0] - 1)
	# 	b = random.randint(1, tsp.dimension[0] - 1)
	#
	# 	if a > b:
	# 		temp = a
	# 		a = b
	# 		b = temp
	#
	# 	childPath = np.zeros(tsp.dimension[0], dtype=int)
	#
	# 	# 1. Choose a random segment and copy it from p1 to child
	# 	childPath[a:b] = p1.path[a:b]
	#
	# 	i = []
	# 	j = []
	#
	# 	# 2. Starting from the first crossover point, look for elements in that swgment that have not been copied
	# 	for x in range(a, b):
	# 		if p2.path[x] not in p1.path[a:b]:
	# 			i.append(p2.path[x])
	#
	# 	# 3. For each of these i, look in the offspring to see whtat element j has been copied in its place from p1
	# 	for x in range(len(i)):
	# 		index = np.where(p2.path == i[x])[0][0]
	# 		j.append(p1.path[index])
	#
	# 		# 4. Place i into the position occuped by j in p2, since we will not be putting j there (as j is already in the offspring)
	# 		index = np.where(p2.path == j[x])[0][0]
	#
	# 		if childPath[index] == 0:
	# 			childPath[index] = i[x]
	# 		else:
	# 			# 5. If the place occuped by j in p2 has already been filled in the offspring by k, put i in the positioin occuped by k in p2
	# 			kIndex = index
	# 			while(True):
	# 				k = p1.path[kIndex]
	# 				kIndex = np.where(p2.path == k)[0][0]
	# 				if childPath[kIndex] == 0:
	# 					childPath[kIndex] = i[x]
	# 					break
	#
	# 	# 6. Having dealt with the elements from the crossover segment, the rest of the offspring can be filled from p2
	# 	for x in range(1, tsp.dimension[0]):
	# 		if childPath[x] == 0:
	# 			childPath[x] = p2.path[x]
	#
	# 	ind = Individual(tsp)
	# 	ind.path = childPath
	#
	# 	return ind

	def mutate(self, ind):
		if random.random() < ind.alpha:
			# Mutate elimination parameter (self-adaptivity)
			ind.k2 += round(np.random.normal(loc=0.0, scale=1))
			ind.k2 = max(ind.k2, 2)
			ind.k2 = min(ind.k2, 10)

			# Mutate mutation parameters (self-adaptivity)
			ind.alpha += np.random.normal(loc=0.0, scale=0.1)
			ind.alpha = max(ind.alpha, 0.05)
			ind.alpha = min(ind.alpha, 0.5)

			# Mutate lso parameter (self-adaptivity)
			ind.beta += np.random.normal(loc=0.0, scale=0.1)
			ind.beta = max(ind.beta, 0.05)
			ind.beta = min(ind.beta, 0.5)

			# ind.mutationStep += round(np.random.normal(loc=0.0, scale=1))
			# ind.mutationStep = max(ind.mutationStep, 1)
			# ind.mutationStep = min(ind.mutationStep, tsp.maxMutationStep)

			mutationStep = int(np.around(expon.rvs(loc=self.defaultMutationStep, scale=1)))

			# Inverse mutation: reverse order of sub path
			self.inversionMutation(ind, mutationStep)

			# Randomly swap two elements
			# for l in range(5):
			# 	i = random.randrange(1, len(ind.path))
			# 	j = random.randrange(1, len(ind.path))
			#
			# 	temp = ind.path[i]
			# 	ind.path[i] = ind.path[j]
			# 	ind.path[j] = temp
		return

	def inversionMutation(self, ind, mutation_step):
		i = random.randrange(1, len(ind.path) - mutation_step - 1)
		j = i + mutation_step

		ind.path[i:j] = ind.path[i:j][::-1]

	# lambda + mu elimination
	def eliminateNew(self, tsp, population, offspring):
		combined = np.concatenate([population, offspring])

		# combinedSorted = combined[np.apply_along_axis(lambda l: tsp.length(l), 0, combined).argsort()]
		# np.sorted(combined, key=lambda x: tsp.length(x))

		l = list(combined)
		l.sort(key=lambda x: tsp.length(x))
		combinedSorted = np.array(l)
		# combinedSorted = sorted(combined, key=lambda x: tsp.length(x))
		result = combinedSorted[0:self.lambdaa]

		# while combined.size > self.lambdaa:
		#
		# 	selected = np.random.choice(combined, self.k2, False)
		#
		# 	values = list(map(lambda a: tsp.length(a), selected))
		#
		# 	maxIndex = values.index(max(values))
		#
		# 	combined = np.delete(combined, maxIndex)

		return result

	# lambda + mu k-tournament elimination
	def eliminate(self, tsp, population, offspring, best):
		combined = np.concatenate([population, offspring])
		combined = list(combined)
		while len(combined) > self.lambdaa:

			selected = np.random.choice(combined, self.k2, False)

			values = list(map(lambda a: tsp.length(a), selected))
			maxIndex = values.index(max(values))
			combined.remove(selected[maxIndex])

			# # Crowding
			# closestInd = self.crowding(selected[maxIndex], combined, tsp)
			# combined.remove(closestInd)
		combined.insert(0, best)
		# print("tijdens elim: ", tsp.length(best))
		# print("tijdens elim: ", tsp.length(combined[0]))
		return np.array(combined)

	def crowding(self, ind, population, tsp):
		closestInd = None
		shortestDistance = tsp.dimension[0] + 1
		inds = np.random.choice(population, self.k2, False)
		for i in inds:
			if hamming_distance(ind, i) < shortestDistance:
				shortestDistance = hamming_distance(ind, i)
				closestInd = i
		return closestInd


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
		distanceMatrix[distanceMatrix > 1e308] = 100000000000
		tsp = TravelingSalespersonProblem(distanceMatrix)
		solver = Solver(tsp)
		# for j in range(solver.lambdaa):
		# 	for i in range(tsp.dimension[0] - 2):
		# 		if tsp.cost[solver.population[j].path[i]][solver.population[j].path[i + 1]] > tsp.cost[solver.population[j].path[i + 1]][solver.population[j].path[i]]:
		# 			temp = solver.population[j].path[i]
		# 			solver.population[j].path[i] = solver.population[j].path[i + 1]
		# 			solver.population[j].path[i + 1] = temp

		iteration = 0
		while True:
			# Your code here.

			# Recombination
			offspring = []
			rankedPopulation = solver.rankCandidates(tsp)
			selected = solver.select(rankedPopulation, iteration, tsp)
			best = copy.deepcopy(selected[0])
			# print("voor elim", tsp.length(best))
			# for i in selected:
			# 	print("selected: ", i)

			for j in range(solver.mu):
				p1 = selected[j]
				p2 = selected[len(selected)-j-1]

				offspring.append(solver.recombine(tsp, p1, p2))
				# if random.random() < 0.05:
				# 	lso_insert(offspring[j], tsp)
				solver.mutate(offspring[j])
			offspring = np.array(offspring)

			# Mutation
			for j in range(solver.lambdaa):
				solver.mutate(solver.population[j])
				# if random.random() < 0.1:
				# 	solver.population[j].path = two_opt(solver.population[j], tsp)

			# Elimination
			# print("vlak voor elim", tsp.length(best))
			solver.population = solver.eliminate(tsp, solver.population, offspring, best)
			# print("na elim", tsp.length(solver.population[0]))

			# Calculate statistics
			objectives = []

			for j in range(len(solver.population)):
				objectives.append(tsp.length(solver.population[j]))

			meanObjective = np.mean(objectives)
			bestObjective = np.min(objectives)
			bestIndex = objectives.index(bestObjective)
			bestSolution = solver.population[bestIndex].path

			print(iteration, "Mean length: ", meanObjective, " Best fitnesses: ", bestObjective)
			iteration += 1
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

test.optimize('tour250.csv')
