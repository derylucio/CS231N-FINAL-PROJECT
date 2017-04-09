# Utils for Evaulating the Fitness function.
import numpy 
import image # Created this

# Returns the fitness score between two cells.
# TODO: Cache values. 
name = "man"
def getFitness(chromosome):

	rval, dval = 0.0, 0.0
	for i in xrange(len(chromosome[0])):		# N
		for j in xrange(len(chromosome) - 1):		# M - 1
			print i, j
			# print chromosome[i][j]
			rval += getRightDisimilarity(getPixelData(chromosome[i][j]), getPixelData(chromosome[i][j + 1]))
	for i in xrange(len(chromosome[0]) - 1):	# N - 1
		for j in xrange(len(chromosome)): 		# M
			dval += getDownDisimilarity(getPixelData(chromosome[i][j]), getPixelData(chromosome[i + 1][j]))
	return rval + dval

def getRightDisimilarity(left, right):
	result = 0.0
	width, height, depth = left.shape
	for k in xrange(height):
		for b in xrange(depth): 
			result += numpy.square(left[width - 1][k][b] - right[0][k][b])
	result = numpy.sqrt(result)
	# return sqrt(result)
	return result

def getDownDisimilarity(top, bottom):
	result = 0.0
	width, height, depth = top.shape
	for k in xrange(width):
		for b in xrange(depth): 
			result += numpy.square(top[k][height - 1][b] - bottom[k][0][b])
	result = numpy.sqrt(result)
	return numpy.sqrt(result)

def getPixelData(val):
	i, j = val
	print name + "_" + str(j).rjust(2, '0') + "_" + str(i).rjust(2, '0') + ".png"
	x = image.loadImageAsMatrix(name + "_" + str(j).rjust(2, '0') + "_" + str(i).rjust(2, '0') + ".png")
	return x

print "Fitness: " = getFitness([[(1,1), (1,2)],[(2,1), (2,2)]])