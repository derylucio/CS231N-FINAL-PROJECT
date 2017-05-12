''' 
TESTING
Sample Input: 
width = 4
x = xrange(1,width + 1)
pieces = [zip([i]*width, x) for i in range(1, width + 1)]
numRows, numCols = len(pieces), len(pieces[0])
ordering = list(xrange(1, numRows * numCols + 1))

Functions: 
visualize('face', pieces, ordering)
shuffled_pieces, ordering = pieces_shuffle(pieces)
print order_pieces(shuffled_pieces, ordering) #testing shuffle
'''

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import math 
from PIL import Image
import random
from scipy.misc import imread, imsave

outputExt = ".png"
previewSize = 600   
zeroPad = 2 		
maxWidth = 99

def getImageFilename(prefix, (i, j)):
	return prefix + "_" + str(j).rjust(zeroPad, '0') + "_" + \
						  str(i).rjust(zeroPad, '0') + outputExt
    
def mergeAndShowImage(prefix, permutation):
	width = len(permutation)
	reducedSize = previewSize / width
	imgWidth, imgHeight = reducedSize, reducedSize
	result = Image.new("RGB", (previewSize, previewSize))
	for i in xrange(len(permutation)):
		for j in xrange(len(permutation[i])):
			x = i * imgWidth
			y = j * imgHeight
			filename = getImageFilename(prefix, permutation[i][j])
			print "Getting filename %s:" % filename
			img = Image.open(filename)
			img.thumbnail((reducedSize, reducedSize), Image.ANTIALIAS)
			imgWidth, imgHeight = img.size
			print('pos {0},{1} size {2},{3}'.format(x, y, imgWidth, imgHeight))
			result.paste(img, (x, y, x + imgWidth, y + imgHeight))
	result.crop((0, 0, (imgWidth + 1) * width, (imgHeight + 1) * width))
	r_i = np.array(result)
	plt.imshow(r_i)
	plt.show()
    
# order pieces according to the ordering given. 
def order_pieces(pieces, ordering):
    assert len(pieces) > 0
    numRows, numCols = len(pieces), len(pieces[0])
    ordered = [[0] * numCols for _ in range(numRows)]
    for i, val in enumerate(ordering):
        val -= 1
        o_row, row = int(math.floor (i/float(numRows))), int(math.floor (val/float(numRows)))    
        o_col, col  = i % numCols, val % numCols
        ordered[row][col] = pieces[o_row][o_col]
    return ordered

# Input: prefix of image, permutation of pieces, and ordering. 
# Visualizes image by placing pieces[i] the position specified by 
# (order[i] / numRows, order[i] % numCols).
def visualize(prefix, pieces, ordering):
    assert len(pieces) > 0
    numRows, numCols = len(pieces), len(pieces[0])
    assert len(ordering) == numRows * numCols
    ordered_pieces = order_pieces(pieces, ordering)
    mergeAndShowImage("face", ordered_pieces)

# Shuffles pieces and return it with the corresponding ordering.  
def pieces_shuffle(pieces):
    assert len(pieces) > 0
    numRows, numCols = len(pieces), len(pieces[0])
    ordering = list(xrange(1, numRows * numCols + 1))
    
    dt=np.dtype('int,int')
    flat_pieces = np.array(pieces, dt).flatten().tolist()
    
    zipped = list(zip(flat_pieces, ordering))
    random.shuffle(zipped)
    shuffled_pieces, ordering = zip(*zipped)
    shuffled_pieces = list(shuffled_pieces)
    shuffled_pieces = np.array(shuffled_pieces, dt).reshape(numRows, numCols).tolist()
    return shuffled_pieces, list(ordering)