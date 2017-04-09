# Utils for manipulating images
import image_slicer 
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
from PIL import Image


extension = ".png"
previewSize = 600
zeroPad = 2 # TODO: Change depending on how many pieces


def loadImageAsMatrix(filename):
	image_data = imread(filename).astype(np.float32)
	print "Size,  ", image_data.size
	print "Shape, ", image_data.shape
	return image_data
	# scaled_image_data = image_data / 255 # imshow requires values between 0 and 1
	# plt.imshow(scaled_image_data)
	# plt.show()
	# exit()

def splitImage(filename, destination, width):
	# Assert the file exists
	print "Splitting into %d-by-%d tiles." % (width, width)
	tiles = image_slicer.slice(filename, width * width, save=False)
	basename = os.path.basename(filename)
	prefix = basename.split('.')[0]
	image_slicer.save_tiles(tiles, directory=destination, prefix=prefix)

def showImage(filename):
	im = Image.open(filename)
	im.show()


# TODO: Note functional -- hardcoded-values
def mergeAndDisplayImage(prefix, permutation_list, width):
	reducedSize = previewSize / width
	imgWidth, imgHeight = reducedSize, reducedSize
	result = Image.new("RGB", (previewSize, previewSize))
	for (i, j) in permutation_list:
		x = (i - 1) * imgWidth
		y = (j - 1) * imgHeight
		filename = prefix + "_" + str(j).rjust(2, '0') + "_" + str(i).rjust(2, '0') + extension
		print "Getting filename %s" % filename
		img = Image.open(filename)
		img.thumbnail((reducedSize, reducedSize), Image.ANTIALIAS)
		imgWidth, imgHeight = img.size
		print('pos {0},{1} size {2},{3}'.format(x, y, imgWidth, imgHeight))
		result.paste(img, (x, y, x + imgWidth, y + imgHeight))
	result.show()
	result.save("new_image.BMP")

# TESTING
# splitImage("man.jpeg", ".", 256)
# showImage("man.jpeg")
# mergeAndDisplayImage("man", [(1, 1), (1, 2), (2, 2), (2, 1)], 2)