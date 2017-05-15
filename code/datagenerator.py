import numpy as np
import scipy.misc
import scipy.ndimage
import os
import sys

sys.path.append('../utils/')
import fitness_vectorized as fv

NUM_TEST = 20
NUM_TRAIN = 20
NUM_VAL = 80
NUM_DATA = NUM_TEST+NUM_TRAIN+NUM_VAL
DATA_DIR = "../data"

def getData(puzzle_height, puzzle_width, batch_size=-1):
	'''
	returns data : such that data['val'] = (x_val, y_val, val_seq_len)
							data['train'] = (x_train, y_train, train_seq_len)
							data['test'] = (x_test, y_test, test_seq_len) 
							# x_train = [batch1, batch2, ... batchn] - each batch should be the image
							# seq_len = [batch_size, ] # for each image in that batch, the number of pieces it is cut int
	'''
	# TODO: Verify what puzzle_height, puzzle_width are.
	X_flat = generateImageData(NUM_DATA, puzzle_height, puzzle_width)
	data = prepareDataset(X_flat, NUM_TRAIN, NUM_TEST, NUM_VAL)
	assert(data['X_train'].shape == (20, 32, 32, 3))
	assert(data['X_val'].shape == (20, 32, 32, 3))
	assert(data['X_test'].shape == (80, 32, 32, 3))
	return data

def prepareDataset(X_flat, num_train, num_test, num_val):
	'''
	Splits and preprocessed dimension-formatted data into 
	train, test and validation data. 

	Returns:

	'''
	print("Preparing Dataset...")
	X_train, y_train = None, None
	X_test, y_test = None, None
	X_val, y_val = None, None
	return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


def generateImageData(N, H, W, dims=(32,32,3)):
	'''
	Prepares images from the data dir and returns an N*W*H*C numpy array.
	TODO: Add more transformations.   
	'''
	print("Generating Image Data...")
	imgList = []
	for imgName in sorted(os.listdir(DATA_DIR)):
		imgList.append(scipy.ndimage.imread(DATA_DIR + os.sep + imgName)) # W, H, C
		if (len(imgList) > 1): break
	print("Loaded %d images from %s." % (len(imgList), DATA_DIR))
	print("Augmenting images by flipping.")
	imgListFlipped = [np.fliplr(img) for img in imgList]
	print("Flipped %d images from %s." % (len(imgListFlipped), DATA_DIR))
	imgList.extend(imgListFlipped)

	X = np.array(imgList[:N]).astype(dtype=float)

	# TODO: Check this. How Numpy works with different varying sized arrays. 
	X -= np.mean(X, axis=0)
	X /= np.std(X, axis=0)

	X_flat = np.array((NUM_DATA, H * W, -1))
	for i, img in enumerate(imgList):
		X_flat[i] = fv.splitImage(H, W, img, dims)
	return X_flat


# TEST
print("========= TESTING ==========")
getData(1, 1)
# print(getImageData(1), len(getImageData(1)))
# assert(len(getImageData(1)) == 60)
print("========= ALL TESTS PASS =======")
