import numpy as np
import scipy.misc
import scipy.ndimage
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append('../utils/')
import fitness_vectorized as fv

NUM_TEST = 20
NUM_TRAIN = 80
NUM_VAL = 20
NUM_DATA = NUM_TEST + NUM_TRAIN + NUM_VAL
numRows, numCols = (3, 3)

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
	return data

def prepareDataset(X_flat, num_train, num_test, num_val):
	'''
	Splits and preprocessed dimension-formatted data into 
	train, test and validation data. 
	Returns:
	'''
	print("Preparing Dataset...")
	N, S, W, H, C = X_flat.shape
	xs = np.empty_like(X_flat)
	ys = np.zeros((N, S), dtype=np.uint8)
	ys += np.arange(S, dtype=np.uint8)

	np.random.shuffle(X_flat)
	for i in np.arange(X_flat.shape[0]):
		np.random.shuffle(ys[i])
		xs[i,:] = X_flat[i,ys[i]]

	X_train, X_val, X_test = np.split(xs, [num_train, num_train + num_val])
	y_train, y_val, y_test = np.split(ys, [num_train, num_train + num_val])
	print("Prepared Dataset!")
	return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


def generateImageData(N, H, W, dims=(100,100,3)):
	'''
	Prepares images from the data dir and returns an N*W*H*C numpy array.
	TODO: Add more transformations.   
	'''
	print("Generating Image Data...")
	imgList = []
	for imgName in sorted(os.listdir(DATA_DIR)):
		imgList.append(scipy.ndimage.imread(DATA_DIR + os.sep + imgName)) # W, H, C
		# if (len(imgList) > 1): break
	print("Loaded %d images from %s." % (len(imgList), DATA_DIR))
	print("Augmenting images by flipping.")
	imgListFlipped = [np.fliplr(img) for img in imgList] # Expensive. 
	print("Flipped %d images from %s." % (len(imgListFlipped), DATA_DIR))
	imgList.extend(imgListFlipped)

	X = np.array(imgList[:N]).astype(dtype=float)

	# TODO: Check this. How Numpy works with different varying sized arrays. 
	# X -= np.mean(X, axis=0)
	# X /= np.std(X, axis=0)

	X_arr = [] # np.zeros([NUM_DATA, H * W, 32, 32, 3])
	for i, img in enumerate(imgList):
		X_arr.append(np.array(fv.splitImage(H, W, img, dims)))
	assert(np.linalg.norm(X_arr[0]) > 0)
	print("Generated Data!")
	print(np.array(X_arr).shape)
	return np.array(X_arr)

def reassemble(data, numRows, numCols):
	print("Reassembling...")
	# Choose random val for train, test and val
	train_idx = np.random.randint(NUM_TRAIN)
	X_train0 = data['X_train'][train_idx]
	y_train0 = data['y_train'][train_idx]
	test_idx = np.random.randint(NUM_TEST)
	X_test0 = data['X_test'][test_idx]
	y_test0 = data['y_test'][test_idx]
	val_idx = np.random.randint(NUM_VAL)
	X_val0 = data['X_val'][val_idx]
	y_val0 = data['y_val'][val_idx]

	gs = gridspec.GridSpec(numRows, numCols)
	gs.update(wspace=0.0, hspace=0.0)
	ax = [plt.subplot(gs[i]) for i in np.arange(numRows * numCols)]

	# Print Train Image
	for i in np.arange(len(X_train)):
		idx = np.where(y == i)[0]
		img = x[idx][0,:]
		print x[idx]
		ax[i].axis('off')
		ax[i].imshow(img)
		ax[i].set_aspect('equal')
	plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
	plt.show()

# TEST
print("========= TESTING ==========")
numRows, numCols = 3, 3
data = getData(numRows, numCols).copy()
reassemble(data, numRows, numCols)
print("========= ALL TESTS PASS =======")
