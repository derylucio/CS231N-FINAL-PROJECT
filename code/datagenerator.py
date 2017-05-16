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
DIMS=(100,100,3)

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
	X_flat = generateImageData(NUM_DATA, puzzle_height, puzzle_width, dims=DIMS)
	data = prepareDataset(X_flat)
	return data

def prepareDataset(X_flat):
	'''
	Splits and preprocessed dimension-formatted data into 
	train, test and validation data. 
	Returns:
	'''
	print("Preparing Dataset...")
	print(X_flat.shape)
	N, L, W, H, C = X_flat.shape
	xs = np.empty_like(X_flat)
	ys = np.zeros((N, L), dtype=np.uint8)
	ys += np.arange(L, dtype=np.uint8)

	np.random.shuffle(X_flat)
	for i in np.arange(X_flat.shape[0]):
		np.random.shuffle(ys[i])
		xs[i,:] = X_flat[i,ys[i]]

	X_train, X_val, X_test = np.split(xs, [NUM_TRAIN, NUM_TRAIN + NUM_VAL])
	y_train, y_val, y_test = np.split(ys, [NUM_TRAIN, NUM_TRAIN + NUM_VAL])

	print("Prepared Flattened Dataset!")
	X_train = X_train.reshape(NUM_TRAIN, L, -1)
	X_val = X_val.reshape(NUM_VAL, L, -1)
	X_test = X_test.reshape(NUM_TEST, L, -1)

	# Create one-hot vectors of these arrays. 
	y_train_onehot = np.where(y_train[:,:,np.newaxis] == np.arange(L), 1, 0)
	y_val_onehot = np.where(y_val[:,:,np.newaxis] == np.arange(L), 1, 0)  
	y_test_onehot = np.where(y_test[:,:,np.newaxis] == np.arange(L), 1, 0)  

	return {
      'train': (X_train, y_train_onehot, train_seq),  
      'val'  : (X_val, y_val_onehot, val_seq), X_val,  
      'test' : (X_test, y_test_onehot, test_seq)
    }

def generateImageData(N, H, W, dims=(32,32,3)):
	'''
	Prepares images from the data dir and returns an N*W*H*C numpy array.
	TODO: Add more transformations.   
	'''
	print("Generating Image Data...")
	imgList = []
	for imgName in sorted(os.listdir(DATA_DIR)):
		imgList.append(scipy.ndimage.imread(DATA_DIR + os.sep + imgName)) 
		# if len(imgList) > 1: break
	print("Loaded %d images from %s." % (len(imgList), DATA_DIR))

	print("Augmenting images by flipping.")
	imgListFlipped = [np.fliplr(img) for img in imgList] # Expensive. 
	print("Flipped %d images from %s." % (len(imgListFlipped), DATA_DIR))
	imgList.extend(imgListFlipped)

	X_arr = []
	for i, img in enumerate(imgList):
		# TODO: Check this to confirm it's good. 
		img = img.astype(dtype=np.float64)
		np.subtract(img, np.mean(img), out=img, casting="safe")
		np.divide(img, np.std(img), out=img, casting="safe") 
		X_arr.append(np.array(fv.splitImage(H, W, img, dims)))
	print("Generated Data!")
	return np.array(X_arr, dtype=float)

def reassemble(data, numRows, numCols):
	print("Reassembling...")
	train_idx = np.random.randint(NUM_TRAIN)
	X_train0 = data['X_train'][train_idx]
	y_train0 = data['y_train'][train_idx]

	test_idx = np.random.randint(NUM_TEST)
	X_test0 = data['X_test'][test_idx]
	y_test0 = data['y_test'][test_idx]

	val_idx = np.random.randint(NUM_VAL)
	X_val0 = data['X_val'][val_idx]
	y_val0 = data['y_val'][val_idx]

	xs = [X_train0, X_test0, X_val0]
	ys = [y_train0, y_test0, y_val0]

	for i in np.arange(3):
		x, y = xs[i], ys[i]
		plt.figure(i)
		gs = gridspec.GridSpec(numRows, numCols)
		gs.update(wspace=0.0, hspace=0.0)
		ax = [plt.subplot(gs[i]) for i in np.arange(numRows * numCols)]

		for i in np.arange(len(x)):
			assert(sum(y[i]) == 1)
			idx = np.where(y[i] == 1)[0]
			print i, idx
			img = x[i].reshape(DIMS)
			ax[int(idx)].axis('off')
			ax[int(idx)].imshow(img)
			ax[int(idx)].set_aspect('equal')
	
	plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
	plt.show()

# TEST
print("========= TESTING ==========")
data = getData(numRows, numCols) 
for n, d in data.items():
	print n, d.shape
reassemble(data, numRows, numCols)
print("========= ALL TESTS PASS =======")
