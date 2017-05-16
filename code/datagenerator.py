import numpy as np

def getData(puzzle_height, puzzle_width, batch_size):
	'''
	returns data : such that data['val'] = (x_val, y_val, val_seq_len)
							data['train'] = (x_train, y_train, train_seq_len)
							data['test'] = (x_test, y_test, test_seq_len) 
							# x_train = [batch1, batch2, ... batchn] - each batch should be the image
							# seq_len = [batch_size, ] # for each image in that batch, the number of pieces it is cut into
	making toy dataset 
	'''
	data = {}
	x_train, y_train, train_seq = [], [], []
	for i in range(batch_size):
		perm = np.random.permutation(puzzle_width*puzzle_height)
		x_train.append([[j] for j in perm])
		y = np.zeros((puzzle_width*puzzle_height, puzzle_width*puzzle_height))
		y[range(len(perm)), perm] = 1
		y_train.append(y)
		train_seq.append(puzzle_width*puzzle_height)
	x_train, y_train, train_seq = np.array(x_train), np.array(y_train), np.array(train_seq)

	x_val, y_val, val_seq = [], [], []
	for i in range(batch_size):
		perm = np.random.permutation(puzzle_width*puzzle_height)
		x_val.append([[j] for j in perm])
		y = np.zeros((puzzle_width*puzzle_height, puzzle_width*puzzle_height))
		y[range(len(perm)), perm] = 1
		y_val.append(y)
		val_seq.append(puzzle_width*puzzle_height)
	x_val, y_val, val_seq = np.array(x_val), np.array(y_val), np.array(val_seq)

	data['train'] = (x_train, y_train, train_seq)
	data['val'] = (x_val, y_val, val_seq)
	return data

# res = getData(2, 2, 5)
# res = res['train']
# print res[2].shape, res[2]
# print res[1].shape, res[1]
# print res[0].shape, res[0]
