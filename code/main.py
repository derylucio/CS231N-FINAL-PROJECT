from optparse import OptionParser
from datagenerator import getData
import numpy as np
from ptrNet import PointerNetwork
import tensorflow as tf
# TODO: Trainable initial state might be needed
# TODO: Implement glimpses

def getOptimizer(options):
	lr = options.lr  # might want to do more complicated things 
	if options.optim == "Adam":
		return tf.train.AdamOptimizer(learning_rate=lr)
	elif options.optim == "RMSProp":
		return tf.train.RMSPropOptimizer(learning_rate=lr)
	elif options.optim == "Adadelta":
		return tf.train.AdadeltaOptimizer(learning_rate=lr)
	else:
		return tf.train.GradientDescentOptimizer(learning_rate=lr)

def get_corrected_inputs(inputs, targets, seq_lens):
	indices = tf.cast(tf.argmax(targets, 1),  tf.int32)
	# modified from https://github.com/devsisters/pointer-network-tensorflow/blob/master/layers.py
	replicated_first_indices = tf.range(tf.shape(indices)[0])
	rank = len(indices.get_shape())
	if rank == 2:
	    replicated_first_indices = tf.tile(
	        tf.expand_dims(replicated_first_indices, dim=1),
	        [1, tf.shape(indices)[1]])
	nd_indices = tf.stack([replicated_first_indices, indices], axis=rank)
	return tf.stop_gradient(tf.gather_nd(inputs, nd_indices))

def model_string(options):
	model_str = 'will_do_this_later'
	return model_str


def buildModel(options):
	
	# most probably will have to set up the visual model first but let's make sure that the ptr net works! 
	ptr_net = PointerNetwork(options.max_len, options.input_dim, options.hidden_dim, options.bidirectional, options.fc_dim, options.batch_size, options.inter_dim)
	placeholder_inputs, seq_lens, targets = ptr_net.get_input_placeholders(options.use_cnn) # will change this to visual model input once time permits
	with tf.variable_scope('encoder'):
		if options.use_cnn:
			inputs, attention_states, encoder_end_states, inputfn = ptr_net.encode(placeholder_inputs, seq_lens, process='CNN', is_training=True)
		else: 
			inputs, attention_states, encoder_end_states = ptr_net.encode(placeholder_inputs, seq_lens, process='FC', is_training=True)
	

	with tf.variable_scope("decoder"):
		# need to transform inputs
		train_inputs = get_corrected_inputs(inputs, targets, seq_lens) # reshuffles the inputs to be the correct shape
		outputs, final_state  = ptr_net.decode(train_inputs, attention_states, encoder_end_states, seq_lens, train=1)

	with tf.variable_scope("decoder", reuse=True):
		# if options.use_cnn:
		# 	predictions, final_state = ptr_net.decode(test_inputs, test_attention_states, test_encoder_end_states, seq_lens, train=0)
		# else:
			predictions, final_state = ptr_net.decode(inputs, attention_states, encoder_end_states, seq_lens, train=0)
	outputs = tf.reshape(outputs, [-1, options.max_len])
	targets_flat = tf.reshape(targets, [-1, options.max_len])
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets_flat))

	l2 = options.reg * sum(
	    tf.nn.l2_loss(tf_var)
	        for tf_var in tf.trainable_variables()
	        if not ("Bias" in tf_var.name)
	)
	loss += l2

	optimizer = getOptimizer(options)
	train_op = optimizer.minimize(loss)

	flat_pred = tf.reshape(predictions, [-1, options.max_len])
	val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_pred, labels=targets_flat))

	if options.use_cnn:
		return placeholder_inputs, targets, seq_lens, predictions, val_loss, loss, train_op, inputfn
	return placeholder_inputs, targets, seq_lens, predictions, val_loss, loss, train_op



def train(data, options):
	tf.reset_default_graph()
	model_str = model_string(options)
	x_train, y_train, train_seq_lens = data['train'] 
	x_val , y_val, val_seq_lens  = data["val"]
	X_test, y_test, test_seq_lens = data["test"]

	if options.use_cnn:
		input_x, input_y, seq_lens,  predictions, val_loss, loss, train_op, inputfn = buildModel(options)
	else:
		input_x, input_y, seq_lens,  predictions, val_loss, loss, train_op = buildModel(options)

	# merged = tf.summary.merge_all()
	saver = tf.train.Saver()
	sess = tf.Session()
	with sess.as_default():
		# train_writer = tf.summary.FileWriter(options.log_dir + "/" + model_str + "/train", sess.graph())
		# test_writer = tf.summary.FileWriter(options.log_dir + "/" + model_str + "/test", sess.graph())
		init = tf.global_variables_initializer()
		sess.run(init)
		if options.use_cnn: sess.run(inputfn)
		if options.load_ckpt:
			print "Loading from checkpoint..."
			saver.restore(sess, options.ckpt_dir + "/" + model_str + "/model.ckpt")
        print "Training Network"
        epoch_losses = []
        for epoch in xrange(options.nb_epoch):
			epoch_loss = 0.0
			num_batches = len(x_train) / options.batch_size
			permutation = np.random.permutation(len(x_train))
			print x_train.shape, y_train.shape, type(train_seq_lens), train_seq_lens.shape
			x_train, y_train, train_seq_lens = x_train[permutation], y_train[permutation], train_seq_lens[permutation]
			# x_val, y_val, val_seq_lens = x_val[permutation], y_val[permutation], val_seq_lens[permutation]
			for i in xrange(num_batches):
				curr_iter = (epoch * num_batches + i)
				batch_x = x_train[(i*options.batch_size) : (i + 1)*options.batch_size]
				batch_y = y_train[(i*options.batch_size) : (i + 1)*options.batch_size]
				batch_seq = train_seq_lens[(i*options.batch_size) : (i + 1)*options.batch_size]

				feed_dict = {input_x : batch_x, input_y : batch_y, seq_lens : batch_seq}
				loss_val, _ = sess.run([loss, train_op], feed_dict = feed_dict)
				# train_writer.add_summary(summary, curr_iter)
				if curr_iter % options.iter_per_chkpt == 0:
					saver.save(sess, options.ckpt_dir + "/" + model_str + "/model.ckpt")

				print("Epoch :  ", epoch, " Batch : " , i,  ", Minibatch Train Loss= " + "{:.6f}".format(loss_val))
				epoch_loss += loss_val

			feed_dict = {input_x : x_val, input_y : y_val, seq_lens : val_seq_lens}
			v_loss = sess.run([val_loss], feed_dict = feed_dict)[0]
			print("Epoch :  ", epoch, ", Val Loss= " + "{:.6f}".format(v_loss))
			print("-----------------------------")
			epoch_losses.append(epoch_loss/num_batches)

        feed_dict = {input_x : X_test, input_y : y_test, seq_lens : test_seq_lens}
        test_pred = sess.run([predictions], feed_dict = feed_dict)[0]

	return epoch_losses, test_pred





if __name__ == "__main__":
	parser = OptionParser()
	parser.add_option("--batch_size", type=int, dest='batch_size', default = 128)
	parser.add_option("--epochs", type=int, dest="nb_epoch", default=200)
	parser.add_option("--lr", type=float, dest="lr", default=1e-4) # TO DO : FIND BEST 
	parser.add_option("--optim", type=str, dest="optim", default="SGD") # TO DO : FIND BEST 
	parser.add_option("--bidirect", action="store_true", dest="bidirectional", default=True) # TO DO : FIND BEST 
	parser.add_option("--puzzle_width", type=int, dest="puzzle_width", default=7)
	parser.add_option("--puzzle_height", type=int, dest="puzzle_height", default=7)
	parser.add_option("--log_dir", type=str, dest="log_dir", default="logs")
	parser.add_option("--ckpt_dir", type=str, dest="ckpt_dir", default="checkpoints")
	parser.add_option("--load_ckpt", type=str, dest="load_ckpt", default="")
	parser.add_option("--iter_per_chkpt", type=int, dest="iter_per_chkpt", default=10)
	parser.add_option("--max_len", type=int, dest="max_len", default=49)
	parser.add_option("--ptrnet_input_dim", type=int, dest="input_dim", default=12288)
	parser.add_option("--ptrnet_hidden_dim", type=int, dest="hidden_dim", default=64) # TO DO : FIND BEST 
	parser.add_option("--fc_dim", type=int, dest='fc_dim', default=512) # TO DO : FIND BEST 
	parser.add_option("--inter_dim", type=int, dest='inter_dim', default=2048) # TO DO : FIND BEST 
	parser.add_option("--reg", type=float, dest='reg', default=0) # TO DO : FIND BEST 
	parser.add_option("--use_cnn", action="store_true", dest="use_cnn", default=True)
	options, args = parser.parse_args()

	print 'Loading Data'
	
	data = getData(options.puzzle_height, options.puzzle_width, options.batch_size, options.use_cnn)
	# import cPickle as pickle
	# with open(r'data_file', 'rb') as savef:
	# 	data = pickle.load(savef) #pickle.dump(data, savef) #

	print 'Beginning Training'
	_, test_pred = train(data, options)
	np.save('test_predictions', test_pred)
	# test(data)