import optparse import OptionParser
from datagenerator import getData
import numpy as np
from ptrNet import PointerNetwork
# TODO: Trainable initial state might be needed
# TODO: Implement glimpses
# TODO: 
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
	indices = tf.argmax(targets, 1)
	first_ind = tf.range()
	indices = tf.reshape(indices, tf.shape(inputs)[:2])
	retun tf.gather_nd(inputs, indices)

def model_string(options):
	model_str = 
	return model_str

def buildModel(options):
	# most probably will have to set up the visual model first but let's make sure that the ptr net works! 
	ptr_net = PointerNetwork(options.max_len, options.input_dim, options.hidden_dim, options.bidirectional)
	inputs, seq_lens, targets = ptr_net.get_input_placeholders() # will change this to visual model input once time permits
	attention_states, encoder_end_states = ptr_net.encode(inputs, seq_lens)

	with tf.variable_scope("decoder"):
		# need to transform inputs
		train_inputs = get_corrected_inputs(attention_states, targets, seq_lens) # reshuffles the inputs to be the correct shape
		outputs, final_state, final_context_state = ptr_net.decode(train_inputs, targets, attention_states, encoder_end_states, seq_lens, train=1)

	with tf.variable_scope("decoder", reuse=True):
		predictions, final_state, final_context_state = ptr_net.decode(inputs, attention_states, encoder_end_states, seq_lens, train=0)

	outputs = tf.reshape(outputs, [-1, options.max_len])
	loss = tf.reduce_mean(tf.softmax_crossentropy_with_logits(targets, outputs))
	optimizer = getOptimizer(optimizer)
	train_op = optimizer.minimize(loss)
	return inputs, targets, predictions, loss, train_op



def train(data, options):
	model_str = model_string(options)
	x_train, y_train, train_seq_lens = data['train']
	x_val , y_val, val_seq_lens  = data["val"]

	init = tf.global_varialbles_initializer()
	input_x, input_y, predictions, loss, train_op = buildModel()

	merged = tf.merge_all_summaries()
	saver = tf.train.saver()
	with tf.Session() as sess:
		train_writer = tf.train.SummaryWriter(options.log_dir + "/" + model_str + "/train", sess.graph())
		test_writer = tf.train.SummaryWriter(options.log_dir + "/" + model_str + "/test", sess.graph())
		sess.run(init)
		if options.load_ckpt:
			print "Loading from checkpoint..."
            saver.restore(sess, options.ckpt_dir + "/" + model_str + "/model.ckpt")
        print "Training Network"
		epoch_losses = []
		for epoch in xrange(options.nb_epoch):
			epoch_loss = 0.0
			num_batches = len(x_train)
			permutation = np.random.permutation(num_batches)
			x_train, y_train, train_seq_lens = x_train[permutation], y_train[permutation], train_seq_lens[permutation]
			x_val, y_val, val_seq_lens = x_val[permutation], y_val[permutation], val_seq_lens[permutation]

			for i in xrange(num_batches):
				curr_iter = (epoch * num_batches + i)

				feed_dict = {input_x : x_train[i], input_y : x_train[i]}
				summary, loss, _ = sess.run([merged, loss, train_op], feed_dict = feed_dict)
				train_writer.add_summary(summary, curr_iter)
				if curr_iter % options.iter_per_chkpt == 0:
					saver.save(sess, options.ckpt_dir + "/" + model_str + "/model.ckpt")

				feed_dict = {input_x : x_val[i], input_y : x_val[i]}
				summary, val_loss = sess.run([merged, loss], feed_dict = feed_dict)
				test_writer.add_summary(summary, curr_iter)

				print("Epoch :  ", epoch, " Batch : " , i,  ", Minibatch Train Loss= " + "{:.6f}".format(loss))
				print("Epoch :  ", epoch, " Batch : " , i,  ", Minibatch Val Loss= " + "{:.6f}".format(val_loss))
				print("-----------------------------")
				epoch_loss += loss
			epoch_losses.append(epoch_loss/num_batches)
	return epoch_losses





if __name__ == "__main__":
	parser = OptionParser()
	parser.add_option("--batch_size", type=int, dest='batch_size', default=64)
	parser.add_option("--epochs", type=int, dest="nb_epoch", default=20)
	parser.add_option("--lr", type=float, dest="lr", default=1e-3)
	parser.add_option("--optim", type=string, dest="optim", default="Adam")
	parser.add_option("--bidirect", actions="store_true", dest="bidirectional", default=True)
	parser.add_option("--puzzle_width", type=int, dest="puzzle_width", default=5)
	parser.add_option("--puzzle_height", type=int, dest="puzzle_height", default=5)
	parser.add_option("--log_dir", type=string, dest="log_dir", default="logs")
	parser.add_option("--ckpt_dir", type=string, dest="ckpt_dir", default="checkpoints")
	parser.add_option("--load_ckpt", type=string, dest="load_ckpt", default="")
	parser.add_option("--iter_per_chkpt", type=int, dest="iter_per_chkpt", default="10")
	parser.add_option("--max_len", type=int, dest="max_len", default=25)
	parser.add_option("--ptrnet_input_dim", type=int, dest="input_dim", default=100)
	parser.add_option("--ptrnet_hidden_dim", type=int, dest="hidden_dim", default=100)

	options, args = parser.parse_args()

	data = getData(options.puzzle_height, options.puzzle_width, options.batch_size)
	train(data, options)
	# test(data)