import optparse import OptionParser
from datagenerator import getData
import numpy as np

def getOptimizer(options):
	lr = options.lr 
	if options.optim == "Adam":
		return tf.train.AdamOptimizer(learning_rate=lr)
	elif options.optim == "RMSProp":
		return tf.train.RMSPropOptimizer(learning_rate=lr)
	elif options.optim == "Adadelta":
		return tf.train.AdadeltaOptimizer(learning_rate=lr)
	else:
		return tf.train.GradientDescentOptimizer(learning_rate=lr)

def model_string(options):

def createFeedDict():

def train(data, options):
	model_str = model_string(options)
	x_train, y_train = data['train']
	x_val , y_val = data["val"]

	init = tf.global_varialbles_initializer()
	input_x, input_y = #
	loss = #
	acc = #
	predictions = #

	optimizer = getOptimizer(optimizer)
	train_op = optimizer.minimize(loss)

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
			x_train, y_train = x_train[permutation], y_train[permutation]
			x_val, y_val = x_val[permutation], y_val[permutation]

			for i in xrange(num_batches):
				curr_iter = (epoch * num_batches + i)

				feed_dict = {input_x : x_train[i], input_y : x_train[i]}
				summary, loss, acc, _ = sess.run([merged, loss, acc, train_op], feed_dict = feed_dict)
				train_writer.add_summary(summary, curr_iter)
				if curr_iter % options.iter_per_chkpt == 0:
					saver.save(sess, options.ckpt_dir + "/" + model_str + "/model.ckpt")

				feed_dict = {input_x : x_val[i], input_y : x_val[i]}
				summary, val_loss, val_acc= sess.run([merged, loss, acc], feed_dict = feed_dict)
				test_writer.add_summary(summary, curr_iter)

				print("Epoch :  ", epoch, " Batch : " , i,  ", Minibatch Train Loss= " + \
	              "{:.6f}".format(loss) + ", Training Accuracy= " + \
	              "{:.5f}".format(acc))
				print("Epoch :  ", epoch, " Batch : " , i,  ", Minibatch Val Loss= " + \
	              "{:.6f}".format(val_loss) + ", Val Accuracy= " + \
	              "{:.5f}".format(val_acc))
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
	parser.add_option("--bidirect", actions="store_true", default=True)
	parser.add_option("--puzzle_width", type=int, dest="puzzle_width", default=5)
	parser.add_option("--puzzle_height", type=int, dest="puzzle_height", default=5)
	parser.add_option("--log_dir", type=string, dest="log_dir", default="logs")
	parser.add_option("--ckpt_dir", type=string, dest="ckpt_dir", default="checkpoints")
	parser.add_option("--load_ckpt", type=string, dest="load_ckpt", default="")
	parser.add_option("--iter_per_chkpt", type=int, dest="iter_per_chkpt", default="10")


	options, args = parser.parse_args()

	data = getData(options.puzzle_height, options.puzzle_width, options.batch_size)
	train(data, options)
	test(data)