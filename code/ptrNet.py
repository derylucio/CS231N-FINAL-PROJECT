import tensorflow as tf 
from tf.contrib.layers import variance_scaling_initializer
from tensorflow.python.ops import variable_scope as vs

class PointerNetwork(object):

	def __init__(self, max_len, input_dim, hidden_dim, bidirectional):
		self.max_len = max_len
		self.input_dim = input_dim
		self.bidirectional = bidirectional
		self.hidden_dim = hidden_dim
		self.init = variance_scaling_initializer()


		# self.inputs, self.seq_lengths = self.get_placeholders(max_len, input_dim) will need this for initial testing

		
	def get_input_placeholders(self):
		inputs = tf.placeholder(tf.float32, [None, self.max_len, self.input_dim], name='Encoder_input')
		seq_lengths = tf.placeholder(tf.int32, [None, ], name='seq_input')
		targets = tf.placeholder(tf.int32, [None, self.max_len], name="targets")
		return inputs, seq_lengths, targets

	def encode(self, inputs, seq_lengths):
		if not self.bidirectional:
			cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, initializer = self.init)
			output, output_state = tf.nn.dynamic_rnn(cell, inputs)
		else:
			cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim, initializer = self.init)
			cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim, initializer = self.init)
			output_fw_bw, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs , sequence_length=seq_lengths)
			output = output_fw_bw[0] + output_fw_bw[1] # assuming we are just summing the forward and backwared activations
			output_state = output_states[0] + output_states[1]

		return output, output_state

# might need to stop gradients for inputs when doing tests
# should probably implement glimpses : https://github.com/devsisters/pointer-network-tensorflow/blob/master/layers.py
	def decode(self, inputs, targets, enc_out, enc_end_state, seq_lengths, train=1):
		# note that here seq_lengths is a numpy array 
		with vs.variable_scope("decoder_scope") as scope:
			maximum_length = tf.convert_to_tensor(self.max_len, tf.int32)
			def attention(hidden, scope="Attention"):
				with vs.variable_scope(scope):
					if hidden is None:
						return tf.zeros([tf.shape(inputs)[0], self.max_len], tf.float32), tf.zeros([tf.shape(inputs)[0], self.hidden_dim], tf.float32)

					att_W_in = vs.get_variable("att_W_in", (1, self.hidden_dim, self.hidden_dim), self.init)
					att_W_h  =  vs.get_variable("att_W_h", (self.hidden_dim, self.hidden_dim),  self.init)
					att_V = vs.get_variable("att_V", (self.hidden_dim, ), self.init)

					enc_ref = tf.nn.conv1d(enc_out, att_W_in, 1, "VALID", name="intermediate")
					enc_query = tf.expand_dims(tf.matmul(hidden, att_W_h), 1, name="exp_enc_query")
					enc_query = tf.tile(enc_query, [1, self.max_len, 1], name = "tiled_enc_query")
					scores = tf.reduce_sum(att_V*tf.tanh(enc_ref + enc_query), [-1]) # doing the sum of the last dimension as dot product
					inds = tf.argmax(scores, 1)
					next_inputs = tf.stop_gradient(tf.gather(inputs, inds))
					return scores, next_inputs

			decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, initializer = self.init)
			if train:
				decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_end_state, scope=scope)
			else:
				def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
					cell_output, next_input = attention(cell_output)
					if cell_state is None:
						cell_state = enc_end_state
						next_input = cell_input
						done = tf.zeros([tf.shape(inputs)[0], ], dtype=tf.bool)
					done = tf.cond(tf.greater(time, maximum_length), lambda: tf.ones([tf.shape(inputs)[0],], dtype=tf.bool), lambda: done)
					return done, cell_state, next_input, cell_output, context_state

			outputs, final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, \
				decoder_fn, inputs, sequence_length = seq_lengths, scope=scope)

		return outputs, final_state, final_context_state




		




