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

		
	def get_placeholders(self, max_len, input_dim):
		inputs = tf.placeholder(tf.float32, [None, max_len, input_dim], name='Encoder_input')
		seq_lengths = tf.placeholder(tf.int32, [None, ])
		return inputs, seq_lengths

	def encode(self, inputs, seq_lengths):
		if !self.bidirectional:
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
	def decode(self, inputs, enc_out, enc_end_state, seq_lengths):
		# note that here seq_lengths is a numpy array 
		with vs.variable_scope("decoder_scope") as scope:
			def attention(hidden, inputs, to_attend, valid_len, scope="Attention"):
				with vs.variable_scope(scope):
					att_W_in = vs.get_variable("att_W_in", (self.hidden_dim, self.hidden_dim), self.init)
					att_W_h  =  vs.get_variable("att_W_h", (self.hidden_dim, self.hidden_dim),  self.init)
					att_V = vs.get_variable("att_V", (self.hidden_dim, ), self.init)
					intermediate = tf.tanh(tf.matmul(enc_out, att_W_in) + tf.matmul(hidden, att_W_h))
					logits = tf.matmul(intermediate, att_V)
					activations = tf.softmax(logits)
					next_input = tf.gather(inputs, tf.argmax(activations))
					return activations, next_input


			decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, initializer = self.init)
			outputs = []
			batch_size = inputs.get_shape()[0]
			for i in xrange(batch_size):
				if i > 0:
                	vs.get_variable_scope().reuse_variables()
				input_seq = tf.squeeze(tf.gather(inputs, i))
				attn_seq = tf.squeeze(tf.gather(enc_out, i))
				prev_hidden = tf.squeeze(tf.gather(enc_end_state, i))
				prev_in = tf.zeros((1, self.input_dim), dtype=tf.float32, name="decoder_starter")
				for j in xrange(seq_lengths[i]):
					prev_hidden = decoder_cell(prev_in, prev_hidden)
					out, prev_in = attention(prev_hidden, inputs, attn_seq, seq_lengths[i])
					outputs.append(out)

		return tf.stack(outputs)





		




