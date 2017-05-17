import tensorflow as tf 
# from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


class PointerNetwork(object):

	def __init__(self, max_len, input_dim, hidden_dim, bidirectional, fc_dim, batch_size):
		self.max_len = max_len
		self.input_dim = input_dim
		self.bidirectional = bidirectional
		self.hidden_dim = hidden_dim
		self.init =  tf.contrib.layers.variance_scaling_initializer()
		self.fc_dim = fc_dim
		self.batch_size = batch_size
		self.decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, initializer = self.init)
		self.trainable_init_state = self.trainable_initial_state()


		# self.inputs, self.seq_lengths = self.get_placeholders(max_len, input_dim) will need this for initial testing

	def trainable_initial_state(self,  name="initial_state"):
	  flat_state_size = nest.flatten(self.fc_dim)

	  names = ["{}_{}".format(name, i) for i in xrange(len(flat_state_size))]
	  tiled_states = []

	  for name, size in zip(names, flat_state_size):
	    shape_with_batch_dim = [1, size]
	    initial_state_variable = tf.get_variable(
	        name, shape=shape_with_batch_dim, initializer = self.init)

	    tiled_state = tf.tile(initial_state_variable,
	                          [self.batch_size, 1], name=(name + "_tiled"))
	    tiled_states.append(tiled_state)

	  return nest.pack_sequence_as(structure=self.fc_dim,
	                               flat_sequence=tiled_states)

		
	def get_input_placeholders(self):
		inputs = tf.placeholder(tf.float32, shape=(None, self.max_len, self.input_dim), name='Encoder_input')
		seq_lengths = tf.placeholder(tf.int32, [None, ], name='seq_input')
		targets = tf.placeholder(tf.int32, [None, self.max_len, self.max_len], name="targets")
		return inputs, seq_lengths, targets

	def encode(self, orig_inputs, seq_lengths, process='FC'):
		if process == 'FC':
			orig_inputs = tf.reshape(orig_inputs, [-1, self.input_dim])
			inputs = tf.layers.dense(orig_inputs, self.fc_dim, activation=tf.nn.relu, kernel_initializer=self.init)
			inputs = tf.reshape(inputs, [-1, self.max_len, self.fc_dim])
		elif process == 'CNN':
			pass

		if not self.bidirectional:
			cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, initializer = self.init)
			output, output_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_lengths, dtype=tf.float32)
		else:
			cell_fw =  tf.contrib.rnn.LSTMCell(self.hidden_dim, initializer = self.init)
			cell_bw =  tf.contrib.rnn.LSTMCell(self.hidden_dim, initializer = self.init)
			output_fw_bw, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs , sequence_length=seq_lengths, dtype=tf.float32)

			fw, bw = tf.unstack(output_fw_bw, axis=0)
			output = fw + bw # assuming we are just summing the forward and backwared activations
			fw_state, bw_state = tf.unstack(output_states, axis=0)
			c_fw, h_fw = tf.unstack(fw_state, axis = 0)
			c_bw, h_bw = tf.unstack(bw_state, axis = 0)
			output_state = tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(c_fw + c_bw, h_fw + h_bw)

		return inputs, output, output_state

# might need to stop gradients for inputs when doing tests
# should probably implement glimpses : https://github.com/devsisters/pointer-network-tensorflow/blob/master/layers.py
	def decode(self, inputs, enc_out, enc_end_state, seq_lengths, train=1):
		# note that here seq_lengths is a numpy array 
		with vs.variable_scope("decoder_scope") as scope:
			maximum_length = tf.convert_to_tensor(self.max_len, tf.int32)
			def attention(hidden, scope="Attention"):
				with vs.variable_scope(scope):
					if hidden is None:
						return tf.zeros([tf.shape(enc_out)[0], self.hidden_dim], tf.float32), tf.zeros([tf.shape(enc_out)[0], tf.shape(inputs)[2]], tf.float32)

					att_W_in = vs.get_variable("att_W_in", (1, self.hidden_dim, self.hidden_dim), initializer=self.init)
					att_W_h  =  vs.get_variable("att_W_h", (self.hidden_dim, self.hidden_dim),  initializer=self.init)
					att_V = vs.get_variable("att_V", (self.hidden_dim, ), initializer=self.init)

					enc_ref = tf.nn.conv1d(enc_out, att_W_in, 1, "VALID", name="intermediate")
					_, hidden = tf.unstack(hidden, axis=0)
					# print c, hidden, att_W_h, tf.matmul(hidden, att_W_h)
					enc_query = tf.expand_dims(tf.matmul(hidden, att_W_h), 1, name="exp_enc_query")
					enc_query = tf.tile(enc_query, [1, self.max_len, 1], name = "tiled_enc_query")
					scores = tf.reduce_sum(att_V*tf.tanh(enc_ref + enc_query), [-1], name='scores') # doing the sum of the last dimension as dot product
					
					inds = tf.cast(tf.argmax(scores, 1),  tf.int32)
					replicated_first_indices = tf.range(tf.shape(inds)[0])
					inds = tf.stack([inds, replicated_first_indices], axis=1)

					next_inputs = tf.stop_gradient(tf.gather_nd(inputs, inds))
					return scores, next_inputs

			batch_size = tf.shape(inputs)[0]  # Needed for reshaping.
	        input_size = tf.shape(inputs)[2]
	        attn_length = tf.shape(enc_out)[1]
	        attn_size = tf.shape(enc_out)[2]
	        outputs = []

	        states = [enc_end_state]
	        inp = tf.zeros([tf.shape(enc_out)[0], self.fc_dim], tf.float32)
	        inputs = tf.transpose(inputs, [1, 0, 2])
	        for i in xrange(self.max_len):
	            if i > 0:
	                vs.get_variable_scope().reuse_variables()

	            if train:
	           		inp = tf.gather(inputs, i)
	                
	            # Merge input and previous attentions into one vector of the right size.
	            cell_output, new_state = self.decoder_cell(inp, states[-1])
	            # print inp, cell_output, new_state , '\n'
	            states.append(new_state)
	            # Run the attention mechanism.
	            output, inp = attention(new_state)
	            outputs.append(output)
		outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
		return outputs, states




