from keras import backend as K
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import _time_distributed_dense  as TD_Dense
import numpy as np
# In case I fuck this up https://github.com/vyraun/seq2set-keras

class PtrNet(Recurrent):

	def __init__(self, units, **kwargs):
		super(PtrNet, self).__init__(units, **kwargs)
		self.units = units  # will make more complicated once it is working 

	def build(self, input_shape):
		self.input_dim = input_shape[2]
		self.kernel = self.add_weight(shape = (self.input_dim, self.units), name = 'attn_input_kernel', \
			initializer = uniform, trainable=True) # W1 for attention model 
		self.recurrent_kernel = self.add_weight(shape = (self.self.units, self.units), name = 'attn_recurrent_kernel', \
			initializer = uniform, trainable=True) # W2 for attention model 
		self.attn_vec = self.add_weight(shape = (self.self.units, ), name = 'attn_vec', \
			initializer = uniform, trainable=True) # v for attention model
		super(PtrNet, self).build(input_shape)

	def call(self, x, mask=None):
		self.encoder_input = x
		self.mask = mask
		super(PtrNet, self).call(x, mask)

	def step(self, inputs, states):
		# ignore this input and come up with input based on attention model of prev-state.
		attn_input, _  = #
		hidden, info = super(PtrNet, self).step(attn_input, states)

		return hidden, info

	def _attend(x):
		# TAKEN FROM KERAS SOURCE CODE!
	    input_dim = K.shape(x)[2]
	    timesteps = K.shape(x)[1]
	    output_dim = K.shape(w)[1]
	    # collapse time dimension and batch dimension together
	    x = K.reshape(x, (-1, input_dim))
	    x = K.dot(x, w)
	    if b is not None:
	        x = K.bias_add(x, b)
	    # reshape to 3D tensor
	    if K.backend() == 'tensorflow':
	        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
	        x.set_shape([None, None, output_dim])
	    else:
	        x = K.reshape(x, (-1, timesteps, output_dim))

	    return x




