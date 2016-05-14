from keras import backend as K
from keras.engine.topology import Layer


class Bias(Layer):
	"""Custom keras layer that simply adds a scalar bias to each location in the input

	Largely copied from the keras docs:
	http://keras.io/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers
	"""
	def __init__(self, **kwargs):
		super(Bias, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = K.zeros(input_shape[1:])
		self.trainable_weights = [self.W]

	def call(self, x, mask=None):
		return x + self.W
