
# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute

class MinhModel:

	def __init__(self, shape=(84, 84), window_length=4, actions=4):

		self.input_shape = (window_length,) + shape

		self.model = Sequential()
		self.model.add(Permute((2, 3, 1), input_shape=self.input_shape))
		self.model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
		self.model.add(Activation('relu'))
		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.model.add(Dense(actions))
		self.model.add(Activation('linear'))

	def summary(self):
		return self.model.summary()
