import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten
from nn_util import NeuralNetBase, neuralnet
from keras.layers import Dense, convolutional


@neuralnet
class CNNValue(NeuralNetBase):
    """A convolutional neural network to guess the reward at the end of the
    game for a given board state, under the optimal policy.
    """

    def normalize(self, nn_output, percentage):
        # value network has tanh output (-1/1)
        # convert to (0/1) or (0/100)
        if percentage:
            # convert to range (0/100)
            values = [(value[0] + 1.) / 2. * 100 for value in nn_output]
        else:
            # convert to range (0/1)
            values = [(value[0] + 1.) / 2. for value in nn_output]

        return values

    def eval_state(self, state, percentage=False):
        """Given a GameState object, returns a value
           in range (0/1) or if percentage (0/100)
        """
        tensor = self.preprocessor.state_to_tensor(state)

        # run the tensor through the network
        network_output = self.forward(tensor)

        return self.normalize(network_output, percentage)[0]

    def batch_eval_state(self, states, percentage=False):
        """Given a list with GameState objects, returns a list values
           in range (0/1) or if percentage (0/100)
        """
        # concatenate together all one-hot encoded states along the 'batch' dimension
        nn_input = np.concatenate([self.preprocessor.state_to_tensor(s) for s in states], axis=0)

        # pass all input through the network at once (backend makes use of
        # batches if len(states) is large)
        network_output = self.forward(nn_input)

        return self.normalize(network_output, percentage)

    @staticmethod
    def create_network(**kwargs):
        """construct a convolutional neural network.
           Keword Arguments:
           - input_dim:            depth of features to be processed by first layer (no default)
           - board:                width of the go board to be processed (default 19)
           - filters_per_layer:    number of filters used on every layer (default 128)
           - filters_per_layer_K:  (where K is between 1 and <layers>) number of filters
                                   used on layer K (default #filters_per_layer)
           - layers:               number of convolutional steps (default 12)
           - dense:                number of neurons in dense layer (default 256)
           - filter_width_K:       (where K is between 1 and <layers>) width of filter on
                                   layer K (default 3 except 1st layer which defaults to 5).
                                   Must be odd.
        """

        defaults = {
            "board": 19,
            "filters_per_layer": 128,
            "layers": 13,  # layers 2-12 are identical to policy net
            "filter_width_1": 5,
            "dense": 256
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)

        # create the network:
        # a series of zero-paddings followed by convolutions
        # such that the output dimensions are also board x board
        network = Sequential()

        # create first layer
        network.add(convolutional.Conv2D(
            input_shape=(params["input_dim"], params["board"], params["board"]),
            filters=params.get("filters_per_layer_1", params["filters_per_layer"]),
            kernel_size=(params["filter_width_1"], params["filter_width_1"]),
            kernel_initializer='uniform',
            activation='relu',
            padding='same',
            kernel_constraint=None,
            activity_regularizer=None,
            trainable=True,
            strides=[1, 1],
            use_bias=True,
            bias_regularizer=None,
            bias_constraint=None,
            data_format="channels_first",
            kernel_regularizer=None,
            name="con_first"))

        for i in range(2, params["layers"] + 1):
            # use filter_width_K if it is there, otherwise use 3
            filter_width_key = "filter_width_%d" % i
            filter_width = params.get(filter_width_key, 3)

            # use filters_per_layer_K if it is there, otherwise use #filters_per_layer
            filter_count_key = "filters_per_layer_%d" % i
            filter_nb = params.get(filter_count_key, params["filters_per_layer"])

            network.add(convolutional.Conv2D(
                filters=filter_nb,
                kernel_size=(filter_width, filter_width),
                kernel_initializer='uniform',
                activation='relu',
                padding='same',
                kernel_constraint=None,
                activity_regularizer=None,
                trainable=True,
                strides=[1, 1],
                use_bias=True,
                bias_regularizer=None,
                bias_constraint=None,
                data_format="channels_first",
                kernel_regularizer=None,
                name="con_" + str(i)))

        # the last layer maps each <filters_per_layer> feature to a number
        network.add(convolutional.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            kernel_initializer='uniform',
            activation='relu',
            padding='same',
            kernel_constraint=None,
            activity_regularizer=None,
            trainable=True,
            strides=[1, 1],
            use_bias=True,
            bias_regularizer=None,
            bias_constraint=None,
            data_format="channels_first",
            kernel_regularizer=None,
            name="con_last"))

        network.add(Flatten())

        network.add(Dense(input_dim=params["board"] * params["board"],
                          units=params["dense"],
                          kernel_initializer='uniform',
                          activation='relu',
                          bias_regularizer=None,
                          bias_constraint=None,
                          activity_regularizer=None,
                          trainable=True,
                          kernel_constraint=None,
                          kernel_regularizer=None,
                          use_bias=True,
                          name="dense_1"))

        network.add(Dense(input_dim=params["dense"],
                          units=1,
                          kernel_initializer='uniform',
                          activation="tanh",
                          bias_regularizer=None,
                          bias_constraint=None,
                          activity_regularizer=None,
                          trainable=True,
                          kernel_constraint=None,
                          kernel_regularizer=None,
                          use_bias=True,
                          name="dense_2"))

        return network
