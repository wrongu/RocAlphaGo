from keras.models import Sequential
from AlphaGo.util import flatten_idx
from keras.layers import convolutional
from keras.layers.core import Activation, Flatten
from nn_util import Bias, NeuralNetBase, neuralnet


@neuralnet
class CNNRollout(NeuralNetBase):
    """A convolutional neural network to compute a probability
       distribution over the next action
    """

    def _select_moves_and_normalize(self, nn_output, moves, size):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        distribution = distribution / distribution.sum()
        return zip(moves, distribution)

    def eval_state(self, state, moves=None):
        """Given a GameState object, returns a list of (action, probability) pairs
        according to the network outputs

        If a list of moves is specified, only those moves are kept in the distribution
        """
        tensor = self.preprocessor.state_to_tensor(state)

        # run the tensor through the network
        network_output = self.forward(tensor)

        moves = moves or state.get_legal_moves()
        return self._select_moves_and_normalize(network_output[0], moves, state.size)

    @staticmethod
    def create_network(**kwargs):
        """construct a fast rollout neural network.
           Keword Arguments:
           - input_dim:            depth of features to be processed by first layer (no default)
           - board:                width of the go board to be processed (default 19)
        """

        defaults = {
            "board": 19
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)

        # create the network:
        network = Sequential()

        # create one convolutional layer
        network.add(convolutional.Conv2D(
            input_shape=(params["input_dim"], params["board"], params["board"]),
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
            kernel_regularizer=None))

        # reshape output to be board x board
        network.add(Flatten())
        # add a bias to each board location
        network.add(Bias())
        # softmax makes it into a probability distribution
        network.add(Activation('softmax'))

        return network
