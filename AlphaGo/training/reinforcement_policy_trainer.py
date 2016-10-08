import os
import json
import numpy as np
from shutil import copyfile
from keras.optimizers import Optimizer
import keras.backend as K
import theano.tensor as T
from AlphaGo.ai import ProbabilisticPolicyPlayer
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.util import flatten_idx
from keras.engine.topology import Input


def log_loss(y_true, y_pred):
    """Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the positive gradient will make that action more likely.
    """
    return y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))


class RLPolicyTrainer(object):
    """TODO
    """

    def __init__(self, keras_model, lr=0.001, n_parallel=1, batch_size=16):
        """TODO
        """
        if K.backend() != 'theano':
            raise ValueError("RLPolicyTrainer only works in Theano.")
        self.game_idx = Input(shape=(1,), dtype='uint8')
        self.model = keras_model
        self.input_batch = [None] * batch_size
        self.batch_next = 0
        self.n_parallel = n_parallel
        self._accumulate_gradients = self._make_gradient_accumulator_function()
        self._apply_gradients = self._make_gradient_apply_function()

    def _make_gradient_accumulator_function(self):
        """Create a Function that accumulates gradients.
        """
        gradients = K.gradients(log_loss, self.model)
        # Create a set of accumulated gradients, one for each parallel game.
        shapes = [K.get_variable_shape(g) for g in gradients]
        # Create gradient for each set of trainable weights, duplicated n_parallel times.
        self.all_gradients = [K.zeros((self.n_parallel,) + shape) for shape in shapes]
        # Update gets applied only to the gradient at index game_idx
        # Note: currently this relies on T.inc_subtensor and is not compatible with tensorflow.
        grad_updates = [(ag, T.inc_subtensor(ag[self.game_idx], g))
                        for g, ag in zip(gradients, self.all_gradients)]
        # TODO - make this work with batches of inputs and indices.
        return K.function(self.model.inputs + [self.game_idx], [], updates=grad_updates)

    def _make_gradient_apply_function(self):
        """Create a Function that applies all accumulated gradients.
        """
        # TODO (but see apply_gradients(results) below)
        pass

    def _reset_batch(self):
        self.input_batch = [None] * len(self.input_batch)
        self.batch_next = 0

    def apply_gradients(self, results):
        """Apply accumulated gradients in the directions specified in 'results'.

        Inputs:
        results - list of +/- 1 values, one for each parallel game.
        """
        # First, make sure no batches are left unprocessed
        if self.batch_next != 0:
            # TODO - is Theano OK with None in input_batch?
            self._accumulate_gradients(zip(*self.input_batch))
            self._reset_batch()
        self._apply_gradients([self.all_gradients, results])

    def process_xy_pair(self, X, y, game_idx=0):
        """Queue (X, y) pair in batch. Computes gradients when batch is full.
        """
        self.input_batch[self.batch_next] = (X, y, game_idx)
        self.batch_next += 1


# TODO - update things below here


def _make_training_pair(st, mv, preprocessor):
    # Convert move to one-hot
    st_tensor = preprocessor.state_to_tensor(st)
    mv_tensor = np.zeros((1, st.size * st.size))
    mv_tensor[(0, flatten_idx(mv, st.size))] = 1
    return (st_tensor, mv_tensor)


def run_n_games(optimizer, learner, opponent, num_games):
    """Run num_games games to completion, calling train_batch() on each position
    the learner sees.

    (Note: optimizer only accumulates gradients in its update function until
    all games have finished)

    """
    board_size = learner.policy.model.input_shape[-1]
    states = [GameState(size=board_size) for _ in range(num_games)]
    learner_net = learner.policy.model

    # Start all odd games with moves by 'opponent'. Even games will have 'learner' black.
    learner_color = [go.BLACK if i % 2 == 0 else go.WHITE for i in range(num_games)]
    odd_states = states[1::2]
    moves = opponent.get_moves(odd_states)
    for st, mv in zip(odd_states, moves):
        st.do_move(mv)

    current = learner
    other = opponent
    # Need to keep track of the index of unfinished states so that we can communicate which one is
    # being updated to the optimizer.
    idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}
    while len(idxs_to_unfinished_states) > 0:
        # Get next moves by current player for all unfinished states.
        moves = current.get_moves(idxs_to_unfinished_states.values())
        just_finished = []
        # Do each move to each state in order.
        for (idx, state), mv in zip(idxs_to_unfinished_states.iteritems(), moves):
            # Order is important here. We must first get the training pair on the unmodified state.
            # Next, the state is updated and checked to see if the game is over. If it is over, the
            # optimizer is notified via set_result. Finally, train_on_batch is called, which
            # will trigger an update of all parameters only if set_result() has been called
            # for all games already (so set_result must come before train_on_batch).
            is_learnable = current is learner and mv is not go.PASS_MOVE
            if is_learnable:
                (X, y) = _make_training_pair(state, mv, learner.policy.preprocessor)
            state.do_move(mv)
            if state.is_end_of_game:
                learner_is_winner = state.get_winner() == learner_color[idx]
                optimizer.set_result(idx, learner_is_winner)
                just_finished.append(idx)
            if is_learnable:
                optimizer.set_current_game(idx)
                learner_net.train_on_batch(X, y)

        # Remove games that have finished from dict.
        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        # Swap 'current' and 'other' for next turn.
        current, other = other, current

    # Return the win ratio.
    wins = sum(state.get_winner() == pc for (state, pc) in zip(states, learner_color))
    return float(wins) / num_games


def run_training(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Perform reinforcement learning to improve given policy network. Second phase of pipeline.')  # noqa: E501
    parser.add_argument("model_json", help="Path to policy model JSON.")
    parser.add_argument("initial_weights", help="Path to HDF5 file with inital weights (i.e. result of supervised training).")  # noqa: E501
    parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")  # noqa: E501
    parser.add_argument("--learning-rate", help="Keras learning rate (Default: 0.001)", type=float, default=0.001)  # noqa: E501
    parser.add_argument("--policy-temp", help="Distribution temperature of players using policies (Default: 0.67)", type=float, default=0.67)  # noqa: E501
    parser.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 500)", type=int, default=500)  # noqa: E501
    parser.add_argument("--game-batch", help="Number of games per mini-batch (Default: 20)", type=int, default=20)  # noqa: E501
    parser.add_argument("--move-limit", help="Maximum number of moves per game", type=int, default=500)  # noqa: E501
    parser.add_argument("--iterations", help="Number of training batches/iterations (Default: 10000)", type=int, default=10000)  # noqa: E501
    parser.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # Baseline function (TODO) default lambda state: 0  (receives either file
    # paths to JSON and weights or None, in which case it uses default baseline 0)
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    ZEROTH_FILE = "weights.00000.hdf5"

    if args.resume:
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without existing output directory")

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print "creating output directory {}".format(args.out_directory)
        os.makedirs(args.out_directory)

    if not args.resume:
        # make a copy of weights file, "weights.00000.hdf5" in the output directory
        copyfile(args.initial_weights, os.path.join(args.out_directory, ZEROTH_FILE))
        if args.verbose:
            print "copied {} to {}".format(args.initial_weights,
                                           os.path.join(args.out_directory, ZEROTH_FILE))
        player_weights = ZEROTH_FILE
    else:
        # if resuming, we expect initial_weights to be just a
        # "weights.#####.hdf5" file, not a full path
        args.initial_weights = os.path.join(args.out_directory,
                                            os.path.basename(args.initial_weights))
        if not os.path.exists(args.initial_weights):
            raise ValueError("Cannot resume; weights {} do not exist".format(args.initial_weights))
        elif args.verbose:
            print "Resuming with weights {}".format(args.initial_weights)
        player_weights = os.path.basename(args.initial_weights)

    # Set initial conditions
    policy = CNNPolicy.load_model(args.model_json)
    policy.model.load_weights(args.initial_weights)
    player = ProbabilisticPolicyPlayer(policy, temperature=args.policy_temp,
                                       move_limit=args.move_limit)

    # different opponents come from simply changing the weights of 'opponent.policy.model'. That
    # is, only 'opp_policy' needs to be changed, and 'opponent' will change.
    opp_policy = CNNPolicy.load_model(args.model_json)
    opponent = ProbabilisticPolicyPlayer(opp_policy, temperature=args.policy_temp,
                                         move_limit=args.move_limit)

    if args.verbose:
        print "created player and opponent with temperature {}".format(args.policy_temp)

    if not args.resume:
        metadata = {
            "model_file": args.model_json,
            "init_weights": args.initial_weights,
            "learning_rate": args.learning_rate,
            "temperature": args.policy_temp,
            "game_batch": args.game_batch,
            "opponents": [ZEROTH_FILE],  # which weights from which to sample an opponent each batch
            "win_ratio": {}  # map from player to tuple of (opponent, win ratio) Useful for
                             # validating in lieu of 'accuracy/loss'
        }
    else:
        with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)

    # Append args of current run to history of full command args.
    metadata["cmd_line_args"] = metadata.get("cmd_line_args", []).append(vars(args))

    def save_metadata():
        with open(os.path.join(args.out_directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

    optimizer = BatchedReinforcementLearningSGD(lr=args.learning_rate, ng=args.game_batch)
    player.policy.model.compile(loss=log_loss, optimizer=optimizer)
    for i_iter in xrange(1, args.iterations + 1):
        # Randomly choose opponent from pool (possibly self), and playing
        # game_batch games against them.
        opp_weights = np.random.choice(metadata["opponents"])
        opp_path = os.path.join(args.out_directory, opp_weights)

        # Load new weights into opponent's network, but keep the same opponent object.
        opponent.policy.model.load_weights(opp_path)
        if args.verbose:
            print "Batch {}\tsampled opponent is {}".format(i_iter, opp_weights)

        # Run games (and learn from results). Keep track of the win ratio vs
        # each opponent over time.
        win_ratio = run_n_games(optimizer, player, opponent, args.game_batch)
        metadata["win_ratio"][player_weights] = (opp_weights, win_ratio)

        # Save all intermediate models.
        player_weights = "weights.%05d.hdf5" % i_iter
        player.policy.model.save_weights(os.path.join(args.out_directory, player_weights))

        # Add player to batch of oppenents once in a while.
        if i_iter % args.save_every == 0:
            metadata["opponents"].append(player_weights)
        save_metadata()

if __name__ == '__main__':
    run_training()
