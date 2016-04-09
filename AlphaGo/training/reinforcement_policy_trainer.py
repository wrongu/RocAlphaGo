import argparse
from AlphaGo.ai import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import flatten_idx
from keras.optimizers import SGD
import numpy as np
# np.set_printoptions(linewidth=160)
import os
#
# from ipdb import set_trace as BP


def make_training_pairs(player, opp, features, mini_batch_size):
	"""Make training pairs for batch of matches, utilizing player.get_moves (parallel form of
	player.get_move), which calls `CNNPolicy.batch_eval_state`.

	Args:
	player -- player that we're always updating
	opp -- batch opponent
	feature_list -- game features to be one-hot encoded
	mini_batch_size -- number of games in mini-batch

	Return:
	X_list -- list of 1-hot board states associated with moves.
	y_list -- list of 1-hot moves associated with board states.
	winners -- list of winners associated with each game in batch
	"""

	def do_move(states, states_prev, moves, X_list, y_list):
		bsize_flat = bsize * bsize
		for st, st_prev, mv, X, y in zip(states, states_prev, moves, X_list, y_list):
			if not st.is_end_of_game:
				# Only do more moves if not end of game already
				st.do_move(mv)
				if st.current_player != go.BLACK and mv is not go.PASS_MOVE:
					# Convert move to one-hot
					state_1hot = preprocessor.state_to_tensor(st_prev)
					move_1hot = np.zeros(bsize_flat)
					move_1hot[flatten_idx(mv, bsize)] = 1
					X.append(state_1hot)
					y.append(move_1hot)
		return states, X_list, y_list

	# Lists of game training pairs (1-hot)
	X_list = [list()] * mini_batch_size
	y_list = [list()] * mini_batch_size
	preprocessor = Preprocess(features)
	bsize = player.policy.model.input_shape[-1]
	states = [GameState() for i in xrange(mini_batch_size)]
	while True:
		# Cache states before moves
		states_prev = [st.copy() for st in states]
		# Get moves (batch)
		moves = player.get_moves(states)
		# Do moves (player)
		states, X_list, y_list = do_move(states, states_prev, moves, X_list, y_list)
		# Do moves (opponent)
		moves_opp = opp.get_moves(states)
		states, X_list, y_list = do_move(states, states_prev, moves_opp, X_list, y_list)
		# If all games have ended, we're done. Get winners.
		done = [st.is_end_of_game for st in states]
		if all(done):
			break
	winners = [st.get_winner() for st in states]
	# Concatenate tensors across turns within each game
	for i in xrange(mini_batch_size):
		X_list[i] = np.concatenate(X_list[i], axis=0)
		y_list[i] = np.vstack(y_list[i])
	return X_list, y_list, winners


def train_batch(player, X_list, y_list, winners, lr):
	"""Given the outcomes of a mini-batch of play against a fixed opponent,
	   update the weights with reinforcement learning.

	   Args:
	   player -- player object with policy weights to be updated
	   training_pairs_list -- List of one-hot encoded state-action pairs.
	   winners -- List of winners corresponding to each item in
				  training_pairs_list
	   lr -- Keras learning rate

	   Return:
	   player -- same player, with updated weights.
	"""

	for X, y, winner in zip(X_list, y_list, winners):
		# Update weights in + direction if player won, and - direction if player lost.
		# Setting learning rate negative is hack for negative weights update.
		if winner == -1:
			player.policy.model.optimizer.lr.set_value(-lr)
		player.policy.model.fit(X, y, nb_epoch=1, batch_size=len(X))
	return player


def run(player, args, opponents, features):
	# Set SGD and compile
	sgd = SGD(lr=args.learning_rate)
	player.policy.model.compile(loss='binary_crossentropy', optimizer=sgd)
	player_wins_per_batch = []
	for i_iter in xrange(args.iterations):
		# Train mini-batches
		for i_batch in xrange(args.save_every):
			# Randomly choose opponent from pool
			opp = np.random.choice(opponents)
			# Make training pairs and do RL
			X_list, y_list, winners = make_training_pairs(
				player, opp, features, args.game_batch_size)
			n_wins = np.sum(np.array(winners) == 1)
			player_wins_per_batch.append(n_wins)
			print 'Number of wins this batch: {}/{}'.format(n_wins, args.game_batch_size)
			player = train_batch(player, X_list, y_list, winners, args.learning_rate)
		# TODO: Save policy to model folder
		# Add snapshot of player to pool of opponents
		opponents.append(player.copy())
	return opponents


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Perform reinforcement learning '
									 'to improve given policy network. Second phase of pipeline.')
	parser.add_argument("initial_weights", help="Path to file with weights to start from.")
	parser.add_argument("initial_json", help="Path to file with initial network params.")
	parser.add_argument("--model_folder", help="Path to folder where the model "
						"params will be saved after each epoch. Default: None", default=None)
	parser.add_argument(
		"--learning_rate", help="Keras learning rate (Default: .03)", type=float,
		default=.03)
	parser.add_argument(
		"--save_every", help="Save policy every n mini-batches (Default: 500)",
		type=int, default=500)
	parser.add_argument(
		"--game_batch_size", help="Number of games per mini-batch (Default: 20)",
		type=int, default=20)
	parser.add_argument(
		"--iterations", help="Number of training iterations (i.e. mini-batch) "
		"(Default: 20)",
		type=int, default=20)
	# parser.add_argument(
	# 	"--nb_workers", help="Number of threads to use when training in parallel. "
	# 	"Requires appropriately set Theano flags.", type=int, default=1)
	# game batch size
	# Baseline function (TODO) default lambda state: 0  (receives either file 
	# paths to JSON and weights or None, in which case it uses default baseline 0)
	args = parser.parse_args()
	# Load policy from file
	# policy = model_from_json(open(args.initial_json).read())
	# policy.load_weights(args.initial_weights)
	# player = ProbabilisticPolicyPlayer(model)
	#############################################
	# Just for now, while we get the model directories set up...
	features = ["board", "ones", "turns_since", "liberties", "capture_size",
				"self_atari_size", "liberties_after",
				"sensibleness", "zeros"]
	policy = CNNPolicy(features)
	player = ProbabilisticPolicyPlayer(policy)
	#############################################
	# Load opponent pool
	opponents = []
	if args.model_folder is not None:
		# TODO
		opponent_files = next(os.walk(args.model_folder))[2]
		if len(args.model_folder) == 0:
			# No opponents yet, so play against self
			opponents = [player]
		else:
			# TODO
			pass
	else:
		opponents = [player]
	opponents = run(player, args, opponents, features)
