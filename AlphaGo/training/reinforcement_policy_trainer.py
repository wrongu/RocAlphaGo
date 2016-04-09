import argparse
from AlphaGo.ai import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess
from interface.Play import play_match
# from joblib import Parallel, delayed
from keras.optimizers import SGD
# from keras.models import model_from_json
# from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
# np.set_printoptions(linewidth=160)
import os
# import threading
#
from ipdb import set_trace as BP


def _make_training_pairs(player, opp, features, mini_batch_size):
	"""Helper function to make_training_pairs. Make training pairs for batch of matches,
	utilizing player.get_moves, which calls `CNNPolicy.batch_eval_state`.

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
		for st, st_prev, mv, X, y in zip(states, states_prev, moves, X_list, y_list):
			if not st.is_end_of_game:
				# Only do more moves if not end of game already
				st.do_move(mv)
				if st.current_player != go.BLACK and mv is not go.PASS_MOVE:
					# Convert move to one-hot
					move_1hot = np.zeros((1, bsize, bsize))
					move_1hot[0][mv] = 1
					state_1hot = preprocessor.state_to_tensor(st_prev)
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
	# Concatenate tensors with each game
	for i in xrange(mini_batch_size):
		X_list[i] = np.concatenate(X_list[i], axis=0)
		y_list[i] = np.concatenate(y_list[i], axis=0)
	BP()
	# X = []
	# y = []
	# match = play_match(player, opp)
	# bsize = match.state.board.shape[0]
	# matches = [play_match(player, opp) for i in xrange(mini_batch_size)]
	# # Play out game
	# while True:
	# 	# Cached copy of previous board state, so that training pairs are
	# 	# between latest move and the board state when it was being considered.
	# 	state = match.state.copy()
	# 	# Do move
	# 	end_of_game = match.play()
	# 	print "turns played:", match.state.turns_played
	# 	move = match.state.history[-1]
	# 	# Only add training pairs and tensors for BLACK moves and not pass moves
	# 	if match.state.current_player != go.BLACK and \
	# 		move is not go.PASS_MOVE:
	# 		# Convert move to one-hot
	# 		move_1hot = np.zeros((1, bsize, bsize))
	# 		move_1hot[0][move] = 1
	# 		state_1hot = preprocessor.state_to_tensor(state)
	# 		X.append(state_1hot)
	# 		y.append(move_1hot)
	# 		# Add training pairs
	# 		# training_pairs.append((state_1hot,
	# 		# 					   move_1hot))
	# 	# Print out board states for debugging purposes
	# 	# if match.state.turns_played % 1 == 0:
	# 	# 	print match.state.board
	# 	# End game prematurely for debugging
	# 	if match.state.turns_played > 400:
	# 		break
	# 	# Detect end of game
	# 	if end_of_game:
	# 		break
	# winner = match.state.get_winner()
	# X = np.concatenate(X, axis=0)
	# y = np.concatenate(y, axis=0)
	return X_list, y_list, winners


# def gen_training_pairs(player, opp, features):
# 	"""Generator that yields training pairs, by playing games
# 	   against fixed opponent.
# 	"""
# 	# Make training pairs
# 	while True:
# 		X, y, winner = _make_training_pairs(player, opp, features)
# 		n_turns = len(X)
# 		# Make sample weights vector
# 		if winner == 1:
# 			sw = np.ones(n_turns)
# 		else:
# 			sw = np.ones(n_turns) * -1
# 		for turn in game:
# 			yield (X, y, sw)


def make_training_pairs(player, opp, mini_batch_size, features,
						nb_workers):
	"""Make all n training pairs for a single batch, by playing n games
	   against fixed opponent.
	"""

	# Make training pairs
	winners = []
	training_pairs_list = []
	X, y, winner = _make_training_pairs(player, opp, features, mini_batch_size)
	training_pairs_list.append((X, y))
	winners.append(winner)
	return training_pairs_list, winners


def train_batch(player, training_pairs_list, winners, lr):
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
	# player.policy.model.compile(loss='binary_crossentropy', optimizer=sgd)
	# for training_pairs, winner in zip(training_pairs_list, winners):
		# Concatenate input and target tensors
		# input_tensors = []
		# target_tensors = []
		# for t in training_pairs:
		# 	input_tensors.append(t[0])
		# 	target_tensors.append(t[1])
		# input_tensors = np.concatenate(input_tensors, axis=0)
		# target_tensors = np.concatenate(target_tensors, axis=0)
		# if winner == 1:
		# 	sw = np.ones(len(input_tensors))
		# else:
		# 	sw = np.ones(len(input_tensors)) * -1
		# player.policy.model.fit(input_tensors, target_tensors, nb_epoch=1,
		# 			   			batch_size=len(input_tensors), sample_weight=sw)
	for X, y, winner in zip(training_pairs_list, winners):
		# Update weights in + direction if player won, and - direction if player lost.
		# Setting sample_weight negative is hack for negative weights update.
		if winner == -1:
			player.policy.model.optimizer.lr.set_value(-lr)
		player.policy.model.fit(X, y, nb_epoch=1, batch_size=len(X))
	return player


def run(player, args, opponents, features):
	# Set SGD and compile
	# sgd = SGD(lr=args.learning_rate)
	# player.policy.model.compile(loss='binary_crossentropy', optimizer=sgd)
	for i_iter in xrange(args.iterations):
		# Train mini-batches
		for i_batch in xrange(args.save_every):
			# Randomly choose opponent from pool
			opp = np.random.choice(opponents)
			# Make training pairs and do RL
			training_pairs_list, winners = make_training_pairs(
				player, opp, args.game_batch_size, features, args.nb_workers)
			player = train_batch(player, training_pairs_list, winners)

			# Parallelize with Keras
			# gen = gen_training_pairs(player.copy(), opp, features)
			# player.policy.model.fit_generator(gen, args.game_batch_size, 1,
			# 								  nb_worker=args.nb_workers)
		BP()
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
		"--nb_workers", help="Number of threads to use when training in parallel. "
		"Requires appropriately set Theano flags.", type=int, default=1)
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




# def make_training_pairs(match, feature_list):
# 	while True:
# 		training_pairs = []
# 		preprocessor = Preprocess(feature_list)
# 		bsize = match.state.board.shape[0]
# 		while True:
# 			# Cached copy of previous board state, so that training pairs are
# 			# between latest move and the board state when it was being considered.
# 			state = match.state.copy()
# 			# Do move
# 			end_of_game = match.play()
# 			print "turns played:", match.state.turns_played
# 			move = match.state.history[-1]
# 			# Only add training pairs and tensors for BLACK moves and not pass moves
# 			if match.state.current_player != go.BLACK and \
# 				move is not go.PASS_MOVE:
# 				# Convert move to one-hot
# 				move_1hot = np.zeros((1, bsize, bsize))
# 				move_1hot[0][move] = 1
# 				# Add training pairs
# 				training_pairs.append((preprocessor.state_to_tensor(state),
# 									   move_1hot))
# 			# Print out board states for debugging purposes
# 			if match.state.turns_played % 1 == 0:
# 				print match.state.board
# 			# End game prematurely for debugging
# 			if match.state.turns_played > 10:
# 				break
# 			# Detect end of game
# 			if end_of_game:
# 				break
# 		yield training_pairs

# class RL_policy_trainer(object):
# 	"""Train reinforcement learning policy. First, extract training tuples from
# 	game (`get_training_pairs`), then use those tuples for reinforcement learning.
# 	Call `train_policy.train()` to first extract training pairs and then run
# 	training. Initialize object with any two policy players (must have `get_move`
# 	function that returns a single move tuple given board state array).
# 	"""
# 	def __init__(self, player1, player2, feature_list, learning_rate=0.01):
# 		self.player1 = player1
# 		self.player2 = player2
# 		self.match = play_match(player1, player2, 'test')
# 		# Player1 is by convention the network model to be updated by reinforcement
# 		self.model = player1.policy.model  # Keras model, inherited from supervised training phase
# 		self.learning_rate = learning_rate  # TODO
# 		self.preprocessor = Preprocess(feature_list)

# 	def save_model(self):
# 		# TODO
# 		pass
# 		# self.fname = self.save_dir
# 		# self.model.save_weights('fname.h5')
# 		# self.model.save_json('fname.json')

# 	def train(self, training_pairs):
# 		# Calculate which player won
# 		self.winner = self.match.state.get_winner()
# 		# Concatenate input and target tensors
# 		input_tensors = []
# 		target_tensors = []
# 		for t in training_pairs:
# 			input_tensors.append(t[0])
# 			target_tensors.append(t[1])
# 		input_tensors = np.concatenate(input_tensors, axis=0)
# 		target_tensors = np.concatenate(target_tensors, axis=0)
# 		# Initialize SGD and load keras net
# 		sgd = SGD(lr=self.learning_rate)
# 		self.model.compile(loss='binary_crossentropy', optimizer=sgd)
# 		# Update weights in + direction if player won, and - direction if player lost.
# 		# Setting sample_weight negative is hack for negative weights update.
# 		# states = np.array([s[0] for s in samples if s[1] is not None])
# 		if self.winner == 1:
# 			sw = np.ones(len(input_tensors))
# 		else:
# 			sw = np.ones(len(input_tensors)) * -1
# 		self.model.fit(input_tensors, target_tensors, nb_epoch=1,
# 					   batch_size=len(input_tensors), sample_weight=sw)
# 		# Save new weights
# 		# TODO
# 		self.save_model()
# 		return
