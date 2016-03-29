import argparse
from AlphaGo.ai import GreedyPolicyPlayer
import AlphaGo.go as go
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess
from interface.Play import play_match
from keras.optimizers import SGD
import numpy as np
import os


def make_training_pairs(match, feature_list):
	while True:
		training_pairs = []
		preprocessor = Preprocess(feature_list)
		bsize = match.state.board.shape[0]
		while True:
			# Cached copy of previous board state, so that training pairs are
			# between latest move and the board state when it was being considered.
			state = match.state.copy()
			# Do move
			end_of_game = match.play()
			print "turns played:", match.state.turns_played
			move = match.state.history[-1]
			# Only add training pairs and tensors for BLACK moves and not pass moves
			if match.state.current_player != go.BLACK and \
				move is not go.PASS_MOVE:
				# Convert move to one-hot
				move_1hot = np.zeros((1, bsize, bsize))
				move_1hot[0][move] = 1
				# Add training pairs
				training_pairs.append((preprocessor.state_to_tensor(state),
									   move_1hot))
			# Print out board states for debugging purposes
			if match.state.turns_played % 1 == 0:
				print match.state.board
			# End game prematurely for debugging
			if match.state.turns_played > 10:
				break
			# Detect end of game
			if end_of_game:
				break
		yield training_pairs


class RL_policy_trainer(object):
	"""Train reinforcement learning policy. First, extract training tuples from
	game (`get_training_pairs`), then use those tuples for reinforcement learning.
	Call `train_policy.train()` to first extract training pairs and then run
	training. Initialize object with any two policy players (must have `get_move`
	function that returns a single move tuple given board state array).
	"""

	def __init__(self, player1, player2, feature_list, learning_rate=0.01):
		self.player1 = player1
		self.player2 = player2
		self.match = play_match(player1, player2, 'test')
		# Player1 is by convention the network model to be updated by reinforcement
		self.model = player1.policy.model  # Keras model, inherited from supervised training phase
		self.learning_rate = learning_rate  # TODO
		self.preprocessor = Preprocess(feature_list)

	def save_model(self):
		# TODO
		pass
		# self.fname = self.save_dir
		# self.model.save_weights('fname.h5')
		# self.model.save_json('fname.json')

	def train(self, training_pairs):
		# Make training tuples
		self.make_training_pairs()
		# Calculate which player won
		self.winner = self.match.state.get_winner()
		# Concatenate input and target tensors
		input_tensors = []
		target_tensors = []
		for t in training_pairs:
			input_tensors.append(t[0])
			target_tensors.append(t[1])
		input_tensors = np.concatenate(input_tensors, axis=0)
		target_tensors = np.concatenate(target_tensors, axis=0)
		# Initialize SGD and load keras net
		sgd = SGD(lr=self.learning_rate)
		self.model.compile(loss='binary_crossentropy', optimizer=sgd)
		# Update weights in + direction if player won, and - direction if player lost.
		# Setting sample_weight negative is hack for negative weights update.
		# states = np.array([s[0] for s in samples if s[1] is not None])
		if self.winner == 1:
			sw = np.ones(len(input_tensors))
		else:
			sw = np.ones(len(input_tensors)) * -1
		self.model.fit(input_tensors, target_tensors, nb_epoch=1,
					   batch_size=len(input_tensors), sample_weight=sw)
		# Save new weights
		# TODO
		self.save_model()
		return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Perform reinforcement learning '
									 'to improve given policy network. Second phase of pipeline.')
	parser.add_argument("initial_weights", help="Path to file with weights to start from.")
	parser.add_argument("initial_json", help="Path to folder of training samples")
	parser.add_argument("--model_folder", help="Path to folder where the model "
						"params will be saved after each epoch. Default: None", default=None)
	parser.add_argument(
		"--learning_rate", help="Keras learning rate (Default: .03)", type=float,
		default=.03)
	parser.add_argument(
		"--nb_worker", help="Number of threads to use when training in parallel. "
		"Requires appropriately set Theano flags.", type=int, default=1)
	parser.add_argument(
		"--save_every", help="Save policy every n mini-batches (Default: 500)",
		type=int, default=500)
	parser.add_argument(
		"--game_batch_size", help="Number of games per mini-batch (Default: 10)",
		type=int, default=10)
	# game batch size
	# Baseline function (TODO) default lambda state: 0  (receives either file 
	# paths to JSON and weights or None, in which case it uses default baseline 0)
	args = parser.parse_args()
	trainer = RL_policy_trainer


	# from ipdb import set_trace as BP
	# import pickle
	# np.set_printoptions(linewidth=160)
	# features = ["board", "ones", "turns_since", "liberties", "capture_size",
	# 			"self_atari_size", "liberties_after",
	# 			"sensibleness", "zeros"]
	# policy1 = CNNPolicy(features)
	# policy2 = CNNPolicy(features)
	# player1 = GreedyPolicyPlayer(policy1)
	# player2 = GreedyPolicyPlayer(policy2)
	# train = train_policy(player1, player2, features)

# Test get_training_pairs:
# train.make_training_pairs()
# with open('game.pkl', 'wb') as fid:
#   pickle.dump(train.training_pairs, fid)
# with open('tensors.pkl', 'wb') as fid:
#   pickle.dump(train.tensors, fid)
# with open('history.pkl', 'wb') as fid:
#   pickle.dump(train.match.state.history, fid)
# training_pairs = train.train()
# with open('board.pkl', 'wb') as fid:
#   pickle.dump(train.match.state.board, fid)
# np.save('board.npy', train.match.state.board)

# BP()
# with open('game.pkl', 'rb') as fid:
#   train.training_pairs = pickle.load(fid)
# with open('tensors.pkl', 'rb') as fid:
#   train.tensors = pickle.load(fid)
# with open('history.pkl', 'rb') as fid:
#   train.match.state.history = pickle.load(fid)
# # with open('board.pkl', 'rb') as fid:
# #     train.match.state.board = pickle.load(fid)
# train.match.state.board = np.load('board.npy')
# train.train()
