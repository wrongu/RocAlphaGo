from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.models.policy import CNNPolicy
from interface.Play import play_match
import numpy as np


class train_policy(object):
	"""Train reinforcement learning policy. First, extract training tuples from
	game (`get_training_pairs`), then use those tuples for reinforcement learning.
	Call `train_policy.train()` to first extract training pairs and then run
	training. Initialize object with any two policy players (must have `get_move`
	function that returns a single move tuple given board state array).
	"""

	def __init__(self, player1, player2):
		self.player1 = player1
		self.player2 = player2
		self.match = play_match(player1, player2, 'test')
		self.training_pairs = []

	def get_training_pairs(self):
		while True:
			end_of_game = self.match.play()
			print "turns played:", self.match.state.turns_played
			# Append training pair
			self.training_pairs.append((self.match.state.board.copy(), self.match.state.history[-1]))
			# if self.match.state.turns_played % 1 == 0:
			# 	print self.match.state.board
			if end_of_game:
				break
		return self.training_pairs

	def train(self):
		training_pairs = self.get_training_pairs()
		# TODO...
		return
