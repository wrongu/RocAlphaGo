import numpy as np
from AlphaGo import go


class GreedyPolicyPlayer(object):
	"""A player that uses a greedy policy (i.e. chooses the highest probability
	move each turn)
	"""

	def __init__(self, policy_function):
		self.policy = policy_function

	def get_move(self, state):
		sensible_moves = [move for move in state.get_legal_moves() if not state.is_eye(move, state.current_player)]
		if len(sensible_moves) > 0:
			move_probs = self.policy.eval_state(state, sensible_moves)
			max_prob = max(move_probs, key=lambda (a, p): p)
			return max_prob[0]
		# No 'sensible' moves available, so do pass move
		return go.PASS_MOVE


class SamplingPolicyPlayer(object):
	"""A player that samples a move in proportion to the probability given by the
	policy.

	By manipulating the 'temperature', moves can be pushed towards totally random
	(high temperature) or towards greedy play (low temperature)
	"""

	def __init__(self, policy_function, temperature=1.0):
		assert(temperature > 0.0)
		self.policy = policy_function
		self.exp = 1.0 / temperature

	def get_move(self, state):
		sensible_moves = [move for move in state.get_legal_moves() if not state.is_eye(move, state.current_player)]
		if len(sensible_moves) > 0:
			move_probs = self.policy.eval_state(state, sensible_moves)
			# zip(*list) is like the 'transpose' of zip; zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
			moves, probabilities = zip(*move_probs)
			probabilities = np.array(probabilities)
			probabilities = probabilities ** self.exp
			probabilities = probabilities / probabilities.sum()
			# numpy interprets a list of tuples as 2D, so we must choose an _index_ of moves then apply it in 2 steps
			choice_idx = np.random.choice(range(len(moves)), p=probabilities)
			return moves[choice_idx]
		return go.PASS_MOVE
