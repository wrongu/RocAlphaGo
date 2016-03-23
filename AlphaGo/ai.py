class GreedyPolicyPlayer(object):
	"""A CNN player that uses a greedy policy (i.e. chooses the highest probability
	   move at each point)
	"""
	def __init__(self, policy_function):
		self.policy = policy_function

	def get_move(self, state):
		action_probs = self.policy.eval_state(state)
		if len(action_probs) > 0:
			sensible_actions = [a for a in action_probs if not state.is_eye(
				a[0], state.current_player)]
			if len(sensible_actions) > 0:
				max_prob = max(sensible_actions, key=lambda (a, p): p)
				return max_prob[0]
			else:
				# No legal moves available, do so pass move
				return None
		else:
			# No legal moves available, do so pass move
			return None
