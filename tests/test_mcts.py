from AlphaGo.go import GameState
from AlphaGo.mcts import MCTS, TreeNode
import numpy as np
import unittest


class TestMCTS(unittest.TestCase):

	def setUp(self):
		self.gs = GameState()
		self.mcts = MCTS(dummy_value, dummy_policy, dummy_rollout, n_playout=2)

	def test_treenode_selection(self):
		treenode = TreeNode(None, 1.0)
		treenode.expand(dummy_policy(self.gs))
		action, node = treenode.select()
		self.assertEqual(action, (18, 18))  # according to the dummy policy below
		self.assertIsNotNone(node)

	def test_mcts_playout(self):
		self.mcts._playout(self.gs.copy(), 8)
		self.assertEqual(1, self.mcts._root._children[(18, 18)]._n_visits, 'playout visits incorrect')

	def test_mcts_get_move(self):
		move = self.mcts.get_move(self.gs)
		self.mcts.update_with_move(move)
		# success if no errors


# A distribution over positions that is smallest at (0,0) and largest at (18,18)
dummy_distribution = np.arange(361, dtype=np.float)
dummy_distribution = dummy_distribution / dummy_distribution.sum()


def dummy_policy(state):
	moves = state.get_legal_moves(include_eyes=False)
	return zip(moves, dummy_distribution)


def dummy_value(state):
	# it's not very confident
	return 0.0


def dummy_rollout(state):
	# just another policy network
	return dummy_policy(state)


if __name__ == '__main__':
	unittest.main()
