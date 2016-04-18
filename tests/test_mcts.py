from AlphaGo.go import GameState
from AlphaGo.mcts import MCTS, TreeNode
import numpy as np
import unittest


class TestMCTS(unittest.TestCase):

	def setUp(self):
		self.gs = GameState()
		self.mcts = MCTS(self.gs, value_network, policy_network, rollout_policy, n_search=2)

	def test_treenode_selection(self):
		treenode = TreeNode(None, 1.0)
		treenode.expansion(policy_network(self.gs))
		action, node = treenode.selection()
		self.assertEqual(action, (18, 18))  # according to the policy below
		self.assertIsNotNone(node)

	def test_mcts_DFS(self):
		treenode = TreeNode(None, 1.0)
		self.mcts._DFS(8, treenode, self.gs.copy())
		self.assertEqual(1, treenode.children[(18, 18)].nVisits, 'DFS visits incorrect')

	def test_mcts_getMove(self):
		move = self.mcts.get_move(self.gs)
		self.mcts.update_with_move(move)
		# success if no errors


def policy_network(state):
	moves = state.get_legal_moves(include_eyes=False)
	# 'random' distribution over positions that is smallest
	# at (0,0) and largest at (18,18)
	probs = np.arange(361, dtype=np.float)
	probs = probs / probs.sum()
	return zip(moves, probs)


def value_network(state):
	# it's not very confident
	return 0.0


def rollout_policy(state):
	# just another policy network
	return policy_network(state)


if __name__ == '__main__':
	unittest.main()
