from AlphaGo.training.go import GameState
from AlphaGo.training.mcts import MCTS
from AlphaGo.training.mcts import TreeNode
import numpy as np
import unittest


class TestMCTS(unittest.TestCase):
    
    def setUp(self):
        self.s = GameState()
        self.mcts = MCTS()
        self.treenode = TreeNode()


    def test_treenode_selection(self):
        actions = self.mcts.priorProb(self.s)
        self.treenode.expansion(actions)
        self.treenode.updateU_value(actions)
        selectednode, selectedaction = self.treenode.selection()
        self.assertEqual(max(actions, key = lambda x:x[1]), selectednode.toValue(), 'incorrect node selected')

    def test_mcts_DFS(self):
        treenode = self.mcts.DFS(3, self.treenode, self.s) 
        self.assertEqual(1, treenode.nVisits, 'incorrect visit count')
      

    def test_mcts_getMove(self):
        action = self.mcts.getMove(2, 3)
        self.assertIsNotNone(action,'no output action') 


    
if __name__ == '__main__':
    unittest.main()


