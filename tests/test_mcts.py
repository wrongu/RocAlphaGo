from AlphaGo.go import GameState
from AlphaGo.mcts import MCTS
from AlphaGo.mcts import TreeNode
import random
import numpy as np
import unittest


class TestMCTS(unittest.TestCase):
    
    def setUp(self):
        self.s = GameState()
        self.mcts = MCTS(s, value_network, policy_network, rollout_policy)
        self.treenode = TreeNode()


    def test_treenode_selection(self):
        actions = self.mcts.priorProb(self.s)
        self.treenode.expansion(actions)
        self.treenode.updateU_value(actions)
        selectednode, selectedaction = self.treenode.selection()
        self.assertEqual(max(actions, key = lambda x:x[1]), selectednode.toValue(), 'incorrect node selected')
        self.assertEqual(max(actions, key = lambda x:x[0]), selectedaction, 'incorrect action selected')
    
    def test_mcts_DFS(self):
        value = self.mcts.leafEvaluation(self.s)
        treenode = self.mcts.DFS(3, self.treenode, self.s) 
        self.assertEqual(1, treenode.nVisits, 'incorrect visit count')
        self.assertEqual(value/2*9, treenode.Q_value, 'incorrect Q_value') 

    def test_mcts_getMove(self):
        action = self.mcts.getMove(2, 3)
        self.assertIsNotNone(action,'no output action') 
        print("getMove checked")

def policy_network(state):
    actions = []
    for i in range(0, 10):
        actions.append(((i, i+1), random.uniform(0, 1)))                              
    return actions 
   
def value_network(state):

    return 0.5

def rollout_policy(state):

    return 1

if __name__ == '__main__':
    unittest.main()


