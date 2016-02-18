import numpy as np
import policy
import value
import shallow_policy
import random
import go

SIMULATIONS = 100
LAMBDA = 0.5
L = 20

class TreeNode(object):
         
        def __init__(self):
                self.nVisits = 0
                self.toValue = 0
                self.children = []
                # get the number of legal moves from a given state
                self.nActions =  
                 
        
        def selectAction(self):
                visited=[] 

        def selection(self):
                # select among children nodes with maximum value
                maxValue = 0
                selected = TreeNode()
                for treenode in self.children:
                    if(treenode.toValue > maxValue):
                        selected = treenode
                        maxValue = treenode.toValue
                return selected

        def expansion(self):
                # expand children nodes
                for i in range(0, nActions): 
                    self.children.append(TreeNode())
                    
        def isLeaf(self):
                # check if reaches leaf state
                return self.children == []

        def updateStats(self, value):
                # update the number of visits and values
                self.nVisits += 1
                self.toValue += value
        
        def evaluation(self, value, rollout):
                #each treenode is evaluated using a weighted average between value network and fast rollout policy
                return (1-LAMBDA)*value+LAMBDA*rollout         

        def backup(self):
                        

class MCTS(object):
    pass
    

class ParallelMCTS(MCTS):
	pass
