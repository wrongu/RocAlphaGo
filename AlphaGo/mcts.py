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


        def DFS(self, Depth):
                # need to move this part to MCTS class
                visited = [] 
                visited.insert(0,self)
                for n in range(0,Depth):
                    self.expansion(nActions)
                    self = self.selection()
                    visited.insert(0,self)
                for treenode in visited:
                    treenode.updateVisits()
                    if(treenode.isLeaf()==False):
                        treenode.backup()
                    else:
                        

        def selection(self):
                # select among children nodes with maximum value
                maxValue = 0
                selected = TreeNode()
                for treenode in self.children:
                    if(treenode.toValue > maxValue):
                        selected = treenode
                        maxValue = treenode.toValue
                return selected

        def expansion(self, nActions):
                # expand children nodes
                for i in range(0, nActions): 
                    self.children.append(TreeNode())
                    
        def isLeaf(self):
                # check if reaches leaf state
                return self.children == []

        def updateVisits(self):
                # update the number of visits
                self.nVisits += 1

        def updateValues(self, value):
                # update values 
                self.toValue += value

        def backup(self):
                # Backpropagate values by averaging values of subtree 
                sumValue = 0
                for treenode in self.children:
                    sumValue += treenode.toValue
                self.toValue = sumValue/len(self.children)
                
                

class MCTS(object):
    pass
    

class ParallelMCTS(MCTS):
	pass
