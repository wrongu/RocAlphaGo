from AlphaGo.go import GameState
import numpy as np
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.models.value import value_trainer
import random


LAMBDA = 0.5



class TreeNode(object):
"""Tree Representation of MCTS that covers Selection, Expansion, Evaluation, Backup
"""
        def __init__(self):
                
                self.nVisits = 0
                self.Q_value = 0
                self.u_value = 0
                self.children = {} 


        def expansion(self, actions):
        """Expand subtree --a dictionary with a tuple of (x,y) position as keys, TreeNode object as values

        Keyword arguments:
        Output from policy function-- a list of tuples of (x, y) position and prior probability

        Return:
        None

        """
                for action in actions: 
                    self.children[actions[0]] = TreeNode()

        def selection(self):
        """Select among subtree to get the position that gives maximum action value Q plus bonus u(P)

        Keyword arguments:
        None. 

        Return:
        action -- a tuple of (x, y)
        treenode object
        

        """
                selectednode = self.children.values()[0]
                selectedaction = self.children.keys()[0]
                maxValue = selectednode.toValue()
                
                for child in self.children.items():
                    if(child[1].toValue() > maxValue):
                        selectednode = child[1]
                        maxValue = child[1].toValue()
                        selectedaction = child[0]

                return selectednode, selectedaction


                    
        def isLeaf(self):
        """Check if leaf state is reached
        """

                return self.children == {}

        def updateVisits(self):
        """Update the count of visit times 
        """
                self.nVisits += 1

        def updateQ_value(self, value):
        """Update the action value Q 
        """
                (self.Q_value * self.nVisits + value) / (self.nVisits + 1)
            
        def updateU_value(self, actions): 

        """Update the bonus value u(P)--proportional to the prior probability but decays with the number of visits to encourage exploration

        Keyword arguments:
        Output from policy function-- a list of tuples of (x, y) position and prior probability

        Return:
        None

        """

                for index in range(0, len(self.children)):  
                    self.children.values()[index].u_value = actions[index][1] / (1 + self.children.values()[index].nVisits)

        def backUp(self, value):

        """Track the mean value of evaluations in the subtrees

        Keyword arguments:
        value of traversed subtree evaluation each simulation

        Return:
        Mean value

        """
                return value / len(self.children)

                
        def toValue(self):
        """Return action value Q plus bonus u(P)
        """
                return self.Q_value + self.u_value

class MCTS(object):
        """Monte Carlo tree search, takes an input of game state, outputs an action after lookahead search is complete.
        """

        def __init__(self, state):        
                
                self.state = GameState()
                self.treenode = TreeNode()
                
        def DFS(self, nDepth = 20, treenode, state):
        """Monte Carlo tree search over a certain depth per simulation, at the end of simulation, 
        the action values and visits of counts of traversed treenode are updated.

        Keyword arguments:
        Initial GameState object
        Initial TreeNode object 
        Search Depth

        Return:
        TreeNode object with updated statistics(visit count N, action value Q)
        """
                
                visited = []
                visited.insert(0, (state, treenode))
                
                for index in range(0, nDepth-1):      
                    actions = self.priorProb(state)
                    treenode.expansion(actions)
                    treenode.updateU_value(actions)
                    treenode, action = treenode.selection() 
                    state = state.do_move(action).copy()
                    visited.insert(0, (state, treenode)) 
                
                for index in range(0, len(visited)-1): 
                    if(visited[index][1].isLeaf() == True):
                        value = self.leafEvaluation(visited[index][0])
                    else:    
                        value = visited[index][1].backUp(value)
                
                visited[-1][1].updateQ_value(value)
                visited[-1][1].updateVisits()    
                return visited[-1][1]

        def leafEvaluation(self, state):
        """Calculate leaf evaluation, a weighted average using a mixing parameter LAMBDA, combined outcome z
        of a random rollout using the fast rollout policy and value network output v.

        Keyword arguments:
        GameState object

        Return:
        value
        """

        """
        Use random generated values for now
        """
                z = np.random.randint(2)
                v = random.uniform(0, 1) 
                return (1-LAMBDA) * v + LAMBDA * z  
        
        def priorProb(self, state):
        """Get a list of (action, probability) pairs according to policy network outputs

        Keyword arguments:
        GameState object

        Return:
        list of tuples ((x,y), probability)
        """
    
                policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])
                actions = policy.eval_state(state)     
       
                return actions

        def getMove(self, nSimulations):
                
        """After running simulations for a certain number of times, when the search is complete, an action is selected 
        from root state

        Keyword arguments:
        Number of Simulations

        Return:
        action -- a tuple of (x, y)
        """

                actions = self.priorProb(self.state)
                self.treenode.expansion(actions)

                for n in range(0, nSimulations):                    
                
                    self.treenode.updateU_value(actions)
                    treenode, action = self.treenode.selection()     
                    state = self.state.do_move(action).copy()
                    treenode = self.DFS(nDepth, treenode, state)
                    self.treenode.children[action] = treenode
                
                self.treenode.updateU_value(actions)   
                treenode, action = self.treenode.selection()
                
                return action

class ParallelMCTS(MCTS):
        pass
