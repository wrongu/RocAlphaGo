from __future__ import division
import numpy as np
import policy
import value
import shallow_policy
import random
import go


LAMBDA = 0.5



class TreeNode(object):
         
        def __init__(self):
                
                self.nVisits = 0
                self.Q_value = 0
                self.u_value = 0
                self.children = {} 


        def selection(self):
                # select among children nodes with maximum value, return selected treenode and action
                selectednode = self.children.values()[0]
                action = self.children.keys()[0]
                maxValue = selectednode.toValue()
                
                for child in self.children.items():
                    if(child[1].toValue() > maxValue):
                        selectednode = child[1]
                        maxValue = child[1].toValue()
                        action = child[0]
                return selectednode, action

        def expansion(self, actions):
                # expand children nodes to a dict with action(x, y) as keys, and TreeNode object as values
                for action in actions: 
                    self.children[actions[0]] = TreeNode()
                    
        def isLeaf(self):
                # check if reaches leaf state
                return self.children == {}

        def updateVisits(self):
                # update the number of visits
                self.nVisits += 1

        def updateQ_value(self, value):
                # update Q value for leaf state
                self.Q_value = value
            
        def updateU_value(self, actions): 
                # update u value from a list of prior probability from policy function
                for index in range(0, len(self.children)):  
                    self.children.values()[index].u_value = actions[index][1] / (1 + self.children.values()[index].nVisits)

        def backUp(self):
                # Backpropagate values by averaging values of subtree 
                sumValue = 0
                for child in self.children.items():
                    sumValue += child[1].Q_value
                self.Q_value = sumValue / len(self.children)
                
        def toValue(self):
                # evaluate the value of treenode with both Q value and u value
                return self.Q_value + self.u_value

class MCTS(object):
    
        def __init__(self, state):        
                
                self.state = state
                self.treenode = TreeNode()
                
        def DFS(self, nDepth, treenode, state):
                # Depth First Search Tree Traverse from start state over certain depths, keep track and update statistics for childrenlist of (state, treenode) pair
                visited = []
                visited.insert(0, (state, treenode))
                
                for index in range(0, nDepth-1):      
                    #actions = self.priorProb(state)
                    actions = self.priorProb()
                    treenode.expansion(actions)
                    treenode.updateU_value(actions)
                    treenode, action = treenode.selection() 
                    state = state.do_move(action).copy()
                    visited.insert(0, (state, treenode)) 
                
                   # value = self.leafEvaluation(visited[0][0])
                    value = self.leafEvaluation()
                for index in range(1, len(visited)): 
                    value /= len(visited[index][1].children)
                   # if(visited[index][1].isLeaf() == True):
                        #value = self.leafEvaluation(visited[index][0])
                    #    value = self.leafEvaluation()
                    #    visited[index][1].updateQ_value(value)
                   # else:    
                    #    visited[index][1].backUp()
                visited[-1][1].updateQ_value(value)
                visited[-1][1].updateVisits()    
                return visited[-1][1]

        def leafEvaluation(self): #state):
                # return weighted average between rollout and value function 
                # z = somerolloutfunction(state)
                # for testing purposes, use random rollout instead
                z = np.random.randint(2)
                # v = somevaluefunction(state)
                # for testing purposes, use random values instead
                v = random.uniform(0, 1) 
                return (1-LAMBDA) * v + LAMBDA * z  
        
        def priorProb(self): #state):
                # return a list of (action, prior probality) pair from policy function of a given state
                #actions = somepolicyfunction(state)
                # for testing purposes, use random values instead
                actions = []
                for i in range(0, random.randrange(50, 300)):
                    actions.append(((np.random.randint(18), np.random.randint(18)), random.uniform(0, 1))) 
                return actions

        def getMove(self, nSimulations, nDepth = 20):
                # run MSTC simulations for a number of times and return the best move 
                #actions = self.priorProb(self.state)
                actions = self.priorProb()
                self.treenode.expansion(actions)

                for n in range(0, nSimulations):                    
                
                    self.treenode.updateU_value(actions)
                    treenode, action = self.treenode.selection()     
                    state = state.do_move(action).copy()
                    treenode = self.DFS(nDepth, treenode, state)
                    self.treenode.children[action] = treenode
                
                self.treenode.updateU_value(actions)   
                treenode, action = self.treenode.selection()
                return action

class ParallelMCTS(MCTS):
        pass
