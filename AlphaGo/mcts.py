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
                self.children = []  


        def selection(self):
                # select among children nodes with maximum value
                maxValue = 0
                action = (0,0)
                selectednode = TreeNode()
                for child in self.children:
                    if(child[1].toValue() > maxValue):
                        selectednode = child[1]
                        maxValue = child[1].toValue()
                        action = child[0]
                return selectednode, action

        def expansion(self, probs):
                # expand children nodes
                for prob in probs: 
                    self.children.append((prob[0],TreeNode()))
                    
        def isLeaf(self):
                # check if reaches leaf state
                return self.children == []

        def updateVisits(self):
                # update the number of visits
                self.nVisits += 1

        def updateLeafStats(self, value):
            
                # update Q value and counts of visits for leaf state
                self.Q_value = self.Q_value * self.nVisits + value
                self.nVisits += 1
                self.Q_value = self.Q_value / self.nVisits
        
        def updateBonus(self, probs):
                
                # update u value from a list of prior probability from policy function
                for index in range(0, len(self.children)):  
                    self.children[index].u_value = probs[index][1] / (1 + self.children[index].nVisits)

        def backUp(self):
                # Backpropagate values by averaging values of subtree 
                sumValue = 0
                for treenode in self.children:
                    sumValue += treenode.Q_value
                self.toValue = sumValue/len(self.children)
                
        def toValue(self):

                # evaluate value of treenode from both Q value and u value
                return self.Q_value + self.u_value



class MCTS(object):
    
        def __init__(self, state):        
                
                self.state = state
                self.treenode = TreeNode()
                

        def DFS(self, nDepth, traversed):
                # Depth First Search Tree Traverse from start state over certain depths, keep track and update statistics for all of the traversed edges: (state, action, treenode)pair
                visited = []
                treenode = self.treenode
                state = self.state

                for index in range(0, nDepth):
                        
                    probs=priorProb(state)
                    treenode.expansion(probs)
                    treenode.updateBonus(probs)
                    treenode, action = treenode.selection() 
                    # need do_move(action) function to return updated state
                    state = state.do_move(action) 
                    visited.insert(0, (state, action, treenode)) 
                
                for index in range(0, nDepth):
                    
                    if(visited[index][2].isLeaf() == False):
                        for item in traversed[index]:
                            #need function from GameState class to check if two state equals
                            if(item[0].equalstate(visited[index][0]):
                                item[2].updateVisits()
                                item[2].backUp()
                        
                        visited[index][2].updatedVisits()
                        visited[index][2].backUp()
                        traversed[index].append(visited[index])
                    else:
                        value = self.leaf_evaluation(visited[index][0])
                        for item in traversed[index]:
                            #need function from GameState class to check if two state equals
                            if(item[0].equalstate(visited[index][0]):
                                item[2].updateLeafStats(value)

                        visited[index][2].updateLeafStats(value)
                        traversed[index].append(visited[index])
                    
                        
        def leafEvaluation(self, state):
                # return weighted average between ramdom rollout by fast policy and value function 
                z = somerolloutfunction(state)
                v = somevaluefunction(state)
                return (1-LAMBDA)*v+LAMBDA*z  
        
        def priorProb(self, state):
                
                # return a list of action, prior probality of from policy function of a given state
                probs = somepolicyfunction(state)
                return probs

        def getMove(self, nSimulations, nDepth = 20):
               
                # run MSTC simulations for a number of times and return the best move 
                traversed = []

                for index in range(0, nDepth):
                    sublist = []
                    traversed.append(sublist)

                for n in range(0, nSimulations):
                    self.DFS(nDepth, traversed)

                treenode, action = self.treenode.selection() 
                return action





class ParallelMCTS(MCTS):
        pass
