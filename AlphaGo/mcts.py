
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
            self.children[action[0]] = TreeNode()

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
        self.Q_value = (self.Q_value * self.nVisits + value) / (self.nVisits + 1)
            
    def updateU_value(self, actions): 

        """Update the bonus value u(P)--proportional to the prior probability but decays with the number of visits to encourage exploration

        Keyword arguments:
        Output from policy function-- a list of tuples of (x, y) position and prior probability

        Return:
        None

        """

        for index in range(0, len(self.children)):  
            self.children[actions[index][0]].u_value = actions[index][1] / (1 + self.children[actions[index][0]].nVisits)

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
    """Monte Carlo tree search, takes an input of game state, value network function, policy network function, rollout policy function, outputs an action after lookahead search is complete.
    """

    def __init__(self, state, value_network, policy_network, rollout_policy):        
                
        self.state = state
        self.treenode = TreeNode()
        self._value = value_network
        self._policy = policy_network
        self._rollout = rollout_policy
                
    def DFS(self, nDepth, treenode, state):
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
            state.do_move(action)
            visited.insert(0, (state, treenode)) 
                
        for index in range(0, len(visited)): 
            if(visited[index][1].isLeaf() == True):
                value = self.leafEvaluation(visited[index][0])
            else:    
                value = visited[index][1].backUp(value)
        
        visited[-1][1].updateQ_value(value)
        visited[-1][1].updateVisits()
        return visited[-1][1]

    def leafEvaluation(self, state):
        """Calculate leaf evaluation, a weighted average using a mixing parameter LAMBDA, combined outcome z
        of fast rollout policy function and value network function output v.

        Keyword arguments:
        GameState object

        Return:
        value
        """
        z = self._rollout(state)
        v = self._value(state) 
        return (1-LAMBDA) * v + LAMBDA * z  
        
    def priorProb(self, state):
        """Get a list of (action, probability) pairs according to policy network function outputs

        Keyword arguments:
        GameState object

        Return:
        list of tuples ((x,y), probability)
        
        """
        actions = self._policy(state)

        return actions   
       

    def getMove(self, nDepth, nSimulations):
                
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
            state = self.state.copy()
            state.do_move(action)
            treenode = self.DFS(nDepth, treenode, state)
            self.treenode.children[action] = treenode

        self.treenode.updateU_value(actions)   
        treenode, action = self.treenode.selection()
        return action

class ParallelMCTS(MCTS):
      pass





