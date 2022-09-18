# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from inspect import currentframe
from stat import FILE_ATTRIBUTE_NOT_CONTENT_INDEXED
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remains
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoods = newFood.asList()
        nearestGhostDistance = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()])

        if min(newScaredTimes) > nearestGhostDistance:
            return 10 / nearestGhostDistance

        if nearestGhostDistance < 3:
            return - len(newFoods) - 200

        minManhattanDistance = 10000
        for foodPos in newFoods:
            currManhattanDistance = util.manhattanDistance(newPos, foodPos)
            if currManhattanDistance < minManhattanDistance:
                minManhattanDistance = currManhattanDistance
        return - len(newFoods) + 1 / (minManhattanDistance + 1)

def scoreEvaluationFunction(currentGameState: GameState):
    """)
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agentList = ['Pacman'] + ['Ghost'] * (gameState.getNumAgents() - 1)

        def maxAndmin(currState, currDepth, index):
            currAgent = agentList[index]

            if currAgent == 'Pacman':
                nextDepth = currDepth - 1
                agentMode, optimalValue = max, -float('inf')
            elif currAgent == "Ghost":
                nextDepth = currDepth
                agentMode, optimalValue = min, float('inf')

            if currState.isWin() or currState.isLose() or nextDepth < 0:
                return (self.evaluationFunction(currState), None)

            nextAgentIndex = index + 1 if index != len(agentList) - 1 else 0
            optimalAction = None

            for childAction in currState.getLegalActions(index):
                currentValue = maxAndmin(currState.generateSuccessor(index, childAction), nextDepth, nextAgentIndex)[0]
                if agentMode(optimalValue, currentValue) == currentValue:
                    optimalValue, optimalAction = currentValue, childAction
            return (optimalValue, optimalAction)

        return maxAndmin(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentList = ['Pacman'] + ['Ghost'] * (gameState.getNumAgents() - 1)

        def alphaBeta(currState, currDepth, index, alpha, beta):
            currAgent = agentList[index]

            if currAgent == 'Pacman':
                nextDepth = currDepth - 1
                agentMode, optimalValue = max, -float('inf')
            elif currAgent == "Ghost":
                nextDepth = currDepth
                agentMode, optimalValue = min, float('inf')
                
            if currState.isWin() or currState.isLose() or nextDepth < 0:
                return (self.evaluationFunction(currState), None)

            nextAgentIndex = index + 1 if index != len(agentList) - 1 else 0
            optimalAction = None

            for childAction in currState.getLegalActions(index):
                currentValue = alphaBeta(currState.generateSuccessor(index, childAction), nextDepth, nextAgentIndex, alpha, beta)[0]
                if agentMode(optimalValue, currentValue) == currentValue:
                    optimalValue, optimalAction = currentValue, childAction
            
                if agentMode == max:
                    if optimalValue > beta:
                        return (optimalValue, optimalAction)
                    alpha = agentMode(alpha, optimalValue)
                else:
                    if optimalValue < alpha:
                        return (optimalValue, optimalAction)
                    beta = agentMode(beta, optimalValue)

            return (optimalValue, optimalAction)

        return alphaBeta(gameState, self.depth, 0, -float('inf'), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentList = ['Pacman'] + ['Ghost'] * (gameState.getNumAgents() - 1)

        def expectiMax(currState, currDepth, index):
            currAgent = agentList[index]

            if currAgent == 'Pacman':
                nextDepth = currDepth - 1
                agentMode, optimalValue = max, -float('inf')
            elif currAgent == "Ghost":
                nextDepth = currDepth
                agentMode, optimalValue = 'exp', 0

            if currState.isWin() or currState.isLose() or nextDepth < 0:
                return (self.evaluationFunction(currState), None)

            nextAgentIndex = index + 1 if index != len(agentList) - 1 else 0
            optimalAction = None

            for childAction in currState.getLegalActions(index):
                currentValue = expectiMax(currState.generateSuccessor(index, childAction), nextDepth, nextAgentIndex)[0]
                
                if agentMode == max:
                    if agentMode(optimalValue, currentValue) == currentValue:
                        optimalValue, optimalAction = currentValue, childAction
                
                else: #expecti
                    optimalValue += 1 / len(currState.getLegalActions(index)) * currentValue
            return (optimalValue, optimalAction)

        return expectiMax(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Priority; 1. Avoid Stop 2. Ghost Hunting 3. Avoid Lose 4. Food-gobbling


    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFoods = currentGameState.getFood().asList()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    nearestGhostDistance = min([util.manhattanDistance(currPos, ghostPos) for ghostPos in currentGameState.getGhostPositions()])

    if len(currentGameState.getLegalActions()) < 3 and not currentGameState.isLose() and not currentGameState.isWin():
        return - len(currFoods) - 500

    if min(currScaredTimes) > nearestGhostDistance:
        return 10 / nearestGhostDistance

    if nearestGhostDistance < 2:
        return - len(currFoods) - 200

    minManhattanDistance = 10000
    for foodPos in currFoods:
        currManhattanDistance = util.manhattanDistance(currPos, foodPos)
        if currManhattanDistance < minManhattanDistance:
            minManhattanDistance = currManhattanDistance

    return - len(currFoods) + 1 / (minManhattanDistance + 1)

# Abbreviation
better = betterEvaluationFunction
