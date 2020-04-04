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


import random
import math
import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa

#Worst/Best case score constants
BEST_SCORE = float("inf")
WORST_SCORE = float("-inf")
WORST_POSITION = (BEST_SCORE, BEST_SCORE)
WORST_DISTANCE = BEST_SCORE

#Returns the euclidean distance between 2 points
def euclideanDistance(pt1, pt2):
  return math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        #Generate next game state based on supplied action
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        #Pacmans location
        newPos = successorGameState.getPacmanPosition()

        #Grid of Food and the number of food
        newFood = successorGameState.getFood()
        foodCount = successorGameState.getNumFood()
        #0 if nothing eaten by action, 1 otherwise
        foodDiff = currentGameState.getNumFood() - foodCount

        #Ghost information
        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        #0 if capsule not eaten by action, 1 otherwise
        newCapsules = successorGameState.getCapsules()
        capDiff = len(currentGameState.getCapsules()) - len(newCapsules)

        #if action causes a loss return worst score
        if newPos in ghostPositions:
          return WORST_SCORE
        #if action causes a win or capsule to be eaten return best score
        if foodCount == 0 or capDiff == 1:
          return BEST_SCORE
       

        #0 will ignore the ghosts. If a ghost is close this becomes negative, subtracting from the score
        ghostWeight = 0.0
        foodWeight = 1.0
        #closestFood stores the average of the manhattan distance and the euclidean distance for an element
        closestFood = WORST_DISTANCE
        #closestFoodPos stores the grid location of the closest food
        closestFoodPos = WORST_POSITION

        #If we don't eat anything with the action we need to find the closest food
        if foodDiff == 0:
          avg = 0.0
          #get the locations of food
          food = newFood.asList()
          #compute all their distances
          dist = [(manhattanDistance(newPos, element))for element in food]
          #find the closest distance and its position tuple
          closestFood = min(dist)
          if dist.count(closestFood) > 1:
            dist = [euclideanDistance(newPos, element) for element in food]
            closestFood = min(dist)
            
          closestFoodPos = food[dist.index(closestFood)]
         
        else:
          #the action resulted in us eating, so return 1, which will always be preffered to 1/closestFood
          closestFood = 1.0

        #stores the closest ghosts manhattan distance
        closestGhost = WORST_DISTANCE
        #stores the grid location of the closest ghost
        closestGhostPos = WORST_POSITION
        closestGhostState = None
 
        #Finds the closest ghost
        for ghostState in newGhostStates:
          ghost = ghostState.getPosition()
          gmanDist = manhattanDistance(newPos, ghost)
          if gmanDist < closestGhost: 
            closestGhost = gmanDist
            closestGhostPos = ghost
            closestGhostState = ghostState
          
        

         #best to wait it till the ghost moves
        if action != "Stop" and closestFoodPos == closestGhostPos:
          return WORST_SCORE
        #if the ghost is away from the closest food, waiting isn't helpful
        if action == "Stop" and closestFoodPos != closestGhostPos:
          return WORST_SCORE
        
        if closestGhostState.scaredTimer > 5 or all(ghost.scaredTimer > 5 for ghost in newGhostStates):
          return (1.0/closestFood)
        #if the closest ghost is within 5 squares and its scaredTimer is less than 5, 
        # change ghost weight to a larger negative value so the closer the ghost
        #the more negative this value is. 1 block away => -6, 5 blocks away => -2. Otherwise ghostWeight is 0 and doest effect
        #the return values
        if closestGhost < 5.0 and closestGhostState.scaredTimer < 5:
          ghostWeight = -1.0*(7.0 - closestGhost)
        
        return ((foodWeight/closestFood) + (ghostWeight/closestGhost))


def scoreEvaluationFunction(currentGameState):
    """
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def moveAgent(self, agentID, gameState,  currentDepth = 0):
      """
        Params:
          agentID: Identifier of the playing agent (agentID = 0 => pacman, agentID >= 1 => ghost)
          gameState: The current game state to expand
          currentDepth: The amount of levels explored (1 means pacman has moved once and ghosts have moved once)
        Returns:
          max score when agentID is 0
          min score when agentID is >= 1 
      """
      #if the game has been won or lost return the evaluation of the gameState, no further expantion needed
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      
      #Get legal actions for the current agent
      agentActions = gameState.getLegalActions(agentID)

      #Return the evaluation of the board if the agent has no actions availible
      if len(agentActions) == 0:
        return self.evaluationFunction(gameState)

      #Generate the agents successor states based on the list of availible actions
      agentStates = [gameState.generateSuccessor(agentID, move) for move in agentActions]

      #Pacman is the agent
      if agentID == 0:
        #start moving the ghosts for each successor state generated above, and increase our current depth 
        scores = [self.moveAgent(1, state, currentDepth + 1) for state in agentStates]
        return max(scores)

      #A ghost is the agent
      else:
        agentID += 1
        #We have expanded all the ghosts
        if agentID == gameState.getNumAgents():
          #If we have reached the max depth and we've expanded all the ghosts, evaluate the states 
          if currentDepth == self.depth:
            scores = [self.evaluationFunction(state) for state in agentStates]
          #We haven't reached the depth limit yet, so we move pacman next
          else:
            scores = [self.moveAgent(0, state, currentDepth) for state in agentStates]
        #Expand the next ghost
        else:
          scores = [self.moveAgent(agentID, state, currentDepth) for state in agentStates]
        
        return min(scores)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

        """
        #If the pacman has won or lost, respond with action Stop
        if gameState.isWin() or gameState.isLose(): 
          return Directions.STOP
        #Get all of pacmans availible actions
        pacmanLegalActions = gameState.getLegalActions()
        #Generate states from the availible actions
        pacmanStates = [gameState.generateSuccessor(0, move) for move in pacmanLegalActions]
        #Expand each state and create a list of scores from them
        pacmanScores = [self.moveAgent(1, state, 1) for state in pacmanStates]
        #store the index of the maximum score
        bestMove = pacmanScores.index(max(pacmanScores))

        #Since the states were created using pacmans "Legal" actions and the scores were generated from those states
        #the index of the max score value will also be the index of the instruction that resulted in that score
        return pacmanLegalActions[bestMove]
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def moveAgent(self, agentID, gameState,  currentDepth = 0):
      """
        Params:
          agentID: Identifier of the playing agent (agentID = 0 => pacman, agentID >= 1 => ghost)
          gameState: The current game state to expand
          currentDepth: The amount of levels explored (1 means pacman has moved once and ghosts have moved once)
        Returns:
          max score when agentID is 0
          min score when agentID is >= 1 
      """
      #if pacman has win or lost return the evaluation of the gameState, no further exploration needed
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      
      #Get legal actions for the current agent
      agentActions = gameState.getLegalActions(agentID)

      #Return the evaluation of the board if the agent has no actions availible
      if len(agentActions) == 0:
        return self.evaluationFunction(gameState)

      #Generate the agents successor states based on the list of availible actions
      agentStates = [gameState.generateSuccessor(agentID, move) for move in agentActions]

      #Pacman is the agent
      if agentID == 0:
        #start moving the ghosts for each successor state generated above, and increase our current depth 
        scores = [self.moveAgent(1, state, currentDepth + 1) for state in agentStates]
        return float(max(scores))

      #A ghost is the agent
      else:
        agentID += 1
        #We have expanded all the ghosts
        if agentID == gameState.getNumAgents():
          #If we have reached the max depth and we've expanded all the ghosts, evaluate the states 
          if currentDepth == self.depth:
            scores = [self.evaluationFunction(state) for state in agentStates]
          #We haven't reached the depth limit yet, so we move pacman next
          else:
            scores = [self.moveAgent(0, state, currentDepth) for state in agentStates]
        #Expand the next ghost
        else:
          scores = [self.moveAgent(agentID, state, currentDepth) for state in agentStates]
        
        return float(sum(scores)) / float(len(scores))

    def getAction(self, gameState):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
      """
      "*** YOUR CODE HERE ***"
      #If the pacman has won or lost, respond with action Stop
      if gameState.isWin() or gameState.isLose(): 
        return Directions.STOP
      #Get all of pacmans availible actions
      pacmanLegalActions = gameState.getLegalActions()
      #Generate states from the availible actions
      pacmanStates = [gameState.generateSuccessor(0, move) for move in pacmanLegalActions]
      #Expand each state and create a list of scores from them
      pacmanScores = [self.moveAgent(1, state, 1) for state in pacmanStates]
      #store the index of the maximum score
      bestMove = pacmanScores.index(max(pacmanScores))

      #Since the states were created using pacmans "Legal" actions and the scores were generated from those states
      #the index of the max score value will also be the index of the instruction that resulted in that score
      return pacmanLegalActions[bestMove]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
