import random
import numpy as np

class NaiveAuctionAlgorithm:

    def __init__(self, numObjects, numAgents, data):
        self.numAgents = numAgents  # n
        self.numObjects = numObjects # n

        self.agents = np.arange(self.numAgents) # N
        self.objects = np.arange(self.numObjects) # X

        self.dataMatrix  = data
        self.epsilon = 1.0 / numAgents # O(1/n)

    def run(self):
        # initialize
        self.initialize()
        while not (self.isFeasible(self.assignments)):
            # bidding step. i  = any unassigned agent
            i = random.choice(np.delete(self.agents, [x[0] for x in self.assignments]))

            # Find an object j in X that offers i maximal value at current prices:
            j = self.getBestMatch(i) # for each object k, get the max of (value(i, k) - p_k)

            # Compute i's bid increment for j (b_i)
            # This is the difference between the value to i of the best and second-best objects at current prices
            # (note that i's bid will be the current price plus this bid increment).
            # dif in utility b/t best and second best obj
            bidIncr = self.utility(i, j) - self.getSecondBestMatch(i, j) + self.epsilon

            # Assignment Step:
            self.assignments.append((i, j))

            #if there is another pair (i', j) then
            assignedObjects = [x[1] for x in self.assignments]
            assignment = self.assignments[assignedObjects.index(j)]
            if (j in assignedObjects and assignment[0] is not i):
                self.assignments.remove(assignment)
            self.prices[j] += bidIncr

        # self.printAssignments()
        # self.printAgentValues()

    def initialize(self):
        self.assignments = []
        self.prices = []

        # for each object, set price to 0
        for j in range(0, self.numObjects):
            self.prices.append(0)

    def getBestMatch(self, i):
        """ give me an object s.t. it maximizes

        :param i:
        :return:
        """
        utils = []

        for k in range(0, self.numObjects):
            utils.append(self.utility(i, k))

        return self.objects[utils.index(max(utils))]

    def getSecondBestMatch(self, i, j):
        utils = []

        for k in range(0, self.numObjects):
            if (k != j):
                utils.append(self.utility(i, k))

        return max(utils)

    def utility(self, i, j):
        return self.value(i, j) - self.prices[j]

    def value(self, i, j):
        return self.dataMatrix[i][j]

    def isFeasible(self, assignments):
        """ Check if a set of assignments is feasible. The assignment set is feasible
        if all agents are assigned to an object.

        :param assignments:
        :return: boolean
        """
        agentsListCopy = self.agents.tolist()
        assignedAgents = [x[0] for x in assignments]

        return set(assignedAgents) == set(agentsListCopy)

    def printAssignments(self):
        print(self.assignments)

    def printAgentValues(self):
        for pair in self.assignments:
            print("agent " + str(pair[0]) + ": " + str(self.value(pair[0], pair[1])))

    def getPerAgentValue(self):
        agentValsDict = dict()
        for pair in self.assignments:
            agent = pair[0]
            agentValue = self.value(pair[0], pair[1])
            agentValsDict[agent] = agentValue
        return agentValsDict