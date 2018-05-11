import NaiveAuctionAlgorithm as naa
from time import time

from random import randint
import AssignmentLP as lp
from pymprog import *


class AssignmentSolver:

    def __init__(self, dataFile):
        self.data = self.parseDataFile(dataFile)
        self.numObjects = len(self.data[0])
        self.numAgents = len(self.data)
        self.algorithm = naa.NaiveAuctionAlgorithm(self.numObjects, self.numAgents, self.data)

    def solve(self):
        self.algorithm.run()

    def parseDataFile(self, file):
        data = []
        dataFile = open(file, "r")

        for row in dataFile:
            # convert list of str to list of int
            dataRow = row.split()
            dataRow = map(int, dataRow)
            data.append(dataRow)

        return data

    def generateRandomData(self, n, M, fileName="generatedData.txt"):
        """

        :param n: number of agents (= number of objects)
        :param M: value of each assignment is an int sampled uniformly at random [0, M - 1]
        :return:
        """
        numAgents = n
        numObjects = n

        dataFile = open(fileName, "w")

        for i in range(numAgents):
            for j in range(numObjects):
                value = randint(1, M)
                optionalSpace =  " " if j != numObjects - 1 else ""
                dataFile.write(str(value) + optionalSpace)
            dataFile.write("\n")

    def simulateProblem1b(self):
        M = 100
        for i in range(8):
            n = 2**(i+1)
            print("n = " + str(n))
            self.generateRandomData(n, M)
            data = self.parseDataFile("generatedData.txt")
            numObjects = len(data[0])
            numAgents = len(data)
            algorithm = naa.NaiveAuctionAlgorithm(numObjects, numAgents, data)
            perAgentAvgs = dict()

            numRuns = 1000
            for runNum in range(numRuns):
                algorithm.run()

                agentValsDict = algorithm.getPerAgentValue()

                for key, value in agentValsDict.iteritems():
                    if key not in perAgentAvgs:
                        perAgentAvgs[key] = value
                    else:
                        perAgentAvgs[key] += value
            arr = []
            perAgentAvgAverage = 0
            for key, value in perAgentAvgs.iteritems():
                perAgentAvgs[key] /= float(numRuns)
                perAgentAvgAverage += perAgentAvgs[key]
                print(str(perAgentAvgs[key]))
            perAgentAvgAverage /= numAgents
            print ("perAgentAvgAverage: "+str(perAgentAvgAverage))

    def simulateProblem1c(self):
        n = 128
        for i in range(7):
            M = 10 ** (i + 1)

            print("M = " + str(M))
            # assign = report()
            numRuns = 100
            totalTime = 0
            for runNum in range(numRuns):

                self.generateRandomData(n, M)
                dataFile = "generatedData.txt"
                data = self.parseDataFile(dataFile)
                numObjects = len(data[0])
                numAgents = len(data)
                # algorithm = lp.AssignmentLP(data, numAgents, numObjects)
                algorithm = naa.NaiveAuctionAlgorithm(numObjects, numAgents, data)
                t0 = time()
                algorithm.run()
                # self.lpSolver(numAgents, numObjects, data)
                # agentValsDict = algorithm.getPerAgentValue()
                totalTime += round(time() - t0, 3)
            totalTime /= numRuns
            print ("avg simulation time:", totalTime, "s")
    def lpSolver(self, m, n, data):
        # problem data
        M = range(m)  # set of agents
        N = range(n)  # set of tasks
        c = data

        begin("assign")
        # verbose(True) # for model output
        A = iprod(M, N)  # Descartan product
        x = var('x', A)  # assignment decisions
        # use parameters for automatic model update
        c = par('c', c)  # when their values change
        maximize(sum(c[i][j] * x[i, j] for i, j in A))
        # each agent is assigned to at most one task
        for k in M: sum(x[k, j] for j in N) <= 1
        # each task must be assigned to somebody
        for k in N: sum(x[i, k] for i in M) == 1
        solve()
        # print("Total Cost = %g" % vobj())
        assign = [(i, j) for i in M for j in N
                  if x[i, j].primal > 0.5]
        # for i, j in assign:
        #     # print("Agent %d gets Task %d" % (i, j))

        agentValsDict = dict()
        for i,j in assign:
            agent = i
            agentValue = data[i][j]
            agentValsDict[agent] = agentValue
        return agentValsDict

        # return assign


solver = AssignmentSolver("data.txt")

solver.simulateProblem1c()
