# reference: http://pymprog.sourceforge.net/advanced.html#assign

from pymprog import *

class AssignmentLP:

    def __init__(self, data, numAgents, numObjects):
        # problem data
        self.c = data
        self.data = data
        m = numAgents
        self.M = range(m)
        n = numObjects
        self.N = range(n)

    def run(self):

        begin("assign")

        # verbose(True) # for model output
        A = iprod(self.M, self.N)  # Descartan product

        self.x = var('x', A)  # assignment decisions

        # use parameters for automatic model update when their values change
        self.c = par('c', self.c)

        maximize(sum(self.c[i][j] * self.x[i, j] for i, j in A))

        # each agent is assigned to at most one task
        for k in self.M: sum(self.x[k, j] for j in self.N) <= 1

        # each task must be assigned to somebody
        for k in self.N: sum(self.x[i, k] for i in self.M) == 1

        solve()

        self.report()

        end()


    def report(self):
        print("Total Cost = %g" % vobj())
        assign = [(i, j) for i in self.M for j in self.N
                  if self.x[i, j].primal > 0.5]
        for i, j in assign:
            print("Agent %d -> Object %d, value: %d" % (i, j, self.data[i][j]))
        return assign



