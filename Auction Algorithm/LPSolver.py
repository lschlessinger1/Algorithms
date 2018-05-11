from pymprog import *

def lpSolver(m, n, data):
    #problem data
    m = 8 # agents
    M = range(m) #set of agents
    n = 8 # tasks
    N = range(n) #set of tasks
    c = [ #cost
    (13,21,20,12,8,26,22,11),
    (12,36,25,41,40,11,4,8),
    (35,32,13,36,26,21,13,37),
    (34,54,7,8,12,22,11,40),
    (21,6,45,18,24,34,12,48),
    (42,19,39,15,14,16,28,46),
    (16,34,38,3,34,40,22,24),
    (26,20,5,17,45,31,37,43)]



    begin("assign")
    #verbose(True) # for model output
    A = iprod(M, N) # Descartan product
    x = var('x', A) # assignment decisions
    # use parameters for automatic model update
    c = par('c', c) # when their values change
    maximize(sum(c[i][j]*x[i,j] for i,j in A))
    # each agent is assigned to at most one task
    for k in M: sum(x[k,j] for j in N)<=1
    # each task must be assigned to somebody
    for k in N: sum(x[i,k] for i in M)==1
    solve()

def report(M, N, x):
    print("Total Cost = %g"%vobj())
    assign = [(i,j) for i in M for j in N
                if x[i,j].primal>0.5]
    for i,j in assign:
        print("Agent %d gets Task %d"%(i, j))
    return assign

lpSolver(8,8,[])
# assign = report()