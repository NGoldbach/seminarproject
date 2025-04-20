from typing import List
import numpy as np

#input: data array of tuples, array of prototypes,amount of max iterations
#output: [membershipList,objective function value]
def kmeans(data: List, initialPrototypes: List, iter=100):
    membershipList = [0] * len(data)
    prototypes = np.array(initialPrototypes)
    k = len(initialPrototypes)

    for i in range(iter):
        membershipCheck = membershipList.copy()

        #update memberships
        for j in range(len(data)):
            distances = np.linalg.norm(prototypes - data[j],axis=1)
            bestFit = np.argmin(distances)
            membershipList[j] = bestFit

        #update prototypes
        for j in range(k):
            centoid = (0, 0)
            counter = 0
            for v in range(len(data)):
                if membershipList[v] == j:
                    centoid = (centoid[0]+data[v][0],centoid[1]+data[v][1])
                    counter += 1
            if counter != 0:
                centoid = (centoid[0]/counter,centoid[1]/counter)
            prototypes[j] = centoid

        #Convergence Check
        if np.array_equal(membershipCheck,membershipList):
            break

    #Calculate total distance for objective function metric
    objFunc = 0
    for v in range(len(data)):
        for p in k:
            if membershipList[v]==p:
                objFunc += np.linalg.norm(data[v]-prototypes[p])

    return [membershipList,objFunc]