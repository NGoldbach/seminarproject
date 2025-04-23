from typing import List
import numpy as np
import math

#input: data array of tuples, array of prototypes
#optional inputs:
#   max iterations(default 100), iter
#   noise variant on/off (default off), nvBool
#   noise variant distance scalar (default 0.275), nvScalar
#output: [membershipList (0 to k-1),objective function value]
def kmeans(data, initialPrototypes, nvBool=False,nvScalar=0.275, iter=100):
    membershipList = [0] * len(data)
    prototypes = np.array(initialPrototypes)
    k = len(initialPrototypes)
    nvDistSquared = 0 #noise Prototype

    #Addition of noise prototype
    if nvBool:
        k += 1
        for v in data:
            for p in prototypes:
                tempDist = np.linalg.norm(v - p)
                nvDistSquared += tempDist * tempDist
        nvDistSquared = nvDistSquared/(k-1)
        nvDistSquared = nvDistSquared*nvScalar

    for i in range(iter):
        membershipCheck = membershipList.copy()

        #update memberships
        for j in range(len(data)):
            distances = np.linalg.norm(prototypes - data[j],axis=1)
            if nvBool:
                distances = np.append(distances, math.sqrt(nvDistSquared))
            bestFit = np.argmin(distances)
            membershipList[j] = bestFit

        #update prototypes
        for j in range(k):
            if nvBool and j==k-1:
                nvDistSquared = 0
                for v in data:
                    for p in prototypes:
                        tempDist = np.linalg.norm(v - p)
                        nvDistSquared += tempDist*tempDist
                nvDistSquared = nvDistSquared / (k-1)
                nvDistSquared = nvDistSquared * nvScalar
            else:
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
        for p in range(k):
            if membershipList[v]==p:
                if p==k-1:
                    objFunc += nvDistSquared
                else:
                    distance = np.linalg.norm(data[v]-prototypes[p])
                    objFunc += distance*distance
    for m in range(len(membershipList)):
        membershipList[m] = int(membershipList[m])
    return [membershipList,objFunc]


def dbscan():
    pass