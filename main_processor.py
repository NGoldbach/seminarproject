from typing import List
import numpy as np
import math
import random

#input: data array of tuples, array of prototypes
#optional inputs:
#   max iterations(default 100), iter
#   noise variant on/off (default off), nvBool
#   noise variant distance scalar (default 0.275), nvScalar
#output: [membershipList (0 to k-1),objective function value]
def kmeans(data, k, nvBool=False,nvScalar=0.75, iter=100):
    membershipList = [0] * len(data)
    prototypes = [data[0],data[100],data[200]]
    # for i in range(k):
    #     prototypes.append(random.choice(data))
    nvDistSquared = 0 #noise Prototype
    prototypes = np.array(prototypes)
    #Addition of noise prototype
    if nvBool:
        k += 1
        for v in data:
            for p in prototypes:
                tempDist = np.linalg.norm(v - p)
                nvDistSquared += tempDist * tempDist
        nvDistSquared = nvDistSquared/(len(data)*(k-1))
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
                nvDistSquared = nvDistSquared/(len(data)*(k-1))
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


def dbscan(data, eps=0.05, min_pts=4):
    n_points = len(data)                 # Anzahl der Punkte im Datensatz
    labels = [-1]*n_points               # Alle Labels auf -1 (Rauschen) setzen
    visit = [False] * n_points            # Besuchte Punkte markieren
    cluster_id = 0                        # Cluster-Nummerierung mit 0 starten

    for i in range(n_points):
        if visit[i]:
            continue
        visit[i] = True

        # Suche Nachbarn von Punkt i
        neighbors = []
        for j in range(n_points):
            if np.linalg.norm(np.array(data[i]) - np.array(data[j])) <= eps:
                neighbors.append(j)

        if len(neighbors) >= min_pts:
            # Neuer Cluster
            labels[i] = cluster_id
            queue = neighbors.copy()      # Punkte, die noch bearbeitet werden müssen

            while queue:
                neighbor_id = queue.pop(0)

                if not visit[neighbor_id]:
                    visit[neighbor_id] = True
                    labels[neighbor_id] = cluster_id

                    # Suche Nachbarn für diesen Punkt
                    neighbors_of_neighbor = []
                    for k in range(n_points):
                        if np.linalg.norm(np.array(data[neighbor_id]) - np.array(data[k])) <= eps:
                            neighbors_of_neighbor.append(k)

                    # Nur wenn genug Nachbarn: Cluster erweitern
                    if len(neighbors_of_neighbor) >= min_pts:
                        queue.extend(neighbors_of_neighbor)

            # Nach Bearbeiten aller Nachbarn: Cluster-ID erhöhen
            cluster_id += 1

    for i in range(len(data)):
        if labels[i] == -1:
            labels[i] = cluster_id
    return labels


# Function to plot k-distance graph
#     def plot_k_distance_graph(X, k):
#         neigh = NearestNeighbors(n_neighbors=k)
#         neigh.fit(X)
#         distances, _ = neigh.kneighbors(X)
#         distances = np.sort(distances[:, k-1])
#         plt.figure(figsize=(10, 6))
#         plt.plot(distances)
#         plt.xlabel('Points')
#         plt.ylabel(f'{k}-th nearest neighbor distance')
#         plt.title('K-distance Graph')
#         plt.show()
#     # Plot k-distance graph
#     plot_k_distance_graph(X, k=5)