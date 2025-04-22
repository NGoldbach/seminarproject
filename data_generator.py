import random
import numpy as np

def noiseGenerator(data):
    # add noise points
    # return longer dataset, noise points at the end
    pass


def getDataSets():
    loaded_arrays = []

    with open("dataSets.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tuples = [
                    tuple(map(float, t.strip("()").split(",")))
                    for t in line.split()
                ]
                loaded_arrays.append(tuples)
    return loaded_arrays




def createDataSet(pointAmount, clusterAmount, maxA1, maxA2, clusterShapes=None, clusterDensity=None):
    data = []
    centers = []
    radius = 0.1  # replace with density[cluster] in loops
    buffer = 0.1 #adjust based on radius?

    for clusters in range(clusterAmount):
        valid = False
        center = (0,0)
        while not valid:
            valid = True
            center = np.array((random.uniform(radius, 1 - radius), random.uniform(radius, 1 - radius)))
            for c in centers:
                if np.linalg.norm(c - center) < radius+buffer:
                    valid = False
        centers.append(center)

    for cluster in range(clusterAmount):
        radii = np.sqrt(np.random.rand(pointAmount)) * radius
        angles = np.random.uniform(0, 2 * np.pi, pointAmount)
        points = [(centers[cluster][0] + r * np.cos(a), centers[cluster][1] + r * np.sin(a)) for r, a in zip(radii, angles)]
        points = [(a1*maxA1, a2*maxA2) for (a1, a2) in points]
        for p in points:
            data.append(p)

    with open("dataSets.txt", "a") as f:
        tuple_strs = ["({},{})".format(p[0], p[1]) for p in data]
        f.write(" ".join(tuple_strs) + "\n")
