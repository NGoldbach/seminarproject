import math
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




def createDataSet(pointAmount, clusterAmount, maxA1, maxA2, clusterShapes=['circle','line','ring'], clusterDensity=None):
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
        if clusterShapes[cluster] == 'line':
            line(data, pointAmount, centers, cluster, radius, maxA1, maxA2)
        elif clusterShapes[cluster] == 'circle':
            circle(data,pointAmount,centers,cluster,radius,maxA1,maxA2)
        elif clusterShapes[cluster] == 'ring':
            ring(data,pointAmount,centers,cluster,radius,maxA1,maxA2)


    with open("dataSets.txt", "a") as f:
        tuple_strs = ["({},{})".format(p[0], p[1]) for p in data]
        f.write(" ".join(tuple_strs) + "\n")



def circle(data,pointAmount,centers,cluster,radius,maxA1,maxA2):
    radii = np.abs(np.random.normal(loc=0.0, scale=0.5, size=pointAmount)) * radius
    for i in range(pointAmount):
        while radii[i] > radius:
            radii[i] = np.abs(np.random.normal(loc=0.0, scale=1.0)) * radius
    angles = np.random.uniform(0, 2 * np.pi, pointAmount)
    points = [(centers[cluster][0] + r * np.cos(a), centers[cluster][1] + r * np.sin(a)) for r, a in zip(radii, angles)]
    points = [(a1 * maxA1, a2 * maxA2) for (a1, a2) in points]
    for p in points:
        data.append(p)

def ring(data,pointAmount,centers,cluster,radius,maxA1,maxA2):
    radii = np.sqrt(np.random.uniform(0.8,1.0,pointAmount)) * radius
    angles = np.random.uniform(0, 2 * np.pi, pointAmount)
    points = [(centers[cluster][0] + r * np.cos(a), centers[cluster][1] + r * np.sin(a)) for r, a in zip(radii, angles)]
    points = [(a1 * maxA1, a2 * maxA2) for (a1, a2) in points]
    for p in points:
        data.append(p)

def line(data, pointAmount, centers, cluster, radius, maxA1, maxA2):
    angle = random.uniform(0, 360)
    angle_rad = math.radians(angle)
    v1 = (math.cos(angle_rad) * radius, math.sin(angle_rad) * radius)
    v2 = (math.cos(angle_rad + math.pi/2) * radius * 0.33, math.sin(angle_rad + math.pi/2) * radius * 0.33)
    points = []

    for i in range(pointAmount):
        r1 = random.uniform(-1, 1)
        r2 = random.uniform(-1, 1)
        point = (
            centers[cluster][0] + r1 * v1[0] + r2 * v2[0],
            centers[cluster][1] + r1 * v1[1] + r2 * v2[1]
        )
        point = (point[0] * maxA1, point[1] * maxA2)
        data.append(point)