import math
import random
import numpy as np

def noiseGenerator(data, mode):
    #modes: low/far, high/far, low/near, high/near

    noisePercentage = 0
    noiseMinDistScalar = 0
    if mode == 0:
        noisePercentage = 0.25
        noiseMinDistScalar = 0.5
    elif mode == 1:
        noisePercentage = 0.75
        noiseMinDistScalar = 0.5
    elif mode == 2:
        noisePercentage = 0.25
        noiseMinDistScalar = -0.25
    elif mode == 3:
        noisePercentage = 0.75
        noiseMinDistScalar = -0.25

    clusterCenters = data[1]
    clusterRadius = data[2]
    minDistance = clusterRadius * (1 + noiseMinDistScalar)

    numNoisePoints = int(len(data[0]) * noisePercentage)
    noisePoints = []
    while len(noisePoints) < numNoisePoints:
        x = random.gauss(0.5,0.3)
        y = random.gauss(0.5,0.3)
        valid = True
        for cx, cy in clusterCenters:
            dist = math.hypot(x - cx, y - cy)
            if dist < minDistance:
                valid = False
                break
        if valid:
            noisePoints.append((x, y))

    newData = data[0].copy()
    newData.extend(noisePoints)
    return newData


def getDataSets():
    loaded_arrays = []
    with open("dataSets.txt", "r") as f:
        for l in f:
            lineStripped = l.strip()
            if lineStripped:
                tuples = [
                    tuple(map(float, t.strip("()").split(",")))
                    for t in lineStripped.split()
                ]
                loaded_arrays.append(tuples)
    return loaded_arrays


def clearDataSetFile():
    with open("dataSets.txt", "w") as f:
        pass


def createDataSet(pointAmount, clusterAmount, maxA1=1, maxA2=1, clusterShapes=['circle','circle','circle'], clusterDensity=None):
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

    return [data, centers, radius]


def saveDataSet(data):
    with open("dataSets.txt", "a") as f:
        tuple_strs = ["({},{})".format(p[0], p[1]) for p in data]
        f.write(" ".join(tuple_strs) + "\n")



def circle(data,pointAmount,centers,cluster,radius,maxA1,maxA2):
    radii = np.abs(np.random.normal(loc=0.0, scale=0.5, size=pointAmount)) * radius
    for i in range(pointAmount):
        while radii[i] > radius/2:
            radii[i] = np.abs(np.random.normal(loc=0.0, scale=1.0)) * radius
    angles = np.random.uniform(0, 2 * np.pi, pointAmount)
    points = [(centers[cluster][0] + r * np.cos(a), centers[cluster][1] + r * np.sin(a)) for r, a in zip(radii, angles)]
    points = [(a1 * maxA1, a2 * maxA2) for (a1, a2) in points]
    for p in points:
        data.append(p)

def ring(data,pointAmount,centers,cluster,radius,maxA1,maxA2):
    radii = np.sqrt(np.random.uniform(0.3,0.5,pointAmount)) * radius
    angles = np.random.uniform(0, 2 * np.pi, pointAmount)
    points = [(centers[cluster][0] + r * np.cos(a), centers[cluster][1] + r * np.sin(a)) for r, a in zip(radii, angles)]
    points = [(a1 * maxA1, a2 * maxA2) for (a1, a2) in points]
    for p in points:
        data.append(p)

def line(data, pointAmount, centers, cluster, radius, maxA1, maxA2):
    angle = random.uniform(0, 360)
    angle_rad = math.radians(angle)
    v1 = (math.cos(angle_rad) * radius, math.sin(angle_rad) * radius)
    v2 = (math.cos(angle_rad + math.pi/2) * radius * 0.1, math.sin(angle_rad + math.pi/2) * radius * 0.1)
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