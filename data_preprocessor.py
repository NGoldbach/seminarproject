import numpy as np

#Takes array of tuples, returns array of normalized data
def normalize(data):
    newData = []
    minXY = np.min(data,axis=0)
    maxXY = np.max(data,axis=0)
    for v in data:
        newX = float((v[0]-minXY[0])/(maxXY[0]-minXY[0]))
        newY = float((v[1]-minXY[1])/(maxXY[1]-minXY[1]))
        newData.append((newX,newY))
    return newData

#Same input, output standardized based on mean and stdDeviation
def standardize(data):
    newData = []
    meanXY = np.mean(data,axis=0)
    stdDeviationXY = np.std(data,axis=0)
    for v in data:
        newX = float((v[0]-meanXY[0])/stdDeviationXY[0])
        newY = float((v[1]-meanXY[1])/stdDeviationXY[1])
        newData.append((newX,newY))
    print(newData)
    return newData

#Same input, output robust standardization based on median and IQR
def standardizeRobust(data):
    newData = []
    medianXY = np.median(data,axis=0)
    lowerBoundIQRXY = np.percentile(data,25,axis=0)
    higherBoundIQRXY = np.percentile(data,75,axis=0)
    iqrXY = (higherBoundIQRXY[0]-lowerBoundIQRXY[0],higherBoundIQRXY[1]-lowerBoundIQRXY[0])
    for v in data:
        newX = float((v[0]-medianXY[0])/iqrXY[0])
        newY = float((v[1]-medianXY[1])/iqrXY[1])
        newData.append((newX,newY))
    return newData
