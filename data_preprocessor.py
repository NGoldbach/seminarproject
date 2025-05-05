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
    data = np.array(data, dtype=float)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data.tolist()

#Same input, output robust standardization based on median and IQR
def standardizeRobust(data):
    data = np.array(data)

    if data.ndim == 1:  # 1D data: simple list of numbers
        median = float(np.median(data))
        iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
        return [float((x - median) / iqr) for x in data]

    elif data.ndim == 2 and data.shape[1] == 2:  # 2D data: list of (x, y) pairs
        medianXY = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        iqrXY = q75 - q25

        return [(
            float((x - medianXY[0]) / iqrXY[0]),
            float((y - medianXY[1]) / iqrXY[1])
        ) for x, y in data]


def iqrTrimming(data, iqrScalar=1.5):
    iqrTrimRange = 1.349 * iqrScalar #1.349 is theoretical value for Q3-Q1
    trimmedData = []

    for d in data:
        if abs(d[0]) <= iqrTrimRange and abs(d[1]) <= iqrTrimRange:
            trimmedData.append(d)

    return trimmedData
