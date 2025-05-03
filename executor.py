import data_preprocessor as dp
import evaluator as ev
import main_processor as mp
import visualizer as vs
import data_generator as dg
import numpy as np



#pre set-up: create all datasets for tests and variants: low/high noise, far/near noise -> 4 dataset types, d1,d2,d3,d4
#for all 24 variants(3 algos, d1-d4, dvv on/off)

#tabelle = avg pc, avg total sc, avg total dbi
#visualization = [pc1,pc2,pc3], [avg. sc1,avg. sc2,avg. sc3,..], [dbi1,dbi2,dbi3]




def testrun(dn, algo, dvv):
    avgPercentages = []
    avgTotalSc = []
    avgDbi = []
    percentageArrays = [] #[[]...]
    scArrays = []
    dbiArrays = []
    for i in range(len(dn)):
        currentDataset = dn[i]
        result = None
        if algo == 0:
            result = mp.kmeans(currentDataset,3,False)
        elif algo == 1:
            result = mp.kmeans(currentDataset,3,True, #scalar x if dvv else scalar y)
        else:
            result = mp.dbscan(currentDataset,#eps x if dvv else eps y)

        # in arrays an stelle i jeweils berechnen/eintragen
            # [c1,c1,c1,c1,c1,c2,c2,c2,c2,c2,c3,c3,c3,n,n,n,n,n,n,]
    return [avgPercentages,avgTotalSc,avgDbi,percentageArrays,scArrays,dbiArrays]














#kmeans test
# testData = [(0.3,0.3), (0.401, 0.398), (0.399, 0.402), (0.400, 0.401), (0.403, 0.400), (0.398, 0.399), (0.402, 0.397), (0.400, 0.400), (0.401, 0.403), (0.397, 0.401), (0.400, 0.399), (0.6, 0.6), (0.633, 0.633), (0.667, 0.667), (0.7, 0.7), (0.733, 0.733), (0.767, 0.767), (0.8, 0.8), (0.833, 0.833), (0.867, 0.867), (0.9, 0.9)]
# testData = dp.standardize(testData)
# currentScore = -2
# savedLabels = []
# for i in range(100):
#     print(i)
#     currentResult = mp.kmeans(testData, 2,True)
#     tempScores = ev.silhouette(testData, np.array(currentResult[0]))
#     avgScore = np.mean(tempScores)
#     if avgScore > currentScore:
#         print("now")
#         print(currentScore, avgScore)
#         currentScore = avgScore
#         savedLabels = currentResult
# vs.drawMultipleCAResults([testData],[currentResult[0]],[True])

# data = [
#     -20, 5, 30,
#     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
#     10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11
# ]
#
# stData = dp.standardize(data)
# print(stData)

# for i in range(1):
#     dg.createDataSet(100,2,1,3)
#     preprocessData = dg.getDataSets()
#     preprocessData = preprocessData[-1]
#     resultPre = None
#     currentScore = -2
#     for i in range(10):
#         currentResult = mp.kmeans(preprocessData, 2,True)
#         tempScores = ev.silhouette(preprocessData, np.array(currentResult[0]))
#         avgScore = np.mean(tempScores)
#         if avgScore > currentScore:
#             currentScore = avgScore
#             resultPre = currentResult[0]
#     print("Resultpre: ", resultPre)
#     vs.drawMultipleCAResults([preprocessData],[resultPre],hasNoiseArray=[True])
#     postprocessData = dp.standardize(preprocessData)
#     resultPost = None
#     currentScore = -2
#     for i in range(10):
#         currentResult = mp.kmeans(postprocessData, 2,True)
#         tempScores = ev.silhouette(postprocessData, np.array(currentResult[0]))
#         avgScore = np.mean(tempScores)
#         if avgScore > currentScore:
#             currentScore = avgScore
#             resultPost = currentResult[0]
#     print("Resultpost: ", resultPost)
#     vs.drawMultipleCAResults([postprocessData],[resultPost],hasNoiseArray=[True])


# dbscan test
# dg.createDataSet(100,3,1,1)
# datasets = dg.getDataSets()
# testDBSCAN = mp.dbscan(datasets[-1], 0.05, 4)
# vs.drawCAResult(datasets[-1], testDBSCAN, False)

#dbi test
#dbiData = [12,52,5,9,100,22,75]
#vs.drawDBIGraph(dbiData)

#silouhette plot test
#sptData = [.5,-.4,.1,.2,.3,.9,-.8,.6,.5,-.4,.1,.8,-.9,-.2]
#sptMemb = [2,1,0,0,0,2,1,1,1,1,1,2,2,2]
#vs.drawSilhouttePlot(sptData, sptMemb)

#generator test
# dg.createDataSet(100,3,1,1)
# datasets = dg.getDataSets()
# noNoiseTest = mp.kmeans(datasets[-1],[datasets[-1][11],datasets[-1][111],datasets[-1][211]])
# vs.drawMultipleCAResults([datasets[-1]],[noNoiseTest[0]],[False])

# DBSCAN test II
# dg.createDataSet(100,3,1,1)
# datasets = dg.getDataSets()
# data = datasets[-1]
# labels = mp.dbscan(data)
# vs.drawMultipleCAResults([data], [labels], [False])
# print(labels)

#metriken test
# dg.createDataSet(100,3,1,1)
# testData = dg.getDataSets()[-1]
#
# kmeans_result = mp.kmeans(testData,3)
# kmeans_labels = np.array(kmeans_result[0])  # Konvertiere Labels in NumPy-Array
# dbi_kmeans = ev.dbi(testData, kmeans_labels)
# print("Kmeans DBI: ", dbi_kmeans, "\n")
# silhouette_kmeans = ev.silhouette(testData, kmeans_labels)
# print("Kmeans Silhouette: ", silhouette_kmeans, "\n")
# vs.drawIndexGraph([dbi_kmeans])
# vs.drawSilhouttePlot(silhouette_kmeans, kmeans_labels)
# vs.drawMultipleCAResults([testData],[kmeans_labels],[False])
#
# # DBSCAN Test
# dbscan_labels = np.array(mp.dbscan(testData))  # Ersetze 'data' durch 'testData'
# print("DBSCAN -Cluster Labels: ", dbscan_labels, "\n")
# silhouette_dbscan = ev.silhouette(testData, dbscan_labels)  # Ersetze 'data' durch 'testData'
# print("DBSCAN Silhouette: ", silhouette_dbscan, "\n")
# dbi_dbscan = ev.dbi(testData, dbscan_labels)  # Ersetze 'data' durch 'testData'
# print("DBSCAN -DBI: ", dbi_dbscan, "\n")
# vs.drawIndexGraph([dbi_dbscan])
# vs.drawSilhouttePlot(silhouette_dbscan, dbscan_labels)
# vs.drawMultipleCAResults([testData],[dbscan_labels],[False])