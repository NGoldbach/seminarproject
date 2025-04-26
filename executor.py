import data_preprocessor as dp
import evaluator as ev
import main_processor as mp
import visualizer as vs
import data_generator as dg

#kmeans test
# testData = [(1,1),(2,2),(1.5,1.5),(5,5),(6,6),(5.5,5.5),(1,19)]
# noNoiseTest = mp.kmeans(testData,[(1,1),(5,5)],nvBool=False)
# withNoiseTest = mp.kmeans(testData,[(1,1),(5,5)],nvBool=True)
# vs.drawMultipleCAResults([testData,testData],[noNoiseTest[0],withNoiseTest[0]],[False,True])

#dbi test
#dbiData = [12,52,5,9,100,22,75]
#vs.drawDBIGraph(dbiData)

#silouhette plot test
# sptData = [.5,-.4,.1,.2,.3,.9,-.8,.6,.5,-.4,.1,.8,-.9,-.2]
# sptMemb = [2,1,0,0,0,2,1,1,1,1,1,2,2,2]
# vs.drawSilhouttePlot(sptData, sptMemb)

#generator test
dg.createDataSet(100,3,1,1)
datasets = dg.getDataSets()
noNoiseTest = mp.kmeans(datasets[-1],[datasets[-1][11],datasets[-1][111],datasets[-1][211]])
vs.drawMultipleCAResults([datasets[-1]],[noNoiseTest[0]],[False])
