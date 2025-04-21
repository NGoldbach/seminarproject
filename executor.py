import data_preprocessor as dp
import evaluator as ev
import main_processor as mp
import visualizer as vs
import data_generator as dg


testData = [(1,1),(2,2),(1.5,1.5),(5,5),(6,6),(5.5,5.5),(1,19)]
noNoiseTest = mp.kmeans(testData,[(1,1),(5,5)],nvBool=False)
withNoiseTest = mp.kmeans(testData,[(1,1),(5,5)],nvBool=True)
vs.drawMultipleCAResults([testData,testData],[noNoiseTest[0],withNoiseTest[0]],[False,True])

