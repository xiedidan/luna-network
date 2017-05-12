import caffe
import multiprocessing

class Train(object):
    # constructor
    def __init__(self, dataPath):
        self.dataPath = dataPath
        

        self.solver = caffe.SGDSolver("solver.prototxt")

    # helper
    def dataProcessor(self):

    def trainProcessor(self):

    # interface
    def train(self):

