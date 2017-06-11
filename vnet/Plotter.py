import numpy as np

import matplotlib.pyplot as plt
import matplotlib

class Plotter(object):
    # constructor
    def __init__(self):
        # matplotlib.pyplot.show()
        print("hello plotter")

    # interface
    def plot(self, data, index):
        plt.clf()
        plt.plot(range(0, index), data[0: index], 'b-', label='Loss', linewidth=1)
        plt.pause(0.00000001)
        matplotlib.pyplot.show()