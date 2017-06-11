import numpy as np

import matplotlib.pyplot as plt
import matplotlib

class Plotter(object):
    # constructor
    def __init__(self):
        plt.clf()
        # matplotlib.pyplot.show()

    # interface
    def plot(self, data1, index1, data2, index2):
        plt.clf()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.grid(True)
        self.ax2 = self.ax1.twinx()
        self.ax1.plot(data1[0:index1], 'b-', label='Loss', linewidth=1)
        self.ax2.plot(data2[0:index2], 'g-', label='Accu', linewidth=1)
        # plt.show()
        matplotlib.pyplot.show()