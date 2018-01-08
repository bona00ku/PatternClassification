import numpy as np


def convert2d():
    data = np.loadtxt("simple3.csv",delimiter=',')
    np.reshape(data,
