from geogps import Parser
from geogps.Aux import DefaultOrderedDict
from geogps.Aux import flatten

import collections
import os
import pickle
import sys

# Plotting functions
import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def dd_set():
    return collections.defaultdict(set)


def dd_list():
    return collections.defaultdict(list)


def ddd_list():
    return collections.defaultdict(dd_list)


def dddd_list():
    return collections.defaultdict(ddd_list)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory>"
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    outputPath = sys.argv[2]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputData = Parser.parsePath(inputData, scriptDir)

    with open(inputData + '/parsedData.pck', 'rb') as f:
        rs = pickle.load(f)

    data = rs['count']
    plt.hist(data)
    plt.savefig(outputPath + '/densities.png')
    plt.close()

    localizedBlues = rs['localizedBlues']
    localizedBluesCounter = []
    for user, timePeriods in localizedBlues.items():
        for timePeriod, observations in timePeriods.items():
            if observations:
                for locationIndex, observation in observations.items():
                    ls = [b.time for b in flatten(observation.values())]
                    localizedBluesCounter.extend(ls)

    plt.hist(localizedBluesCounter)
    plt.savefig(outputPath + '/count_localized_blues.png')
    plt.close()

    blues = rs['blues']
    bluesCounter = []
    for user, blue in blues.items():
        for timePeriod, observations in blue.items():
            if observations:
                ls = [b.time for b in observations]
                bluesCounter.extend(ls)

    plt.hist(bluesCounter)
    plt.savefig(outputPath + '/count_blues.png')
    plt.close()
