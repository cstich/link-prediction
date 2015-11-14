from gs import aux
from gs import parser
from gs.dictAux import dd_list

import collections
import math
import numpy as np
import os
import pickle
import pytz
import re
import statistics
import sys
''' Plotting related '''
import matplotlib as mp
mp.use('agg') # Workaround bug in conda
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

amsterdam = pytz.timezone('Europe/Amsterdam')


def calculateEntropy(ls):
    ''' For calculation of entropy see here:
    http://www.johndcook.com/blog/2013/08/17/calculating-entropy/
    '''
    c = collections.Counter(ls)
    totalLen = sum(c.values())
    entropy = float()
    sumOfLogs = sum([x * math.log(x, 2)
                    for x in c.values()])
    entropy = math.log(
        totalLen, 2
    ) - 1/totalLen * sumOfLogs
    return entropy


def resadjust(ax, xres=None, yres=None):
    """
    Send in an axis and I fix the resolution as desired.
    """

    if xres:
        start, stop = ax.get_xlim()
        ticks = np.arange(start, stop + xres, xres)
        ax.set_xticks(ticks)
    if yres:
        start, stop = ax.get_ylim()
        ticks = np.arange(start, stop + yres, yres)
        ax.set_yticks(ticks)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s" % (sys.argv[0]) +
              "<parsed data> "
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    outputPath = sys.argv[2]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    print('loadingData')
    with open(inputData, 'rb') as f:
        rs = pickle.load(f)

    base = os.path.basename(inputData)
    base = base.split('_')[2]
    resultsDirectory = os.path.dirname(inputData)

    def transformKey(x):
        return tuple(map(int, x.split('_')))

    locBluesPattern = re.compile(
        'parsedData_time_' + base + '_localizedBlue_([0-9_]+)\.pck')
    localizedBlues = parser.loadPickles(
        resultsDirectory, locBluesPattern, transformKey=transformKey)
    bluesPattern = re.compile(
        'parsedData_time_' + base + '_blue_([0-9_]+)\.pck')
    blues = parser.loadPickles(
        resultsDirectory, bluesPattern, transformKey=transformKey)
    stopLocations = rs['stopLocs']
    timeIntervalls = list(rs['intervalls'])  # Are ordered in time
    slidingTimeIntervalls = list(rs['slidingIntervalls'])

    # TODO Change around timePeriod and user key
    timeSpentWith = collections.defaultdict(dd_list)
    for timeIntervall, users in blues.items():
        for user, blue in users.items():
            for b in blue:
                timeSpentWith[user][str(b[0])].append(b[1])

    users = rs['users']
    lengthOfInteractions = list()
    peers = list()
    for user, data in timeSpentWith.items():
        for peer, blues in data.items():
                currentInteraction = 0
                beginTime = blues[0]
                for i, v in enumerate(aux.difference(blues)):
                    if v < 601:
                        currentInteraction += v
                        endTime = int(blues[i] / 600) * 600
                    else:
                        endTime = int(blues[i] / 600) * 600
                        peers.append((user, peer, currentInteraction))
                        currentInteraction = 0
                        try:
                            beginTime = int(blues[i+1] / 600) * 600
                        except IndexError:
                            beginTime = endTime
                lengthOfInteractions.append(currentInteraction)
                peers.append((user, peer, currentInteraction))

    lengthOfInteractions = np.asarray(lengthOfInteractions)

    ''' Plot the distribution '''
    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.distplot(lengthOfInteractions)
    plt.savefig(outputPath+'lenght_of_interactions.png',
                bbox_inches='tight')
    plt.close()

    q0 = np.percentile(lengthOfInteractions, 0)
    q1 = np.percentile(lengthOfInteractions, 25)
    q2 = np.percentile(lengthOfInteractions, 50)
    q3 = np.percentile(lengthOfInteractions, 75)
    q4 = np.percentile(lengthOfInteractions, 80)
    q5 = np.percentile(lengthOfInteractions, 85)
    q6 = np.percentile(lengthOfInteractions, 90)
    q7 = np.percentile(lengthOfInteractions, 95)
    q8 = np.percentile(lengthOfInteractions, 100)
    print('quartils')
    print(q0, ' ', q1, ' ', q2, ' ', q3, ' ', q4, ' ', q5, ' ', q6,
          ' ', q7, ' ', q8)

    print('quantils of the counter object')
    distributionOfValues = list(collections.Counter(lengthOfInteractions).keys())
    qv0 = np.percentile(distributionOfValues, 0)
    qv1 = np.percentile(distributionOfValues, 25)
    qv3 = np.percentile(distributionOfValues, 75)
    qv4 = np.percentile(distributionOfValues, 100)
    print(qv0, ' ', qv1, ' ', qv3, ' ', qv4)

    ''' Build list of interactions classified into the three quartil ranges '''
    networkQ1 = collections.defaultdict(list)
    networkIqr = collections.defaultdict(list)
    networkQ3 = collections.defaultdict(list)
    for p in peers:
        if p[2] <= q1:
            networkQ1[p[0]].append(p[1])
        elif p[2] > q1 and p[2] <= q3:
            networkIqr[p[0]].append(p[1])
        elif p[2] > q3:
            networkQ3[p[0]].append(p[1])
        else:
            raise(ValueError)

    entropyQ1 = list()
    for user, peers in networkQ1.items():
        entropyQ1.append(calculateEntropy(peers))
    entropyIqr = list()
    for user, peers in networkIqr.items():
        entropyIqr.append(calculateEntropy(peers))
    entropyQ3 = list()
    for user, peers in networkQ3.items():
        entropyQ3.append(calculateEntropy(peers))

    print('entropy')
    print(statistics.mean(entropyQ1), ' ', statistics.mean(entropyIqr),
          ' ', statistics.mean(entropyQ3))
