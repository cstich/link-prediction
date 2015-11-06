from geogps import Aux
from geogps import Parser
from geogps.DictAux import dd_list, ddd_set
from geogps import TimeAux

import collections
import matplotlib as mp
mp.use('agg')
import math
import numpy as np
import os
import pytz
import re
import statistics
import sys

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
    outptPath = sys.argv[2]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputFilePattern = re.compile('parsedData[\-\_a-zA-Z0-9]*\.pck')
    rs = Parser.loadPickles(inputData, inputFilePattern,
                            matchUser=False)[0]

    localizedBlues = rs['localizedBlues']
    stopLocations = rs['stopLocs']
    blues = rs['blues']
    timeIntervalls = list(rs['intervalls'])  # Are ordered in time
    slidingTimeIntervalls = list(rs['slidingIntervalls'])
    users = list(map(str, list(blues.keys())))

    timeSpentWith = collections.defaultdict(dd_list)
    for user in users:
        for timeIntervall, blue in blues[user].items():
            for b in blue:
                timeSpentWith[user][str(b.peer)].append(b.time)

    lengthOfInteractions = list()
    peers = list()
    for user, data in timeSpentWith.items():
        for peer, blues in data.items():
                currentInteraction = 0
                beginTime = blues[0]
                for i, v in enumerate(Aux.difference(blues)):
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
    q0 = np.percentile(lengthOfInteractions, 0)
    q1 = np.percentile(lengthOfInteractions, 25)
    q3 = np.percentile(lengthOfInteractions, 75)
    q4 = np.percentile(lengthOfInteractions, 100)

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

    print('quartils')
    print(q0, ' ', q1, ' ', q3, ' ', q4)
    print('entropy')
    print(statistics.mean(entropyQ1), ' ', statistics.mean(entropyIqr),
          ' ', statistics.mean(entropyQ3))
