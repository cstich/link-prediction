from geogps import Aux
from geogps import Parser
from geogps.DictAux import dd_list
from geogps import Bluetooth

from Metrics import UserMetrics


import collections
import matplotlib as mp
mp.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pytz
import re
import seaborn as sns
import statistics
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s" % (sys.argv[0]) +
              "<parsed data> "
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    outputPath = sys.argv[2]
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

    ''' Generate featues '''
    print('Generate distribution of meetings')
    interactionsAtContextAtTime = collections.defaultdict(dd_list)
    contextPattern = collections.defaultdict(list)
    for time in timeIntervalls:
        for user in users:
            try:
                stopLocs = stopLocations[user][time]
            except KeyError:
                stopLocs = None
            try:
                locBlues = localizedBlues[user][time]
            except KeyError:
                locBlues = None
            try:
                bl = blues[user][time]
            except KeyError:
                bl = None
            um = UserMetrics(user, bl,
                             locBlues,
                             stopLocs,
                             None, None)
            um.contextTimePattern(contextPattern)
            um.generateMeetingsDistribution(interactionsAtContextAtTime)

    dataToPlot = collections.defaultdict(list)
    for con, values in interactionsAtContextAtTime.items():
        for hour in range(168):
            try:
                m = statistics.mean(values[hour])
            except statistics.StatisticsError:
                m = 0
            dataToPlot[con].append((hour, m))

    f, ax = plt.subplots(figsize=(14, 7))
    ax = plt.plot(list(zip(*dataToPlot['all']))[1], label='all')
    plt.legend()
    plt.ylabel('average amount of peers present')
    plt.xlabel('hour of the week')
    plt.xticks(list(range(0, 192, 24)))
    plt.savefig(outputPath + 'meetingsDistribution.png', bbox_inches='tight')
    plt.close()

    f, ax = plt.subplots(figsize=(14, 7))
    for key, values in contextPattern.items():
        x, y = zip(*collections.Counter(values).items())
        ax = plt.plot(x, y, label=key)
    plt.legend()
    plt.ylabel('number of observations')
    plt.xlabel('hour of the week')
    plt.xticks(list(range(0, 192, 24)))
    plt.savefig(outputPath + 'actitivityPattern.png', bbox_inches='tight')
    plt.close()

    ''' Plot edge life aka length of meetings '''
    users = set(blues.keys())
    timeSpentWith = collections.defaultdict(dd_list)
    for user in users:
        for timeIntervall, blue in blues[user].items():
            for b in blue:
                timeSpentWith[user][str(b.peer)].append(b.time)

    lengthOfInteractions = list()
    for user, data in timeSpentWith.items():
        for peer, times in data.items():
                currentInteraction = 0

                for v in Aux.difference(times):
                    if v < 601:
                        currentInteraction += v
                    else:
                        lengthOfInteractions.append(currentInteraction)
                        currentInteraction = 0
                lengthOfInteractions.append(currentInteraction)

    print('average edge life: ', statistics.mean(lengthOfInteractions))
    f, ax = plt.subplots(figsize=(14, 7))
    x, y = zip(*collections.Counter(lengthOfInteractions).items())
    ax = sns.distplot(x)
    plt.legend()
    plt.ylabel('number of observations')
    plt.xlabel('edge life in seconds')
    plt.savefig(outputPath + 'edgeLife.png', bbox_inches='tight')
    plt.close()
