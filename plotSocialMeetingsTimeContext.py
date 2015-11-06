from geogps import Aux
from geogps import Parser
from geogps.DictAux import dd_list, ddd_set
from geogps import TimeAux

from Metrics import UserMetrics

import collections
import matplotlib as mp
mp.use('agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import pytz
import re
import seaborn as sns
import statistics
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


def generate_triangles(nodes):
    """Generate triangles. Weed out duplicates."""
    visited_ids = set()  # remember the nodes that we have tested already
    for node_a_id in nodes:
        for node_b_id in nodes[node_a_id]:
            if node_b_id == node_a_id:
                print(node_b_id, node_a_id)
                raise ValueError  # nodes shouldn't point to themselves
            if node_b_id in visited_ids:
                continue  # we should have already found b->a->??->b
            for node_c_id in nodes[node_b_id]:
                if node_c_id in visited_ids:
                    continue  # we should have already found c->a->b->c
                if node_a_id in nodes[node_c_id]:
                    yield(node_a_id, node_b_id, node_c_id)
        visited_ids.add(node_a_id)  # don't search a -
        # we already have all those cycles


def countTrianglesAtContext(networkAtContextAtTime, trianglesAtContextAtTime):
    for context, values in networkAtContextAtTime.items():
        for time, network in values.items():
            hourOfTheWeek = TimeAux.epochToHourOfTheWeek(time, amsterdam)

            cycles = list(generate_triangles(network))
            trianglesAtContextAtTime[context][hourOfTheWeek].append(len(cycles))
            trianglesAtContextAtTime['all'][hourOfTheWeek].append(
                len(cycles))


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
    networkAtContextAtTime = collections.defaultdict(ddd_set)
    contextPattern = collections.defaultdict(list)
    contextAtTime = collections.defaultdict(dd_list)
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
            um.generateContextAtTime(contextAtTime)
            um.generateMeetingsDistribution(interactionsAtContextAtTime,
                                            contextAtTime)
            um.generateNetworkAtTime(networkAtContextAtTime, contextAtTime)

    ''' triangles count '''
    trianglesAtContextAtTime = collections.defaultdict(dd_list)
    countTrianglesAtContext(networkAtContextAtTime, trianglesAtContextAtTime)
    trianglesToPlot = collections.defaultdict(list)
    for con, values in trianglesAtContextAtTime.items():
        for hour in range(168):
            try:
                m = statistics.mean(values[hour])
            except statistics.StatisticsError:
                m = 0
            trianglesToPlot[con].append((hour, m))

    f, ax = plt.subplots(figsize=(14, 7))
    averageTriangles = list(zip(*trianglesToPlot['all']))[1]
    ax = plt.plot(averageTriangles, label='all')
    plt.legend()
    plt.ylabel('average amount of peers present')
    plt.xlabel('hour of the week')
    plt.xticks(list(range(0, 168, 24)))
    plt.savefig(outputPath + 'triangles.png', bbox_inches='tight')
    plt.close()

    ''' Plot the different contexts '''
    contexts = ['home', 'university', 'thirdPlace', 'other']
    f, ax = plt.subplots(figsize=(14, 7))
    for con in contexts:
        triangles = list(zip(*trianglesToPlot[con]))[1]
        data = list(map(lambda x, y: x-y, triangles, averageTriangles))
        ax = plt.plot(data, label=con)
    plt.legend()
    plt.ylabel('average amount of peers present')
    plt.xlabel('hour of the week')
    plt.xticks(list(range(0, 168, 24)))
    plt.savefig(outputPath + 'trianglesAllContexts.png',
                bbox_inches='tight')
    plt.close()

    ''' average amount of peers '''
    dataToPlot = collections.defaultdict(list)
    for con, values in interactionsAtContextAtTime.items():
        for hour in range(168):
            try:
                m = statistics.mean(values[hour])
            except statistics.StatisticsError:
                m = 0
            dataToPlot[con].append((hour, m))

    f, ax = plt.subplots(figsize=(14, 7))
    averagePeers = list(zip(*dataToPlot['all']))[1]
    ax = plt.plot(averagePeers, label='all')
    plt.legend()
    plt.ylabel('average amount of peers present')
    plt.xlabel('hour of the week')
    plt.xticks(list(range(0, 168, 24)))
    plt.savefig(outputPath + 'meetingsDistribution.png', bbox_inches='tight')
    plt.close()

    ''' Plot the different contexts '''
    contexts = ['home', 'university', 'thirdPlace', 'other']
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                               sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    for con, ax in zip(contexts, axes):
        peers = list(zip(*dataToPlot[con]))[1]
        # data = list(map(lambda x, y: x-y, peers, averagePeers))
        ax.plot(peers, label=con)
        ax.legend()
        plt.ylabel('absolute deviation from $\overline{x}_{peers}$')
    plt.xlabel('hour of the week')
    plt.xticks(list(range(0, 168, 24)))
    plt.savefig(outputPath + 'meetingsDistributionAllContexts.png',
                bbox_inches='tight')
    plt.close()

    ''' Lag plots '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                               sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    for con, ax in zip(contexts, axes):
        peers = list(zip(*dataToPlot[con]))[1]
        # data = list(map(lambda x, y: x-y, peers, averagePeers))
        s = pd.Series(peers)
        pd.tools.plotting.lag_plot(s, ax=ax)
        ax.legend()
    plt.savefig(outputPath + 'lag_peers.png',
                bbox_inches='tight')
    plt.close()

    ''' Autocorrelation plots '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                               sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    for con, ax in zip(contexts, axes):
        peers = list(zip(*dataToPlot[con]))[1]
        data = list(map(lambda x, y: x-y, peers, averagePeers))
        s = pd.Series(data)
        pd.tools.plotting.autocorrelation_plot(s, ax=ax)
    plt.savefig(outputPath + 'autocorrelation_peers.png',
                bbox_inches='tight')
    plt.close()

    ''' activity pattern '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                               sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    for con, ax in zip(contexts, axes):
        values = contextPattern[con]
        x, y = zip(*collections.Counter(values).items())
        ax.plot(x, y, label=con)
        ax.legend()
        ax.set_xticks(list(range(0, 168, 24)))
    plt.ylabel('number of observations')
    plt.xlabel('hour of the week')
    plt.savefig(outputPath + 'actitivityPattern.png', bbox_inches='tight')
    plt.close()

    ''' Lag plots '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                               sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    for con, ax in zip(contexts, axes):
        values = contextPattern[con]
        x, y = zip(*collections.Counter(values).items())
        # data = list(map(lambda x, y: x-y, peers, averagePeers))
        s = pd.Series(y)
        pd.tools.plotting.lag_plot(s, ax=ax)
        ax.legend()
    plt.savefig(outputPath + 'lag_activity.png',
                bbox_inches='tight')
    plt.close()

    ''' Plot edge life aka length of meetings '''
    users = set(blues.keys())
    timeSpentWith = collections.defaultdict(dd_list)
    for user in users:
        for timeIntervall, blue in blues[user].items():
            for b in blue:
                timeSpentWith[user][str(b.peer)].append(b.time)

    lengthOfInteractions = list()
    lengthOfInteractionsAtContextAtTime = collections.defaultdict(dd_list)
    for user, data in timeSpentWith.items():
        for peer, blues in data.items():
                currentInteraction = 0
                beginTime = blues[0]
                for i, v in enumerate(Aux.difference(blues)):
                    if v > 601:
                        currentInteraction += v
                        endTime = int(blues[i] / 600) * 600
                    else:
                        lengthOfInteractions.append(currentInteraction)
                        endTime = int(blues[i] / 600) * 600
                        for t in range(beginTime, endTime, 600):
                            con = contextAtTime[user][t]
                            if len(con) == 0:
                                con = 'other'
                            else:
                                con = Aux.maxMode(con)
                            hour = TimeAux.epochToHourOfTheWeek(t, amsterdam)
                            lengthOfInteractionsAtContextAtTime[con][hour]\
                                .append(currentInteraction)
                            lengthOfInteractionsAtContextAtTime['all'][hour]\
                                .append(currentInteraction)
                        currentInteraction = 0
                        try:
                            beginTime = int(blues[i+1] / 600) * 600
                        except IndexError:
                            beginTime = endTime
                lengthOfInteractions.append(currentInteraction)
                for t in range(beginTime, endTime, 600):
                    hour = TimeAux.epochToHourOfTheWeek(t, amsterdam)
                    lengthOfInteractionsAtContextAtTime[con][hour]\
                        .append(currentInteraction)
                    lengthOfInteractionsAtContextAtTime['all'][hour]\
                        .append(currentInteraction)

    print('average edge life: ', statistics.mean(lengthOfInteractions))
    f, ax = plt.subplots(figsize=(14, 7))
    x, y = zip(*collections.Counter(lengthOfInteractions).items())
    ax = sns.distplot(x)
    plt.legend()
    plt.ylabel('number of observations')
    plt.xlabel('edge life in seconds')
    plt.savefig(outputPath + 'edgeLife.png', bbox_inches='tight')
    plt.close()

    ''' plot edge life for each context '''
    dataToPlot = collections.defaultdict(list)
    for con, values in lengthOfInteractionsAtContextAtTime.items():
        for hour in range(168):
            try:
                m = statistics.mean(values[hour])
            except statistics.StatisticsError:
                m = 0
            dataToPlot[con].append((hour, m))

    averageEdgeLife = list(zip(*dataToPlot['all']))[1]
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                               sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    for con, ax in zip(contexts, axes):
        edgeLife = list(zip(*dataToPlot[con]))[1]
        # data = list(map(lambda x, y: x-y, edgeLife, averageEdgeLife))
        ax.plot(edgeLife, label=con)
        ax.legend()
        ax.set_xticks(list(range(0, 168, 24)))
    plt.savefig(outputPath + 'edge_life_contexts.png',
                bbox_inches='tight')
    plt.close()

    ''' Lag plots '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                               sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    for con, ax in zip(contexts, axes):
        data = list(zip(*dataToPlot[con]))[1]
        # data = list(map(lambda x, y: x-y, peers, averagePeers))
        s = pd.Series(data)
        pd.tools.plotting.lag_plot(s, ax=ax)
        ax.legend()
    plt.savefig(outputPath + 'lag_edge_life.png',
                bbox_inches='tight')
    plt.close()
