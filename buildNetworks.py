from gs import aux
from gs import dictAux
from gs import parser
from gs import timeAux
from gs.dictAux import ddd_int, ddd_list, dd_list, dd_set, DefaultOrderedDict

import collections
import numpy as np
import os
import pickle
import pytz
import re
import resource
import statistics
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


def calculateDifference(x, y):
    x = set(x)
    y = set(y)

    try:
        return len(x.symmetric_difference(y))/len(x.union(y))
    except ZeroDivisionError:
        if len(x) == 0 and len(y) == 0:
            return 0
        else:
            return 1


def createOutgoingWeightedTies(rawFriends):
    outgoingFriends = collections.defaultdict(ddd_int)
    for user, values in rawFriends.items():
        for timePeriod, observations in values.items():
            for peer, meetings in observations.items():
                meetings = sorted(meetings)
                meetings = list(
                    aux.difference(meetings))
                if len(meetings) > 0:
                    meetings = aux.filterList(
                        meetings, lambda x: x > 0 and x <= 305)
                    meetings = sum(meetings)
                    outgoingFriends[str(user)][timePeriod][peer]\
                        += meetings
    return outgoingFriends


def createIncomingTies(friends):
    result = collections.defaultdict(dd_set)
    for user, timePeriods in friends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                result[str(peer)][timePeriod].add(str(user))
    return result


def createIncomingWeightedTies(friends):
    result = collections.defaultdict(ddd_int)
    for user, timePeriods in friends.items():
        for timePeriod, peers in timePeriods.items():
            for peer, amount in peers.items():
                result[str(peer)][timePeriod][user] = amount
    return result


def combineNetworks(networkA, networkB):
    result = collections.defaultdict(dd_set)
    for user, timePeriods in networkA.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                result[str(user)][timePeriod].add(str(peer))

    for user, timePeriods in networkB.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                result[str(user)][timePeriod].add(str(peer))
    return result


def combineWeightedNetworks(outgoingFriends, incomingFriends):
    friends = collections.defaultdict(ddd_int)
    for user, timePeriods in outgoingFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer, amount in peers.items():
                friends[str(user)][timePeriod][peer] += amount

    for user, timePeriods in incomingFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer, amount in peers.items():
                friends[str(user)][timePeriod][peer] += amount
    return friends


def networkDensitiesOverTime(network, timeIntervalls, N=None):
    densities = []
    connections = DefaultOrderedDict(int)
    if not N:
        N = len(network.keys())
    pc = N * (N - 1)
    for user in network.keys():
        for key in timeIntervalls:
            currentFriends = network[user][key]
            connections[key] += len(currentFriends)
    for c in connections.values():
        density = c / pc
        densities.append(density)
    return densities


def estimateTieChangeOverTime(network, timeIntervalls):
    differences = collections.defaultdict(list)
    for user in network:
        currentFriends = network[user][timeIntervalls[0]]
        for key in timeIntervalls[1:]:
            lastFriends = currentFriends
            currentFriends = network[user][key]
            diff = calculateDifference(lastFriends, currentFriends)
            differences[key].append(diff)
    meanDifferences = collections.OrderedDict()
    for key, diffs in differences.items():
        meanDifferences[key] = statistics.mean(diffs)
    return meanDifferences


def exportNetworks(network, timeIntervalls, outputPath):
    for key in timeIntervalls:
        lines = []
        filename = str(key[0]) + '-' + str(key[1]) + '.csv'
        for user in network.keys():
            if len(network[user][key]) > 0:
                line = user + ',' + ','.join(network[user][key]) + '\n'
            else:
                line = user + '\n'
            lines.append(line)
        with open(outputPath + filename, 'w') as f:
            for line in lines:
                f.write(line)


def exportWeightedNetworks(network, timeIntervalls, outputPath):
    for key in timeIntervalls:
        lines = []
        filename = str(key[0]) + '-' + str(key[1]) + '.csv'
        for user in network.keys():
            line = [user]
            for peer, value in network[user][key].items():
                line.extend([str(peer), str(value)])
            line = ','.join(line) + '\n'
            lines.append(line)
        with open(outputPath + filename, 'w') as f:
            for line in lines:
                f.write(line)


def filterFriends(friends, minNumberOfMeetings):
    result = collections.defaultdict(dd_list)
    for user, timePeriods in friends.items():
        for timePeriod, peers in timePeriods.items():
            filteredPeers = collections.Counter(peers)
            filteredPeers = [k for k, v in filteredPeers.items()
                             if v >= minNumberOfMeetings]
            result[str(user)][timePeriod] = filteredPeers
    return result


def findUniversityMeetings(meetings):
    result = list()
    for m in meetings:
        now = timeAux.epochToLocalTime(m, amsterdam)
        if now.hour <= 18 and now.hour >= 9 and now.weekday() <= 4:
            result.append(m)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: %s " % (sys.argv[0]) +
              "<data path> "
              "<output directory for 'weighted social' friends> "
              "<output directory for 'weighted university' friends> "
              "<output directory for 'weighted all' friends> "
              "<whether to export or to just get statistics>")
        sys.exit(-1)

    ''' Set memory limit '''
    rsrc = resource.RLIMIT_AS
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (1024*1024*1024*30, hard))  # limit is in bytes

    ''' Load data '''
    inputData = sys.argv[1]
    outputPathWeighted = sys.argv[2]
    outputPathWeightedUniversity = sys.argv[3]
    outputPathWeightedAll = sys.argv[4]
    try:
        export = bool(int(sys.argv[5]))
    except ValueError:
        outputPath = export
        export = False

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
    timeIntervalls = list(rs['intervalls'])
    usersLs = rs['users']
    rs = []

    ''' Infer the place context friends for each time period '''
    print('Inferring context friends')
    outgoingWeightedFriends = collections.defaultdict(ddd_int)
    outgoingWeightedUniversityFriends = collections.defaultdict(ddd_int)

    for timePeriod, users in localizedBlues.items():
        for user, observations in users.items():
            if len(observations) > 0:
                for locationIndex, observation in observations.items():
                    con = list(stopLocations[str(user)][timePeriod].values())
                    con = con[locationIndex][1]
                    for peer, meetings in observation.items():
                        meetings = sorted(meetings)
                        meetingsForUniversity = findUniversityMeetings(meetings)
                        meetings = list(aux.difference(
                            meetings))
                        meetingsForUniversity = list(aux.difference(
                            meetings))

                        if meetings:
                            meetings = aux.filterList(
                                meetings, lambda x: x > 0 and x <= 305)
                        meetings = sum(meetings)
                        if con != 'university' and con != 'University':
                            outgoingWeightedFriends[
                                str(user)][timePeriod][peer] += meetings

                        if meetingsForUniversity:
                            meetingsForUniversity = aux.filterList(
                                meetingsForUniversity,
                                lambda x: x > 0 and x <= 305)
                        meetingsForUniversity = sum(meetingsForUniversity)
                        if con == 'university' or con == 'University':
                            outgoingWeightedUniversityFriends[
                                str(user)][timePeriod][peer] += meetings

    ''' Clear unused variables as soon as you don't need them anymore '''
    print('clearing stopLocations')
    stopLocations = list()

    ''' Process context friends '''
    print('Process context friends')
    incomingWeightedFriends = createIncomingWeightedTies(
        outgoingWeightedFriends)
    weightedFriends = combineWeightedNetworks(
        outgoingWeightedFriends, incomingWeightedFriends)

    incomingWeightedUniversityFriends = createIncomingWeightedTies(
        outgoingWeightedUniversityFriends)
    weightedUniversityFriends = combineWeightedNetworks(
        outgoingWeightedUniversityFriends, incomingWeightedUniversityFriends)

    ''' Clear variables you don't need anymore '''
    incomingWeightedFriends = None
    outgoingWeightedFriends = None
    incomingWeightedUniversityFriends = None
    outgoingWeightedUniversityFriends = None

    ''' Build network based on the time people have met '''
    print('Inferring time friends')
    tempFriends = collections.defaultdict(ddd_list)
    tempAllFriends = collections.defaultdict(ddd_list)
    numberOfUsers = 0

    begin = timeIntervalls[0]
    end = timeIntervalls[1] - begin


    # TODO Fix the user/timePeiod key change
    for timePeriod in blues.keys():
        blue = blues[timePeriod]
        print('{:1.4f}'.format((timePeriod[1]-begin)/end)),
             end=' ')
        sys.stdout.flush()
        for user, observations in blue.items():
            if len(observations) > 0:
                for b in observations:
                    now = timeAux.epochToLocalTime(b[1], amsterdam)
                    tempAllFriends[str(user)][timePeriod][b[0]].append(b[1])
                    if now.hour >= 18 or now.hour <= 9 or now.weekday() > 4:
                        tempFriends[str(user)][timePeriod][b[0]].append(b[1])
        ''' Convert to numpy array to save memory '''
        timePeriods = tempFriends[timePeriod]
        dictAux.castLeaves(timePeriods, np.asarray)
        timePeriods = tempAllFriends[timePeriod]
        dictAux.castLeaves(timePeriods, np.asarray)
        ''' Clear the timePeriod once you don't need them anymore '''
        blues[timePeriod] = None
        numberOfUsers += 1

    ''' Create outgoing times from the meetings '''
    print('Create outgoing ties for time friends and all friends ''')
    outgoingWeightedTimeFriends = createOutgoingWeightedTies(tempFriends)
    tempFriends = None
    outgoingWeightedAllFriends = createOutgoingWeightedTies(tempAllFriends)
    tempAllFriends = None

    ''' Process time friends '''
    print('Process time friends')
    incomingWeightedTimeFriends = createIncomingWeightedTies(
        outgoingWeightedTimeFriends)
    weightedTimeFriends = combineWeightedNetworks(outgoingWeightedTimeFriends,
                                                  incomingWeightedTimeFriends)
    incomingWeightedTimeFriends = None
    outgoingWeightedTimeFriends = None

    ''' Process weighted friends '''
    incomingWeightedAllFriends = createIncomingWeightedTies(
        outgoingWeightedAllFriends)
    weightedAllFriends = combineWeightedNetworks(
        outgoingWeightedAllFriends, incomingWeightedAllFriends)
    incomingWeightedAllFriends = None
    outgoingWeightedAllFriends = None

    ''' Build hybrid network - both time and context '''
    print('Inferring hybrid friends')
    weightedHybridFriends = combineWeightedNetworks(
        weightedFriends, weightedTimeFriends)

    ''' Caldulate densities for the time slices '''
    print('Calculating densities')
    weightedDensities = networkDensitiesOverTime(weightedFriends,
                                                 timeIntervalls)
    weightedDensitiesUniversity = networkDensitiesOverTime(
        weightedUniversityFriends, timeIntervalls)
    weightedDensitiesTime = networkDensitiesOverTime(weightedTimeFriends,
                                                     timeIntervalls)
    weightedDensitiesHybrid = networkDensitiesOverTime(weightedHybridFriends,
                                                       timeIntervalls)
    weightedDensitiesAll = networkDensitiesOverTime(weightedAllFriends,
                                                    timeIntervalls)

    ''' Estimate change between time slices '''
    weightedDifferencesAll = estimateTieChangeOverTime(weightedAllFriends,
                                                       timeIntervalls)
    if export:
        exportWeightedNetworks(
            weightedHybridFriends, timeIntervalls, outputPathWeighted)
        exportWeightedNetworks(
            weightedUniversityFriends, timeIntervalls,
            outputPathWeightedUniversity)
        exportWeightedNetworks(
            weightedAllFriends, timeIntervalls, outputPathWeightedAll)
    else:
        duration = inputData.split('/')[-1].replace('_', '.')
        duration = float(duration.split('.')[-3])
        lines = []

        for i, (den, dif) in enumerate(zip(weightedDensitiesAll,
                                           weightedDifferencesAll.values())):
            line = ' '.join(map(str, [i, duration, den, dif, '\n']))
            lines.append(line)

        with open(outputPath + 'networkStatistics.ssv', 'a+') as f:
            for line in lines:
                f.write(line)
