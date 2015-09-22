from geogps import Aux
from geogps import Parser
from geogps import TimeAux
from geogps.DictAux import ddd_int, ddd_list, dd_list, dd_set, DefaultOrderedDict

import collections
import os
import pickle
import pytz
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
        now = TimeAux.epochToLocalTime(m.time, amsterdam)
        if now.hour <= 18 and now.hour >= 9 and now.weekday() <= 4:
            result.append(m)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 11:
        print("Usage: %s " % (sys.argv[0]) +
              "<data path> "
              "<min. number of meetings> "
              "<min. amount of seconds> "
              "<output directory for social friends> "
              "<output directory for university friends> "
              "<output directory for 'all' friends> "
              "<output directory for 'weighted social' friends> "
              "<output directory for 'weighted university' friends> "
              "<output directory for 'weighted all' friends> "
              "<whether to export or to just get statistics>")
        sys.exit(-1)

    inputData = sys.argv[1]
    minNumberOfMeetings = int(sys.argv[2])
    minimumTime = int(sys.argv[3])
    outputPathSocial = sys.argv[4]
    outputPathUniversity = sys.argv[5]
    outputPathAll = sys.argv[6]
    outputPathWeighted = sys.argv[7]
    outputPathWeightedUniversity = sys.argv[8]
    outputPathWeightedAll = sys.argv[9]
    export = bool(int(sys.argv[10]))
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputData = Parser.parsePath(inputData, scriptDir)

    with open(inputData, 'rb') as f:
        rs = pickle.load(f)

    localizedBlues = rs['localizedBlues']
    stopLocations = rs['stopLocs']
    blues = rs['blues']
    timeIntervalls = list(rs['intervalls'])

    ''' Infer the place context friends for each time period '''
    outgoingFriends = collections.defaultdict(dd_list)  # You are friends
    # if you have met somewhere else than university for each period '''
    outgoingUniversityFriends = collections.defaultdict(dd_list)  # These are
    # your colleageus
    outgoingWeightedFriends = collections.defaultdict(ddd_int)
    outgoingWeightedUniversityFriends = collections.defaultdict(ddd_int)

    for user, timePeriods in localizedBlues.items():
        for timePeriod, observations in timePeriods.items():
            if observations:
                for locationIndex, observation in observations.items():
                    con = list(stopLocations[str(user)][timePeriod].values())
                    con = con[locationIndex][1]
                    for peer, meetings in observation.items():
                        meetings = sorted(meetings, key=lambda x: x.time)
                        meetingsForUniversity = findUniversityMeetings(meetings)
                        meetings = list(Aux.difference(
                            [m.time for m in meetings]))
                        meetingsForUniversity = list(Aux.difference(
                            [m.time for m in meetingsForUniversity]))

                        if meetings:
                            meetings = Aux.filterList(
                                meetings, lambda x: x > 0 and x <= 305)
                        meetings = sum(meetings)
                        if con != 'university' and con != 'University':
                            outgoingWeightedFriends[str(user)][timePeriod][peer]\
                                += meetings
                            if meetings > minimumTime:
                                outgoingFriends[str(user)][timePeriod].append(peer)

                        if meetingsForUniversity:
                            meetingsForUniversity = Aux.filterList(
                                meetingsForUniversity,
                                lambda x: x > 0 and x <= 305)
                        meetingsForUniversity = sum(meetingsForUniversity)
                        if con == 'university' or con == 'University':
                            outgoingWeightedUniversityFriends[str(user)][timePeriod][peer]\
                                += meetings
                            if meetingsForUniversity > minimumTime:
                                outgoingUniversityFriends[str(user)][timePeriod]\
                                    .append(peer)

    ''' Process context friends '''
    outgoingFriends = filterFriends(outgoingFriends, minNumberOfMeetings)
    incomingFriends = createIncomingTies(outgoingFriends)
    friends = combineNetworks(outgoingFriends, incomingFriends)

    incomingWeightedFriends = createIncomingWeightedTies(
        outgoingWeightedFriends)
    weightedFriends = combineWeightedNetworks(
        outgoingWeightedFriends, incomingWeightedFriends)

    outgoingUniversityFriends = filterFriends(
        outgoingUniversityFriends, minNumberOfMeetings)
    incomingUniversityFriends = createIncomingTies(outgoingUniversityFriends)
    universityFriends = combineNetworks(
        outgoingUniversityFriends, incomingUniversityFriends)

    incomingWeightedUniversityFriends = createIncomingWeightedTies(
        outgoingWeightedUniversityFriends)
    weightedUniversityFriends = combineWeightedNetworks(
        outgoingWeightedUniversityFriends, incomingWeightedUniversityFriends)

    ''' Build network based on the time people have met '''
    outgoingTimeFriends = collections.defaultdict(dd_list)
    outgoingWeightedTimeFriends = collections.defaultdict(ddd_int)
    tempFriends = collections.defaultdict(ddd_list)
    tempAllFriends = collections.defaultdict(ddd_list)
    outgoingAllFriends = collections.defaultdict(dd_list)
    outgoingWeightedAllFriends = collections.defaultdict(ddd_int)

    for user, blue in blues.items():
        for timePeriod, observations in blue.items():
            if observations:
                for b in observations:
                    now = TimeAux.epochToLocalTime(b.time, amsterdam)
                    tempAllFriends[str(user)][timePeriod][b.peer].append(b)
                    if now.hour >= 18 or now.hour <= 9 or now.weekday() > 4:
                        tempFriends[str(user)][timePeriod][b.peer].append(b)

    for user, values in tempFriends.items():
        for timePeriod, observations in values.items():
            for peer, meetings in observations.items():
                meetings = sorted(meetings, key=lambda x: x.time)
                meetings = list(
                    Aux.difference([m.time for m in meetings]))
                if len(meetings) > 0:
                    meetings = Aux.filterList(
                        meetings, lambda x: x > 0 and x <= 305)
                    meetings = sum(meetings)
                    outgoingWeightedTimeFriends[str(user)][timePeriod][peer]\
                        += meetings
                    if meetings > minimumTime:
                        outgoingTimeFriends[str(user)][timePeriod].append(peer)

    for user, values in tempAllFriends.items():
        for timePeriod, observations in values.items():
            for peer, meetings in observations.items():
                meetings = sorted(meetings, key=lambda x: x.time)
                meetings = list(
                    Aux.difference([m.time for m in meetings]))
                if len(meetings) > 0:
                    meetings = Aux.filterList(
                        meetings, lambda x: x > 0 and x <= 305)
                    meetings = sum(meetings)
                    outgoingWeightedAllFriends[str(user)][timePeriod][peer]\
                        += meetings
                    if meetings > minimumTime:
                        outgoingAllFriends[str(user)][timePeriod].append(peer)

    ''' Process time friends '''
    outgoingTimeFriends = filterFriends(outgoingTimeFriends,
                                        minNumberOfMeetings)
    incomingTimeFriends = createIncomingTies(outgoingTimeFriends)
    timeFriends = combineNetworks(outgoingTimeFriends, incomingTimeFriends)

    incomingWeightedTimeFriends = createIncomingWeightedTies(
        outgoingWeightedTimeFriends)
    weightedTimeFriends = combineWeightedNetworks(outgoingWeightedTimeFriends,
                                                  incomingWeightedTimeFriends)

    ''' Process 'all' friends '''
    outgoingAllFriends = filterFriends(outgoingAllFriends, minNumberOfMeetings)
    incomingAllFriends = createIncomingTies(outgoingAllFriends)
    allFriends = combineNetworks(outgoingAllFriends, incomingAllFriends)

    ''' Process weighted friends '''
    incomingWeightedAllFriends = createIncomingWeightedTies(
        outgoingWeightedAllFriends)
    weightedAllFriends = combineWeightedNetworks(
        outgoingWeightedAllFriends, incomingWeightedAllFriends)

    ''' Build hybrid network - both time and context '''
    hybridFriends = combineNetworks(friends, timeFriends)
    weightedHybridFriends = combineWeightedNetworks(
        weightedFriends, weightedTimeFriends)

    ''' Caldulate densities for the time slices '''
    densities = networkDensitiesOverTime(friends, timeIntervalls)
    densitiesUniversity = networkDensitiesOverTime(universityFriends,
                                                   timeIntervalls)
    densitiesTime = networkDensitiesOverTime(timeFriends, timeIntervalls)
    densitiesHybrid = networkDensitiesOverTime(hybridFriends, timeIntervalls)
    densitiesAll = networkDensitiesOverTime(allFriends, timeIntervalls)

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
    differences = estimateTieChangeOverTime(friends, timeIntervalls)
    print()
    for i, d in enumerate(differences):
        print('Average change period (non-university)', i, '/', i+1,
              ': ', d)

    differencesUniversity = estimateTieChangeOverTime(universityFriends,
                                                      timeIntervalls)
    print()
    for i, d in enumerate(differencesUniversity):
        print('Average change period (university)', i, '/', i+1,
              ': ', statistics.mean(d))

    differencesTime = estimateTieChangeOverTime(timeFriends, timeIntervalls)
    print()
    for i, d in enumerate(differencesTime):
        print('Average change period (time)', i, '/', i+1,
              ': ', statistics.mean(d))

    differencesHybrid = estimateTieChangeOverTime(hybridFriends, timeIntervalls)
    print()
    for i, d in enumerate(differencesHybrid):
        print('Average change period (social)', i, '/', i+1,
              ': ', statistics.mean(d))

    differencesAll = estimateTieChangeOverTime(allFriends, timeIntervalls)
    print()
    for i, d in enumerate(differencesAll):
        print('Average change period (all)', i, '/', i+1,
              ': ', statistics.mean(d))

    print()
    for i, (d, u, t, s, a) in enumerate(zip(densities, densitiesUniversity,
                                            densitiesTime, densitiesHybrid,
                                            densitiesAll)):
        print('Densities period ', i, ': Non-Uni: ', d, ' Uni: ', u, ' Time: ',
              t, ' Soc: ', s, ' All: ', a)

    if export:
        exportNetworks(hybridFriends, timeIntervalls, outputPathSocial)
        exportNetworks(universityFriends, timeIntervalls, outputPathUniversity)
        exportNetworks(allFriends, timeIntervalls, outputPathAll)

        exportWeightedNetworks(
            weightedHybridFriends, timeIntervalls, outputPathWeighted)
        exportWeightedNetworks(
            weightedUniversityFriends, timeIntervalls,
            outputPathWeightedUniversity)
        exportWeightedNetworks(
            weightedAllFriends, timeIntervalls, outputPathWeightedAll)
    else:
        duration = float(inputData.split('/')[-2].replace('_', '.'))
        lines = []

        for i, (den, dif) in enumerate(zip(densitiesAll, differencesAll.values())):
            line = ' '.join(map(str, [i, duration, den, dif, '\n']))
            lines.append(line)

        with open('networkStatistics.ssv', 'a+') as f:
            for line in lines:
                f.write(line)
