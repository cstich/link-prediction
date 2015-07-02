from geogps import Parser
from geogps import TimeAux
from geogps.Aux import DefaultOrderedDict

import collections
import os
import pickle
import pytz
import statistics
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


def dd_set():
    return collections.defaultdict(set)


def dd_list():
    return collections.defaultdict(list)


def ddd_list():
    return collections.defaultdict(dd_list)


def dddd_list():
    return collections.defaultdict(ddd_list)


def calculateDifference(x, y):
    x = set(x)
    y = set(y)

    try:
        return len(x.difference(y))/len(x.union(y))
    except ZeroDivisionError:
        if len(x) == 0 and len(y) == 0:
            return 0
        else:
            return 1


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

    localizedBlues = rs['localizedBlues']
    stopLocations = rs['stopLocs']
    blues = rs['blues']
    timeIntervalls = list(rs['intervalls'])

    ''' Infer the friends for each time period. You are friends
    if you have met somewhere else than university of each period '''
    outgoingFriends = collections.defaultdict(dd_list)
    for user, timePeriods in localizedBlues.items():
        for timePeriod, observations in timePeriods.items():
            if observations:
                for locationIndex, observation in observations.items():
                    peers = observation.keys()
                    con = list(stopLocations[str(user)][timePeriod].values())[locationIndex][1]
                    if con != 'university':  # Potentially exclude the home
                        # location because of the dormitories
                        [outgoingFriends[str(user)][timePeriod].append(friend)
                         for friend in peers]

    ''' Filter friends you only meet once '''
    filteredOutgoingFriends = collections.defaultdict(dd_list)
    for user, timePeriods in outgoingFriends.items():
        for timePeriod, peers in timePeriods.items():
            filteredPeers = collections.Counter(peers)
            filteredPeers = [k for k, v in filteredPeers.items() if v > 0]
            filteredOutgoingFriends[str(user)][timePeriod] = filteredPeers

    ''' Creat incoming links '''
    incomingFriends = collections.defaultdict(dd_set)
    for user, timePeriods in filteredOutgoingFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                incomingFriends[str(peer)][timePeriod].add(str(user))

    ''' Make the network symmetric '''
    friends = collections.defaultdict(dd_set)
    for user, timePeriods in outgoingFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                friends[str(peer)][timePeriod].add(str(user))

    for user, timePeriods in incomingFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                friends[str(peer)][timePeriod].add(str(user))

    ''' Build network based on the time people have met '''
    outgoingTimeFriends = collections.defaultdict(dd_list)
    countTimesBlues = collections.defaultdict(list)
    for user, blue in blues.items():
        for timePeriod, observations in blue.items():
            if observations:
                ls = [outgoingTimeFriends[str(user)][timePeriod].append(b.peer)
                      for b in observations
                      if TimeAux.epochToLocalTime(b.time, amsterdam).hour > 17]
                [countTimesBlues[timePeriod].append(
                    TimeAux.epochToLocalTime(b.time, amsterdam).hour)
                 for b in observations
                 if TimeAux.epochToLocalTime(b.time, amsterdam).hour > 17]

    ''' Filter friends you only meet once '''
    filteredOutgoingTimeFriends = collections.defaultdict(dd_list)
    for user, timePeriods in outgoingTimeFriends.items():
        for timePeriod, peers in timePeriods.items():
            filteredPeers = collections.Counter(peers)
            filteredPeers = [k for k, v in filteredPeers.items() if v > 0]
            filteredOutgoingTimeFriends[str(user)][timePeriod] = filteredPeers

    incomingTimeFriends = collections.defaultdict(dd_set)
    for user, timePeriods in filteredOutgoingTimeFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                incomingFriends[str(peer)][timePeriod].add(str(user))

    ''' Make the network symmetric '''
    timeFriends = collections.defaultdict(dd_set)
    for user, timePeriods in filteredOutgoingTimeFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                timeFriends[str(peer)][timePeriod].add(str(user))

    for user, timePeriods in incomingTimeFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                timeFriends[str(peer)][timePeriod].add(str(user))

    ''' Build hybrid network - both time and context '''
    hybridFriends = collections.defaultdict(dd_set)

    for user, timePeriods in friends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                hybridFriends[str(user)][timePeriod].add(str(peer))

    for user, timePeriods in timeFriends.items():
        for timePeriod, peers in timePeriods.items():
            for peer in peers:
                hybridFriends[str(user)][timePeriod].add(str(peer))

    ''' Caldulate densities for the time slices '''
    densitiesTime = []
    connectionsTime = DefaultOrderedDict(int)

    N = max([len(timeFriends.keys()), len(friends.keys()),
             len(hybridFriends.keys())])
    pc = N * (N - 1)
    for user in timeFriends.keys():
        for key in timeIntervalls:
            currentFriends = timeFriends[user][key]
            connectionsTime[key] += len(currentFriends)

    for c in connectionsTime.values():
        density = c / pc
        densitiesTime.append(density)

    densities = []
    connections = DefaultOrderedDict(int)

    for user in friends.keys():
        for key in timeIntervalls:
            currentFriends = friends[user][key]
            connections[key] += len(currentFriends)

    for c in connections.values():
        density = c / pc
        densities.append(density)

    densitiesHybrid = []
    connectionsHybrid = DefaultOrderedDict(int)

    for user in hybridFriends.keys():
        for key in timeIntervalls:
            currentFriends = hybridFriends[user][key]
            connectionsHybrid[key] += len(currentFriends)

    for c in connectionsHybrid.values():
        density = c / pc
        densitiesHybrid.append(density)

    ''' Estimate change between time slices '''
    differencesTime = collections.defaultdict(list)
    for user in timeFriends:
        currentFriends = timeFriends[user][timeIntervalls[0]]
        for key in timeIntervalls[1:]:
            lastFriends = currentFriends
            currentFriends = timeFriends[user][key]
            diff = calculateDifference(lastFriends, currentFriends)
            differencesTime[user].append(diff)

    countDifferencesTime = zip(*differencesTime.values())
    for i, d in enumerate(countDifferencesTime):
        print('Average change period (time)', i, '/', i+1,
              ': ', statistics.mean(d))

    differences = collections.defaultdict(list)
    for user in friends:
        currentFriends = friends[user][timeIntervalls[0]]
        for key in timeIntervalls[1:]:
            lastFriends = currentFriends
            currentFriends = friends[user][key]
            diff = calculateDifference(lastFriends, currentFriends)
            differences[user].append(diff)
    print()
    countDifferences = zip(*differences.values())
    for i, d in enumerate(countDifferences):
        print('Average change period (context)', i, '/', i+1,
              ': ', statistics.mean(d))

    differencesHybrid = collections.defaultdict(list)
    for user in hybridFriends:
        currentFriends = hybridFriends[user][timeIntervalls[0]]
        for key in timeIntervalls[1:]:
            lastFriends = currentFriends
            currentFriends = hybridFriends[user][key]
            diff = calculateDifference(lastFriends, currentFriends)
            differencesHybrid[user].append(diff)
    print()
    countDifferencesHybrid = zip(*differencesHybrid.values())
    for i, d in enumerate(countDifferencesHybrid):
        print('Average change period (hybrid)', i, '/', i+1,
              ': ', statistics.mean(d))
    print()
    for i, (t, d, h) in enumerate(zip(densitiesTime, densities,
                                      densitiesHybrid)):
        print('Densities period ', i, ': Time: ', t, ' context: ', d,
              ' hybrid: ', h)

    ''' Export best network '''
    for key in timeIntervalls:
        lines = []
        filename = str(key[0]) + '-' + str(key[1]) + '.csv'
        for user in hybridFriends.keys():
            if len(hybridFriends[user][key]) > 0:
                line = user + ',' + ','.join(hybridFriends[user][key]) + '\n'
            else:
                line = user + '\n'
            lines.append(line)
        with open('./results/' + filename, 'w') as f:
            for line in lines:
                f.write(line)
