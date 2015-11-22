# Parse data for link prediction

from gs import parser
from gs import timeAux
from gs import estimateSignificantLocations as esl
from gs import partition
from gs.dictAux import defaultdict_defaultdict_DefaultOrderedDict,\
    dddd_list, ddd_list, dd_list, DefaultOrderedDict
from bisect import bisect_left
from delorean import Delorean
# from groupStore import dict2group

import Metrics

# import hickle as hkl
# import joblib
import collections
import numpy as np
import pytz
import re
import resource
import os
import sys
# import tables
import pickle

amsterdam = pytz.timezone('Europe/Amsterdam')


def matchBTToSigLoc(original, ksLower, ksUpper):
    time = original.time
    pos = bisect_left(ksUpper, time, 0, len(ksUpper))
    if pos < len(ksUpper) and time >= ksLower[pos] and time <= ksUpper[pos]:
        return pos
    else:
        return None


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: %s " % (sys.argv[0]) +
              "<directory of significant locations> "
              "<directory of geographic context> "
              "<directory of bluetooth contacts> "
              "<timedelta of the friendship period in seconds> "
              "<number of intervalls in a friendship period> "
              "<output directory>")
        sys.exit(-1)

    print('Version 0.1')

    ''' Set memory limit '''
    rsrc = resource.RLIMIT_AS
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (1024*1024*1024*30, hard))  # limit is in bytes

    ''' Parse in the system arguments '''
    inputLocations = sys.argv[1]
    inputContext = sys.argv[2]
    inputBT = sys.argv[3]
    timeDelta = int(sys.argv[4])
    numberOfIntervalls = int(sys.argv[5])
    outputPath = sys.argv[6]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputLocations = parser.parsePath(inputLocations, scriptDir)
    inputContext = parser.parsePath(inputContext, scriptDir)
    outputPath = parser.parsePath(outputPath, scriptDir)

    ''' Read all sig. locations from the saved pickles '''
    print('reading sig. locations')
    pattern = re.compile('sig_locations_user_([0-9]+)\.pck')
    stopLocs = parser.loadPickles(inputLocations, pattern)

    for user, values in stopLocs.items():
        for intervall, _ in values.items():
            ts = intervall[1] - intervall[0]
            assert ts > 0, 'Error in stopLoc duration. Must be positive'

    ''' Find the first and last midnights bounding the observations '''
    minimum = min([t[0] for v in stopLocs.values() for t in v.values()],
                  key=lambda x: x.time)
    maximum = max([t[0] for v in stopLocs.values() for t in v.values()],
                  key=lambda x: x.time)
    firstMonday = timeAux.localTimeToEpoch(
        Delorean(timeAux.epochToLocalTime(minimum.time,
                                          amsterdam)).last_monday().midnight())
    lastSunday = timeAux.localTimeToEpoch(
        Delorean(timeAux.epochToLocalTime(maximum.time,
                                          amsterdam)).next_monday().midnight())

    users_locations = parser.inferUsers(inputLocations, pattern)

    ''' Read the geo contexts from the saved pickles '''
    print('reading geographic contexts')
    pattern = re.compile('geographic_context_sig_loc_user_([0-9]+)\.pck')
    context = parser.loadPickles(inputContext, pattern)

    ''' Read in the bluetooth filepaths '''
    print('reading bluetooths')
    pattern = re.compile('bluetooth_([0-9]+)$')
    listOfBTFiles = parser.readFiles(inputBT, pattern)

    ''' Infer users '''
    users_blues = parser.inferUsers(inputBT, pattern)
    all_users = users_locations | users_blues
    print('found users: ', len(all_users))

    mappingForContexts = {'college': 'university',
                          'home': 'home',
                          'social': 'social',
                          'Social': 'social',
                          'Home': 'home',
                          'University': 'university',
                          'university': 'university'}

    ''' Remap different contexts '''
    # None becomes mapped to other implicitly
    context = {k: [mappingForContexts[e]
                   if e in mappingForContexts
                   else 'other'
                   for e in v]
               for k, v in context.items()}

    print('create global index of sig. locations')
    # I can do this because I have OrderedDicts
    listOfStopLocs = [e[0] for v in list(stopLocs.values()) for e in v.values()]
    # listOfSigLocs = Aux.flatten([v.values() for v in sigLocs.values()])
    globalSigLocs = esl.clusterLocationsInSpace(listOfStopLocs)
    # Match the labels of the globalSigLoc to the individual locations
    # Create a lookup data structure
    # EDGE CASE: What if a user has no sig. locations? What happens then?
    globalSigLocsLabels = collections.defaultdict(list)
    index = 0
    for user, sls in stopLocs.items():
        for c in sls:
            globalSigLocsLabels[user].append(globalSigLocs[index])
            index += 1

    ''' Add the context and the global label to locations '''
    print('adding context')
    # Am I doing this correctly?
    newLocations = collections.defaultdict(DefaultOrderedDict)  # A very nested
    # dictionary, should probably be ordered
    for user, locations in stopLocs.items():
        for i, (time, ls) in enumerate(locations.items()):
            assert len(ls) == 2
            (l, localLabel) = ls
            con = context[user][i]
            globalLabel = globalSigLocsLabels[user][i]
            newLocations[user][time] = [(time[0], time[1]), con,
                                        localLabel, globalLabel]

    ''' Split the stop locations into bins '''
    print('splitting stop locs')
    slidingTimeDelta = 0
    offset = int(timeDelta / numberOfIntervalls)

    timeIntervalls = list()
    slidingTimeIntervalls = list()
    for i in range(numberOfIntervalls):
        ls = list(range(firstMonday+i*offset,
                        lastSunday, timeDelta))
        ls = list(zip(ls[:-1], ls[1:]))
        timeIntervalls.extend(ls)
        slidingTimeIntervalls.append(ls)
    timeIntervalls.sort(key=lambda x: x[0])

    stopLocations = collections.defaultdict(
        defaultdict_defaultdict_DefaultOrderedDict)
    while slidingTimeDelta < timeDelta:
        for user, locations in newLocations.items():
            partitionedLocations =\
                partition.partitionTimePeriods(
                    locations.values(),
                    timeDelta,
                    lookupKey=lambda x: (x[0][0], x[0][1]),
                    setter=lambda e, t: e.__setitem__(0, t),
                    lowerBound=firstMonday + slidingTimeDelta,
                    upperBound=lastSunday)
            for timePeriod, locations in partitionedLocations.items():
                locations = sorted(list(locations), key=lambda x: x[0][0])
                for l in locations:
                    stopLocations[timePeriod][user][l[0]] = l
        slidingTimeDelta += offset

    ''' Match the bluetooths to sig. locations '''
    localizedBlues = collections.defaultdict(dddd_list)
    blues = collections.defaultdict(dd_list)
    print('matching bluetooths')
    numberOfUsers = 0
    for BTFilename in listOfBTFiles:
        if os.stat(BTFilename).st_size != 0:
            node = parser.parseBTTracesForOneUser(BTFilename, userIDIndex=3,
                                                  time=2, strength=1, peer=0)
            numberOfUsers += 1
            print(numberOfUsers/len(all_users))
            node.discardUnknownBluetooths()
            node.discardWeakBluetooths(lowerBound=-80, upperBound=100)
            node.bluetooths.sort(key=lambda x: x.time)
            partitionedBlues = partition.splitBySlidingTimedelta(
                node.bluetooths, timeDelta, offset,
                lookupKey=lambda x: x.time,
                lowerBound=firstMonday,
                upperBound=lastSunday)

            ''' Strip the bluetooth objects of their object nature to save
            memory and convert them to tuples '''
            newBlues = dict()
            for time, meetings in partitionedBlues.items():
                # ls = tuple([(b.peer, b.time) for b in meetings])
                ls = np.asarray([(b.peer, b.time) for b in meetings])
                blues[time][str(node.name)] = ls
            tempUserBlues = collections.defaultdict(ddd_list)

            ''' Match bluetooth to locations '''
            for timePeriod, users in stopLocations.items():
                locations = users[str(node.name)]
                if locations:
                    ksLower, ksUpper = zip(*locations)
                    for bt in node.bluetooths:
                        posSL = matchBTToSigLoc(bt, ksLower, ksUpper)
                        if posSL:
                            peer = bt.peer
                            tempUserBlues[timePeriod][posSL][peer].append(
                                bt.time)

            ''' Convert the lists of matched bluetooths to numpy arrays '''
            for timePeriod, posSls in tempUserBlues.items():
                for posSl, peers in posSls.items():
                    for peer, bluetooths in peers.items():
                        ls = np.asarray(bluetooths)
                        localizedBlues[timePeriod][str(
                            node.name)][posSl][peer] = ls

    ''' Feature extraction '''
    print('dumping files')
    filename = str(outputPath) + 'parsedData_time_' +\
        str(timeDelta).replace('.', '_')

    for period, values in blues.items():
        with open(filename + '_blue_' + str(period[0]) + '_' +
                  str(period[1]) + '.pck', 'wb') as f:
            pickle.dump(values, f)
    blues = []

    for period, values in localizedBlues.items():
        with open(filename + '_localizedBlue_' +
                  str(period[0]) + '_' + str(period[1]) +
                  '.pck', 'wb') as f:
            pickle.dump(values, f)
    localizedBlues = []

    placeEntropy = Metrics.calculatePlaceEntropy(stopLocations)
    for period, values in stopLocations.items():
        with open(filename + '_stopLocation_' + str(period[0]) + '_' +
                  str(period[1]) + '.pck', 'wb') as f:
            pickle.dump(values, f)
    stopLocations = []

    results = dict()
    # results['localizedBlues'] = localizedBlues
    # results['blues'] = blues
    # results['stopLocs'] = stopLocations
    # results['count'] = countObservations
    results['placeEntropy'] = placeEntropy
    results['intervalls'] = timeIntervalls
    results['slidingIntervalls'] = slidingTimeIntervalls
    results['users'] = all_users

    with open(filename + '_results.pck', 'wb') as f:
        pickle.dump(results, f)

    sys.exit()
