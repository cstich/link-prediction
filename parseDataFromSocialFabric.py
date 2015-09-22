# Parse data for link prediction

from geogps import Parser
from geogps import TimeAux
from geogps import EstimateSignificantLocations as esl
from geogps import Partition
from geogps.DictAux import dddd_list, DefaultOrderedDict
from bisect import bisect_left
from delorean import Delorean

import pytz
import re
import os
import sys
import collections
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

    ''' Parse in the system arguments '''
    inputLocations = sys.argv[1]
    inputContext = sys.argv[2]
    inputBT = sys.argv[3]
    timeDelta = int(sys.argv[4])
    numberOfIntervalls = int(sys.argv[5])
    outputPath = sys.argv[6]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputLocations = Parser.parsePath(inputLocations, scriptDir)
    inputContext = Parser.parsePath(inputContext, scriptDir)
    outputPath = Parser.parsePath(outputPath, scriptDir)

    ''' Read all sig. locations from the saved pickles '''
    pattern = re.compile('sig_locations_user_([0-9]+)\.pck')
    sigLocs = Parser.loadPickles(inputLocations, pattern)

    ''' Find the first and last midnights bounding the observations '''
    minimum = min([t[0] for v in sigLocs.values() for t in v.values()],
                  key=lambda x: x.time)
    maximum = max([t[0] for v in sigLocs.values() for t in v.values()],
                  key=lambda x: x.time)
    firstMonday = TimeAux.localTimeToEpoch(
        Delorean(TimeAux.epochToLocalTime(minimum.time,
                                          amsterdam)).last_monday().midnight())
    lastSunday = TimeAux.localTimeToEpoch(
        Delorean(TimeAux.epochToLocalTime(maximum.time,
                                          amsterdam)).next_monday().midnight())

    ''' Infer users '''
    users = Parser.inferUsers(inputLocations, pattern)

    ''' Read the geo contexts from the saved pickles '''
    pattern = re.compile('geographic_context_sig_loc_user_([0-9]+)\.pck')
    context = Parser.loadPickles(inputContext, pattern)

    ''' Read in the bluetooth filepaths '''
    pattern = re.compile('bluetooth_user_([0-9]+)\.ssv')
    listOfBTFiles = Parser.readFiles(inputBT, pattern)

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

    ''' TO DO: Try what happens if you pass a nested list and what
    happens when you pass a flat list '''
    # I can do this because I have OrderedDicts
    listOfSigLocs = [e[0] for v in list(sigLocs.values()) for e in v.values()]
    # listOfSigLocs = Aux.flatten([v.values() for v in sigLocs.values()])
    globalSigLocs = esl.clusterLocationsInSpace(listOfSigLocs)
    # Match the labels of the globalSigLoc to the individual locations
    # Create a lookup data structure
    # EDGE CASE: What if a user has no sig. locations? What happens then?
    globalSigLocsLabels = collections.defaultdict(list)
    index = 0
    for user, sls in sigLocs.items():
        for c in sls:
            globalSigLocsLabels[user].append(globalSigLocs[index])
            index += 1

    ''' Add the context and the global label to locations '''
    newLocations = collections.defaultdict(DefaultOrderedDict)  # A very nested
    # dictionary, should probably be ordered
    for user, locations in sigLocs.items():
        for i, (time, ls) in enumerate(locations.items()):
            assert len(ls) == 2
            (l, localLabel) = ls
            con = context[user][i]
            globalLabel = globalSigLocsLabels[user][i]
            newLocations[user][time] = tuple([l, con, localLabel, globalLabel])

    ''' Find the densest period to determine the intervalls for training and
    test data '''
    countObservations = list()
    for user, locations in sigLocs.items():
        for time, ls in locations.items():
            l = ls[0]
            begin = time[0]
            end = time[1]
            countObservations.extend(list(range(begin, end, 60)))

    ''' Split the stop locations into bins '''
    # Also take into account the densities of the localized Bluetooth
    # measurements. See also the plots from lookAtDensities.py
    # timeDelta = int((lastSunday - firstMonday)/numberOfIntervalls)
    # cutOff = firstMonday + (lastSunday - firstMonday)/4
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

    stopLocations = collections.defaultdict(collections.OrderedDict)
    while slidingTimeDelta < timeDelta:
        for user, locations in newLocations.items():
            partitionedLocations =\
                Partition.splitByTimedelta(locations.items(),
                                           timeDelta,
                                           lookupKey=lambda x: x[1][0].time,
                                           lowerBound=firstMonday+slidingTimeDelta,
                                           upperBound=lastSunday)
            for timePeriod, locations in partitionedLocations.items():
                stopLocations[user][timePeriod] = collections.OrderedDict(
                    locations)
        slidingTimeDelta += offset

    ''' Match the bluetooths to sig. locations '''
    BTs = collections.defaultdict(dddd_list)
    blues = collections.defaultdict(list)
    for BTFilename in listOfBTFiles:
        node = Parser.parseBTTracesForOneUser(BTFilename, userIDIndex=0,
                                              time=1, strength=4, peer=5)
        print('Matching BTs of user: ' + str(node.name))
        node.discardUnknownBluetooths()
        node.discardWeakBluetooths(lowerBound=-80, upperBound=100)
        node.bluetooths.sort(key=lambda x: x.time)

        blues[str(node.name)] =\
            Partition.splitBySlidingTimedelta(node.bluetooths,
                                              timeDelta,
                                              offset,
                                              lookupKey=lambda x: x.time,
                                              lowerBound=firstMonday,
                                              upperBound=lastSunday)

        ''' Match bluetooth to locations '''
        for timePeriod, locations in stopLocations[str(node.name)].items():
            if locations:
                ksLower, ksUpper = zip(*locations)
                for bt in node.bluetooths:
                    posSL = matchBTToSigLoc(bt, ksLower, ksUpper)
                    if posSL:
                        peer = bt.peer
                        BTs[str(node.name)][timePeriod][posSL][peer].append(bt)

    results = dict()
    results['localizedBlues'] = BTs
    results['blues'] = blues
    results['stopLocs'] = stopLocations
    results['count'] = countObservations
    results['intervalls'] = timeIntervalls
    results['slidingIntervalls'] = slidingTimeIntervalls

    ''' Feature extraction '''
    with open(outputPath + '/parsedData_time_' +
              str(timeDelta/3600/24).replace('.', '_') + '.pck', 'wb') as f:
        pickle.dump(results, f)
