from gs import aux
from gs import parser
from gs.dictAux import dd_dict, dd_set

from Metrics import UserMetrics
from Metrics import generate_triangles
from networksAux import mapSecondsToFriendshipClass

import collections
import copy
import Metrics
import networkx as nx
import numpy as np
import os
import pickle
import pytz
import re
import resource
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


def generateListOfFeatures():
    ls = [
        'metLast',
        'timeSpent',
        'numberOfPeople',
        'timeSpentAtHomeWith',
        'timeSpentAtUniversityWith',
        'timeSpentAtThirdPlaceWith',
        'timeSpentAtOtherPlaceWith',
        'relativeImportance',
        'adamicAdar',
        'jaccard',
        'PA',
        'placeEntropy',
        'propScores',
        'resourceAllocation']
    metAtHourOfTheWeek = generateFeatureVectorStrings('metAtHourOfTheWeek',
                                                      range(196))
    ls.extend(metAtHourOfTheWeek)
    triadicClosure = generateFeatureVectorStrings('triadicClosure', range(6))
    ls.extend(triadicClosure)
    return ls


def generateNetworkFeatures(network, features, networkFunctions, labels):
    for nF, label in zip(networkFunctions, labels):
        for e in nF(network):
            setNetworkFeatures(features, label, e)


def readNetwork(fileObject, nxGraph=None, evaluateKey=None, value=None):
    key = networkFilesPattern.search(fileObject).group(1).split(sep='-')
    key = tuple(map(int, key))
    if evaluateKey:
        if not evaluateKey(key, value):
            return None, None, None
    network = dict()
    nxNetwork = nx.Graph()
    with open(fileObject, 'r') as f:
        for line in f.readlines():
            line = line[:-1].split(sep=',')
            node = line[0]
            peers = copy.copy(line[1:])
            peers = dict(zip(peers[0::2], peers[1::2]))
            network[node] = peers
            if nxGraph:
                for peer, strength in peers.items():
                    nxNetwork.add_edge(node, peer, weight=strength)
    if nxGraph:
        return key, network, nxNetwork
    else:
        return key, network


def setNetworkFeatures(features, name, edge):
    try:
        features[edge[0]][name][edge[1]] = edge[2]
    except (TypeError, KeyError):
        features[edge[0]][name] = collections.defaultdict(float)
        features[edge[0]][name][edge[1]] = edge[2]
    try:
        features[edge[1]][name][edge[0]] = edge[2]
    except (TypeError, KeyError):
        features[edge[1]][name] = collections.defaultdict(float)
        features[edge[1]][name][edge[0]] = edge[2]


def findValuesForFeatures(listOfFeatures, features, user, peer):
    result = []
    for f in listOfFeatures:
        try:
            result.append(features[user][f][peer])
        except (TypeError, KeyError):
            result.append(0)
    return result


def generateFeatureVectorStrings(s, iterable, sep='_'):
    result = []
    iterable = map(str, iterable)
    for i in iterable:
        result.append(sep.join([s, i]))
    return result


def generateTestingTimeIntervalls(trainingTimeIntervalls, length):
    result = list()
    for time in trainingTimeIntervalls:
        intervall = tuple([time[1], time[1]+length])
        result.append(intervall)
    return result


def findTie(network, user, peer, classes=None):
    if classes:
        try:
            result = network[user][peer]
            return classes(result)
        except KeyError:
            return 0
    else:
        return int(peer in network[user])


def unweighNetwork(network):
    result = collections.defaultdict(list)
    for user, links in network.items():
        for peer, strength in links.items():
            if int(strength) > 0:
                result[user].append(peer)
    return result


def binBlues(bluetooths):
    interactionsAtTime = collections.defaultdict(dd_set)
    for user, blues in bluetooths.items():
        for blue in blues:
            peer = str(blue[0])
            if peer == user:
                continue
            # Bin all interactions into 10 minute buckets
            # Consider wider buckets if spatial triadic closure isn't
            # working very well
            time = int(blue[1] / 600) * 600
            # Assume interactions are symmetric
            interactionsAtTime[time][user].add(peer)
            interactionsAtTime[time][peer].add(user)
    return interactionsAtTime


def generateTransformKey(split='-'):
    def transformKey(key):
        key = key.split(split)
        key = tuple(map(int, key))
        return key
    return transformKey


def evaluateKey(key, value):
    if key[1] - key[0] <= value:
        return False
    else:
        return True


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory> "
              "<networks directory> "
              "<length of 1 set of test users> "
              "<length of delta-t period in seconds> "
              "<output directory>")
        sys.exit(-1)

    ''' Set memory limit '''
    print('setting memory limit')
    rsrc = resource.RLIMIT_AS
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (1024**3*30, hard))  # limit is in bytes

    ''' Load data '''
    print('loading data')
    inputData = sys.argv[1]
    inputNetworks = sys.argv[2]
    lengthOfTestUsers = int(sys.argv[3])
    lengthOfDeltaT = int(sys.argv[4])
    outputPath = sys.argv[5]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    with open(inputData, 'rb') as f:
        rs = pickle.load(f)

    base = os.path.basename(inputData)
    base = base.split('_')[2]
    resultsDirectory = os.path.dirname(inputData)
    transformKey = generateTransformKey('_')
    locBluesPattern = re.compile(
        'parsedData_time_'+base+'_localizedBlue_([0-9_]+)\.pck')
    localizedBluesFiles = parser.readFiles(
        resultsDirectory, locBluesPattern, transformKey=transformKey)
    bluesPattern = re.compile(
        'parsedData_time_'+base+'_blue_([0-9_]+)\.pck')
    bluesFiles = parser.readFiles(
        resultsDirectory, bluesPattern, transformKey=transformKey)

    stopLocations = rs['stopLocs']
    placeEntropy = Metrics.calculatePlaceEntropy(stopLocations)
    timeIntervalls = list(rs['intervalls'])  # Are ordered in time
    slidingTimeIntervalls = list(rs['slidingIntervalls'])
    users = rs['users']

    ''' Read networks at different time states '''
    networkFilesPattern = re.compile('([0-9]+\-[0-9-]+)\.csv')
    transformKey = generateTransformKey('-')
    networkFiles = parser.readFiles(
        inputNetworks, networkFilesPattern, generateKey=True,
        transformKey=transformKey)
    networks = collections.defaultdict(dd_dict)
    testingNetworks = collections.defaultdict(dd_dict)
    # nxNetworks = collections.defaultdict(nx.Graph)

    # TODO check the re-factoring
    ''' Generate featues '''
    print('Generate features')
    features = collections.defaultdict(dd_dict)
    networkFunctions = [nx.adamic_adar_index, nx.jaccard_coefficient,
                        nx.preferential_attachment,
                        nx.resource_allocation_index]
    networkLabels = ['adamicAdar', 'jaccard', 'PA', 'resourceAllocation']
    progress = 0
    listOfFeatures = generateListOfFeatures()
    ''' Use the same train and test sample for all timeframes
    aka create the samples of test users before you output your
    data '''
    # Create n-folded training and verification sets
    # 1) Shuffle the list of users
    # 2) Split the random list of users into n-sublists
    # where each sublist represents a set of users that are
    # used for testing
    testUsers = np.random.choice(list(users), len(users), replace=False)
    testUsers = aux.chunkify(testUsers, int(len(users)/lengthOfTestUsers))
    classes = mapSecondsToFriendshipClass
    testingIntervalls = generateTestingTimeIntervalls(timeIntervalls,
                                                      lengthOfDeltaT)
    trainingTestingIntervalls = zip(timeIntervalls,
                                    testingIntervalls)

    # TODO Parallize this
    for trainingIntervall, testingIntervall in trainingTestingIntervalls:
        progress += 1
        print(progress/len(networkFiles))
        print('Working on: ', trainingIntervall, '/', testingIntervall)
        trainingNetworkFile = networkFiles[trainingIntervall]
        testingNetworkFile = networkFiles[testingIntervall]
        key, network, nxNetwork = readNetwork(
            trainingNetworkFile, nxGraph=True)
        testingKey, testingNetwork = readNetwork(testingNetworkFile)
        bluesFile = bluesFiles[key]
        with open(bluesFile, 'rb') as f:
            blues = pickle.load(f)
        localizedBluesFile = localizedBluesFiles[key]
        with open(localizedBluesFile, 'rb') as f:
            localizedBlues = pickle.load(f)

        ''' Generate bluetooth bins '''
        interactionsAtTime = binBlues(blues)

        ''' Generate triangles for all bluetooth bins '''
        triangles = collections.defaultdict(lambda: None)
        for time, interactions in interactionsAtTime.items():
            triangles[time] = generate_triangles(interactions)

        ''' Generate user based features '''
        for user, peers in network.items():
            try:
                stopLocs = stopLocations[user][key]
            except KeyError:
                stopLocs = None
            try:
                locBlues = localizedBlues[user]
            except KeyError:
                locBlues = None
            try:
                bl = blues[user]
            except KeyError:
                bl = None
            um = UserMetrics(user,
                             users,
                             bl,
                             locBlues,
                             stopLocs,
                             interactionsAtTime,
                             triangles,
                             placeEntropy,
                             key[0], key[1])
            um.generateFeatures()
            features[um.user] = um.metrics

        generateNetworkFeatures(
            nxNetwork, features, networkFunctions, networkLabels)

        ''' Save features to file '''
        for j, currentTestSet in enumerate(testUsers):
            test = []
            train = []
            for user in users:
                for peer in users:
                    if user != peer:
                        line = []
                        truth = findTie(testingNetwork,
                                        user, peer, classes)
                        friends = findTie(network, user,
                                          peer, classes)
                        line.extend([truth, user, peer, friends])
                        # Iterate through all the features
                        valuesFeatures = findValuesForFeatures(
                            listOfFeatures, features,
                            user, peer)
                        line.extend(valuesFeatures)
                        if user in currentTestSet:
                            test.append(line)
                        else:
                            train.append(line)
            directory = outputPath + str(testingIntervall[0])
            if not os.path.exists(directory):
                    os.makedirs(directory)
            header = ['edge', 'source', 'destination', 'friends', 'random']
            header.extend(listOfFeatures)
            with open(directory + '/train' +
                      '_sample_' + str(j) + '.csv', 'w') as f:
                f.write(','.join(header) + '\n')
                for line in train:
                    f.write(','.join(map(str, line)) + '\n')
            with open(directory + '/test' +
                      '_sample_' + str(j) + '.csv', 'w') as f:
                f.write(','.join(header) + '\n')
                for line in test:
                    f.write(','.join(map(str, line)) + '\n')
