from gs import parser
from gs.dictAux import dd_dict, dd_float, dd_set

from Metrics import UserMetrics
from Metrics import generate_triangles
from PersonalizedPageRank import PersonalizedPageRank

import networksAux
import nullModel

import collections
import copy
import multiprocessing as mp
import networkx as nx
import numpy as np
import os
import pickle
import pytz
import re
import resource
import sys
import warnings

amsterdam = pytz.timezone('Europe/Amsterdam')


def createDictOfFeatures():
    dictOfFeatures = dict(
        altNullModel=altNullFeatures,
        base=baseFeatures, time=timeFeatures,
        place=placeFeatures, network=networkFeatures,
        social=socialFeatures,
        timeSocialPlace=timeSocialPlaceFeatures,
        full=fullFeatures, past=pastFeatures,
        networkOnly=networkOnlyFeatures,
        node=nodeFeatures)
    return dictOfFeatures


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
        'resourceAllocation',
        'weightedPropScores']
    metAtHourOfTheWeek = generateFeatureVectorStrings('metAtHourOfTheWeek',
                                                      range(168))
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


def setPropScores(features, user, scores, name='propScores'):
    for s in scores:
        peer = s[0]
        try:
            features[user][name][peer] = s[1]
        except TypeError:
            features[user][name] =\
                collections.defaultdict(float)
            features[user][name][peer] = s[1]


def findValuesForFeatures(X, listOfFeatures, features, user, peer):
    for f in listOfFeatures:
        try:
            e = features[user][f][peer]
        except (TypeError, KeyError):
            e = 0
        X[f].append(e)


def generateFeatureVectorStrings(s, iterable, sep=''):
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


def checkFeature(ls, name):
    if max(ls) <= 0:
        warningString = name + ': Has no positive cases'
        warnings.warn(warningString, UserWarning)


def trianglesLambda():
    return None


def generateFeatures(interactionsAtTime, stopLocations, localizedBlues,
                     blues, allUsers, key, triangles, placeEntropy, user):
    try:
        stopLocs = stopLocations[user]
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
    um = UserMetrics(
        user, allUsers, bl, locBlues, stopLocs, interactionsAtTime,
        triangles, placeEntropy, key[0], key[1])
    um.generateFeatures()
    return um.metrics


def selectModel(X, model, y):
    ls = list()
    for key in model:
        ls.append(X[key])
    result = np.asarray(list(zip(*ls)))
    assert len(y) == len(result)
    return result


def featureWrapper(trainingTestingIntervall):
    trainingIntervall, testingIntervall = trainingTestingIntervall
    assert trainingIntervall[1] == testingIntervall[0]
    print('Working on: ', trainingIntervall, '/', testingIntervall)
    trainingNetworkFile = networkFiles[trainingIntervall]
    testingNetworkFile = networkFiles[testingIntervall]
    key, trainingNetwork, nxTrainingNetwork = readNetwork(
        trainingNetworkFile, nxGraph=True)
    testingKey, testingNetwork = readNetwork(testingNetworkFile)

    ''' nullModel filenames '''
    a = range(trainingIntervall[0], trainingIntervall[1],
              lengthOfDeltaT)
    b = range(trainingIntervall[0] + lengthOfDeltaT,
              trainingIntervall[1] + lengthOfDeltaT,
              lengthOfDeltaT)
    nullModelIntervalls = list(zip(a, b))
    nullModelTrainingFilenames = [networkFiles[intervall]
                                  for intervall in nullModelIntervalls]
    nM = nullModel.NullModel(
            nullModelTrainingFilenames, testingNetworkFile, classesSet, allUsers)

    bluesFile = bluesFiles[key]
    with open(bluesFile, 'rb') as f:
        blues = pickle.load(f)
    localizedBluesFile = localizedBluesFiles[key]
    with open(localizedBluesFile, 'rb') as f:
        localizedBlues = pickle.load(f)
    stopLocationsFile = stopLocationsFiles[key]
    with open(stopLocationsFile, 'rb') as f:
        stopLocations = pickle.load(f)

    ''' Generate bluetooth bins '''
    interactionsAtTime = binBlues(blues)

    ''' Generate triangles for all bluetooth bins '''
    triangles = collections.defaultdict(trianglesLambda)
    for time, interactions in interactionsAtTime.items():
        triangles[time] = generate_triangles(interactions)

    ''' Generate user based features '''
    for user in list(trainingNetwork.keys()):
        features[user] = generateFeatures(
            interactionsAtTime, stopLocations, localizedBlues,
            blues, allUsers, key, triangles, placeEntropy, user)

    generateNetworkFeatures(
        nxTrainingNetwork, features, networkFunctions, networkLabels)

    ''' Find the propagation scores '''
    unweightedTrainingNetwork = unweighNetwork(trainingNetwork)
    normalizedTrainingNetwork = collections.defaultdict(dd_float)
    for user, peers in trainingNetwork.items():
        meetings = list(map(lambda x: int(x)+1, list(peers.values())))
        normalizedMeetings = 0
        allMeetings = sum(meetings)

        for peer, value in peers.items():
            normalizedValue = (int(value)+1)/allMeetings
            normalizedTrainingNetwork[user][peer] = normalizedValue
            normalizedMeetings += normalizedValue
    assert normalizedMeetings > 0.9999 and normalizedMeetings < 1.00001

    for user, peers in trainingNetwork.items():
        ppr = PersonalizedPageRank(
            unweightedTrainingNetwork, unweightedTrainingNetwork)
        scores = ppr.pageRank(user, returnScores=True)
        setPropScores(features, user, scores)
        weightedPpr = PersonalizedPageRank(
            unweightedTrainingNetwork, unweightedTrainingNetwork,
            normalizedTrainingNetwork)
        weightedScores = weightedPpr.pageRank(user, returnScores=True,
                                              weighted=True)
        setPropScores(features, user, weightedScores, 'weightedPropScores')

    ''' Tests to make sure we include all features '''
    setFeatures = set(listOfFeatures)
    setFeatures.add('edge')
    assert set(fullFeatures) == setFeatures

    timestep = str(testingIntervall[0])+'-'+str(testingIntervall[1])
    for user in allUsers:
        # X = []  # X are features
        X = collections.defaultdict(list)  # X is the training data
        y = list()  # y are the associated values
        src_dest_nodes = list()
        # organized by feature -> easy to select a certain set

        for peer in allUsers:
            truth = findTie(testingNetwork,
                            user, peer, classes)
            edge = findTie(trainingNetwork, user,
                           peer, classes)
            findValuesForFeatures(
                X, listOfFeatures, features,
                user, peer)

            X['edge'].append(edge)
            y.append(truth)
            src_dest_nodes.append((user, peer))
        nMData = dict(prediction=nM.prediction[user], probabilities=nM.probabilities[user],
                      actuals=nM.stringActuals[user], truths=nM.truths[user])
        user_data = dict(X=X, y=y, sdn=src_dest_nodes,
                         nullModel=nMData)
        filename = outputPath+'/user_'+str(user)+'_timestep_'+timestep+'.pck'
        with open(filename, 'wb') as f:
            pickle.dump(user_data, f)
    open(outputPath + '/threads/timestep_' + timestep + '.pid', 'w').close()

''' Define the different models '''
# TODO Update features

''' Alternative null model '''
altNullFeatures = [
    'timeSpent'
]

''' Past model '''
pastFeatures = [
    'edge'
    ]

''' Basic model '''
baseFeatures = [
    'timeSpent',
    'edge',
    'metLast']

''' Single feature set models '''
networkOnlyFeatures = []
networkOnlyFeatures.extend([
    'jaccard',
    'adamicAdar',
    'PA',
    'propScores',
    'weightedPropScores',
    'resourceAllocation'])

networkFeatures = []
networkFeatures.extend(copy.copy(baseFeatures))
networkFeatures.extend([
    'jaccard',
    'adamicAdar',
    'PA',
    'propScores',
    'weightedPropScores',
    'resourceAllocation'])

timeFeatures = []
metAtHourOfTheWeek = generateFeatureVectorStrings('metAtHourOfTheWeek',
                                                  range(168))
timeFeatures.extend(copy.copy(baseFeatures))
timeFeatures.extend(
    metAtHourOfTheWeek)

socialFeatures = []
socialFeatures.extend(copy.copy(baseFeatures))
triadicClosure = generateFeatureVectorStrings('triadicClosure', range(6))
socialFeatures.extend([
    'numberOfPeople'])
socialFeatures.extend(triadicClosure)

placeFeatures = []
placeFeatures.extend(copy.copy(baseFeatures))
placeFeatures.extend([
    'timeSpentAtHomeWith',
    'timeSpentAtUniversityWith',
    'timeSpentAtThirdPlaceWith',
    'timeSpentAtOtherPlaceWith',
    'placeEntropy',
    'relativeImportance'])

''' Node only features '''
nodeFeatures = [
    'numberOfPeople',
    'triadicClosure1',
    'triadicClosure2',
    'triadicClosure3']
nodeFeatures.extend(copy.copy(baseFeatures))
nodeFeatures.extend(copy.copy(placeFeatures))
# nodeFeatures.extend(copy.copy(socialFeatures))
nodeFeatures.extend(copy.copy(timeFeatures))

''' time social place '''
timeSocialPlaceFeatures = copy.copy(baseFeatures)
timeSocialPlaceFeatures.extend(copy.copy(placeFeatures)[2:])
timeSocialPlaceFeatures.extend(copy.copy(socialFeatures)[2:])
timeSocialPlaceFeatures.extend(copy.copy(timeFeatures)[2:])

''' Full model '''
fullFeatures = copy.copy(baseFeatures)
fullFeatures.extend(copy.copy(placeFeatures)[2:])
fullFeatures.extend(copy.copy(socialFeatures)[2:])
fullFeatures.extend(copy.copy(networkFeatures)[2:])
fullFeatures.extend(copy.copy(timeFeatures)[2:])

featureSets = [pastFeatures, baseFeatures, networkFeatures, timeFeatures,
               socialFeatures, placeFeatures, nodeFeatures,
               timeSocialPlaceFeatures, fullFeatures]

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory> "
              "<networks directory> "
              "<number of folds> "
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
    kfolds = int(sys.argv[3])
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
    stopLocPattern = re.compile(
        'parsedData_time_' + base + '_stopLocation_([0-9_]+)\.pck')
    stopLocationsFiles = parser.readFiles(
        resultsDirectory, stopLocPattern, transformKey=transformKey)

    # Read in earlier completed threads
    completedPattern = re.compile(
        'timestep_([0-9-]+)\.pid')
    completedTimeSteps = parser.readFiles(
        outputPath + '/threads/', completedPattern,
        transformKey=generateTransformKey('-'))
    completedTimesteps = set(completedTimeSteps.keys())

    placeEntropy = rs['placeEntropy']
    timeIntervalls = list(rs['intervalls'])  # Are ordered in time
    slidingTimeIntervalls = list(rs['slidingIntervalls'])
    allUsers = rs['users']

    ''' Read networks at different time states '''
    networkFilesPattern = re.compile('([0-9]+\-[0-9-]+)\.csv')
    transformKey = generateTransformKey('-')
    networkFiles = parser.readFiles(
        inputNetworks, networkFilesPattern, generateKey=True,
        transformKey=transformKey)
    networks = collections.defaultdict(dd_dict)
    testingNetworks = collections.defaultdict(dd_dict)

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
    classes = networksAux.mapSecondsToFriendshipClass
    classesSet = networksAux.classesSet()
    testingIntervalls = generateTestingTimeIntervalls(timeIntervalls,
                                                      lengthOfDeltaT)
    trainingTestingIntervalls = zip(timeIntervalls,
                                    testingIntervalls)

    trainingTestingIntervalls = [key for key in trainingTestingIntervalls if
                                 key not in completedTimesteps]

    # Parallization
    pool = mp.Pool(6)
    pool.map(featureWrapper, trainingTestingIntervalls)
    # list(map(featureWrapper, trainingTestingIntervalls))

    threadPath = outputPath+'threads'
    for the_file in os.listdir(threadPath):
        file_path = os.path.join(threadPath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
