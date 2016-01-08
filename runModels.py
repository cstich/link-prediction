from gs import parser
from gs.dictAux import castLeaves, DefaultOrderedDict, mergeDicts

import predictions
import networksAux

import collections
import copy
import numpy as np
import os
import pickle
import pytz
import re
import resource
import sklearn.cross_validation
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


def generateTestingTimeIntervalls(trainingTimeIntervalls, length):
    result = list()
    for time in trainingTimeIntervalls:
        intervall = tuple([time[1], time[1]+length])
        result.append(intervall)
    return result


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


def generateFeatureVectorStrings(s, iterable, sep=''):
    result = []
    iterable = map(str, iterable)
    for i in iterable:
        result.append(sep.join([s, i]))
    return result


def checkFeature(ls, name):
    if max(ls) <= 0:
        warningString = name + ': Has no positive cases'
        warnings.warn(warningString, UserWarning)


def trianglesLambda():
    return None


def selectModel(X, model, y):
    ls = list()
    for key in model:
        ls.append(X[key])
    result = np.asarray(list(zip(*ls)))
    assert len(y) == len(result)
    return result


def runModel(X_features, y, sdn, models, user, nullModelData):
    '''
    A = range(
        trainingIntervall[0], testingIntervall[0], lengthOfDeltaT)
    B = range(
        trainingIntervall[0] + lengthOfDeltaT,
        testingIntervall[0] + lengthOfDeltaT, lengthOfDeltaT)

    trainingSplitIntervalls = list(zip(A, B))
    nullModelTrainingFilenames = [
        networkFiles[intervall]
        for intervall in trainingSplitIntervalls]
    '''
    # nullModelTestingFilename = networkFiles[testingIntervall]

    results = predictions.createResultsDictionaries()
    featureImportance = DefaultOrderedDict(list)

    sdn = np.asarray(sdn)
    y = np.asarray(y)
    castLeaves(X_features, np.asarray)

    for key, model in models.items():
        kf = sklearn.cross_validation.KFold(len(y), n_folds=5)
        X = selectModel(X_features, model, y)

        for fold, (train_index, test_index) in enumerate(kf):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pred = predictions.Predictions(
                X_train, y_train, X_test, y_test,
                sdn, results, featureImportance,
                nullModelData['probabilities'], nullModelData['truths'],
                nullModelData['actuals'], nullModelData['prediction'],
                True, 8, model, classesSet,
                outputPath)
            pred.run(key)
    pred.scoreNullModel()
    pred.export(user)
    # TODO Export prediction model
    # TODO Parralize model
    # timestep = str(testingIntervall[0]) + '-' + str(testingIntervall[1])
    # open(outputPath + '/threads/timestep_' + timestep + '.pid', 'w').close()


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
    if len(sys.argv) != 4:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory> "
              "<number of folds> "
              "<output directory>")
        sys.exit(-1)

    ''' Set memory limit '''
    print('setting memory limit')
    rsrc = resource.RLIMIT_AS
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (1024**3*30, hard))  # limit is in bytes

    ''' Load data '''
    print('loading data')
    inputDir = sys.argv[1]
    kfolds = int(sys.argv[2])
    outputPath = sys.argv[3]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    userFeature = re.compile(
        'user_([0-9]+)_timestep_([0-9-]+)\.pck')
    userFeatures = parser.readFiles(
        inputDir, userFeature, generateKey=True)

    listOfFeatures = generateListOfFeatures()

    models = createDictOfFeatures()
    classes = networksAux.mapSecondsToFriendshipClass
    classesSet = networksAux.classesSet()
    counters = list()
    # users = list(userFeatures.keys())
    users = ['7', '8']
    # TODO missing a loop over all users

    for user in users:
        Xs = dict()
        ys = list()
        sdns = list()
        nullModelData = dict()
        for _, featureFile in userFeatures[user].items():
            with open(featureFile, 'rb') as f:
                features = pickle.load(f)
                mergeDicts(Xs, features['X'])
                mergeDicts(nullModelData, features['nullModel'])
                ys.extend(features['y'])
                sdns.extend(features['sdn'])
            counters.append(collections.Counter(ys))
        runModel(Xs, ys, sdns, models, user, nullModelData)
