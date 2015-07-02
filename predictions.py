from geogps import Parser
from geogps.Aux import DefaultOrderedDict
from predictLinks import RandomForestLinkPrediction as rf

import collections
import copy
import os
import re
import statistics
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory>"
              "<path to model>"
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    outputPath = sys.argv[2]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputData = Parser.parsePath(inputData, scriptDir)
    trainingPattern = re.compile('train_sample_[0-9]+\.csv')
    trainingFiles = Parser.readFiles(inputData, trainingPattern)
    testPattern = re.compile('test_sample_[0-9]+\.csv')
    testFiles = Parser.readFiles(inputData, testPattern)

    ''' Match training to test files '''
    matchFilesPattern = re.compile('_([0-9]+)\.csv')
    trainingFiles = sorted(trainingFiles,
                           key=lambda x: int(matchFilesPattern.search(x).group(1)))
    testingFiles = sorted(trainingFiles,
                          key=lambda x: int(matchFilesPattern.search(x).group(1)))
    files = zip(trainingFiles, testingFiles)

    ''' Define the different models '''
    ''' Basic model '''
    # To Do: Add features of the social network

    baseFeatures = [
        'timeSpent',
        ]

    ''' Single feature set models '''
    networkFeatures = [
        'jaccard',
        'adamicAdar',
        'PA',
        'propScores',
        'resourceAllocation']
    networkFeatures.extend(copy.copy(baseFeatures))

    timeFeatures = []
    timeFeatures = [
        'metAtHourOfTheWeek',
        'metAtDayOfTheWeek',
        'metAtHourOfTheDay']
    timeFeatures.extend(copy.copy(baseFeatures))

    socialFeatures = []
    socialFeatures = [
        'spatialTriadicClosure',
        'candidatesSpatialTriadicClosure',
        'numberOfPeople']
    socialFeatures.extend(copy.copy(baseFeatures))

    placeFeatures = []
    placeFeatures = [
        'metAtHome',
        'metAtUniversity',
        'metAtThirdPlace']
    placeFeatures.extend(copy.copy(baseFeatures))


    ''' time social place '''
    timeSocialPlaceFeatures = copy.copy(baseFeatures)
    timeSocialPlaceFeatures.extend(copy.copy(placeFeatures))
    timeSocialPlaceFeatures.extend(copy.copy(socialFeatures))
    timeSocialPlaceFeatures.extend(copy.copy(timeFeatures))


    ''' Full model '''
    fullFeatures = copy.copy(baseFeatures)
    fullFeatures.extend(copy.copy(placeFeatures))
    fullFeatures.extend(copy.copy(socialFeatures))
    fullFeatures.extend(copy.copy(networkFeatures))
    fullFeatures.extend(copy.copy(timeFeatures))


    ''' Run the models and score them '''
    modelScores = DefaultOrderedDict(list)
    for train, test in files:
        baseModel = rf(train, test, baseFeatures)
        modelScores['base'].append(baseModel.scorePredictions())
        networkModel = rf(train, test, networkFeatures)
        modelScores['network'].append(networkModel.scorePredictions())
        timeModel = rf(train, test, timeFeatures)
        modelScores['time'].append(timeModel.scorePredictions())
        socialModel = rf(train, test, socialFeatures)
        modelScores['social'].append(socialModel.scorePredictions())
        placeModel = rf(train, test, placeFeatures)
        modelScores['place'].append(placeModel.scorePredictions())
        timeSocialPlaceModel = rf(train, test, timeSocialPlaceFeatures)
        modelScores['timeSocialPlace'].append(timeSocialPlaceModel.scorePredictions())
        fullModel = rf(train, test, fullFeatures)
        modelScores['full'].append(fullModel.scorePredictions())

    for key, values in modelScores.items():
        print(key, ': ', statistics.mean(values))
