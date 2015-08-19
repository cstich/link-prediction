from geogps import Parser
from geogps.DictAux import DefaultOrderedDict
from predictLinks import RandomForestLinkPrediction as rf
from nullModel import NullModel

import copy
import os
import pytz
import re
import statistics
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: %s " % (sys.argv[0]) +
              "<data directory> "
              "<networks directory> "
              "<number of jobs> "
              "<output path>")
        sys.exit(-1)

    inputData = sys.argv[1]
    inputNetworks = sys.argv[2]
    n_jobs = int(sys.argv[3])
    outputPath = sys.argv[4]

    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputData = Parser.parsePath(inputData, scriptDir)
    trainingPattern = re.compile('train_sample_[0-9]+\.csv')
    trainingFiles = Parser.readFiles(inputData, trainingPattern)
    testPattern = re.compile('test_sample_[0-9]+\.csv')
    testFiles = Parser.readFiles(inputData, testPattern)
    lengthOfPeriod = float(inputData.split('/')[-3].replace('_', '.'))
    unixTime = int(inputData.split('/')[-1])
    timestep = float((unixTime-1349042400)/24/3600/lengthOfPeriod)

    ''' Create null model '''
    networkPatternT0 = re.compile('[0-9]+\-' + str(unixTime) + '\.csv')
    networkPatternT1 = re.compile(str(unixTime) + '\-[0-9]+\.csv')
    networkFileT0 = Parser.readFiles(inputNetworks, networkPatternT0)[0]
    networkFileT1 = Parser.readFiles(inputNetworks, networkPatternT1)[0]
    nM = NullModel(networkFileT0, networkFileT1, True).run()
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

    ''' Match training to test files '''
    matchFilesPattern = re.compile('_([0-9]+)\.csv')
    trainingFiles = sorted(trainingFiles,
                           key=lambda x:
                           int(matchFilesPattern.search(x).group(1)))
    testingFiles = sorted(trainingFiles,
                          key=lambda x:
                          int(matchFilesPattern.search(x).group(1)))
    files = zip(trainingFiles, testingFiles)

    testSetNames = [str(matchFilesPattern.search(e).group(1))
                    for e in trainingFiles]

    ''' Define the different models '''
    ''' Random feature model '''
    randomFeatures = [
        'random'
    ]

    ''' Past model '''
    pastFeatures = [
        'friends'
    ]

    ''' Basic model '''
    baseFeatures = [
        'timeSpent',
        'friends'
        ]

    ''' Single feature set models '''
    networkOnlyFeatures = []
    networkOnlyFeatures.extend([
        'jaccard',
        'adamicAdar',
        'PA',
        'propScores',
        'resourceAllocation'])

    networkFeatures = []
    networkFeatures.extend(copy.copy(baseFeatures))
    networkFeatures.extend([
        'jaccard',
        'adamicAdar',
        'PA',
        'propScores',
        'resourceAllocation'])

    timeFeatures = []
    timeFeatures.extend(copy.copy(baseFeatures))
    timeFeatures.extend([
        'metAtHourOfTheWeek',
        'metAtDayOfTheWeek',
        'metAtHourOfTheDay'])

    socialFeatures = []
    socialFeatures.extend(copy.copy(baseFeatures))
    socialFeatures.extend([
        'spatialTriadicClosure',
        'candidatesSpatialTriadicClosure',
        'numberOfPeople'])

    placeFeatures = []
    placeFeatures.extend(copy.copy(baseFeatures))
    placeFeatures.extend([
        'metAtHome',
        'metAtUniversity',
        'metAtThirdPlace',
        'metAtOtherPlace',
        'timeSpentAtHomeWith',
        'timeSpentAtUniversityWith',
        'timeSpentAtThirdPlaceWith',
        'timeSpentAtOtherPlaceWith'])

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

    ''' Run the models and score them '''
    modelScores = DefaultOrderedDict(list)
    featureImportance = DefaultOrderedDict(list)
    for train, test in files:
        randomModel = rf(train, test, randomFeatures, n_jobs=n_jobs)
        modelScores['random'].append(randomModel.scorePredictions())
        featureImportance['random'].append(randomModel.importances)
        pastModel = rf(train, test, pastFeatures, n_jobs=n_jobs)
        modelScores['past'].append(pastModel.scorePredictions())
        featureImportance['past'].append(pastModel.importances)
        baseModel = rf(train, test, baseFeatures, n_jobs=n_jobs)
        modelScores['base'].append(baseModel.scorePredictions())
        featureImportance['base'].append(baseModel.importances)
        networkOnlyModel = rf(train, test, networkOnlyFeatures, n_jobs=n_jobs)
        modelScores['networkOnly'].append(networkOnlyModel.scorePredictions())
        featureImportance['networkOnly'].append(networkOnlyModel.importances)
        networkModel = rf(train, test, networkFeatures, n_jobs=n_jobs)
        modelScores['network'].append(networkModel.scorePredictions())
        featureImportance['network'].append(networkModel.importances)
        timeModel = rf(train, test, timeFeatures, n_jobs=n_jobs)
        modelScores['time'].append(timeModel.scorePredictions())
        featureImportance['time'].append(timeModel.importances)
        timeModel = rf(train, test, timeFeatures)
        socialModel = rf(train, test, socialFeatures, n_jobs=n_jobs)
        modelScores['social'].append(socialModel.scorePredictions())
        featureImportance['social'].append(socialModel.importances)
        placeModel = rf(train, test, placeFeatures, n_jobs=n_jobs)
        modelScores['place'].append(placeModel.scorePredictions())
        featureImportance['place'].append(placeModel.importances)
        timeSocialPlaceModel = rf(train, test, timeSocialPlaceFeatures,
                                  n_jobs=n_jobs)
        modelScores['timeSocialPlace'].append(
            timeSocialPlaceModel.scorePredictions())
        featureImportance['timeSocialPlace'].append(
            timeSocialPlaceModel.importances)
        fullModel = rf(train, test, fullFeatures, n_jobs=n_jobs)
        modelScores['full'].append(fullModel.scorePredictions())
        featureImportance['full'].append(fullModel.importances)
    averageFeatureImportance = DefaultOrderedDict(list)
    for key, values in featureImportance.items():
        ls = zip(*values)
        for v in ls:
            averageFeatureImportance[key].append(statistics.mean(v))

    ''' Ouput section '''
    if not os.path.isfile(outputPath + 'modelTimeseriesData.ssv'):
        with open(outputPath + 'modelTimeseriesData.ssv', 'w') as f:
            header = ' '.join(['timepoint', 'model',
                               'testSet', 'accuracy']) + '\n'
            f.write(header)

    with open(outputPath + 'modelTimeseriesData.ssv', 'a') as f:
        print('Timestep: ', timestep)
        for key, values in modelScores.items():
            for testSet, value in zip(testSetNames, values):
                row = ' '.join([str(timestep), str(key),
                                str(testSet), str(value)]) + '\n'
                f.write(row)
            print(key, ': ', statistics.mean(values))
        print()

    if not os.path.isfile(outputPath + 'featureImportance.ssv'):
        with open(outputPath + 'featureImportance.ssv', 'w') as f:
            header = ' '.join(['timepoint', 'model',
                               'testSet', 'feature',
                               'importance']) + '\n'
            f.write(header)

    dictOfFeatures = dict(base=baseFeatures, time=timeFeatures,
                          place=placeFeatures, network=networkFeatures,
                          social=socialFeatures,
                          timeSocialPlace=timeSocialPlaceFeatures,
                          full=fullFeatures, past=pastFeatures,
                          networkOnly=networkOnlyFeatures,
                          random=randomFeatures)

    with open(outputPath + 'featureImportance.ssv', 'a') as f:
        for key, values in featureImportance.items():
            for testSet, importances in zip(testSetNames, values):
                for feature, importance in zip(dictOfFeatures[key],
                                               importances):
                    row = ' '.join([str(timestep), str(key),
                                    str(testSet), str(feature),
                                    str(importance)]) + '\n'
                    f.write(row)
