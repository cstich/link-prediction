from gs import parser
from gs.DictAux import DefaultOrderedDict
from predictLinks import RandomForestLinkPrediction as rf
from nullModel import NullModel

import copy
import os
import pytz
import re
import statistics
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


def scoreModels(model, dictionaries, key):
    acc = model.acc
    prec = model.precision
    rec = model.recall
    roc_macro = model.calculateROC()[2]['macro']
    roc_micro = model.calculateROC()[2]['micro']
    pr_macro = model.calculatePR()[2]['macro']
    pr_micro = model.calculatePR()[2]['micro']
    scores = acc, prec, rec, roc_macro, roc_micro, pr_macro, pr_micro
    for d, score in zip(dictionaries, scores):
        d[key].append(score)


def scoreNullModel(dictionaries, NMacc, NMprec, NMrec, NMroc_auc, NMpr_auc):
    acc = statistics.mean(NMacc)
    prec = statistics.mean(NMprec)
    rec = statistics.mean(NMrec)
    roc_macro = NMroc_auc['macro']
    roc_micro = NMroc_auc['micro']
    pr_macro = NMpr_auc['macro']
    pr_micro = NMpr_auc['micro']
    scores = acc, prec, rec, roc_macro, roc_micro, pr_macro, pr_micro
    for d, score in zip(dictionaries, scores):
        d['null'].append(score)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: %s " % (sys.argv[0]) +
              "<data directory> "
              "<networks directory> "
              "<number of jobs> "
              "<output path> "
              "<weighted>")
        sys.exit(-1)

    inputData = sys.argv[1]
    inputNetworks = sys.argv[2]
    n_jobs = int(sys.argv[3])
    outputPath = sys.argv[4]
    weighted = bool(int(sys.argv[5]))

    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputData = parser.parsePath(inputData, scriptDir)
    trainingPattern = re.compile('train_sample_[0-9]+\.csv')
    trainingFiles = parser.readFiles(inputData, trainingPattern)
    testPattern = re.compile('test_sample_[0-9]+\.csv')
    testingFiles = parser.readFiles(inputData, testPattern)
    lengthOfPeriod = float(inputData.split('/')[-3].replace('_', '.'))
    unixTime = int(inputData.split('/')[-1])
    timestep = float((unixTime-1349042400)/lengthOfPeriod)

    ''' Null model '''
    networkPatternT0 = re.compile('[0-9]+\-' + str(unixTime) + '\.csv')
    networkPatternT1 = re.compile(str(unixTime) + '\-[0-9]+\.csv')
    networkFileT0 = parser.readFiles(inputNetworks, networkPatternT0)[0]
    networkFileT1 = parser.readFiles(inputNetworks, networkPatternT1)[0]

    ''' Match training to test files '''
    matchFilesPattern = re.compile('_([0-9]+)\.csv')
    trainingFiles = sorted(trainingFiles,
                           key=lambda x:
                           int(matchFilesPattern.search(x).group(1)))
    testingFiles = sorted(testingFiles,
                          key=lambda x:
                          int(matchFilesPattern.search(x).group(1)))
    files = zip(trainingFiles, testingFiles)

    testSetNames = [str(matchFilesPattern.search(e).group(1))
                    for e in trainingFiles]

    ''' Define the different models '''
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
        'numberOfPeople'])

    # TODO Update features
    placeFeatures = []
    placeFeatures.extend(copy.copy(baseFeatures))
    placeFeatures.extend([
        'timeSpentAtHomeWith',
        'timeSpentAtUniversityWith',
        'timeSpentAtThirdPlaceWith',
        'timeSpentAtOtherPlaceWith'])

    ''' Node only features '''
    nodeFeatures = []
    nodeFeatures.extend(copy.copy(baseFeatures))
    nodeFeatures.extend(copy.copy(placeFeatures))
    nodeFeatures.extend(copy.copy(socialFeatures))
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

    ''' Run the models and score them '''
    accs = DefaultOrderedDict(list)
    precs = DefaultOrderedDict(list)
    recs = DefaultOrderedDict(list)
    featureImportance = DefaultOrderedDict(list)
    roc_macros = DefaultOrderedDict(list)
    roc_micros = DefaultOrderedDict(list)
    pr_macros = DefaultOrderedDict(list)
    pr_micros = DefaultOrderedDict(list)
    results = list()
    results.extend([accs, precs, recs, roc_macros, roc_micros,
                    pr_macros, pr_micros])

    # Calculate a separete set of NullModel for each test set
    NMaccs = []
    NMprecs = []
    NMrecs = []
    actuals = []
    NM = NullModel(networkFileT0, networkFileT1, weighted)
    predictions = NM.createPredictions(1000)
    probabilities = NM.predictionsToProbability(predictions)
    _, _, NMroc_auc = NM.roc(probabilities)
    _, _, NMpr_auc = NM.pr(probabilities)

    for prediction in predictions:
        NMacc = NM.acc(prediction)
        NMaccs.append(NMacc)
        NMprec = NM.prec(prediction)
        NMprecs.append(NMprec)
        NMrec = NM.rec(prediction)
        NMrecs.append(NMrec)

    for train, test in files:
        pastModel = rf(train, test, pastFeatures, n_jobs=n_jobs)
        scoreModels(pastModel, results, 'past')
        featureImportance['past'].append(pastModel.importances)

        baseModel = rf(train, test, baseFeatures, n_jobs=n_jobs)
        scoreModels(baseModel, results, 'base')
        featureImportance['base'].append(baseModel.importances)

        networkOnlyModel = rf(train, test, networkOnlyFeatures, n_jobs=n_jobs)
        scoreModels(networkOnlyModel, results, 'networkOnly')
        featureImportance['networkOnly'].append(networkOnlyModel.importances)

        networkModel = rf(train, test, networkFeatures, n_jobs=n_jobs)
        scoreModels(networkModel, results, 'network')
        featureImportance['network'].append(networkModel.importances)

        nodeModel = rf(train, test, nodeFeatures, n_jobs=n_jobs)
        scoreModels(nodeModel, results, 'node')
        featureImportance['node'].append(nodeModel.importances)

        timeModel = rf(train, test, timeFeatures, n_jobs=n_jobs)
        scoreModels(timeModel, results, 'time')
        featureImportance['time'].append(timeModel.importances)

        socialModel = rf(train, test, socialFeatures, n_jobs=n_jobs)
        scoreModels(socialModel, results, 'social')
        featureImportance['social'].append(socialModel.importances)

        placeModel = rf(train, test, placeFeatures, n_jobs=n_jobs)
        scoreModels(placeModel, results, 'place')
        featureImportance['place'].append(placeModel.importances)

        timeSocialPlaceModel = rf(train, test, timeSocialPlaceFeatures,
                                  n_jobs=n_jobs)
        scoreModels(timeSocialPlaceModel, results, 'timeSocialPlace')
        featureImportance['timeSocialPlace'].append(
            timeSocialPlaceModel.importances)

        fullModel = rf(train, test, fullFeatures, n_jobs=n_jobs)
        scoreModels(fullModel, results, 'full')
        featureImportance['full'].append(fullModel.importances)

        # Null model scores
        scoreNullModel(results, NMaccs, NMprecs, NMrecs, NMroc_auc, NMpr_auc)

    averageFeatureImportance = DefaultOrderedDict(list)
    for key, values in featureImportance.items():
        ls = zip(*values)
        for v in ls:
            averageFeatureImportance[key].append(statistics.mean(v))

    ''' Ouput section '''
    if not os.path.isfile(outputPath + 'modelTimeseriesData.ssv'):
        with open(outputPath + 'modelTimeseriesData.ssv', 'w') as f:
            header = ' '.join(['timepoint', 'model',
                               'testSet', 'accuracy',
                               'precision', 'recall',
                               'roc_macro', 'roc_micro',
                               'pr_macro', 'pr_micro']) + '\n'
            f.write(header)

    with open(outputPath + 'modelTimeseriesData.ssv', 'a') as f:
        print('Timestep: ', timestep)
        for key, accs in results[0].items():
            precs = results[1][key]
            recs = results[2][key]
            roc_macros = results[3][key]
            roc_micros = results[4][key]
            pr_macros = results[5][key]
            pr_micros = results[6][key]

            for testSet, acc, prec, rec, roc_mac,\
                roc_mic, pr_mac, pr_mic in zip(
                    testSetNames, accs, precs, recs, roc_macros, roc_micros,
                    pr_macros, pr_micros):
                row = ' '.join([str(timestep), str(key),
                                str(testSet), str(acc),
                                str(prec), str(rec),
                                str(roc_mac), str(roc_mic),
                                str(pr_mac), str(pr_mic)]) + '\n'
                f.write(row)
            print(key, ': ', statistics.mean(roc_macros))
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
                          node=nodeFeatures)

    with open(outputPath + 'featureImportance.ssv', 'a') as f:
        for key, values in featureImportance.items():
            for testSet, importances in zip(testSetNames, values):
                for feature, importance in zip(dictOfFeatures[key],
                                               importances):
                    row = ' '.join([str(timestep), str(key),
                                    str(testSet), str(feature),
                                    str(importance)]) + '\n'
                    f.write(row)
