from geogps import Parser

import collections
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy
import statistics
import sys

import seaborn as sns

def unixTimeToTimestep(unixTime, lengthOfPeriod):
    return int(round((unixTime-1349042400)/24/3600/lengthOfPeriod))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s" % (sys.argv[0]) +
              "<model scores> "
              "<feature importance "
              "<parsed data dir>")
        sys.exit(-1)
    inputModelData = sys.argv[1]
    inputImportanceData = sys.argv[2]
    inputParsedData = sys.argv[3]

    ''' Plot model scores '''
    with open(inputModelData, 'r') as f:
        modelScores = pd.read_csv(f, sep=' ')

    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.tsplot(data=modelScores, time="timepoint", condition="model",
                    unit="testSet", value="accuracy", err_style="ci_band",
                    ci=95)
    plt.savefig('modelScores.png', bbox_inches='tight')
    plt.close()

    ''' Plot feature importance '''
    with open(inputImportanceData, 'r') as f:
        featureImportance = pd.read_csv(f, sep=' ')

    featureDfs = dict()
    keys = set(featureImportance['model'])

    for key in keys:
        featureDfs[key] = featureImportance[featureImportance['model'] == key]

    for key, importances in featureDfs.items():
        try:
            f, ax = plt.subplots(figsize=(14, 7))
            ax = sns.tsplot(data=importances, time="timepoint", condition='feature',
                            unit="testSet", value='importance', err_style="ci_band",
                            ci=95)
            plt.savefig('featureImportance_' + str(key) + '.png',
                        bbox_inches='tight')
            plt.close()
        except ValueError:
            print(key)

    ''' Plot correlation accuracy / amount of observations '''
    inputParsedDataFilePattern  = re.compile('parsedData[\-\_a-zA-Z0-9]*\.pck')
    rs = Parser.loadPickles(inputParsedData, inputParsedDataFilePattern,
                            matchUser=False)[0]
    densitiesLocation = rs['count']
    densitiesLocation = [unixTimeToTimestep(e, 33.6) for e in densitiesLocation]
    densitiesLocation = collections.Counter(densitiesLocation)

    timepoints = list(set(modelScores['timepoint']))
    keys = list(set(modelScores['model']))
    averageValues = collections.defaultdict(dict)
    for key in keys:
        for time in timepoints:
            values = modelScores[(modelScores['model'] == key) &
                                 (modelScores['timepoint'] == time)]
            averageValues[key][time] = statistics.mean(values.accuracy)

    f, ax = plt.subplots(figsize=(14, 7))
    for key in keys:
        accuracies = list()
        numberOfObservations = list()
        for time in timepoints:
            accuracies.append(averageValues[key][time])
            numberOfObservations.append(densitiesLocation[time])
        # accuracies = list(map(lambda x: (1-x), accuracies))
        accuracies = scipy.stats.zmap(accuracies, accuracies)
        numberOfObservations = scipy.stats.zmap(numberOfObservations,
                                                numberOfObservations)
        ax = sns.regplot(y=np.array(accuracies), x=np.array(numberOfObservations))

    plt.xlabel('number of GPS observations (z-transformed)')
    plt.ylabel('accuracy (z-transformed)')
    plt.savefig('corrAccuracyGPS.png',
                    bbox_inches='tight')
    plt.close()

    ''' Plot correlation accuracy / amount of bluetooth '''
    blues = rs['blues']
    bluesCounter = []
    for user, blue in blues.items():
        for timePeriod, observations in blue.items():
            if observations:
                ls = [unixTimeToTimestep(b.time, 33.6) for b in observations]
                bluesCounter.extend(ls)
    bluesCounter = collections.Counter(bluesCounter)

    f, ax = plt.subplots(figsize=(14, 7))
    for key in keys:
        accuracies = list()
        numberOfObservations = list()
        for time in timepoints:
            accuracies.append(averageValues[key][time])
            numberOfObservations.append(bluesCounter[time])
        # accuracies = list(map(lambda x: (1-x), accuracies))
        accuracies = scipy.stats.zmap(accuracies, accuracies)
        numberOfObservations = scipy.stats.zmap(numberOfObservations,
                                                numberOfObservations)
        ax = sns.regplot(y=np.array(accuracies), x=np.array(numberOfObservations))

    plt.xlabel('number of bluetooth observations (z-transformed)')
    plt.ylabel('accuracy (z-transformed)')
    plt.savefig('corrAccuracyBlues.png',
                bbox_inches='tight')
    plt.close()
