import collections
import matplotlib as mp
mp.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz
import seaborn as sns
import statistics
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s" % (sys.argv[0]) +
              "<scores> "
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    outputPath = sys.argv[2]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    with open(inputData, 'r') as f:
        scores = pd.read_csv(f, sep=' ', index_col=False)
    friendshipPeriods = sorted(list(set(scores['lengthFriendshipPeriod'])))
    maxPeriod = max(scores['period'])

    ''' densities plots '''
    f, ax = plt.subplots(figsize=(14, 7))
    for period in friendshipPeriods:
        subset = scores[scores['lengthFriendshipPeriod'] == period]
        print(period, ' ', subset.mean()['density'])
        subsetMaxPeriod = max(subset['period'])
        timeFactor = maxPeriod/subsetMaxPeriod
        y = subset['density']
        x = subset['period'] * timeFactor + period
        ax = plt.plot(x, y, label=period)
    plt.legend()
    plt.ylabel('density')
    plt.xlabel('period')
    plt.savefig(outputPath + 'densitiesTimeSeries.png', bbox_inches='tight')
    plt.close()

    ''' lag plots '''
    for period in friendshipPeriods:
        f = plt.figure(figsize=(14, 7))
        subset = scores[scores['lengthFriendshipPeriod'] == period]
        pd.tools.plotting.lag_plot(subset)
        plt.savefig(outputPath + 'autocorrelation'+str(period)+'.png',
                    bbox_inches='tight')
        plt.close()

    ''' change plots '''
    f, ax = plt.subplots(figsize=(14, 7))
    for period in friendshipPeriods:
        subset = scores[scores['lengthFriendshipPeriod'] == period]
        print(period, ' ', subset.mean()['change'])
        subsetMaxPeriod = max(subset['period'])
        timeFactor = maxPeriod/subsetMaxPeriod
        y = subset['change']
        x = subset['period'] * timeFactor + period
        ax = plt.plot(x, y, label=period)
    plt.legend()
    plt.ylabel('change')
    plt.xlabel('period')
    plt.savefig(outputPath + 'changeTimeSeries.png', bbox_inches='tight')
    plt.close()
