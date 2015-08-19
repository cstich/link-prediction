import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

import seaborn as sns


def unixTimeToTimestep(unixTime, lengthOfPeriod):
    return (unixTime-1349042400)/24/3600/lengthOfPeriod


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s" % (sys.argv[0]) +
              "<model scores all> "
              "<model scores uni> "
              "<model scores social> "
              )
        sys.exit(-1)
    inputDataAll = sys.argv[1]
    inputDataUni = sys.argv[2]
    inputDataSocial = sys.argv[3]

    ''' Read model scores '''
    with open(inputDataAll, 'r') as f:
        modelScoresAll = pd.read_csv(f, sep=' ')

    with open(inputDataUni, 'r') as f:
        modelScoresUni = pd.read_csv(f, sep=' ')

    with open(inputDataSocial, 'r') as f:
        modelScoresSocial = pd.read_csv(f, sep=' ')

    ''' Compare model scores '''
    allScores = modelScoresAll
    allScores['type'] = 'all'
    uniScores = modelScoresUni
    uniScores['type'] = 'uni'
    socialScores = modelScoresSocial
    socialScores['type'] = 'social'
    frames = [allScores, uniScores, socialScores]
    fullDf = pd.concat(frames)
    fullDf.reset_index(drop=True)

    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.tsplot(data=fullDf[fullDf['model'] == 'full'], time="timepoint",
                    condition="type", unit="testSet", value="accuracy",
                    err_style="ci_band",
                    ci=95)
    plt.savefig('compareModels.png', bbox_inches='tight')
    plt.close()

    ''' Added information, i.e. score full - score network '''
    networkScores = fullDf[fullDf['model'] == 'network'].reset_index(drop=True)
    fullScores = fullDf[fullDf['model'] == 'full'].reset_index(drop=True)
    addedInformation = fullScores['accuracy'] - networkScores['accuracy']
    fullScores['addedInformation'] = addedInformation

    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.tsplot(data=fullScores, time="timepoint",
                    condition="type", unit="testSet", value="addedInformation",
                    err_style="ci_band",
                    ci=95)
    plt.savefig('compareAddedInformation.png', bbox_inches='tight')
    plt.close()

    ''' Compare context performance to full model '''
    contextScores = fullDf[fullDf['model'] == 'timeSocialPlace'].reset_index(
        drop=True)
    contextPerformance = contextScores['accuracy']/fullScores['accuracy']
    fullScores['contextPerformance'] = contextPerformance

    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.tsplot(data=fullScores, time="timepoint",
                    condition="type", unit="testSet",
                    value="contextPerformance",
                    err_style="ci_band",
                    ci=95)
    plt.savefig('contextPerformance.png', bbox_inches='tight')
    plt.close()

    ''' Added information, i.e. score full - score network '''
    baseScores = fullDf[fullDf['model'] == 'past'].reset_index(drop=True)
    contextScores = fullDf[fullDf['model'] ==
                           'timeSocialPlace'].reset_index(drop=True)
    contextAdvantage = contextScores['accuracy'] - baseScores['accuracy']
    fullScores['contextAdvantage'] = contextAdvantage

    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.tsplot(data=fullScores, time="timepoint",
                    condition="type", unit="testSet", value="contextAdvantage",
                    err_style="ci_band",
                    ci=95)
    plt.savefig('compareContextAdvantage.png', bbox_inches='tight')
    plt.close()
