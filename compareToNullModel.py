import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

import seaborn as sns


def unixTimeToTimestep(unixTime, lengthOfPeriod):
    return (unixTime-1349042400)/24/3600/lengthOfPeriod


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s" % (sys.argv[0]) +
              " <model scores> "
              )
        sys.exit(-1)
    inputData = sys.argv[1]

    ''' Read model scores '''
    with open(inputData, 'r') as f:
        scores = pd.read_csv(f, sep=' ')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

    ''' Added information, i.e. score full - score null '''
    nullScores = scores[scores['model'] == 'null'].reset_index(drop=True)
    fullScores = scores[scores['model'] == 'full'].reset_index(drop=True)
    contextScores = scores[scores['model'] ==
                           'timeSocialPlace'].reset_index(drop=True)
    fullScores['addedInformation'] = fullScores['accuracy'] - nullScores['accuracy']
    contextScores['addedInformation'] = contextScores['accuracy'] - nullScores['accuracy']

    frames = [fullScores, contextScores]
    fullDf = pd.concat(frames)
    fullDf.reset_index(drop=True)

    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.tsplot(data=fullDf, time="timepoint",
                    condition="model", unit="testSet", value="addedInformation",
                    err_style="ci_band",
                    ci=95)

    plt.savefig('compareAddedInformation.png', bbox_inches='tight')
    plt.close()
