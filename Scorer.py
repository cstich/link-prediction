def calculateAveragePrecision(predictions, actuals):
    '''
    Calculate the average precision of a sequence of predicted followings,
    given the true set.
    See http://www.kaggle.com/c/FacebookRecruiting/details/Evaluation for more
    details on the average precision metric.

    Examples:

    calculateAveragePrecision( [A, B, C], [A, C, X] )
    => (1/1 + 2/3) / 3 ~ 0.56

    calculateAveragePrecision( [A, B], [A, B, C] )
    => (1/1 + 2/2) /3 ~ 0.67
    '''
    numCorrect = 0
    precisions = []
    for i, prediction in enumerate(predictions):
        if prediction in actuals:
            numCorrect += 1
            precisions.append(numCorrect / (i + 1))
        else:
            precisions.append(0)

    if len(actuals) == 0 and len(predictions) == 0:
        return 1
    elif len(actuals) == 0:
        return 0
    else:
        return sum(precisions) / len(actuals)


def calculatePrecision(predictions, actuals):
    ''' Calculate how many of your predictions are wrong '''
    predictions = set(predictions)
    actuals = set(actuals)
    correct = predictions.intersection(actuals)

    if len(actuals) == 0 and len(predictions) == 0:
        return 1
    elif len(predictions) == 0:
        return 0
    else:
        return (len(correct)/len(predictions))


def calculateRecall(predictions, actuals):
    ''' Calculate how many of your predictions are wrong '''
    predictions = set(predictions)
    actuals = set(actuals)
    correct = predictions.intersection(actuals)

    if len(actuals) == 0 and len(predictions) == 0:
        return 1
    elif len(actuals) == 0:
        return 0
    else:
        return (len(correct)/len(actuals))
