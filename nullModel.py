from gs.dictAux import castLeaves, dd_list

import collections
import networksAux
import numpy as np
import predictLinks as pl
import sklearn.metrics


class NullModel(object):
    ''' NullModel unstable implementation '''

    def __init__(self, networkFilenames, testingNetworkFilename, classes, nodes):
        classes = sorted(list(classes))

        examples = collections.defaultdict(dd_list)
        for networkFilename in networkFilenames:
            tempNetwork = networksAux.constructNetworkFromFile(
                networkFilename, True)
            for node in nodes:
                for peer in nodes:
                    if node != peer:
                        try:
                            c = tempNetwork[node][peer]
                            fc = networksAux.mapSecondsToFriendshipClass(c)
                        except KeyError:
                            fc = 0
                        examples[node][peer].append(fc)

        actuals = collections.defaultdict(dict)
        stringActuals = collections.defaultdict(list)
        tempNetwork = networksAux.constructNetworkFromFile(
            testingNetworkFilename, True)

        for node in nodes:
            for peer in nodes:
                if node != peer:
                    try:
                        c = tempNetwork[node][peer]
                        fc = networksAux.mapSecondsToFriendshipClass(c)
                    except KeyError:
                        fc = 0
                    actuals[node][peer] = fc
                    stringActuals[node].append(fc)

        ''' Get probabilities based on frequency '''
        probabilities = collections.defaultdict(list)
        prediction = collections.defaultdict(list)
        for node in nodes:
            for peer in nodes:
                if node != peer:
                    observations = examples[node][peer]
                    count = collections.Counter(observations)
                    length = len(observations)
                    row = [count[cl]/length for cl in classes]
                    assert sum(row) > 0.9999 or sum(row) < 1.0001
                    probabilities[node].append(row)
                    prediction[node].append(count.most_common(1)[0][0])

        # probabilities = np.asarray(probabilities)
        # prediction = np.asarray(prediction)

        # self.examples = examples
        self.classes = classes
        self.nodes = list(nodes)  # Nodes have to be ordered for the alignment
        # of stringActuals and the prediction
        self.examples = examples
        # self.stringExamples = self.createString(examples)
        self.actuals = actuals
        self.stringActuals = stringActuals
        castLeaves(probabilities, np.array)
        self.probabilities = probabilities

        # TODO This should also be a numpy array somehow

        self.prediction = prediction
        truths = dict()
        for node in nodes:
            truths[node] = pl.createTruthArray(self.stringActuals[node],
                                               self.classes)
        self.truths = truths
        # self.truths = pl.createTruthArray(self.stringActuals, self.classes)
        # self.acc = self.acc()

    def acc(self):
        return sklearn.metrics.accuracy_score(
            self.stringActuals, self.prediction)

    def precision(self, average):
        return sklearn.metrics.precision_score(
            self.stringActuals, self.prediction, average=average)

    def calculatePR(self):
        return pl.pr(self.truths, self.probabilities, self.classes)

    def recall(self, average):
        return sklearn.metrics.recall_score(
            self.stringActuals, self.prediction, average=average)

    def calculateROC(self):
        return pl.roc(self.truths, self.probabilities, self.classes)
