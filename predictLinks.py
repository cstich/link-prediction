from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, auc, accuracy_score,\
    precision_score, recall_score, precision_recall_curve

import collections
import csv
import numpy as np

def createTruthArray(actuals, classes):
    ar = np.zeros([len(actuals), len(classes)])
    for i, a in enumerate(actuals):
        for j, c in enumerate(classes):
            if c == a:
                ar[i, j] = 1
    return ar


def roc(truths, probabilities, classes):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for c in classes:
            fpr[c], tpr[c], _ = roc_curve(truths[:, c],
                                          probabilities[:, c])
            roc_auc[c] = auc(fpr[c], tpr[c])

        roc_auc['macro'] = sum(roc_auc.values())/len(classes)
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(truths.ravel(),
                                                  probabilities.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        return fpr, tpr, roc_auc


def pr(truths, probabilities, classes):
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for c in classes:
        precision[c], recall[c], _ = precision_recall_curve(
            truths[:, c], probabilities[:, c])
        pr_auc[c] = average_precision_score(
            truths[:, c], probabilities[:, c])

    pr_auc['macro'] = sum(pr_auc.values())/len(classes)
    # Compute micro-average precision-recall curve
    # AUC under the PR curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        truths.ravel(), probabilities.ravel())
    pr_auc["micro"] = average_precision_score(
        truths, probabilities, average="micro")

    return precision, recall, pr_auc


class RandomForestLinkPrediction(object):
    """
    Given a training set of examples and associated features...
    - Each example is of the form (src, dest, 1 if src follows dest else 0)
    - Features are things like # of followers of src, Jaccard similarity
    between src and dest nodes, etc.

    ...train a machine learning classifier on this set.

    Then apply this same classifier on a set of test src nodes, to form
    a ranked prediction of which dest nodes each src is likely to follow.
    """

    def __init__(self, trainingFilename, candidatesFilename, model, **kwargs):
        # Use this file to train the classifier.
        # The first column in this file is the truth of a (src, dest) edge
        # (i.e., 1 if the edge is known to exist, 0 otherwise).
        # The rest of the columns are features on that edge.
        TRAINING_SET_WITH_FEATURES_FILENAME = trainingFilename

        # This file contains candidate edge pairs to score, along with
        # features on these candidate edges.
        #
        # The first column is the src node, the second is the dest node,
        # the rest of the columns are features.
        CANDIDATES_TO_SCORE_FILENAME = candidatesFilename

        # A list (must keep order) of variables to select which features to use
        # for training and for testing

        ########################################
        # STEP 1: Read in the training examples.
        ########################################
        truths = []  # A truth is 1 (for a known true edge) or 0 (for a false
        # edge).
        training_examples = []  # Each training example is an array of features
        with open(TRAINING_SET_WITH_FEATURES_FILENAME, 'r') as csvF:
            csvReader = csv.DictReader(csvF, delimiter=',')
            for line in csvReader:
                fields = [line[feature] for feature in model]
                truth = int(line['edge'])
                training_example_features = fields

                truths.append(truth)
                training_examples.append(training_example_features)

        ''' Create sample weights '''
        classes = set(map(int, set(truths)))
        lengths = dict()
        for c in classes:
            lengths[c] = len([e for e in truths if e == c])

        sampleWeights = []
        for t in truths:
            for c in classes:
                if t == c:
                    sampleWeights.append(1/lengths[c])
                    break
        assert len(training_examples) == len(sampleWeights)

        #############################
        # STEP 2: Train a classifier.
        #############################
        clf = RandomForestClassifier(n_estimators=500,
                                     oob_score=True, **kwargs)
        clf = clf.fit(training_examples, truths,
                      sample_weight=sampleWeights)

        ###############################
        # STEP 3: Score the candidates.
        ###############################
        src_dest_nodes = []
        examples = []
        actualLinks = collections.defaultdict(set)
        actuals = list()
        with open(CANDIDATES_TO_SCORE_FILENAME, 'r') as csvF:
            csvReader = csv.DictReader(csvF, delimiter=',')
            for line in csvReader:
                fields = [float(line[feature]) for feature in model]
                src = line['source']
                dest = line['destination']
                typ = line['edge']
                src_dest_nodes.append((src, dest))

                if int(line['edge']):
                    actualLinks[src].add((dest, typ))
                actuals.append(int(line['edge']))

                example_features = fields
                examples.append(example_features)

        ''' Create a dictionary of predicted links '''
        predictions = clf.predict(examples)
        predictedLinks = collections.defaultdict(set)
        for i, prediction in enumerate(predictions):
            src = src_dest_nodes[i][0]
            dest = (src_dest_nodes[i][1], str(prediction))
            if int(prediction):
                predictedLinks[src].add(dest)
            else:
                predictedLinks[src]

        ''' Store the stuff '''
        self.actualLinks = actualLinks
        self.actuals = actuals
        self.truths = createTruthArray(actuals, classes)
        self.classes = classes
        self.dest = dest
        self.features = model
        self.importances = clf.feature_importances_
        self.predictions = predictions
        self.predictedLinks = predictedLinks
        self.probabilities = np.asarray(clf.predict_proba(examples))
        self.src = src
        self.acc = accuracy_score(actuals, predictions)
        self.precision = precision_score(actuals, predictions,
                                         average='weighted')
        self.recall = recall_score(actuals, predictions,
                                   average='weighted')
        self.TRAINING = TRAINING_SET_WITH_FEATURES_FILENAME
        self.CANDIDATES = CANDIDATES_TO_SCORE_FILENAME


    def calculatePR(self):
        return pr(self.truths, self.probabilities, self.classes)

    def calculateROC(self):
        return roc(self.truths, self.probabilities, self.classes)
