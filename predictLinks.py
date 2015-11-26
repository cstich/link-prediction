import collections
import numpy as np
import sklearn.ensemble
import sklearn.metrics


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
        validClasses = 0
        for c in classes:
            currentTruths = truths[:, c]
            fpr[c], tpr[c], _ = sklearn.metrics.roc_curve(
                currentTruths, probabilities[:, c])
            try:
                roc_auc[c] = sklearn.metrics.auc(fpr[c], tpr[c])
                validClasses += 1
            except ValueError as e:
                if str(e) == "Input contains NaN, infinity or a value too large for dtype('float64').":
                    roc_auc[c] = np.nan
                else:
                    raise e

        roc_aucValues = np.nan_to_num(list(roc_auc.values()))
        roc_auc['macro'] = sum(roc_aucValues)/validClasses
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(
            truths.ravel(), probabilities.ravel())
        roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
        # Compute micro-average ROC curve and ROC area without 0 class
        fpr["micro-0"], tpr["micro-0"], _ = sklearn.metrics.roc_curve(
            truths[:, 1:].ravel(), probabilities[:, 1:].ravel())
        roc_auc["micro-0"] = sklearn.metrics.auc(fpr["micro-0"], tpr["micro-0"])

        return tpr, fpr, roc_auc


def pr(truths, probabilities, classes):
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    pr_auc = dict()
    validClasses = 0
    for c in classes:
        precision[c], recall[c], _ = sklearn.metrics.precision_recall_curve(
            truths[:, c], probabilities[:, c])
        try:
            pr_auc[c] = sklearn.metrics.average_precision_score(
                truths[:, c], probabilities[:, c])
            validClasses += 1
        except ValueError as e:
            if str(e) == "Input contains NaN, infinity or a value too large for dtype('float64').":
                pr_auc[c] = np.nan
            else:
                raise e

    pr_aucValues = np.nan_to_num(list(pr_auc.values()))
    pr_auc['macro'] = sum(pr_aucValues)/validClasses
    # Compute micro-average precision-recall curve
    # AUC under the PR curve
    precision["micro"], recall["micro"], _ = \
        sklearn.metrics.precision_recall_curve(
        truths.ravel(), probabilities.ravel())
    pr_auc["micro"] = sklearn.metrics.average_precision_score(
        truths, probabilities, average="micro")
    precision["micro-0"], recall["micro-0"], _ =\
        sklearn.metrics.precision_recall_curve(
        truths[:, 1:].ravel(), probabilities[:, 1:].ravel())
    pr_auc["micro-0"] = sklearn.metrics.average_precision_score(
        truths[:, 1:], probabilities[:, 1:], average="micro")

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

    def __init__(self, training_truths, training_examples, examples,
                 actuals, model, src_dest_nodes, **kwargs):
        # Use this file to train the classifier.
        # The first column in this file is the truth of a (src, dest) edge
        # (i.e., 1 if the edge is known to exist, 0 otherwise).
        # The rest of the columns are features on that edge.
        # TRAINING_SET_WITH_FEATURES_FILENAME = trainingFilename

        # This file contains candidate edge pairs to score, along with
        # features on these candidate edges.
        #
        # The first column is the src node, the second is the dest node,
        # the rest of the columns are features.
        # CANDIDATES_TO_SCORE_FILENAME = candidatesFilename

        # A list (must keep order) of variables to select which features to use
        # for training and for testing

        ########################################
        # STEP 1: Read in the training examples.
        ########################################
        '''truths = []  # A truth is an int (for a known true edge) or
        # 0 (for a false edge).
        training_examples = []  # Each training example is an array of features
        with open(TRAINING_SET_WITH_FEATURES_FILENAME, 'r') as csvF:
            csvReader = csv.DictReader(csvF, delimiter=',')
            for line in csvReader:
                fields = [line[feature] for feature in model]
                truth = int(line['edge'])
                training_example_features = fields

                truths.append(truth)
                training_examples.append(training_example_features)
        '''

        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        ''' Filter features according to model '''
        training_examples = [[e[1] for e in t if e[0] in model]
                             for t in training_examples]
        examples = [[e[1] for e in t if e[0] in model]
                    for t in examples]

        ''' Create sample weights '''
        classes = set(map(int, set(training_truths)))
        lengths = dict()
        for c in classes:
            lengths[c] = len([e for e in training_truths if e == c])

        sampleWeights = []
        for t in training_truths:
            for c in classes:
                if t == c:
                    sampleWeights.append(1/lengths[c])
                    break
        assert len(training_examples) == len(sampleWeights)

        #############################
        # STEP 2: Train a classifier.
        #############################
        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=500, oob_score=True, **kwargs)
        clf = clf.fit(training_examples, training_truths,
                      sample_weight=sampleWeights)

        ###############################
        # STEP 3: Score the candidates.
        ###############################
        '''src_dest_nodes = []
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
        '''
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
        self.acc = sklearn.metrics.accuracy_score(actuals, predictions)

    def calculatePR(self):
        return pr(self.truths, self.probabilities, self.classes)

    def calculateROC(self):
        return roc(self.truths, self.probabilities, self.classes)

    def precision(self, average):
        self.precision = sklearn.metrics.precision_score(
            self.actuals, self.predictions, average=average)

    def recall(self, average):
        self.precision = sklearn.metrics.recall_score(
            self.ctuals, self.predictions, average=average)
