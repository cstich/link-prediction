from Scorer import calculateAveragePrecision
from sklearn.ensemble import RandomForestClassifier

import collections
import csv
import statistics


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
            lengths[c] = len([e for e in truth if e == c])

        sampleWeights = []
        for t in truths:
            for c in classes:
                if t == c:
                    sampleWeights.append(lengths[c]/len(truths))
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
        actuals = collections.defaultdict(set)
        with open(CANDIDATES_TO_SCORE_FILENAME, 'r') as csvF:
            csvReader = csv.DictReader(csvF, delimiter=',')
            for line in csvReader:
                fields = [float(line[feature]) for feature in model]
                src = line['source']
                dest = line['destination']
                typ = line['edge']
                src_dest_nodes.append((src, dest))

                if int(line['edge']):
                    actuals[src].add((dest, typ))

                example_features = fields
                examples.append(example_features)

        ''' Create a dictionary of predicted links '''
        predictedLinks = clf.predict(examples)
        predictions = collections.defaultdict(set)
        for i, prediction in enumerate(predictedLinks):
            src = src_dest_nodes[i][0]
            dest = (src_dest_nodes[i][1], str(prediction))
            if int(prediction):
                predictions[src].add(dest)
            else:
                predictions[src]

        ''' Store the stuff '''
        self.actuals = actuals
        self.classes = classes
        self.dest = dest
        self.features = model
        self.importances = clf.feature_importances_
        self.predictions = predictions
        self.probabilities = clf.predict_proba(examples)
        self.src = src
        self.TRAINING = TRAINING_SET_WITH_FEATURES_FILENAME
        self.CANDIDATES = CANDIDATES_TO_SCORE_FILENAME

    def printPredictions(self):
        ''' Print predictions '''
        for i in range(len(self.predictions)):
            print(",".join([str(int(x)) for x in [self.src_dest_nodes[i][0],
                                                  self.src_dest_nodes[i][1],
                                                  self.predictions[i]]]))

    def printProbailities(self):
        ''' Print predictions '''
        for i in range(len(self.predictions)):
            print(",".join([str(int(x)) for x in [self.src_dest_nodes[i][0],
                                                  self.src_dest_nodes[i][1],
                                                  self.predictions[i]]]))

    def scorePredictions(self):
        allPredictions = []
        for src, predictedLinks in self.predictions.items():
            actualLinks = self.actuals[src]
            allPredictions.append(calculateAveragePrecision(predictedLinks,
                                                            actualLinks))
        return statistics.mean(allPredictions)

    def nullModel(self):
        # STEP 4: Create a null model
        oldTies = collections.defaultdict(list)
        newTies = collections.defaultdict(list)

        with open(self.CANDIDATES, 'r') as csvF:
            csvReader = csv.DictReader(csvF, delimiter=',')
            for line in csvReader:
                src = line['source']
                dest = line['destination']
                newTyp = line['edge']
                oldTyp = line['friends']
                oldTies[src][oldTyp].append(dest)
                newTies[src][newTyp].append(dest)

        for node in oldTies:
            for c in self.classes:
                pass
