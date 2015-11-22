from sklearn.metrics import accuracy_score, precision_score, recall_score
from gs.dictAux import dd_int

import collections
import copy
import networksAux
import numpy as np
import random
import predictLinks as pl


class NullModel(object):
    ''' NullModel for an evolving graph that preservers the amount of
    change between timesteps for each "class" of tie. Is very stupid and
    does not preserve the distribution of degrees. Randomly creates and
    dissolves ties between nodes to match the amount of change for each
    timestep '''

    def __init__(self, networkT0Filename, networkT1Filename, classes, nodes):
        # we need to read in the full network and not its sparse representation
        networkT0 = collections.defaultdict(set)
        tempNetwork = networksAux.constructNetworkFromFile(networkT0Filename,
                                                           True)
        examples = collections.defaultdict(dd_int)
        for node, peers in tempNetwork.items():
            for peer, c in peers.items():
                friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                networkT0[friendshipClass].add(frozenset((node, peer)))
                examples[node][peer] = friendshipClass
                examples[peer][node] = friendshipClass

        networkT0 = self.__addSparseTies__(networkT0, nodes)

        networkT1 = collections.defaultdict(set)
        tempNetwork = networksAux.constructNetworkFromFile(networkT1Filename,
                                                           True)

        actuals = collections.defaultdict(dd_int)
        for node, peers in tempNetwork.items():
            nodes.add(node)
            for peer, c in peers.items():
                friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                networkT1[friendshipClass].add(frozenset((node, peer)))
                actuals[node][peer] = friendshipClass
                actuals[peer][node] = friendshipClass

        networkT1 = self.__addSparseTies__(networkT1, nodes)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

        self.networkT0 = networkT0
        self.networkT1 = networkT1
        self.classes = sorted(list(classes))
        self.nodes = list(nodes) # Nodes have to be ordered for the alignment
        # of stringActuals and the prediction
        self.examples = examples
        self.stringExamples = self.createString(examples)
        self.actuals = actuals
        self.stringActuals = self.createString(actuals)
        self.truths = pl.createTruthArray(self.stringActuals, self.classes)

    def __addSparseTies__(self, network, nodes):
        ties = set()
        [ties.add(e) for s in network.values() for e in s]
        for node in nodes:
            for peer in nodes:
                if frozenset((node, peer)) not in ties and peer != node:
                    network[0].add(frozenset((node, peer)))
        return network

    def prediction(self, nonTie=0):
        prediction = collections.defaultdict(dd_int)
        exclusion = set()
        classes = copy.deepcopy(self.classes)

        if len(classes) > 1:
            random.shuffle(classes)

        for c in classes:
            if c == nonTie:
                continue
            oldTies = set(self.networkT0[c])
            newTies = set(self.networkT1[c])
            createdTies = newTies.difference(oldTies)
            dissolvedTies = oldTies.difference(newTies)

            # Add the newly formed ties
            candidates = self.selectCandidates(
                self.nodes, len(createdTies), exclusion)

            for s in candidates:
                s = set(s)
                node = s.pop()
                peer = s.pop()
                assert len(s) == 0
                prediction[node][peer] = c
                prediction[peer][node] = c
                exclusion.add(frozenset((node, peer)))
                # A tie cannot have two types of classes

            # Remove the dissolved ties
            toRemove = self.selectTies(
                oldTies, len(dissolvedTies), [])
            remainingTies = oldTies.difference(toRemove)
            for s in remainingTies:
                s = set(s)
                node = s.pop()
                peer = s.pop()
                assert len(s) == 0
                prediction[node][peer] = c
                prediction[peer][node] = c
                exclusion.add(frozenset((node, peer)))
                # A tie cannot have two types of classes
        return prediction

    def predictions(self, n):
        predictions = list()
        for i in range(n):
            predictions.append(self.createString(self.prediction()))
        return predictions

    def probabilities(self):
        p = collections.defaultdict(dict)
        # dictionary in the format d[condition][probability]
        # for example d[1][0] given class 1 the probability for 0 is x
        for c in self.classes:
            oldTies = self.networkT0[c]
            # newTies = self.networkT1[c]
            # createdTies = newTies.difference(oldTies)
            # dissolvedTies = oldTies.difference(newTies)
            # p[c][c] = 1 - len(createdTies)/len(oldTies)
            for d in self.classes:
                newlyFormedInClass = self.networkT1[d]
                intersection = oldTies.intersection(newlyFormedInClass)
                p[c][d] = len(intersection)
        return p

    def probabilityArray(self, p):
        probs = dict()
        marginal = dict()
        for condition, probabilities in p.items():
            marginal = sum(probabilities.values())
            row = [probabilities[c]/marginal if marginal != 0
                   else 1/len(probabilities.values())
                   for c in self.classes]
            assert sum(row) == 1
            probs[condition] = np.asarray(row)

        result = list()
        for e in self.stringExamples:
            row = probs[e]
            result.append(row)

        result = np.asarray(result)
        return result

    def predictionsToProbability(self, predictions):
        N = len(predictions)
        observations = list(map(list, zip(*predictions)))
        result = list()
        for o in observations:
            count = collections.Counter(o)
            prob = list()
            for c in self.classes:
                prob.append(count[c]/N)
            result.append(prob)
        return np.asarray(result)

    def createString(self, dictionary):
        ls = list()
        assert isinstance(self.nodes, list)
        for node in self.nodes:
            for peer in self.nodes:
                if peer != node:
                    ls.append(dictionary[node][peer])
        ls = list(map(int, ls))
        return ls

    def acc(self, prediction):
        return accuracy_score(self.stringActuals, prediction)

    def prec(self, prediction):
        return precision_score(self.stringActuals,
                               prediction, average='weighted')

    def pr(self, probabilities):
        return pl.pr(self.truths, probabilities, self.classes)

    def rec(self, prediction):
        return recall_score(self.stringActuals,
                            prediction, average='weighted')

    def roc(self, probabilities):
        return pl.roc(self.truths, probabilities, self.classes)

    def selectCandidates(self, population, n, exclusion):
        ''' Randomly sample from the population of nodes a tie.
        Excludes self-loops.
        '''
        if len(population) == 0:
            return list()
        else:
            candidates = list()
            while len(candidates) < n:
                node, peer = random.sample(population, 2)
                if node != peer:
                    candidate = frozenset((node, peer))
                    if candidate not in candidates and candidate not in exclusion:
                        candidates.append(candidate)
            return candidates

    def selectTies(self, ties, n, exclusion):
        ''' Randomly sample from the population of ties a tie.
        Excludes self-loops.
        '''
        if len(ties) == 0:
            return set()
        else:
            result = set()
            while len(result) < n:
                tie = random.sample(ties, 1)[0]
                assert len(tie) == 2
                if tie not in result and tie not in exclusion:
                    result.add(frozenset(tie))
            return result
