from sklearn.metrics import accuracy_score, precision_score, recall_score
from geogps.DictAux import dd_int

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

    def __init__(self, networkT0Filename, networkT1Filename,
                 weighted):
        networkT0 = collections.defaultdict(list)
        tempNetwork = networksAux.constructNetworkFromFile(networkT0Filename,
                                                           weighted)
        classes = set()
        nodes = set()

        for node, peers in tempNetwork.items():
            nodes.add(node)
            if weighted:
                for peer, c in peers.items():
                    friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                    classes.add(friendshipClass)
                    nodes.add(peer)
                    if friendshipClass != 0:
                        networkT0[friendshipClass].append((node, peer))
            else:
                classes.add(1)
                for peer in peers:
                    nodes.add(peer)
                    networkT0[1].append((node, peer))

        networkT1 = collections.defaultdict(list)
        tempNetwork = networksAux.constructNetworkFromFile(networkT1Filename,
                                                           weighted)
        actuals = collections.defaultdict(dd_int)

        for node, peers in tempNetwork.items():
            nodes.add(node)
            if weighted:
                for peer, c in peers.items():
                    friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                    classes.add(friendshipClass)
                    nodes.add(peer)
                    if friendshipClass != 0:
                        networkT1[friendshipClass].append((node, peer))
                        actuals[node][peer] = friendshipClass
            else:
                classes.add(1)
                for peer in peers:
                    nodes.add(peer)
                    networkT1[1].append((node, peer))
                    actuals[node][peer] = 1

        self.networkT0 = networkT0
        self.networkT1 = networkT1
        self.classes = sorted(list(classes))
        self.nodes = list(nodes)
        self.actuals = actuals
        self.stringActuals = self.createString(actuals)
        self.truths = pl.createTruthArray(self.stringActuals, self.classes)

    def predictions(self):
        predictions = collections.defaultdict(dd_int)
        exclusion = []
        classes = copy.deepcopy(self.classes)

        if len(classes) > 1:
            random.shuffle(classes)

        for c in classes:
            oldTies = set(self.networkT0[c])
            newTies = set(self.networkT1[c])
            createdTies = newTies.difference(oldTies)
            dissolvedTies = oldTies.difference(newTies)

            # Add the newly formed ties
            candidates = self.selectCandidates(
                self.nodes, len(createdTies), exclusion)
            for node, peer in candidates:
                predictions[node][peer] = c
                predictions[peer][node] = c
                exclusion.append((node, peer))
                exclusion.append((peer, node))
                # A tie cannot have two types of classes

            # Remove the dissovled ties
            toRemove = self.selectCandidates(
                oldTies, len(dissolvedTies), [])
            remainingTies = oldTies.difference(toRemove)
            for node, peer in remainingTies:
                predictions[node][peer] = c
                predictions[peer][node] = c
        return predictions

    def createPredictions(self, n):
        predictions = list()
        for i in range(n):
            predictions.append(self.createString(self.predictions()))
        return predictions

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
        @Return: A tie in the format (node, peer)
        '''
        if len(population) == 0:
            return []
        else:
            candidates = []
            while len(candidates) < n:
                try:
                    node, peer = random.sample(population, 1)[0]
                except ValueError:
                    node = random.sample(population, 1)[0]
                    peer = random.sample(population, 1)[0]
                if node != peer:
                    candidate = (node, peer)
                    if candidate not in exclusion:
                        candidates.append(candidate)
                    candidate = (peer, node)
                    if candidate not in exclusion:
                        candidates.append(candidate)
            return candidates
