from geogps.DictAux import dd_list

import networksAux

import collections
import random


class NodeNullModel(object):

    def __init__(self, networkT0Filename, networkT1Filename,
                 weighted):
        networkT0 = collections.defaultdict(dd_list)
        tempNetwork = networksAux.constructNetworkFromFile(networkT0Filename,
                                                           weighted)
        classes = set()
        for node, peers in tempNetwork.items():
            networkT0[node]
            for peer, c in peers.items():
                friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                classes.add(friendshipClass)
                networkT0[node][friendshipClass].extend(peer)

        networkT1 = collections.defaultdict(dd_list)
        tempNetwork = networksAux.constructNetworkFromFile(networkT0Filename,
                                                           weighted)
        for node, peers in tempNetwork.items():
            networkT0[node]
            for peer, c in peers.items():
                friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                classes.add(friendshipClass)
                networkT1[node][friendshipClass].extend(peer)

        self.networkT0 = networkT0
        self.networkT1 = networkT1
        self.classes = classes
        self.nodes = set(self.networkT0.keys())
        self.nodes.update(set(self.networkT1.keys()))

    def run(self):
        # STEP 4: Create a null model
        prediction = collections.defaultdict(list)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

        for node in self.nodes:
            exclusionForNode = [node]
            for c in self.classes:
                # TODO What happens at class 0?
                oldTies = set(self.networkT0[node][c])
                newTies = set(self.networkT1[node][c])
                createdTies = newTies.difference(oldTies)
                dissolvedTies = oldTies.difference(newTies)

                # Add the newly formed ties
                candidates = self.selectCandidates(
                    self.nodes, len(createdTies), exclusionForNode)
                for candidate in candidates:
                    prediction[node].append((candidate, c))

                # Remove the dissovled ties
                toRemove = self.selectCandidates(
                    oldTies, len(dissolvedTies), [])
                remainingTies = oldTies.difference(toRemove)
                for candidate in remainingTies:
                    prediction[node].append((candidate, c))

        return prediction

    def selectCandidates(self, population, n, exclusion):
        if len(population) == 0:
            return []
        else:
            candidates = []
            while len(candidates) < n:
                candidate = random.sample(population, 1)
                if candidate not in exclusion:
                    candidates.extend(candidate)
            return candidates


class NetworkNullModel(object):
    ''' NullModel for an evolving graph that preservers the amount of
    change between timesteps for each "class" of tie '''

    def __init__(self, networkT0Filename, networkT1Filename,
                 weighted):
        networkT0 = collections.defaultdict(dd_list)
        tempNetwork = networksAux.constructNetworkFromFile(networkT0Filename,
                                                           weighted)
        classes = set()
        for node, peers in tempNetwork.items():
            networkT0[node]
            for peer, c in peers.items():
                friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                classes.add(friendshipClass)
                networkT0[friendshipClass][node].extend(peer)

        networkT1 = collections.defaultdict(dd_list)
        tempNetwork = networksAux.constructNetworkFromFile(networkT0Filename,
                                                           weighted)
        for node, peers in tempNetwork.items():
            networkT0[node]
            for peer, c in peers.items():
                friendshipClass = networksAux.mapSecondsToFriendshipClass(c)
                classes.add(friendshipClass)
                networkT1[friendshipClass][node].extend(peer)

        self.networkT0 = networkT0
        self.networkT1 = networkT1
        self.classes = list(classes)
        self.nodes = set()
        for c in classes:
            self.nodes.update(set(self.networkT0[c].keys()))
            self.nodes.update(set(self.networkT1[c].keys()))
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

    def run(self):
        predictions = list()
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

        exclusion = []
        random.shuffle(self.classes)
        for c in self.classes:
            # TODO Randomly iterate through
            # the classes
            oldTies = set(self.networkT0[c])
            newTies = set(self.networkT1[c])
            createdTies = newTies.difference(oldTies)
            dissolvedTies = oldTies.difference(newTies)

            # Add the newly formed ties
            candidates = self.selectCandidates(
                self.nodes, len(createdTies), exclusion)
            for node, peer in candidates:
                predictions.append((node, peer, c))
                exclusion.append(node, peer)
                # A tie cannot have two types of classes
                # TODO Check the prediction format of the RandomForest

            # Remove the dissovled ties
            toRemove = self.selectCandidates(
                oldTies, len(dissolvedTies), [])
            remainingTies = oldTies.difference(toRemove)
            for node, peer in remainingTies:
                predictions.append((node, peer, c))

        return predictions

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
                node = random.sample(population, 1)
                peer = random.sample(population, 1)
                if node != peer:
                    candidate = (node, peer)
                    if candidate not in exclusion:
                        candidates.extend(candidate)
            return candidates
