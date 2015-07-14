'''
Author: Christoph Stich
Date: 2015-06-30
Uses place friends as candidates for the machine learning stage of the link-prediction task
'''

import copy
import collections
import random


class PlaceFriendsPageRank(object):
    '''

    '''

    def __init__(self, blues,
                 maxNodesToKeep=25,
                 directed=False, alpha=0.5):
        '''
        @maxNodesToKeep: How many nodes to keep. If None, returns all
        @directed: If true only consider outgoing links as the original PageRank
        '''
        self.inLinks = copy.deepcopy(blues)  # Followers as a dictionary where
        self.maxNodesToKeep = maxNodesToKeep
        self.directed = directed

    def placeRank(self, user, returnScores=False):
        '''
        Calculate a personalized PageRank around the given user, and return a
        list of the nodes with the highest personalized PageRank scores.
        Care was taken to break ties randomly.
        @user: The user to calculat the Personalized PageRank for
        algorithm does. If false, uses any tie to propagate the score
        @return A list of (node, probability of landing at this node after
        running a personalize PageRank for K iterations) pairs.
        '''
        probs = {}
        probs[user] = 1

        pageRankProbs = self.pageRankHelper(user, probs, self.numOfIterations,
                                            self.directed, self.alpha)
        pageRankProbs = list(pageRankProbs.items())
        # Reshuffle results of the PPR to make sure ties are broken at random
        random.shuffle(pageRankProbs)

        if returnScores:
            return pageRankProbs
        else:
            # Return the n-highest scoring nodes
            pageRankProbs = sorted(pageRankProbs, key=lambda x: x[1],
                                   reverse=True)[:self.maxNodesToKeep]
            return [e[0] for e in pageRankProbs]

    def pageRankHelper(self, start, probs, numIterations, directed, alpha):
        if numIterations <= 0:
            return probs
        else:
            # This map holds the updated set of probabilities, after the
            # current iteration.
            probsPropagated = collections.defaultdict(int)
            # With probability 1 - alpha, we teleport back to the start node.
            probsPropagated[start] = 1 - alpha

            # Propagate the previous probabilities...
            for node, prob in probs.items():
                forwards = self.getOutLinks(node)
                if directed:
                    neighbors = list()
                    neighbors + forwards
                else:
                    backwards = self.getInLinks(node)
                    neighbors = set(forwards + backwards)

                if len(neighbors) != 0:
                    probToPropagate = alpha * prob / len(neighbors)
                # With probability alpha, we move to a follower...
                # And each node equally distributes its current probability to
                # its neighbors.
                for neighbor in neighbors:
                    probsPropagated[neighbor] += probToPropagate

        return self.pageRankHelper(start, probsPropagated,
                                   numIterations - 1, directed, alpha)

    def getInLinks(self, user):
        try:
            return self.inLinks[user]
        except KeyError:
            return list()

    def getOutLinks(self, user):
        try:
            return self.outLinks[user]
        except KeyError:
            return list()
