'''
Author: Christoph Stich
Date: 2015-06-16
This is a port from Edwin Chen's Scala code to Python 3
My contribution:
    - I fixed a bug where only a symmetric relationship would
    contribute to the propagation of the personalized PageRank score. Now you
    can either just use the outgoing links or any links
    - Breaks ties randomly

Also known as Rooted Page Rank. The citation is as follows:
       Liben-Nowell D, Kleinberg J M. The link-prediction problem for social
       networks.  Journal of the American Society for Information Science and
       Technology, 2007, 58:  1019{1031
'''

import copy
import collections
import random


class PersonalizedPageRank(object):
    '''
    Given a directed graph of follower and following edges, compute personalized
    PageRank scores around specified starting nodes.

    A personalized PageRank is similar to standard PageRank, except that when
    randomly teleporting to a new node, the surfer always teleports back to the
    given source node being personalized (rather than to a node chosen uniformly
    at random, as in the standard PageRank algorithm)

    In other words, the random surfer in the personalized PageRank model works
    as follows:
        - He starts at the source node X that we want to calculate a
        personalized PageRank around.
        - At step i: with probability p, the surfer moves to a neighboring node
                     chosen uniformly at random;
                     with probability $1-p$, the surfer instead teleports back
                     to the original source node X.

    The limiting probability that the surfer is at node N is the
    personalized PageRank score of node N around X.
    '''

    def __init__(self, inLinks, outLinks,
                 numOfIterations=3, maxNodesToKeep=25,
                 directed=False, alpha=0.5):
        '''
        @maxNodesToKeep: How many nodes to keep. If None, returns all
        @directed: If true only consider outgoing links as the original PageRank
        '''
        self.inLinks = copy.deepcopy(inLinks)  # Followers as a dictionary where
        # key: node, values: peers
        self.outLinks = copy.deepcopy(outLinks)  # Followees as a dictionary where
        # key: node, values: peers
        self.numOfIterations = numOfIterations
        self.maxNodesToKeep = maxNodesToKeep
        self.directed = directed
        self.alpha = alpha

    def pageRank(self, user, returnScores=False):
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
