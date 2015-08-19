from geogps import Aux
from geogps import Parser
from geogps.DictAux import dd_dict, dd_list

from Metrics import UserMetrics
from Metrics import friendFriends
from networkAux import mapSecondsToFriendshipClass

from PersonalizedPageRank import PersonalizedPageRank

import collections
import copy
import networkx as nx
import numpy as np
import os
import pytz
import re
import random
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


def setNetworkFeatures(features, name, edge):
    try:
        features[time][edge[0]][name][edge[1]] = edge[2]
    except (TypeError, KeyError):
        features[time][edge[0]][name] = collections.defaultdict(float)
        features[time][edge[0]][name][edge[1]] = edge[2]
    try:
        features[time][edge[1]][name][edge[0]] = edge[2]
    except (TypeError, KeyError):
        features[time][edge[1]][name] = collections.defaultdict(float)
        features[time][edge[1]][name][edge[0]] = edge[2]
    return features


def findValuesForFeatures(listOfFeatures, features, time, user, peer):
    valuesFeature = []
    for f in listOfFeatures:
        # Split into training and test set
        try:
            valuesFeature.append(
                features[time][user][f][peer])
        except (TypeError, KeyError):
            valuesFeature.append(0)
    return valuesFeature


def findTie(network, user, peer, classes=None):
    if classes:
        try:
            result = network[user][peer]
            return classes(result)
        except KeyError:
            return 0
    else:
        return int(peer in network[user])


def unweighNetwork(network):
    result = collections.defaultdict(list)
    for user, links in network.items():
        for peer, strength in links.items():
            if int(strength) > 0:
                result[user].append(peer)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory> "
              "<networks directory> "
              "<weighted networks 1/0> "
              "<length of 1 set of test users> "
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    inputNetworks = sys.argv[2]
    weightedNetworks = sys.argv[3]
    lengthOfTestUsers = int(sys.argv[4])
    outputPath = sys.argv[5]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputFilePattern = re.compile('parsedData[\-\_a-zA-Z0-9]*\.pck')
    rs = Parser.loadPickles(inputData, inputFilePattern,
                            matchUser=False)[0]

    localizedBlues = rs['localizedBlues']
    stopLocations = rs['stopLocs']
    blues = rs['blues']
    timeIntervalls = list(rs['intervalls'])  # Are ordered in time
    slidingTimeIntervalls = list(rs['slidingIntervalls'])
    users = list(map(str, list(blues.keys())))

    ''' Read networks at different time states '''
    print('Read network files')
    networkFilesPattern = re.compile('([0-9]+\-[0-9-]+)\.csv')
    networkFiles = Parser.readFiles(inputNetworks, networkFilesPattern)
    if weightedNetworks:
        networks = collections.defaultdict(dd_dict)
    else:
        networks = collections.defaultdict(dd_list)
    nxNetworks = collections.defaultdict(nx.Graph)

    for f in networkFiles:
        key = networkFilesPattern.search(f).group(1).split(sep='-')
        key = tuple(map(int, key))
        with open(f, 'r') as f:
            for line in f.readlines():
                line = line[:-1].split(sep=',')
                node = line[0]
                if weightedNetworks:
                    peers = copy.copy(line[1:])
                    peers = dict(zip(peers[0::2], peers[1::2]))
                    networks[key][node] = peers
                    for peer, strength in peers.items():
                        nxNetworks[key].add_edge(node, peer)
                        # TODO Add tie strength for networkX networks
                else:
                    peers = copy.copy(line[1:])
                    networks[key][node].extend(peers)
                    for peer in peers:
                        nxNetworks[key].add_edge(node, peer)

    ''' Generate featues '''
    print('Generate user features')
    features = collections.defaultdict(dd_dict)
    for time, network in networks.items():
        ff = friendFriends(network)
        for user, peers in network.items():
            try:
                stopLocs = stopLocations[user][time]
            except KeyError:
                stopLocs = None
            try:
                locBlues = localizedBlues[user][time]
            except KeyError:
                locBlues = None
            try:
                bl = blues[user][time]
            except KeyError:
                bl = None
            um = UserMetrics(user, bl,
                             locBlues,
                             stopLocs,
                             network, ff)
            um.generateFeatures()
            features[time][um.user] = um.metrics

    ''' Add network measures to features '''
    print('Generate network features')
    for time, network in nxNetworks.items():
        for e in nx.adamic_adar_index(network):  # Adamic adar
            features = setNetworkFeatures(features, 'adamicAdar', e)
        for e in nx.jaccard_coefficient(network):  # Jaccard coefficient
            features = setNetworkFeatures(features, 'jaccard', e)
        for e in nx.preferential_attachment(network):  # Preferential attachment
            features = setNetworkFeatures(features, 'PA', e)
        for e in nx.resource_allocation_index(network):  # Ressource allocation
            features = setNetworkFeatures(features, 'resourceAllocation', e)
        '''
        for e in nx.cn_soundarajan_hopcroft(network):  # Commnon neighbors
        for e in nx.ra_index_soundarajan_hopcroft(network):
        for e in nx.within_inter_cluster(network):
        '''

    ''' Find candidates '''
    '''
    candidates = collections.defaultdict(dict)
    for time, network in networks.items():
        ppr = PersonalizedPageRank(network, network)
        for user, peers in network.items():
            candidates[time][user] = ppr.pageRank(user)
    '''

    ''' Find the propagation scores '''
    for time, network in networks.items():
        ppr = PersonalizedPageRank(network, network)
        for user, peers in network.items():
            # features[time]['propScores'] = collections.defaultdict(dd_float)
            if weightedNetworks:
                ppr = PersonalizedPageRank(unweighNetwork(network),
                                           unweighNetwork(network))
            else:
                ppr = PersonalizedPageRank(network, network)
            scores = ppr.pageRank(user, returnScores=True)
            for s in scores:
                peer = s[0]
                try:
                    features[time][user]['propScores'][peer] = s[1]
                except TypeError:
                    features[time][user]['propScores'] = \
                        collections.defaultdict(float)
                    features[time][user]['propScores'][peer] = s[1]

    ''' Write features to files '''
    listOfFeatures = [
        'metAtHourOfTheWeek',
        'metAtDayOfTheWeek',
        'metAtHourOfTheDay',
        'timeSpent',
        'spatialTriadicClosure',
        'candidatesSpatialTriadicClosure',
        'numberOfPeople',
        'metAtHome',
        'metAtUniversity',
        'metAtThirdPlace',
        'metAtOtherPlace',
        'timeSpentAtHomeWith',
        'timeSpentAtUniversityWith',
        'timeSpentAtThirdPlaceWith',
        'timeSpentAtOtherPlaceWith',
        'relativeImportance',
        'adamicAdar',
        'jaccard',
        'PA',
        'propScores',
        'resourceAllocation',
        'commonNeighborsCommunity',
        'ressourceAllocationCommunity',
        'withinInterCluster']

    ''' Use the same train and test sample for all timeframes
    aka create the samples of test users before you output your
    data '''
    # Create n-folded training and verification sets
    # 1) Shuffle the list of users
    # 2) Split the random list of users into n-sublists
    # where each sublist represents a set of users that are
    # used for testing
    testUsers = np.random.choice(users, len(users), replace=False)
    testUsers = Aux.chunkify(testUsers, int(len(users)/lengthOfTestUsers))

    if weightedNetworks:
        classes = mapSecondsToFriendshipClass
    else:
        classes = None

    for slidingTimeIntervall in slidingTimeIntervalls:
        print('Working on: ', slidingTimeIntervall)
        for i in range(1, len(slidingTimeIntervall)):
            t_0 = slidingTimeIntervall[i-1]
            t_1 = slidingTimeIntervall[i]
            print('Outputing timestep ', str(t_0))
            for j, currentTestSet in enumerate(testUsers):
                test = []
                train = []
                ''' This is an experiment without domain restriction '''
                for user in users:
                    for peer in users:
                        if user != peer:
                            line = []
                            truth = findTie(networks[t_1], user, peer, classes)
                            friends = findTie(networks[t_0], user,
                                              peer, classes)
                            randomFeature = int(random.getrandbits(1))
                            line.extend([truth, user, peer, friends,
                                         randomFeature])
                            # Iterate through all the features
                            valuesFeatures = findValuesForFeatures(
                                listOfFeatures, features, t_0, user, peer)
                            line.extend(valuesFeatures)
                            if user in currentTestSet:
                                test.append(line)
                            else:
                                train.append(line)

                directory = outputPath + str(t_1[0])
                if not os.path.exists(directory):
                        os.makedirs(directory)
                header = ['edge', 'source', 'destination', 'friends', 'random']
                header.extend(listOfFeatures)
                with open(directory + '/train' +
                          '_sample_' + str(j) + '.csv', 'w') as f:
                    f.write(','.join(header) + '\n')
                    for line in train:
                        f.write(','.join(map(str, line)) + '\n')
                with open(directory + '/test' +
                          '_sample_' + str(j) + '.csv', 'w') as f:
                    f.write(','.join(header) + '\n')
                    for line in test:
                        f.write(','.join(map(str, line)) + '\n')
