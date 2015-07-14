from geogps import Aux
from geogps import Parser
from geogps.DictAux import dd_dict, dd_list

from Metrics import UserMetrics
from Metrics import friendFriends

from PersonalizedPageRank import PersonalizedPageRank

import collections
import copy
import networkx as nx
import numpy as np
import os
import pytz
import re
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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory> "
              "<networks directory> "
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    inputNetworks = sys.argv[2]
    outputPath = sys.argv[3]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputFilePattern = re.compile('parsedData[\-\_a-zA-Z0-9]*\.pck')
    rs = Parser.loadPickles(inputData, inputFilePattern,
                            matchUser=False)[0]

    localizedBlues = rs['localizedBlues']
    stopLocations = rs['stopLocs']
    blues = rs['blues']
    timeIntervalls = list(rs['intervalls'])  # Are ordered in time
    users = list(map(str, list(blues.keys())))

    ''' Read networks at different time states '''
    networkFilesPattern = re.compile('([0-9]+\-[0-9-]+)\.csv')
    networkFiles = Parser.readFiles(inputNetworks, networkFilesPattern)
    networks = collections.defaultdict(dd_list)
    nxNetworks = collections.defaultdict(nx.Graph)

    for file in networkFiles:
        key = networkFilesPattern.search(file).group(1).split(sep='-')
        key = tuple(map(int, key))
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line[:-1].split(sep=',')
                node = line[0]
                peers = copy.copy(line[1:])
                networks[key][node].extend(peers)
                for peer in peers:
                    nxNetworks[key].add_edge(node, peer)

    ''' Generate featues '''
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
    # TODO Try weighing candidates by something
    candidates = collections.defaultdict(dict)
    for time, network in networks.items():
        ppr = PersonalizedPageRank(network, network)
        for user, peers in network.items():
            candidates[time][user] = ppr.pageRank(user)

    ''' Find the propagation scores '''
    for time, network in networks.items():
        for user, peers in network.items():
            # features[time]['propScores'] = collections.defaultdict(dd_float)
            scores = ppr.pageRank(user, returnScores=True)
            for s in scores:
                peer = s[0]
                try:
                    features[time][user]['propScores'][peer] = s[1]
                except TypeError:
                    features[time][user]['propScores'] = collections.defaultdict(float)
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

    for i in range(1, len(timeIntervalls)):
        t_0 = timeIntervalls[i-1]
        t_1 = timeIntervalls[i]
        print('Working on timestep ', str(t_0))
        listOfTestUsers = []
        # Create n-folded training and verification sets
        # 1) Shuffle the list of users
        # 2) Split the random list of users into n-sublists
        # where each sublist represents a set of users that are
        # used for testing
        testUsers = np.random.choice(users, 131, replace=False)
        testUsers = Aux.chunkify(testUsers, 10)
        for j, currentTestSet in enumerate(testUsers):
            test = []
            train = []
            for user, cans in candidates[t_0].items():
                for c in cans:
                    line = []
                    truth = int(c in networks[t_1][user])
                    line.extend([truth, user, c])
                    # Iterate through all the features
                    valuesFeature = []
                    for f in listOfFeatures:
                        # Split into training and test set
                        try:
                            valuesFeature.append(features[t_0][user][f][c])
                        except TypeError:
                            valuesFeature.append(0)

                    line.extend(valuesFeature)
                    if user in currentTestSet:
                        test.append(line)
                    else:
                        train.append(line)

            directory = outputPath + str(t_0[0])
            if not os.path.exists(directory):
                    os.makedirs(directory)
            header = ['edge', 'source', 'destination']
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
