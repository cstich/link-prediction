from geogps import Aux
from geogps import Parser

from Metrics import UserMetrics
from Metrics import friendFriends

from PersonalizedPageRank import PersonalizedPageRank

import collections
import copy
import networkx as nx
import numpy as np
import os
import pickle
import pytz
import re
import sys

amsterdam = pytz.timezone('Europe/Amsterdam')


def dd_dict():
    return collections.defaultdict(dict)


def dd_set():
    return collections.defaultdict(set)


def dd_float():
    return collections.defaultdict(float)


def dd_list():
    return collections.defaultdict(list)


def ddd_float():
    return collections.defaultdict(dd_float)


def ddd_list():
    return collections.defaultdict(dd_list)


def dddd_list():
    return collections.defaultdict(ddd_list)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s" % (sys.argv[0]) +
              "<data directory>"
              "<networks directory>"
              "<output directory>")
        sys.exit(-1)

    inputData = sys.argv[1]
    inputNetworks = sys.argv[2]
    outputPath = sys.argv[3]
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    inputData = Parser.parsePath(inputData, scriptDir)

    with open(inputData + '/parsedData.pck', 'rb') as f:
        rs = pickle.load(f)

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

    ''' Find candidates '''
    candidates = collections.defaultdict(dict)
    for time, network in networks.items():
        ppr = PersonalizedPageRank(network, network)
        for user, peers in network.items():
            candidates[time][user] = ppr.pageRank(user)

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
            try:
                features[time][e[0]]['adamicAdar'][e[1]] = e[2]
            except TypeError:
                features[time][e[0]]['adamicAdar'] = collections.defaultdict(float)
                features[time][e[0]]['adamicAdar'][e[1]] = e[2]
            try:
                features[time][e[1]]['adamicAdar'][e[0]] = e[2]
            except TypeError:
                features[time][e[1]]['adamicAdar'] = collections.defaultdict(float)
                features[time][e[1]]['adamicAdar'][e[0]] = e[2]
        for e in nx.jaccard_coefficient(network):  # Jaccard coefficient
            try:
                features[time][e[0]]['jaccard'][e[1]] = e[2]
            except TypeError:
                features[time][e[0]]['jaccard'] = collections.defaultdict(float)
                features[time][e[0]]['jaccard'][e[1]] = e[2]
            try:
                features[time][e[1]]['jaccard'][e[0]] = e[2]
            except TypeError:
                features[time][e[1]]['jaccard'] = collections.defaultdict(float)
                features[time][e[1]]['jaccard'][e[0]] = e[2]
        for e in nx.preferential_attachment(network):  # Preferential attachment
            try:
                features[time][e[0]]['PA'][e[1]] = e[2]
            except TypeError:
                features[time][e[0]]['PA'] = collections.defaultdict(float)
                features[time][e[0]]['PA'][e[1]] = e[2]
            try:
                features[time][e[1]]['PA'][e[0]] = e[2]
            except TypeError:
                features[time][e[1]]['PA'] = collections.defaultdict(float)
                features[time][e[1]]['PA'][e[0]] = e[2]
        for e in nx.resource_allocation_index(network):  # Ressource allocation
            try:
                features[time][e[0]]['resourceAllocation'][e[1]] = e[2]
            except TypeError:
                features[time][e[0]]['resourceAllocation'] = collections.defaultdict(float)
                features[time][e[0]]['resourceAllocation'][e[1]] = e[2]
            try:
                features[time][e[1]]['resourceAllocation'][e[0]] = e[2]
            except TypeError:
                features[time][e[1]]['resourceAllocation'] = collections.defaultdict(float)
                features[time][e[1]]['resourceAllocation'][e[0]] = e[2]
        '''
        for e in nx.cn_soundarajan_hopcroft(network):  # Commnon neighbors
            # using community information
            try:
                features[time][e[0]]['commonNeighborsCommunity'][e[1]] = e[2]
            except TypeError:
                features[time][e[0]]['commonNeighborsCommunity'] = collections.defaultdict(float)
                features[time][e[0]]['commonNeighborsCommunity'][e[1]] = e[2]
            try:
                features[time][e[1]]['commonNeighborsCommunity'][e[0]] = e[2]
            except TypeError:
                features[time][e[1]]['commonNeighborsCommunity'] = collections.defaultdict(float)
                features[time][e[1]]['commonNeighborsCommunity'][e[0]] = e[2]
        for e in nx.ra_index_soundarajan_hopcroft(network):
            try:
                features[time][e[0]]['ressourceAllocationCommunity'][e[1]] = e[2]
            except TypeError:
                features[time][e[0]]['ressourceAllocationCommunity'] = collections.defaultdict(float)
                features[time][e[0]]['ressourceAllocationCommunity'][e[1]] = e[2]
            try:
                features[time][e[1]]['ressourceAllocationCommunity'][e[0]] = e[2]
            except TypeError:
                features[time][e[1]]['ressourceAllocationCommunity'] = collections.defaultdict(float)
                features[time][e[1]]['ressourceAllocationCommunity'][e[0]] = e[2]
        for e in nx.within_inter_cluster(network):
            try:
                features[time][e[0]]['withinInterCluster'][e[1]] = e[2]
            except TypeError:
                features[time][e[0]]['withinInterCluster'] = collections.defaultdict(float)
                features[time][e[0]]['withinInterCluster'][e[1]] = e[2]
            try:
                features[time][e[1]]['withinInterCluster'][e[0]] = e[2]
            except TypeError:
                features[time][e[1]]['withinInterCluster'] = collections.defaultdict(float)
                features[time][e[1]]['withinInterCluster'][e[0]] = e[2]
        '''

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
        'relativeImportance',
        'adamicAdar',
        'jaccard',
        'PA',
        'propScores',
        'resourceAllocation',
        'commonNeighborsCommunity',
        'ressourceAllocationCommunity',
        'withinInterCluster']

    for i in range(1, len(timeIntervalls[0:4])):
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
