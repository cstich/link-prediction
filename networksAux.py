import collections
import copy


def constructNetworkFromFile(networkFile, weightedNetworks):

    if weightedNetworks:
        network = collections.defaultdict(dict)
    else:
        network = collections.defaultdict(list)

    with open(networkFile, 'r') as f:
        for line in f.readlines():
            line = line[:-1].split(sep=',')
            node = line[0]
            network[node]
            if weightedNetworks:
                peers = copy.copy(line[1:])
                peers = dict(zip(peers[0::2], peers[1::2]))
                network[node] = peers
            else:
                peers = copy.copy(line[1:])
                network[node].extend(peers)

    return network


def mapSecondsToFriendshipClass(seconds):
    seconds = int(seconds)
    if seconds <= 0:
        return 0
    elif seconds > 0 and seconds <= 900:
        return 1
    elif seconds > 900 and seconds <= 3600:
        return 2
    else:
        return 3
