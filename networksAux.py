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
    if seconds >= 0 and seconds < 300:
        return 1
    elif seconds >= 300 and seconds < 603:
        return 2
    elif seconds >= 603 and seconds < 1203:
        return 3
    elif seconds >= 1203:
        return 4
    else:
        raise ValueError
