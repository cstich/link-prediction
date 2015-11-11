from gs import aux
from gs import timeAux
from gs.dictAux import dd_int
from delorean import Delorean

import collections
import copy
import math
import pytz
import statistics

amsterdam = pytz.timezone('Europe/Amsterdam')


def calculatePlaceEntropy(stopLocs):
    ''' For calculation of entropy see here:
    http://www.johndcook.com/blog/2013/08/17/calculating-entropy/
    '''
    timeSpentAtLocation = collections.defaultdict(int)
    timeSpentAtLocationPerUser = collections.defaultdict(dd_int)
    for user, values in stopLocs.items():
        for intervall, RENAMEME in values.items():
            for timeSpent, stopLoc in RENAMEME.items():
                ts = timeSpent[1] - timeSpent[0]
                assert ts > 0, 'Time between measurments must be > 0'
                locationIndex = stopLoc[2]
                timeSpentAtLocation[locationIndex] += ts
                timeSpentAtLocationPerUser[locationIndex][user] += ts
    placeEntropy = collections.defaultdict(float)
    for place, timeSpent in timeSpentAtLocation.items():
        sumOfLogs = sum([x * math.log(x, 2)
                         for x in timeSpentAtLocationPerUser[place].values()])
        placeEntropy[place] = math.log(
            timeSpentAtLocation[place], 2
        ) - 1/timeSpentAtLocation[place] * sumOfLogs
    return placeEntropy


def generate_triangles(nodes):
    ''' Generate triangles. Weed out duplicates. Undirected '''
    visited_ids = set()  # remember the nodes that we have tested already
    triangles = set()
    for node_a_id in nodes:
        for node_b_id in nodes[node_a_id]:
            if node_b_id == node_a_id:
                print(node_b_id, node_a_id)
                raise ValueError  # nodes shouldn't point to themselves
            if node_b_id in visited_ids:
                continue  # we should have already found b->a->??->b
            for node_c_id in nodes[node_b_id]:
                if node_c_id in visited_ids:
                    continue  # we should have already found c->a->b->c
                if node_a_id in nodes[node_c_id]:
                    triangles.add(frozenset([node_a_id, node_b_id, node_c_id]))
        visited_ids.add(node_a_id)  # don't search a -
        # we already have all those cycles
    return triangles


def friendFriends(friendsDict):
    ''' Given a dictionary of friends, returns
    a dicionary of friend friends '''
    friendsFriends = collections.defaultdict(set)
    lookupFriendsDict = copy.deepcopy(friendsDict)
    iterFriendsDict = copy.deepcopy(dict(friendsDict))

    for k, friends in iterFriendsDict.items():
        for friend in friends:
            candidates = [f for f in lookupFriendsDict[friend] if f != k]
            friendsFriends[k].update(candidates)
    return friendsFriends


def maxMode(ls):
    ''' Returns the mode of a list. If ambigous, return the highest value
    item '''
    return max(statistics._counts(ls), key=lambda x: x[0])[0]


class UserMetrics(object):

    def __init__(self, user, users, blues, localizedBlues, stopLocs,
                 interactionsAtTime, triangles, placeEntropy, beginTime,
                 endTime):
        self.user = user
        self.users = users
        self.blues = blues
        self.localizedBlues = localizedBlues
        self.stopLocs = stopLocs
        self.interactionsAtTime = interactionsAtTime
        self.triangles = triangles
        self.placeEntropy = placeEntropy
        self.metrics = collections.defaultdict(dict)
        self.beginTime = beginTime
        self.endTime = endTime

    def generateNetworkAtTime(self, networkAtContextAtTime, contextAtTime):
        if self.blues is not None:
            for blue in self.blues:
                # Bin all interactions into 10 minute buckets
                time = int(blue[1] / 600) * 600
                peer = str(blue[0])

                if len(contextAtTime[self.user][time]) == 0:
                    context = 'other'
                else:
                    context = maxMode(contextAtTime[self.user][time])

                if self.user == peer:
                    continue
                networkAtContextAtTime[context][time][self.user].add(peer)
                networkAtContextAtTime[context][time][peer].add(self.user)

    def generateMeetingsDistribution(self, interactionsAtContextAtTime,
                                     contextAtTime):
        self.metrics['timeSpentAt'] = self.timeSpentAtLocation()
        interactionsAtTime = collections.defaultdict(set)

        if self.blues is not None:
            for blue in self.blues:
                # Bin all interactions into 10 minute buckets
                time = int(blue[1] / 600) * 600
                peer = str(blue[0])
                if self.user == peer:
                    continue
                interactionsAtTime[time].add(peer)
        for time, peers in interactionsAtTime.items():
            hourOfTheWeek = timeAux.epochToHourOfTheWeek(time, amsterdam)

            if len(contextAtTime[self.user][time]) == 0:
                context = 'other'
            else:
                context = maxMode(contextAtTime[self.user][time])
            interactionsAtContextAtTime[context][hourOfTheWeek].append(
                len(peers))
            interactionsAtContextAtTime['all'][hourOfTheWeek].append(
                len(peers))

    def generateContextAtTime(self, contextAtTime):
        self.metrics['timeSpentAt'] = self.timeSpentAtLocation()
        # Create a lookup for time and context
        importanceDict = self.relativePersonalImportance()

        if self.stopLocs is not None:
            for time, sl in self.stopLocs.items():
                con = sl[1]
                slLabel = sl[2]
                if con == 'other' or con is None:
                    if importanceDict[slLabel] >= 0.1:
                        con = 'thirdPlace'
                if con == 'social':
                    con = 'home'
                begin, end = time
                begin = int(begin / 600) * 600
                end = int(end / 600) * 600
                for t in range(begin, end, 600):
                    contextAtTime[self.user][t].append(con)

    def contextTimePattern(self, result):
        self.metrics['timeSpentAt'] = self.timeSpentAtLocation()
        contextAtTime = collections.defaultdict(list)

        # Create a lookup for time and context
        importanceDict = self.relativePersonalImportance()
        contextAtTime = collections.defaultdict(list)

        if self.stopLocs is not None:
            for time, sl in self.stopLocs.items():
                con = sl[1]
                slLabel = sl[2]
                if con == 'other' or con is None:
                    if importanceDict[slLabel] >= 0.1:
                        con = 'thirdPlace'
                if con == 'social':
                    con = 'home'
                begin, end = time
                for t in range(begin, end, 600):
                    t = timeAux.epochToHourOfTheWeek(t, amsterdam)
                    contextAtTime[t].append(con)

        for hour, values in contextAtTime.items():
            for con in values:
                result[con].append(hour)

    def getTimeAndSocialFeatures(self):
        ''' Time features independent from place '''
        # Count how much time people spend together
        timeSpentWith = collections.defaultdict(list)
        timeSpentWithDict = collections.defaultdict(int)
        metAtHourOfTheWeek = collections.defaultdict(list)
        metAtDayOfTheWeek = collections.defaultdict(list)
        metAtHourOfTheDay = collections.defaultdict(list)
        metLast = collections.defaultdict(int)

        for blue in self.blues:
            if metLast[str(blue[0])] < blue[1]:
                metLast[str(blue[0])] = blue[1]
            # Count how much time people spend together
            timeSpentWith[str(blue[0])].append(blue[1])
            # At which time people have met
            localTime = timeAux.epochToLocalTime(blue[1], amsterdam)
            lastMonday = Delorean(localTime).last_monday().midnight()
            timeDelta = localTime - lastMonday

            hour = localTime.hour
            day = localTime.weekday()
            days, seconds = timeDelta.days, timeDelta.seconds
            hourOfTheWeek = (days * 24) + (seconds // 3600)

            metAtHourOfTheWeek[str(blue[0])].append(hourOfTheWeek)
            metAtDayOfTheWeek[str(blue[0])].append(day)
            metAtHourOfTheDay[str(blue[0])].append(hour)

        # Find the mode for now of when people have mostly met
        self.saveTimeMetric(metAtHourOfTheWeek, 'metAtHourOfTheWeek')
        self.saveTimeMetric(metAtDayOfTheWeek, 'metAtDayOfTheWeek')
        self.saveTimeMetric(metAtHourOfTheDay, 'metAtHourOfTheDay')
        for peer, lastMeeting in metLast.items():
            self.metrics['metLast'][peer] = lastMeeting

        ''' Social features '''
        # If the time difference between two bluetooth observations is less
        # than 601 seconds regard the interaction as continous
        for peer, times in timeSpentWith.items():
            for v in aux.difference(times):
                if v < 601:
                    timeSpentWithDict[peer] += v
        self.metrics['timeSpent'] = timeSpentWithDict

        amountOfOtherPeopleAtInteraction = collections.defaultdict(list)
        for time, interactions in self.interactionsAtTime.items():
            interaction = interactions[self.user]
            for peer in interaction:
                amountOfOtherPeopleAtInteraction[peer].append(
                    len(interaction))

        # Average the amount of other people
        averageAmountOfOtherPeople = collections.defaultdict(int)
        for peer, people in amountOfOtherPeopleAtInteraction.items():
            averageAmountOfOtherPeople[peer] = statistics.median(people)

        self.metrics['numberOfPeople'] = averageAmountOfOtherPeople

        # Spatial-triadic closure
        t0, t1, t2, t3, t4, t5 = self.triadicClosure()
        self.metrics['triadicClosure_0'] = t0
        self.metrics['triadicClosure_1'] = t1
        self.metrics['triadicClosure_2'] = t2
        self.metrics['triadicClosure_3'] = t3
        self.metrics['triadicClosure_4'] = t4
        self.metrics['triadicClosure_5'] = t5

    def getPlaceFeatures(self):
        ''' Create the place features - context, relative importance '''
        typeDict = collections.defaultdict(list)
        timeTypeDict = collections.defaultdict(list)
        importanceDict = self.relativePersonalImportance()
        relativeImportance = collections.defaultdict(list)
        locationEntropy = collections.defaultdict(list)

        if self.localizedBlues is not None:
            for i, blue in self.localizedBlues.items():
                con = list(self.stopLocs.values())[i][1]
                slLabel = list(self.stopLocs.values())[i][2]
                for peer, bs in blue.items():
                    ''' Place features '''
                    # Create a distribution of the relative importance of
                    # venues where people meet
                    locImportance = importanceDict[slLabel]
                    locEntropy = self.placeEntropy[slLabel]
                    relativeImportance[str(peer)].append(locImportance)
                    locationEntropy[str(peer)].append(locEntropy)
                    if con == 'other' or con is None:
                        if importanceDict[slLabel] >= 0.1:
                            con = 'thirdPlace'
                    typeDict[str(peer)].append(con)
                    for e in bs:
                        timeTypeDict[str(peer)].append((con, e))

        ''' Context features '''
        ''' Count the time spent at each context with each peer '''
        timeSpentAtHomeWith = collections.defaultdict(int)
        timeSpentAtUniversityWith = collections.defaultdict(int)
        timeSpentAtThirdPlaceWith = collections.defaultdict(int)
        timeSpentAtOtherPlaceWith = collections.defaultdict(int)
        contexts = ['home', 'university', 'thirdPlace', 'other', None]
        for peer, times in timeTypeDict.items():
            times.sort(key=lambda x: x[1])
            splitTimes = aux.isplit(times, contexts)
            for e in splitTimes:
                con, times = zip(*e)
                con = con[0]
                con = con.replace(' ', '')
                for v in aux.difference(times):
                    if v < 601:
                        if con == 'home':
                            timeSpentAtHomeWith[peer] += v
                        if con == 'social':
                            timeSpentAtHomeWith[peer] += v
                        if con == 'university':
                            timeSpentAtUniversityWith[peer] += v
                        if con == 'thirdPlace':
                            timeSpentAtThirdPlaceWith[peer] += v
                        if con == 'other':
                            timeSpentAtOtherPlaceWith[peer] += v
                        if con is None:
                            timeSpentAtOtherPlaceWith[peer] += v

        self.metrics['timeSpentAtHomeWith'] = timeSpentAtHomeWith
        self.metrics['timeSpentAtUniversityWith'] =\
            timeSpentAtUniversityWith
        self.metrics['timeSpentAtThirdPlaceWith'] =\
            timeSpentAtThirdPlaceWith
        self.metrics['timeSpentAtOtherPlaceWith'] =\
            timeSpentAtOtherPlaceWith

        ''' Relative importance features '''
        tempDict = collections.defaultdict(float)
        for peer, values in relativeImportance.items():
            tempDict[str(peer)] = max(values)
        self.metrics['relativeImportance'] = tempDict

        ''' Place entropy features '''
        tempDict = collections.defaultdict(float)
        for peer, values in locationEntropy.items():
            tempDict[str(peer)] = min(values)
        self.metrics['placeEntropy'] = tempDict

    def generateFeatures(self):
        # If ther person hasn't met anybody, she obviously can't have any
        # features for that time period
        if self.blues is not None:
            # stopLocs = list(self.stopLocs.values())
            ''' Place features derived from the individual '''
            self.metrics['timeSpentAt'] = self.timeSpentAtLocation()
            self.getTimeAndSocialFeatures()
            self.getPlaceFeatures()

    def saveTimeMetric(self, meetingsAtTime, nameOfMetric):
        for peer, meetings in meetingsAtTime.items():
            for time, value in collections.Counter(meetings).items():
                self.metrics[nameOfMetric+str(time)][peer] = value

    def timeSpentAtLocation(self):
        '''
        Counts how much time a person spent at each stop location.
        Organizes it via the significant location label.
        Expects an expanded dictionary of stop locations,
        where {key: (beginTime, endTime), value: (location, label)
        '''
        if self.stopLocs is not None:
            d = collections.defaultdict(int)
            for time, sl in self.stopLocs.items():
                slIndex = sl[2]
                timeSpent = int(time[1] - time[0])
                d[slIndex] += timeSpent
            return d
        else:
            return None

    def relativePersonalImportance(self):
        '''
        Assesses the relative importance for each stop location
        by the amount of time spent there
        '''
        if self.stopLocs is not None:
            d = collections.defaultdict(float)
            totalTime = sum(self.timeSpentAtLocation().values())
            for k, v in self.metrics['timeSpentAt'].items():
                d[k] = v / totalTime
            return d
        else:
            return None

    def triadicClosure(self):
        '''
        Triadic0 counts how many times nobody was there
        Triadic1 counts how often s meets somebody that was not t
        Triadic2 counts how many times s meets t, and only t
        Triadic3 counts how often s meets t and somebody (x) else was there too
        Triadic4 counts how often s and t are only connected via x
        Triadic5 counts triadic closure events between s and t and x
        '''
        singleMeetings = 0
        triadic0 = 0
        triadic1 = collections.defaultdict(int)
        triadic2 = collections.defaultdict(int)
        triadic3 = collections.defaultdict(int)
        triadic4 = collections.defaultdict(int)
        triadic5 = collections.defaultdict(int)
        allMeetings = 0
        for time in range(self.beginTime, self.endTime, 600):
            peers = self.interactionsAtTime[time][self.user]
            ''' Select all triangles with s '''
            triangles = self.triangles[time]
            if triangles is not None:
                trianglesS = [t for t in triangles if self.user in t]
            if len(peers) == 0:
                triadic0 += 1  # Create a defaultdict with
                # this as its default value
            elif len(peers) == 1:
                for p in peers:
                    triadic2[p] += 1
                    singleMeetings += 1
                    allMeetings += 1
            else:
                for p in peers:
                    triadic3[p] += 1
                    connectionsOfPeer = self.interactionsAtTime[time][p]
                    for cp in connectionsOfPeer:
                        if cp in peers:
                            pass
                        else:
                            triadic4[cp] += 1
                    allMeetings += 1
                if trianglesS is not None:
                    for t in trianglesS:
                        for p in t:
                            if p != self.user:
                                triadic5[p] += 1
                                allMeetings += 1

        triadic0 = collections.defaultdict(lambda: triadic0)
        for peer, metings in triadic1.items():
            triadic1[peer] = singleMeetings - triadic2[peer]
        return triadic0, triadic1, triadic2, triadic3, triadic4, triadic5
