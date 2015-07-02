from geogps import Aux
from geogps import TimeAux
from delorean import Delorean


import collections
import copy
import pytz
import statistics

amsterdam = pytz.timezone('Europe/Amsterdam')




def dd_d():
    return collections.defaultdict(dict)


def dd_int():
    return collections.defaultdict(int)


def friendFriends(friendsDict):
    ''' Given a dictionary of friends, returns
    a dicionary of friend friends '''
    friendsFriends = collections.defaultdict(list)
    friendsDict = copy.deepcopy(friendsDict)

    for k, friends in friendsDict.items():
        for friend in friends:
            candidates = [f for f in friendsDict[friend] if f != k]
            friendsFriends[k].extend(candidates)
    return friendsFriends


def maxMode(ls):
    ''' Returns the mode of a list. If ambigous, return the highest value
    item '''
    return max(statistics._counts(ls), key=lambda x: x[0])[0]


class UserMetrics(object):

    def __init__(self, user, blues, localizedBlues, stopLocs, friends,
                 friendFriends):
        self.user = user
        self.blues = blues
        self.localizedBlues = localizedBlues
        self.stopLocs = stopLocs
        self.friends = friends   # A dictionary of who your friendss
        self.friendFriends = friendFriends
        self.metrics = collections.defaultdict(lambda: None)

    def generateFeatures(self):
        # If ther person hasn't met anybody, she obviously can't have any
        # features for that time period
        if self.blues is not None:
            '''
            TO DO HERE:
                - add the place entropy of where you have met
                - add all the social features
            '''

            # stopLocs = list(self.stopLocs.values())
            ''' Place features derived from the individual '''
            self.metrics['timeSpentAt'] = self.timeSpentAtLocation()

            ''' Time features independent from place '''
            # Count how much time people spend together
            timeSpentWith = collections.defaultdict(list)
            timeSpentWithDict = collections.defaultdict(int)
            metAtHourOfTheWeek = collections.defaultdict(list)
            metAtDayOfTheWeek = collections.defaultdict(list)
            metAtHourOfTheDay = collections.defaultdict(list)

            for blue in self.blues:
                # Count how much time people spend together
                timeSpentWith[str(blue.peer)].append(blue.time)
                # At which time people have met
                localTime = TimeAux.epochToLocalTime(blue.time, amsterdam)
                lastMonday = Delorean(localTime).last_monday().midnight()
                timeDelta = localTime - lastMonday

                hour = localTime.hour
                day = localTime.weekday()
                days, seconds = timeDelta.days, timeDelta.seconds
                hourOfTheWeek = (days * 24) + (seconds // 3600)

                metAtHourOfTheWeek[str(blue.peer)].append(hourOfTheWeek)
                metAtDayOfTheWeek[str(blue.peer)].append(day)
                metAtHourOfTheDay[str(blue.peer)].append(hour)

            # Find the mode for now of when people have mostly met
            temp = collections.defaultdict(int)
            for peer, meetings in metAtHourOfTheDay.items():
                temp[peer] = maxMode(meetings)
            self.metrics['metAtHourOfTheWeek'] = temp
            temp = collections.defaultdict(int)
            for peer, meetings in metAtDayOfTheWeek.items():
                temp[peer] = maxMode(meetings)
            self.metrics['metAtDayOfTheWeek'] = temp
            temp = collections.defaultdict(int)
            for peer, meetings in metAtHourOfTheWeek.items():
                temp[peer] = maxMode(meetings)
            self.metrics['metAtHourOfTheDay'] = temp

            ''' Social features '''
            # If the time difference between two bluetooth observations is less
            # than 601 seconds regard the interaction as continous
            for peer, times in timeSpentWith.items():
                for v in Aux.difference(times):
                    if v < 601:
                        timeSpentWithDict[peer] -= v
            self.metrics['timeSpent'] = timeSpentWithDict

            # Count the average amount of people that are present when X and Y
                # meet
            interactionAtTime = collections.defaultdict(set)
            amountOfOtherPeopleAtInteraction = collections.defaultdict(list)
            for blue in self.blues:
                # Bin all interactions into 10 minute buckets
                # Consider wider buckets of spatial triadic closure isn't
                # working very well
                time = int(blue.time / 600) * 600
                peer = str(blue.peer)
                interactionAtTime[time].add(peer)

            for time, interaction in interactionAtTime.items():
                for peer in interaction:
                    amountOfOtherPeopleAtInteraction[peer].append(len(interaction))

            # Average the amount of other people
            averageAmountOfOtherPeople = collections.defaultdict(int)
            for peer, people in amountOfOtherPeopleAtInteraction.items():
                averageAmountOfOtherPeople[peer] = statistics.mean(people)

            self.metrics['numberOfPeople'] = averageAmountOfOtherPeople

            # Spatial-triadic closure
            candidatesSpatialTriadicClosureDict = collections.defaultdict(int)
            spatialTriadicClosureDict = collections.defaultdict(int)
            for time, peers in interactionAtTime.items():
                for p in self.candidatesTriadicClosure(peers):
                    candidatesSpatialTriadicClosureDict[p] += 1
                for p in self.triadicClosure(peers):
                    spatialTriadicClosureDict[p] += 1

            self.metrics['spatialTriadicClosure'] = spatialTriadicClosureDict
            self.metrics['candidatesSpatialTriadicClosure'] =\
                candidatesSpatialTriadicClosureDict

            ''' Create the place features - context, relative importance '''
            metAtHome = collections.defaultdict(int)
            metAtUniversity = collections.defaultdict(int)
            metAtThirdPlace = collections.defaultdict(int)
            typeDict = collections.defaultdict(list)
            importanceDict = self.relativePersonalImportance()
            relativeImportance = collections.defaultdict(list)

            if self.localizedBlues is not None:
                for i, blue in self.localizedBlues.items():
                    con = list(self.stopLocs.values())[i][1]
                    slLabel = list(self.stopLocs.values())[i][2]
                    peers = blue.keys()
                    for peer, bs in blue.items():
                        ''' Place features '''
                        # Counts how often a person meets a peer in which
                        # context
                        typeDict[str(peer)].append(con)
                        # Create a distribution of the relative importance of
                        # venues where people meet
                        locImportance = importanceDict[slLabel]
                        relativeImportance[str(peer)].append(locImportance)

            ''' Context features '''
            for peer, values in typeDict.items():
                tempDict = collections.Counter(values)
                for con, amount in tempDict.items():
                    if con == 'home':
                        metAtHome[str(peer)] += amount
                    elif con == 'university':
                        metAtUniversity[str(peer)] += amount
                    elif con == 'other':
                        metAtThirdPlace[str(peer)] += amount
                    elif con is None:
                        metAtThirdPlace[str(peer)] += amount
            self.metrics['metAtHome'] = metAtHome
            self.metrics['metAtUniversity'] = metAtUniversity
            self.metrics['metAtThirdPlace'] = metAtThirdPlace
            # self.metrics['relativeImportance'] = importanceDict

            ''' Relative importance features '''
            tempDict = collections.defaultdict(float)
            for peer, values in relativeImportance.items():
                tempDict[str(peer)] = statistics.mean(values)
            self.metrics['relativeImportance'] = tempDict

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

    def candidatesTriadicClosure(self, peers):
        friends = set(self.friends[self.user])
        friendFriends = set(self.friendFriends[self.user])
        peers = set(peers)
        presentFriends = peers & friends
        presentFriendFriends = peers & friendFriends
        # The difference between an empty set and a non-empty set is always
        # the empty set
        # TO DO: FIGURE OUT IF THAT IS WHAT YOU WANT OR NOT
        return presentFriends - presentFriendFriends

    def triadicClosure(self, peers):
        friends = set(self.friends[self.user])
        friendFriends = set(self.friendFriends[self.user])
        peers = set(peers)
        presentFriends = peers & friends
        presentFriendFriends = peers & friendFriends
        return presentFriends & presentFriendFriends


class PlaceMetrics(object):

    def placeEntropy():
        pass
