from gs.dictAux import DefaultOrderedDict
from predictLinks import RandomForestLinkPrediction as rf
from nullModel import NullModel

import os
import pytz
import statistics

amsterdam = pytz.timezone('Europe/Amsterdam')


def scoreModels(model, dictionaries, key):
    acc = model.acc
    prec = model.precision
    rec = model.recall
    roc_macro = model.calculateROC()[2]['macro']
    roc_micro = model.calculateROC()[2]['micro']
    pr_macro = model.calculatePR()[2]['macro']
    pr_micro = model.calculatePR()[2]['micro']
    scores = acc, prec, rec, roc_macro, roc_micro, pr_macro, pr_micro
    for d, score in zip(dictionaries, scores):
        d[key].append(score)


def scoreNullModel(dictionaries, NMacc, NMprec, NMrec, NMroc_auc, NMpr_auc):
    acc = statistics.mean(NMacc)
    prec = statistics.mean(NMprec)
    rec = statistics.mean(NMrec)
    roc_macro = NMroc_auc['macro']
    roc_micro = NMroc_auc['micro']
    pr_macro = NMpr_auc['macro']
    pr_micro = NMpr_auc['micro']
    scores = acc, prec, rec, roc_macro, roc_micro, pr_macro, pr_micro
    for d, score in zip(dictionaries, scores):
        d['null'].append(score)


def createResultsDictionaries():
    accs = DefaultOrderedDict(list)
    precs = DefaultOrderedDict(list)
    recs = DefaultOrderedDict(list)
    roc_macros = DefaultOrderedDict(list)
    roc_micros = DefaultOrderedDict(list)
    pr_macros = DefaultOrderedDict(list)
    pr_micros = DefaultOrderedDict(list)
    results = [accs, precs, recs, roc_macros, roc_micros,
               pr_macros, pr_micros]
    return results


class Predictions(object):
    def __init__(self, training_truths, training_examples, examples,
                 actuals, src_dest_nodes, results, featureImportance,
                 networkT0f, networkT1f, weighted, n_jobs, listOfFeatures,
                 classesSet, candidates, allUsers, outputPath, **kwargs):

        self.training_truths = training_truths
        self.training_examples = training_examples
        self.examples = examples
        self.actuals = actuals
        self.src_dest_nodes = src_dest_nodes
        self.results = results
        self.featureImportance = featureImportance
        self.networkT0f = networkT0f
        self.networkT1f = networkT1f
        self.weighted = weighted
        self.classesSet = classesSet
        self.candidate = candidates
        self.allUsers = allUsers
        self.n_jobs = n_jobs

    def runNullModel(self):
        NMaccs = []
        NMprecs = []
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        NMrecs = []
        NM = NullModel(
            self.networkT0f, self.networkT1f, self.classesSet, self.allUsers)
        predictions = NM.predictions(1)
        p = NM.probabilities()
        probabilities = NM.probabilityArray(p)
        _, _, NMroc_auc = NM.roc(probabilities)
        _, _, NMpr_auc = NM.pr(probabilities)

        for prediction in predictions:
            NMacc = NM.acc(prediction)
            NMaccs.append(NMacc)
            NMprec = NM.prec(prediction)
            NMprecs.append(NMprec)
            NMrec = NM.rec(prediction)
            NMrecs.append(NMrec)

        scoreNullModel(self.results, NMaccs, NMprecs, NMrecs,
                       NMroc_auc, NMpr_auc)

    def run(self, modelName, modelFeatures):
        ''' Run the models and score them '''
        model = rf(self.training_truths, self.training_examples,
                   self.examples, self.actuals,
                   modelFeatures, self.src_dest_nodes, n_jobs=self.n_jobs)
        scoreModels(model, self.results, modelName)
        self.featureImportance[modelName].append(model.importances)

    def getScores(self):
        return self.results

    def getFeatureImportance(self):
        return self.featureImportance

    def export(self, timestep, testSetNames, dictOfFeatures):
        ''' Ouput section '''
        if not os.path.isfile(self.outputPath + 'modelTimeseriesData.ssv'):
            with open(self.outputPath + 'modelTimeseriesData.ssv', 'w') as f:
                header = ' '.join(['timepoint', 'model',
                                   'testSet', 'accuracy',
                                   'precision', 'recall',
                                   'roc_macro', 'roc_micro',
                                   'pr_macro', 'pr_micro']) + '\n'
                f.write(header)

        with open(self.outputPath + 'modelTimeseriesData.ssv', 'a') as f:
            print('Timestep: ', timestep)
            for key, accs in self.results[0].items():
                precs = self.results[1][key]
                recs = self.results[2][key]
                roc_macros = self.results[3][key]
                roc_micros = self.results[4][key]
                pr_macros = self.results[5][key]
                pr_micros = self.results[6][key]

                for testSet, acc, prec, rec, roc_mac,\
                    roc_mic, pr_mac, pr_mic in zip(
                        testSetNames, accs, precs, recs, roc_macros, roc_micros,
                        pr_macros, pr_micros):
                    row = ' '.join([str(timestep), str(key),
                                    str(testSet), str(acc),
                                    str(prec), str(rec),
                                    str(roc_mac), str(roc_mic),
                                    str(pr_mac), str(pr_mic)]) + '\n'
                    f.write(row)
                print(key, ': ', statistics.mean(roc_macros))
            print()

        if not os.path.isfile(self.outputPath + 'featureImportance.ssv'):
            with open(self.outputPath + 'featureImportance.ssv', 'w') as f:
                header = ' '.join(['timepoint', 'model',
                                   'testSet', 'feature',
                                   'importance']) + '\n'
                f.write(header)

        with open(self.outputPath + 'featureImportance.ssv', 'a') as f:
            for key, values in self.featureImportance.items():
                for testSet, importances in zip(testSetNames, values):
                    for feature, importance in zip(dictOfFeatures[key],
                                                   importances):
                        row = ' '.join([str(timestep), str(key),
                                        str(testSet), str(feature),
                                        str(importance)]) + '\n'
                        f.write(row)
