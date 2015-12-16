from gs.dictAux import DefaultOrderedDict
from predictLinks import RandomForestLinkPrediction as rf
from nullModel import NullModel

import os
import pytz
import statistics

amsterdam = pytz.timezone('Europe/Amsterdam')


def scoreModels(model, dictionaries, key):
    acc = model.acc
    precMacro = model.precision('macro')
    precMicro = model.precision('micro')
    recMacro = model.recall('macro')
    recMicro = model.recall('micro')
    roc_macro = model.calculateROC()[2]['macro']
    roc_micro = model.calculateROC()[2]['micro']
    pr_macro = model.calculatePR()[2]['macro']
    pr_micro = model.calculatePR()[2]['micro']
    scores = acc, precMacro, precMicro, recMacro, recMicro,\
        roc_macro, roc_micro, pr_macro, pr_micro
    for d, score in zip(dictionaries, scores):
        d[key].append(score)


def scoreNullModel(dictionaries, NMacc, NMprecMacro, NMprecMicro,
                   NMrecMacro, NMrecMicro, NMroc_auc, NMpr_auc):
    acc = statistics.mean(NMacc)
    precMacro = statistics.mean(NMprecMacro)
    recMacro = statistics.mean(NMrecMacro)
    precMicro = statistics.mean(NMprecMicro)
    recMicro = statistics.mean(NMrecMicro)
    roc_macro = NMroc_auc['macro']
    roc_micro = NMroc_auc['micro']
    pr_macro = NMpr_auc['macro']
    pr_micro = NMpr_auc['micro']
    scores = acc, precMacro, precMicro, recMacro, recMicro,\
        roc_macro, roc_micro, pr_macro, pr_micro
    for d, score in zip(dictionaries, scores):
        d['null'].append(score)


def createResultsDictionaries():
    accs = DefaultOrderedDict(list)
    precMacros = DefaultOrderedDict(list)
    precMicros = DefaultOrderedDict(list)
    recMacros = DefaultOrderedDict(list)
    recMicros = DefaultOrderedDict(list)
    roc_macros = DefaultOrderedDict(list)
    roc_micros = DefaultOrderedDict(list)
    pr_macros = DefaultOrderedDict(list)
    pr_micros = DefaultOrderedDict(list)
    results = [accs, precMacros, precMicros, recMacros, recMicros,
               roc_macros, roc_micros, pr_macros, pr_micros]
    return results


class Predictions(object):
    def __init__(self, X_train, y_train, X_test, y_test,
                 src_dest_nodes, results, featureImportance,
                 networkT0f, networkT1f, weighted, n_jobs,
                 model, timestep, classesSet, allUsers, outputPath, **kwargs):

        self.y_train = y_train
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.src_dest_nodes = src_dest_nodes
        self.results = results
        self.featureImportance = featureImportance
        self.networkT0f = networkT0f
        self.networkT1f = networkT1f
        self.weighted = weighted
        self.classesSet = classesSet
        self.classesList = sorted(list(classesSet))
        self.allUsers = allUsers
        self.n_jobs = n_jobs
        self.model = model
        self.timestep = timestep
        self.outputPath = outputPath
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def runNullModel(self):
        NMaccs = []
        NMprecMacros = []
        NMprecMicros = []
        NMrecMacros = []
        NMrecMicros = []
        NM = NullModel(
            self.networkT0f, self.networkT1f, self.classesSet, self.allUsers)
        predictions = NM.predictions(1)
        p = NM.probabilities()
        probabilities = NM.probabilityArray(p)
        # probabilities = NM.predictionsToProbability(predictions)
        tpr, fpr, NMroc_auc = NM.roc(probabilities)
        precision, recall, NMpr_auc = NM.pr(probabilities)

        for prediction in predictions:
            NMacc = NM.acc(prediction)
            NMaccs.append(NMacc)
            NMprecMacro = NM.prec(prediction, 'macro')
            NMprecMacros.append(NMprecMacro)
            NMrecMacro = NM.rec(prediction, 'macro')
            NMrecMacros.append(NMrecMacro)
            NMprecMicro = NM.prec(prediction, 'micro')
            NMprecMicros.append(NMprecMicro)
            NMrecMicro = NM.rec(prediction, 'micro')
            NMrecMicros.append(NMrecMicro)

        scoreNullModel(self.results, NMaccs, NMprecMacros, NMprecMicros,
                       NMrecMacros, NMrecMicros,
                       NMroc_auc, NMpr_auc)

    def run(self, modelName):
        model = rf(self.X_train, self.y_train,
                   self.X_test, self.y_test,
                   self.src_dest_nodes,
                   self.classesList, n_jobs=self.n_jobs)
        scoreModels(model, self.results, modelName)
        self.featureImportance[modelName].append(model.importances)

    def getScores(self):
        return self.results

    def getFeatureImportance(self):
        return self.featureImportance

    def export(self, fold):
        if not os.path.isfile(self.outputPath + 'modelTimeseriesData_' +
                              str(self.timestep[0]) + '-' +
                              str(self.timestep[1]) + '.ssv'):
            with open(self.outputPath + 'modelTimeseriesData.ssv', 'w') as f:
                header = ' '.join(['timepoint', 'model',
                                   'testSet', 'accuracy',
                                   'precision_macro', 'precision_micro',
                                   'recall_macro', 'recall_micro',
                                   'roc_macro', 'roc_micro',
                                   'pr_macro', 'pr_micro']) + '\n'
                f.write(header)

        with open(self.outputPath + 'modelTimeseriesData.ssv', 'a') as f:
            print('Exporting: ', self.timestep)
            for key, accs in self.results[0].items():
                prec_macros = self.results[1][key]
                prec_micros = self.results[2][key]
                rec_macros = self.results[3][key]
                rec_micros = self.results[4][key]
                roc_macros = self.results[5][key]
                roc_micros = self.results[6][key]
                pr_macros = self.results[7][key]
                pr_micros = self.results[8][key]

                for acc, prec_mac, prec_mic, rec_mac, rec_mic,\
                    roc_mac, roc_mic, pr_mac, pr_mic in zip(
                        accs, prec_macros, prec_micros, rec_macros,
                        rec_micros, roc_macros, roc_micros,
                        pr_macros, pr_micros):
                    row = ' '.join([str(self.timestep[0]), str(key),
                                    str(fold), str(acc),
                                    str(prec_mac), str(prec_mic),
                                    str(rec_mac), str(rec_mic),
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
                for importances in values:
                    for feature, importance in zip(self.model,
                                                   importances):
                        row = ' '.join([str(self.timestep[0]), str(key),
                                        str(fold), str(feature),
                                        str(importance)]) + '\n'
                        f.write(row)
