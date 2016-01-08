from gs.dictAux import DefaultOrderedDict
from predictLinks import RandomForestLinkPrediction as rf
from nullModel import NullModel

import predictLinks as pl

import numpy as np
import os
import pytz
import sklearn.metrics
import statistics

amsterdam = pytz.timezone('Europe/Amsterdam')


def scoreModels(model, dictionaries, key):
    acc = model.acc
    precWeight = model.precision('weighted')
    precMacro = model.precision('macro')
    precMicro = model.precision('micro')
    recWeight = model.recall('weighted')
    recMacro = model.recall('macro')
    recMicro = model.recall('micro')
    try:
        roc_auc = model.calculateROC()
        roc_weight = roc_auc[2]['weighted']
        roc_macro = roc_auc[2]['macro']
        roc_micro = roc_auc[2]['micro']
        roc_micro_0 = roc_auc[2]['micro-0']
        pr_auc = model.calculatePR()
        pr_weight = pr_auc[2]['weighted']
        pr_macro = pr_auc[2]['macro']
        pr_micro = pr_auc[2]['micro']
        pr_micro_0 = pr_auc[2]['micro-0']
    except ValueError:
        roc_weight = np.nan
        roc_macro = np.nan
        roc_micro = np.nan
        roc_micro_0 = np.nan
        pr_weight = np.nan
        pr_macro = np.nan
        pr_micro = np.nan
        pr_micro_0 = np.nan

    scores = acc, precWeight, precMacro, precMicro,\
        recWeight, recMacro, recMicro,\
        roc_weight, roc_macro, roc_micro, roc_micro_0,\
        pr_weight, pr_macro, pr_micro, pr_micro_0
    for d, score in zip(dictionaries, scores):
        d[key].append(score)


def createResultsDictionaries():
    accs = DefaultOrderedDict(list)
    precWeights = DefaultOrderedDict(list)
    precMacros = DefaultOrderedDict(list)
    precMicros = DefaultOrderedDict(list)
    recWeights = DefaultOrderedDict(list)
    recMacros = DefaultOrderedDict(list)
    recMicros = DefaultOrderedDict(list)
    roc_weights = DefaultOrderedDict(list)
    roc_macros = DefaultOrderedDict(list)
    roc_micros = DefaultOrderedDict(list)
    roc_micro_0s = DefaultOrderedDict(list)
    pr_weights = DefaultOrderedDict(list)
    pr_macros = DefaultOrderedDict(list)
    pr_micros = DefaultOrderedDict(list)
    pr_micro_0s = DefaultOrderedDict(list)
    results = [accs, precWeights, precMacros, precMicros, recWeights,
               recMacros, recMicros, roc_weights, roc_macros, roc_micros,
               roc_micro_0s, pr_weights, pr_macros, pr_micros, pr_micro_0s]
    return results


class Predictions(object):
    def __init__(self, X_train, y_train, X_test, y_test,
                 src_dest_nodes, results, featureImportance,
                 nullModelProbabilities, nullModelTruths,
                 nullModelStringActuals, nullModelPrediction,
                 weighted, n_jobs, model, classesSet,
                 outputPath, **kwargs):

        self.y_train = y_train
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.src_dest_nodes = src_dest_nodes
        self.results = results
        self.featureImportance = featureImportance
        self.nullModelTruths = nullModelTruths
        self.nullModelProbabilities = nullModelProbabilities
        self.nullModelStringActuals = nullModelStringActuals
        self.nullModelPrediction = nullModelPrediction
        self.weighted = weighted
        self.classesSet = classesSet
        self.classesList = sorted(list(classesSet))
        self.n_jobs = n_jobs
        self.model = model
        self.outputPath = outputPath
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def scoreNullModel(self):
        acc = sklearn.metrics.accuracy_score(
            self.nullModelStringActuals,
            self.nullModelPrediction)

        precWeight = sklearn.metrics.precision_score(
            self.nullModelStringActuals,
            self.nullModelPrediction,
            'weighted')
        precMacro = sklearn.metrics.precision_score(
            self.nullModelStringActuals,
            self.nullModelPrediction,
            'macro')
        precMicro = sklearn.metrics.precision_score(
            self.nullModelStringActuals,
            self.nullModelPrediction,
            'micro')
        recWeight = sklearn.metrics.recall_score(
            self.nullModelStringActuals,
            self.nullModelPrediction,
            'weighted')
        recMacro = sklearn.metrics.recall_score(
            self.nullModelStringActuals,
            self.nullModelPrediction,
            'macro')
        recMicro = sklearn.metrics.recall_score(
            self.nullModelStringActuals,
            self.nullModelPrediction,
            'micro')

        try:
            roc_auc = pl.roc(self.nullModelTruths, self.nullModelProbabilities,
                             self.classesList)
            roc_weight = roc_auc[2]['weighted']
            roc_macro = roc_auc[2]['macro']
            roc_micro = roc_auc[2]['micro']
            roc_micro_0 = roc_auc[2]['micro-0']
            pr_auc = pl.pr(self.nullModelTruths, self.nullModelProbabilities,
                           self.classesList)
            pr_weight = pr_auc[2]['weighted']
            pr_macro = pr_auc[2]['macro']
            pr_micro = pr_auc[2]['micro']
            pr_micro_0 = pr_auc[2]['micro-0']
        except ValueError:
            roc_weight = np.nan
            roc_macro = np.nan
            roc_micro = np.nan
            roc_micro_0 = np.nan
            pr_weight = np.nan
            pr_macro = np.nan
            pr_micro = np.nan
            pr_micro_0 = np.nan

        scores = acc, precWeight, precMacro, precMicro,\
            recWeight, recMacro, recMicro,\
            roc_weight, roc_macro, roc_micro, roc_micro_0,\
            pr_weight, pr_macro, pr_micro, pr_micro_0
        for d, score in zip(self.results, scores):
            d['null'].append(score)

    def run(self, modelName):
        model = rf(self.X_train, self.y_train,
                   self.X_test, self.y_test,
                   self.src_dest_nodes,
                   self.classesList, n_jobs=self.n_jobs)
        scoreModels(model, self.results, modelName)
        self.featureImportance[modelName].append(model.importances)

    def export(self, user):
        if not os.path.isfile(self.outputPath + 'modelData.ssv'):
            with open(self.outputPath + 'modelData.ssv', 'w') as f:
                header = ' '.join(['user', 'model',
                                   'accuracy', 'precision_weighted',
                                   'precision_macro', 'precision_micro',
                                   'recall_weighted',
                                   'recall_macro', 'recall_micro',
                                   'roc_weighted', 'roc_macro',
                                   'roc_micro', 'roc_micro-0',
                                   'pr_weighted', 'pr_macro',
                                   'pr_micro', 'pr_micro-0']) + '\n'
                f.write(header)

        with open(self.outputPath + 'modelData.ssv', 'a') as f:
            print('Exporting: ', user)
            for key, accs in self.results[0].items():
                prec_weights = self.results[1][key]
                prec_macros = self.results[2][key]
                prec_micros = self.results[3][key]
                rec_weights = self.results[4][key]
                rec_macros = self.results[5][key]
                rec_micros = self.results[6][key]
                roc_weights = self.results[7][key]
                roc_macros = self.results[8][key]
                roc_micros = self.results[9][key]
                roc_micro_0s = self.results[10][key]
                pr_weights = self.results[11][key]
                pr_macros = self.results[12][key]
                pr_micros = self.results[13][key]
                pr_micro_0s = self.results[14][key]

                for acc, prec_weight, prec_mac, prec_mic,\
                    rec_weight, rec_mac, rec_mic,\
                    roc_weight, roc_mac, roc_mic, roc_mic_0,\
                    pr_weight, pr_mac, pr_mic, pr_mic_0 in zip(
                        accs, prec_weights, prec_macros, prec_micros,
                        rec_weights, rec_macros, rec_micros,
                        roc_weights, roc_macros, roc_micros, roc_micro_0s,
                        pr_weights, pr_macros, pr_micros, pr_micro_0s
                        ):
                    row = ' '.join([str(user), str(key),
                                    str(acc),
                                    str(prec_weight), str(prec_mac),
                                    str(prec_mic),
                                    str(rec_weight), str(rec_mac), str(rec_mic),
                                    str(roc_weight), str(roc_mac),
                                    str(roc_mic), str(roc_mic_0),
                                    str(pr_weight), str(pr_mac),
                                    str(pr_mic), str(pr_mic_0)
                                    ]) + '\n'
                    f.write(row)

                roc_values = [v for v in roc_macros if not np.isnan(v)]
                pr_values = [v for v in pr_macros if not np.isnan(v)]
                pr_w_values = [v for v in pr_micro_0s if not np.isnan(v)]
                print(key, ': ', statistics.mean(roc_values))
                print(key, ': ', statistics.mean(pr_values))
                print(key, ': ', statistics.mean(pr_w_values))
            print()

        if not os.path.isfile(self.outputPath + 'featureImportance.ssv'):
            with open(self.outputPath + 'featureImportance.ssv', 'w') as f:
                header = ' '.join(['user', 'model',
                                   'testSet', 'feature',
                                   'importance']) + '\n'
                f.write(header)

        with open(self.outputPath + 'featureImportance.ssv', 'a') as f:
            for key, values in self.featureImportance.items():
                for importances in values:
                    for feature, importance in zip(self.model,
                                                   importances):
                        row = ' '.join([str(user), str(key),
                                        str(feature),
                                        str(importance)]) + '\n'
                        f.write(row)
