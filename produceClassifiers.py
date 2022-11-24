import pandas as pd
from PreprocessingND import Preprocessing
from sklearn.model_selection import StratifiedKFold
from N_1 import N_1Classifier
from makePredictions import makePredictions
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tkinter import *
from matplotlib.figure import Figure
from decimal import Decimal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statistics import mean
import os
from MC import MCTS
import warnings

warnings.filterwarnings('ignore')
import timeit


class Retrieve:
    def __init__(self, dataset, kFold, mValues, jaccardValues, iterationsNumber, minimumSupport):
        self.dataset = dataset
        self.kFold = kFold
        self.mValues = mValues
        self.jValues = jaccardValues
        self.iterationsNumber = iterationsNumber
        self.minimumSupport = minimumSupport

    def sortDataset(self, ds):
        # sort the original
        ds = ds.reindex(sorted(ds.columns), axis=1)
        for i in ds.columns:
            # Make sure the class column is the last column:
            if i.lower() == 'class':
                if ds.columns.get_loc(i) == ds.columns.size - 1:
                    pass
                else:
                    column = ds[[i]]
                    ds.drop(columns=ds.columns[ds.columns.get_loc(i)], axis=1, inplace=True)
                    ds = ds.join(column)
        return ds

    def produce(self):
        # the name of ds:
        name, extension = os.path.splitext(self.dataset)
        name = os.path.basename(name)

        # read a csv file:
        df = pd.read_csv(self.dataset)
        # apply the pre-processing phase:
        obj1 = Preprocessing(df)
        # df is clean now:
        df, original = obj1.run()

        # the plot settings:
        root = Tk()
        figure = Figure(figsize=(10, 8), dpi=100)
        plot = figure.add_subplot(1, 1, 1)

        # Starting with the Jaccard classifiers:

        # max1: the JS max accuracy got:
        max1_Acc = 0
        max1_mValue = 0
        max1_Jaccard_value = 0
        max1_var = 0

        # max2: the SC max accuracy got:
        max2_acc = 0
        max2_mValue = 0
        max2_var = 0

        with open("C:\\Users\\moham\Downloads\\results", 'a') as fle:
            Jcounter = 1
            for jValue in self.jValues:
                print(f"For Jaccard Value = {jValue}")
                fle.write(f"For Jaccard Value = {jValue} \n")
                scorePlot = []

                for mValue in self.mValues:
                    print(f"For m = {mValue}")  # each jValue will produce m classifiers.
                    fle.write(f"For m = {mValue} \n")

                    # Apply the K-fold cross validation:
                    strtkfold = StratifiedKFold(n_splits=self.kFold, shuffle=True, random_state=1)
                    X = df.loc[:, df.columns != df.columns[-1]]
                    y = df.iloc[:, -1]

                    kfold = strtkfold.split(X, y)
                    scores = []
                    scores22 = []
                    counter = []
                    start = timeit.default_timer()

                    for k, (train, test) in enumerate(kfold):
                        trainData = df.iloc[train]
                        trainData = trainData.reset_index(drop=True)

                        testData = df.iloc[test]
                        testData = testData.reset_index(drop=True)

                        X_test = testData.loc[:, testData.columns != testData.columns[-1]]
                        y_test = testData.iloc[:, -1].to_numpy()

                        # start:
                        labels = pd.unique(trainData.iloc[:, -1].values.ravel())

                        # Generate a pool of patterns for each class:
                        if original is None:
                            obj2 = MCTS(trainingSet=trainData, numIteration=self.iterationsNumber,
                                        minSupport=self.minimumSupport, mValue=mValue)
                            result = obj2.monteCarloND()
                            patternsPerClass = result[0]
                            counter.append(result[1])
                            print(counter)

                        else:
                            obj2 = MCTS(trainingSet=trainData, numIteration=self.iterationsNumber,
                                        minSupport=self.minimumSupport, mValue=mValue)
                            patternsPerClass = obj2.monteCarloCD()

                        # produce a Jaccard similarity-based Classifier:

                        obj3 = N_1Classifier(trainingData=trainData, mValue=mValue, patternsPerClass=patternsPerClass,
                                             jaccardValue=jValue, name=name, original=original)
                        jsClassifier = obj3.run1()
                        # modified added:
                        newadded_scClassifier = obj3.run2()

                        # make the predictions:
                        if original is None:
                            obj4 = makePredictions(labels, jsClassifier, X_test)
                            predictedValues = obj4.predict()
                            # new added:
                            if Jcounter == len(self.jValues):
                                obj5 = makePredictions(labels, newadded_scClassifier, X_test)
                                modified_predictedValues = obj5.predict()

                        else:
                            obj4 = makePredictions(labels, jsClassifier, X_test)
                            predictedValues = obj4.predictCD()

                        # print(classification_report(y_test, predictedValues))
                        score = accuracy_score(y_test, predictedValues)
                        if Jcounter == len(self.jValues):
                            score2 = accuracy_score(y_test, modified_predictedValues)

                        print('Fold: %2d, JS Accuracy: %.3f' % (k + 1, score))
                        fle.write('Fold: %2d, JS Accuracy: %.3f' % (k + 1, score))
                        fle.write("\n")

                        if Jcounter == len(self.jValues):
                            print('Fold: %2d, SC Accuracy: %.3f' % (k + 1, score2))
                            fle.write('Fold: %2d, SC Accuracy: %.3f' % (k + 1, score2))
                            fle.write("\n")


                        print(
                            '-------------------------------------------------------------------------------------------')
                        fle.write("\n")
                        fle.write(
                            '-------------------------------------------------------------------------------------------')
                        fle.write("\n")


                        scores.append(score)
                        if Jcounter == len(self.jValues):
                            scores22.append(score2)

                    stop = timeit.default_timer()


                    fle.write(f"\n\nJS Scores: {scores}")
                    fle.write("\n")

                    if Jcounter == len(self.jValues):
                        fle.write(f"SC Scores: {scores22}")
                        fle.write("\n")

                    print('\n\nJS Cross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
                    fle.write('\n\nJS Cross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
                    fle.write("\n")


                    print('SC Cross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores22), np.std(scores22)))
                    fle.write('SC Cross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores22), np.std(scores22)))
                    fle.write("\n")

                    print(f'The average generated unique patterns : {np.mean(counter)}')
                    fle.write(f'The average generated unique patterns : {np.mean(counter)}')
                    fle.write("\n")

                    print('Time: ', stop - start)
                    fle.write(f'Time: {stop - start}')
                    fle.write("\n")


                    print('-------------------------------------------------------------------------------------------')
                    fle.write('-------------------------------------------------------------------------------------------')
                    fle.write("\n")

                Jcounter += 1

                '''
                    scorePlot.append(np.mean(scores))
    
                plot.plot(self.mValues, scorePlot, marker="X", label=f'JS with J={jValue}', linestyle=':')
                for i, j in zip(self.mValues, scorePlot):
                    plot.annotate(round(Decimal(str(j)), 4), xy=(i, j))
                plot.set_xscale('log', nonpositive='clip')
    
                if max1_Acc < mean(scorePlot):
                    max1_Acc = mean(scorePlot)
                    max1_mValue = self.mValues[scorePlot.index(max(scorePlot))]
                    max1_Jaccard_value = jValue
                    max1_var = np.var(scorePlot)
    
            print("The best result using JS")
            print(f"Mean Accuracy = {max1_Acc}, Corresponding m-value: {max1_mValue}, Corresponding Jaccard value: "
                  f"{max1_Jaccard_value}, variance = {max1_var}")
    
            # For the SC classifiers:
            print('----------------------------Separate and Conquer Approach----------------------------------')
            scorePlot2 = []
            for mValue in self.mValues:
                print(f"For m = {mValue}")
    
                # Apply the K-fold cross validation:
                strtkfold = StratifiedKFold(n_splits=self.kFold, shuffle=True, random_state=1)
                X = df.loc[:, df.columns != df.columns[-1]]
                y = df.iloc[:, -1]
                kfold = strtkfold.split(X, y)
    
                scores2 = []
    
                start = timeit.default_timer()
                counter = []
    
                for k, (train, test) in enumerate(kfold):
                    trainData = df.iloc[train]
                    trainData = trainData.reset_index(drop=True)
    
                    testData = df.iloc[test]
                    testData = testData.reset_index(drop=True)
    
                    X_test = testData.loc[:, testData.columns != testData.columns[-1]]
                    y_test = testData.iloc[:, -1].to_numpy()
    
                    # start:
                    labels = pd.unique(trainData.iloc[:, -1].values.ravel())
    
                    # Generate a pool of patterns for each class:
    
                    if original is None:
                        obj2 = MCTS(trainingSet=trainData,
                                    numIteration=self.iterationsNumber, minSupport=self.minimumSupport, mValue=mValue)
                        result = obj2.monteCarloND()
                        patternsPerClass = result[0]
                        counter.append(result[1])
                        print(counter)
    
                    else:
                        obj2 = MCTS(trainingSet=trainData,
                                    numIteration=self.iterationsNumber, minSupport=self.minimumSupport, mValue=mValue)
                        patternsPerClass = obj2.monteCarloCD()
    
    
                    # produce a SC-based Classifier:
                    obj3 = N_1Classifier(trainingData=trainData, mValue=mValue, patternsPerClass=patternsPerClass,
                                         jaccardValue=None, name=name, original=original)
                    scClassifier = obj3.run2()
    
                    # make the predictions:
                    if original is None:
                        obj4 = makePredictions(labels, scClassifier, X_test)
                        predictedValues = obj4.predict()
                    else:
                        obj4 = makePredictions(labels, scClassifier, X_test)
                        predictedValues = obj4.predictCD()
    
                    #print(classification_report(y_test, predictedValues))
    
                    score = accuracy_score(y_test, predictedValues)
    
                    print('Fold: %2d, Accuracy: %.3f' % (k + 1, score))
                    scores2.append(score)
                    print('-------------------------------------------------------------------------------------------')
    
                stop = timeit.default_timer()
                print('\n\nCross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores2), np.std(scores2)))
                print(f'The average generated unique patterns : {np.mean(counter)}')
                print('Time: ', stop - start)
                print('-------------------------------------------------------------------------------------------')
    
                scorePlot2.append(np.mean(scores2))
    
            plot.plot(self.mValues, scorePlot2, marker="X", label='S&C method')
            for i, j in zip(self.mValues, scorePlot2):
                plot.annotate(round(Decimal(str(j)), 4), xy=(i, j))
            plot.set_xscale('log', nonpositive='clip')
    
            max2_acc = mean(scorePlot2)
            max2_mValue = self.mValues[scorePlot2.index(max(scorePlot2))]
            max2_var = np.var(scorePlot2)
    
            print("The best result using SC")
            print(f"Mean Accuracy = {max2_acc}, Corresponding m-value: {max2_mValue}, Variance ={max2_var}")
    
    
            
    
            canvas = FigureCanvasTkAgg(figure, root)
            canvas.get_tk_widget().grid(row=0, column=0)
            plot.grid(True)
            plot.set_xlabel("M-estimate Value")
            plot.set_ylabel(f"Average Accuracy After ({self.kFold})-fold Cross Validation")
            plot.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            figure.tight_layout()
            figure.savefig(f'{name}.png')
    
            print('------------------------------------ Final Result ------------------------------------------')
    
            # after selecting the best parameters, we need to use them to generate the classifier:
            def continue_u():
    
                if max1_Acc > max2_acc or (max1_Acc == max2_acc and max1_var < max2_var):
                    print('Jaccard Similarity wins!')
                    print(f"Selected M-Value: {max1_mValue}")
                    print(f'Selected Jaccard values: {max1_Jaccard_value}')
                    # Generate a pool of patterns but based on the whole training data and the selected mValue:
                    obj = MCTS(trainingSet=df,
                               numIteration=self.iterationsNumber, minSupport=self.minimumSupport, mValue=max1_mValue)
                    result = obj2.monteCarloND()
                    patternsPerClas = result[0]
                    # produce a Jaccard similarity-based Classifier:
                    obj22 = N_1Classifier(trainingData=df, mValue=max1_mValue, patternsPerClass=patternsPerClas,
                                          jaccardValue=max1_Jaccard_value, name=name, original=original)
                    obj22.run1()
    
                elif max2_acc > max1_Acc or (max1_Acc == max2_acc and max2_var < max1_var):
                    print('Separate and Conquer Similarity wins!')
                    print(f"Selected M-Value: {max2_mValue}")
                    # Generate a pool of patterns but based on the whole training data and the selected mValue:
                    obj = MCTS(trainingSet=df,
                               numIteration=self.iterationsNumber, minSupport=self.minimumSupport, mValue=max2_mValue)
                    result = obj2.monteCarloND()
                    patternsPerClas = result[0]
    
                    # produce a SC-based Classifier:
                    obj33 = N_1Classifier(trainingData=df, mValue=max2_mValue, patternsPerClass=patternsPerClas,
                                          jaccardValue=None, name=name, original=original)
                    obj33.run2()
    
                quit()
    
            root.after(0, continue_u)
            root.mainloop()
            '''
