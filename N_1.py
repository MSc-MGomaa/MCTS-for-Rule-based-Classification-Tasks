from Ruleset_based_SC import SandC
from Ruleset_based_JS import JS

from displayResult import Show_Pattern_set


class N_1Classifier:
    def __init__(self, trainingData, original, mValue, patternsPerClass, jaccardValue, name):
        self.patternsPerClass = patternsPerClass
        self.trainingData = trainingData
        self.original = original
        self.mValue = mValue
        self.jValue = jaccardValue
        self.name = name

    def run1(self):
        # 1: generate a rule-set per class:
        allPatternSets_based_JS = []
        if self.original is None:
            for pool in self.patternsPerClass:
                obj = JS(dataframe=self.trainingData, all_patterns=pool[1], sita=self.mValue, root=pool[2],
                         label=pool[0], M_value=self.mValue, minimum_covered=1)
                allPatternSets_based_JS.append([pool[0], obj.Jaccard_Sim_ND()])
        else:
            for pool in self.patternsPerClass:
                obj = JS(dataframe=self.trainingData, all_patterns=pool[1], sita=self.mValue, root=pool[2],
                         label=pool[0], M_value=self.mValue, minimum_covered=1)
                allPatternSets_based_JS.append([pool[0], obj.Jaccard_Sim_CD()])

        # 2: retrieve the pattern_sets with the minimum covered negatives to represent the classifier:
        coveredNegatives = []
        for pool in allPatternSets_based_JS:
            N = 0
            for i in pool[1][1]:
                tup = []
                for j in i:
                    tup.append(len(j))
                N += tup[1]
            coveredNegatives.append(N)
        sorting = sorted(zip(coveredNegatives, allPatternSets_based_JS), reverse=False)[:len(coveredNegatives) - 1]

        classifier = []
        for i in sorting:
            classifier.append(i[1])

        # to print the classifier:
        if self.original is None:
            obj = Show_Pattern_set(df=self.trainingData, classifier=classifier, path=f'{self.name}.txt')
            #print("Jaccard-based Classifier")
            #print(obj.show_ND())
        else:
            obj = Show_Pattern_set(df=self.trainingData, classifier=classifier, path=f'{self.name}.txt')
            #print("Jaccard-based Classifier")
            #print(obj.show_CD())

        return classifier

    def run2(self):
        # 1: generate a rule-set per class:
        allPatternSets_based_SC = []

        if self.original is None:
            for pool in self.patternsPerClass:
                obj = SandC(Data=pool[1], dataset=self.trainingData, label=pool[0], root=pool[2], m_value=self.mValue,
                            minimum_covered=1)
                allPatternSets_based_SC.append([pool[0], obj.S_C_ND()])
        else:
            for pool in self.patternsPerClass:
                obj = SandC(Data=pool[1], dataset=self.trainingData, label=pool[0], root=pool[2], m_value=self.mValue,
                            minimum_covered=1)
                allPatternSets_based_SC.append([pool[0], obj.S_C_CD()])

        # 2: retrieve the pattern_sets with the minimum covered negatives to represent the classifier:
        coveredNegatives = []
        for pool in allPatternSets_based_SC:
            N = 0
            for i in pool[1][1]:
                tup = []
                for j in i:
                    tup.append(len(j))
                N += tup[1]
            coveredNegatives.append(N)
        sorting = sorted(zip(coveredNegatives, allPatternSets_based_SC), reverse=False)[:len(coveredNegatives) - 1]

        classifier = []
        for i in sorting:
            classifier.append(i[1])
        # The Best Classifier is represented as:
        # [pool[0],[[Pattern_Set], [Pattern_Set_Info]]]....

        # to print the classifier:
        if self.original is None:
            obj = Show_Pattern_set(df=self.trainingData, classifier=classifier, path=f'{self.name}.txt')
            #print("Separate and Conquer-based Classifier")
            #print(obj.show_ND())
        else:
            obj = Show_Pattern_set(df=self.trainingData, classifier=classifier, path=f'{self.name}.txt')
            #print("Separate and Conquer-based Classifier")
            #print(obj.show_CD())

        return classifier
