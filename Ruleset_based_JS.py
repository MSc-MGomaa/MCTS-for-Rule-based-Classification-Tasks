import itertools
from Extent import Calculate


class JS:
    def __init__(self, dataframe, all_patterns, sita, root, label, M_value, minimum_covered):
        self.df = dataframe
        self.All_patterns = all_patterns
        self.sita = float(sita)
        self.root = root
        self.label = label
        self.m_estimate = M_value
        self.minimum_covered = minimum_covered

    def Indices_CL(self):
        Indices_of_current_label = []
        last_column = self.df.iloc[:, -1].tolist()
        for i in range(len(last_column)):
            if last_column[i] == self.label:
                Indices_of_current_label.append(i)

        return Indices_of_current_label

    def Diff(self, li1, li2):
        return list(set(li1) - set(li2))

    # Jaccard Similarity to handle Numeric_Datasets:
    def jaccard_similarity_ND(self, parent, child):
        obj1 = Calculate(dataset=self.df, pattern=parent, root=self.root, label=self.label, mValue=self.m_estimate)
        obj2 = Calculate(dataset=self.df, pattern=child, root=self.root, label=self.label, mValue=self.m_estimate)
        samples_covered_by_parent = obj1.extentND()[0]
        samples_covered_by_child = obj2.extentND()[0]
        intersection = list(set(samples_covered_by_parent) & set(samples_covered_by_child))
        union = list(set().union(samples_covered_by_parent, samples_covered_by_child))
        similarity = len(intersection) / len(union)

        return similarity

    # Jaccard Similarity to handle Categorical_Datasets:
    def jaccard_similarity_CD(self, parent, child):
        obj1 = Calculate(dataset=self.df, pattern=parent, root=self.root, label=self.label, mValue=self.m_estimate)
        obj2 = Calculate(dataset=self.df, pattern=child, root=self.root, label=self.label, mValue=self.m_estimate)
        samples_covered_by_parent = obj1.extentCD()[2]
        samples_covered_by_child = obj2.extentCD()[2]
        intersection = list(set(samples_covered_by_parent) & set(samples_covered_by_child))
        union = list(set().union(samples_covered_by_parent, samples_covered_by_child))
        try:
            similarity = len(intersection) / len(union)
        except ZeroDivisionError:
            similarity = 1

        return similarity

    def Sorting_PatternsND(self):
        lst = []
        for pattern in self.All_patterns:
            obj = Calculate(dataset=self.df, root=self.root, pattern=pattern, label=self.label, mValue=self.m_estimate)
            lst.append(float(obj.mEstimateND()))

        A = [x for _, x in sorted(zip(lst, self.All_patterns), reverse=True)]
        A = list(k for k, _ in itertools.groupby(A))
        return A

    def Sorting_PatternsCD(self):
        lst = []
        for pattern in self.All_patterns:
            obj = Calculate(dataset=self.df, root=self.root, pattern=pattern, label=self.label, mValue=self.m_estimate)

            lst.append(float(obj.mEstimateCD()))

        # sorted patterns with respect to the quality measure:
        A = [x for _, x in sorted(zip(lst, self.All_patterns), reverse=True)]
        # remove the redundant patterns:
        A = list(k for k, _ in itertools.groupby(A))

        return A

    # ----------------------------------------------------------------------------------------
    # NUMERIC DATASETS
    def Jaccard_Sim_ND(self):
        Indices_of_current_label = self.Indices_CL()
        A = self.Sorting_PatternsND()
        Intermediate_Pattern_Set = [A[0]]
        Intermediate_Pattern_Set_Info = []
        covered_indices = []
        all_covered_falses = []
        counter = 0
        # starting with the best pattern:
        obj = Calculate(dataset=self.df, pattern=A[0], root=self.root, label=self.label, mValue=self.m_estimate)
        extent_of_best = obj.extentND()[0]
        UT = list(set(extent_of_best).intersection(Indices_of_current_label))
        UF = self.Diff(extent_of_best, Indices_of_current_label)
        # new added
        all_covered_falses.extend(UF)
        Intermediate_Pattern_Set_Info.append([UT, UF])

        covered_indices.extend(UT)
        un_covered = self.Diff(Indices_of_current_label, covered_indices)

        # while True continue, break when False.
        break_while = False
        while len(un_covered) != 0:
            # Searching for the next un_similar pattern:

            for next_rule in A[A.index(Intermediate_Pattern_Set[counter]) + 1:]:
                # Next_un_similar_Rule
                if (self.jaccard_similarity_ND(Intermediate_Pattern_Set[counter], next_rule)) < self.sita:
                    # It should cover NEW_uncovered samples:
                    obj2 = Calculate(dataset=self.df, pattern=next_rule, root=self.root, label=self.label,
                                     mValue=self.m_estimate)
                    extent = obj2.extentND()[0]
                    intersection = list(set(extent).intersection(un_covered))
                    if len(intersection) > 0:
                        intersection = list(set(extent).intersection(Indices_of_current_label))
                        U_TP = self.Diff(intersection, covered_indices)
                        # ----------------------- The false examples -------------------------
                        falses = self.Diff(extent, Indices_of_current_label)
                        U_FP = self.Diff(falses, all_covered_falses)
                        all_covered_falses.extend(U_FP)
                        all_covered_falses = [i for j, i in enumerate(all_covered_falses) if
                                              i not in all_covered_falses[:j]]
                        # --------------------------------------------------------------------------------------------
                        # remove the redundant elements
                        covered_indices.extend(extent)
                        covered_indices = [i for j, i in enumerate(covered_indices) if i not in covered_indices[:j]]
                        # --------------------------------------------------------------------------------------------
                        Intermediate_Pattern_Set.append(next_rule)
                        Intermediate_Pattern_Set_Info.append([U_TP, U_FP])
                        break

                    if A.index(next_rule) + 1 == len(A):
                        break_while = True
                        break
                    else:
                        pass

                if A.index(next_rule) + 1 == len(A):
                    break_while = True
                    break

                else:
                    pass

            if break_while:
                break

            else:
                pass

            counter += 1
            un_covered = self.Diff(Indices_of_current_label, covered_indices)

        dropped_indices = []

        for i in range(len(Intermediate_Pattern_Set_Info)):
            # in case the + is less than the desired number, or + less than the - or even equal.
            if len(Intermediate_Pattern_Set_Info[i][0]) < self.minimum_covered \
                    or len(Intermediate_Pattern_Set_Info[i][0]) < len(Intermediate_Pattern_Set_Info[i][1]) or \
                    len(Intermediate_Pattern_Set_Info[i][0]) == len(Intermediate_Pattern_Set_Info[i][1]) or \
                    len(Intermediate_Pattern_Set_Info[i][0]) == 1:
                dropped_indices.append(i)
            else:
                pass

        Pattern_Set = []
        Pattern_Set_Info = []

        for index in range(len(Intermediate_Pattern_Set)):
            if index not in dropped_indices:
                Pattern_Set.append(Intermediate_Pattern_Set[index])
                Pattern_Set_Info.append(Intermediate_Pattern_Set_Info[index])
            else:
                pass

        return [Pattern_Set, Pattern_Set_Info]

    # ------------------------------------------------------------------------------------------------------------

    # CATEGORICAL DATASETS
    def Jaccard_Sim_CD(self):
        Indices_of_current_label = self.Indices_CL()
        A = self.Sorting_PatternsCD()
        Intermediate_Pattern_Set = [A[0]]
        Intermediate_Pattern_Set_Info = []
        covered_indices = []
        all_covered_falses = []
        counter = 0
        # starting with the best pattern:
        obj = Calculate(dataset=self.df, pattern=A[0], root=self.root, label=self.label, mValue=self.m_estimate)
        extent_of_best = obj.extentCD()[0]
        UT = list(set(extent_of_best).intersection(Indices_of_current_label))
        UF = self.Diff(extent_of_best, Indices_of_current_label)
        Intermediate_Pattern_Set_Info.append([UT, UF])

        covered_indices.extend(UT)
        un_covered = self.Diff(Indices_of_current_label, covered_indices)

        # while True continue, when it becomes False, break.
        while len(un_covered) != 0:
            # Searching for the next un_similar pattern:
            try:
                for next_rule in A[A.index(Intermediate_Pattern_Set[counter]) + 1:]:
                    # Next_un_similar_Rule
                    if (self.jaccard_similarity_CD(Intermediate_Pattern_Set[counter], next_rule)) < self.sita:
                        # It should cover NEW_uncovered samples:
                        obj2 = Calculate(dataset=self.df, pattern=next_rule, root=self.root, label=self.label,
                                         mValue=self.m_estimate)
                        extent = obj2.extentCD()[0]
                        intersection = list(set(extent).intersection(un_covered))
                        if len(intersection) > 0:
                            intersection = list(set(extent).intersection(Indices_of_current_label))
                            U_TP = self.Diff(intersection, covered_indices)
                            # ----------------------- The false examples -------------------------
                            falses = self.Diff(extent, Indices_of_current_label)
                            U_FP = self.Diff(falses, all_covered_falses)
                            all_covered_falses.extend(U_FP)
                            all_covered_falses = [i for j, i in enumerate(all_covered_falses) if
                                                  i not in all_covered_falses[:j]]
                            # --------------------------------------------------------------------------------------------
                            # remove the redundant elements
                            covered_indices.extend(extent)
                            covered_indices = [i for j, i in enumerate(covered_indices) if i not in covered_indices[:j]]
                            # --------------------------------------------------------------------------------------------
                            Intermediate_Pattern_Set.append(next_rule)
                            Intermediate_Pattern_Set_Info.append([U_TP, U_FP])
                            break

                        else:
                            pass
                    else:
                        pass

            except IndexError:
                break

            counter += 1
            un_covered = self.Diff(Indices_of_current_label, covered_indices)

        dropped_indices = []
        for i in range(len(Intermediate_Pattern_Set_Info)):
            # in case the + is less than the desired number, or + less than the - or even equal.
            if len(Intermediate_Pattern_Set_Info[i][0]) < self.minimum_covered \
                    or len(Intermediate_Pattern_Set_Info[i][0]) < len(Intermediate_Pattern_Set_Info[i][1]) or \
                    len(Intermediate_Pattern_Set_Info[i][0]) == len(Intermediate_Pattern_Set_Info[i][1]):
                dropped_indices.append(i)
            else:
                pass
        Pattern_Set = []
        Pattern_Set_Info = []

        for index in range(len(Intermediate_Pattern_Set)):
            if index not in dropped_indices:
                Pattern_Set.append(Intermediate_Pattern_Set[index])
                Pattern_Set_Info.append(Intermediate_Pattern_Set_Info[index])
            else:
                pass

        return [Pattern_Set, Pattern_Set_Info]
