from Extent import Calculate
import itertools


class SandC:
    def __init__(self, Data, dataset, label, root, m_value, minimum_covered):
        self.P = Data
        self.df = dataset
        self.df_Const = dataset
        self.label = label
        self.root = root
        self.m_estimate = m_value
        self.minimum_covered = minimum_covered

    def Diff(self, li1, li2):
        return list(set(li1) - set(li2))

    def best_pattern_ND(self, current_df):
        lst = []
        for pattern in self.P:
            obj = Calculate(dataset=current_df, root=self.root, pattern=pattern, label=self.label,
                            mValue=self.m_estimate)
            lst.append(float(obj.mEstimateND()))

        A = [x for _, x in sorted(zip(lst, self.P), reverse=True)]
        A = list(k for k, _ in itertools.groupby(A))
        return A[0]

    def best_pattern_CD(self, patterns, current_df):
        lst = []
        for pattern in patterns:
            obj = Calculate(dataset=current_df, root=self.root, pattern=pattern, label=self.label,
                            mValue=self.m_estimate)
            lst.append(float(obj.mEstimateCD()))

        A = [x for _, x in sorted(zip(lst, patterns), reverse=True)]
        A = list(k for k, _ in itertools.groupby(A))
        return A[0]

    def S_C_ND(self):
        # ----------------------------- Indices Of Current Label --------------------------------------------
        Indices_of_current_label = []
        last_column = list(self.df.iloc[:, -1])
        for i in range(len(last_column)):
            if last_column[i] == self.label:
                Indices_of_current_label.append(i)
        # ----------------------------------------------------------------------------------------------------
        covered_indices = []
        all_covered_falses = []
        Intermediate_Pattern_Set = []
        Intermediate_Pattern_Set_Info = []
        counter = 1
        un_covered = self.Diff(Indices_of_current_label, covered_indices)
        # while True continue, when it becomes False, break.
        while len(un_covered) != 0:
            # the covered indices will contain the evaluation of the best rule w.r.t the whole dataset:
            best_rule = self.best_pattern_ND(self.df)
            obj = Calculate(dataset=self.df_Const, root=self.root, pattern=best_rule, label=self.label,
                            mValue=self.m_estimate)
            extent = obj.extentND()[0]
            intersection = list(set(extent).intersection(Indices_of_current_label))
            U_TP = self.Diff(intersection, covered_indices)
            # ----------------------- The false examples ----------------------------------------------------------
            falses = self.Diff(extent, Indices_of_current_label)
            U_FP = self.Diff(falses, all_covered_falses)
            all_covered_falses.extend(U_FP)
            all_covered_falses = [i for j, i in enumerate(all_covered_falses) if i not in all_covered_falses[:j]]
            # -----------------------------------------------------------------------------------------------------
            # remove the redundant elements
            covered_indices.extend(extent)
            covered_indices = [i for j, i in enumerate(covered_indices) if i not in covered_indices[:j]]
            # ----------------------------------------------------------------------------------------------------
            # Intermediate.append([best_rule, U_TP, U_FP])
            Intermediate_Pattern_Set.append(best_rule)
            Intermediate_Pattern_Set_Info.append([U_TP, U_FP])
            obj2 = Calculate(dataset=self.df, root=self.root, pattern=best_rule, label=self.label,
                             mValue=self.m_estimate)
            extent2 = obj2.extentND()[0]
            self.df = self.df.drop(labels=extent2, axis=0)
            self.df = self.df.reset_index(drop=True)
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

    def S_C_CD(self):
        # ----------------------------- Indices Of Current Label --------------------------------------------
        Indices_of_current_label = []
        last_column = list(self.df.iloc[:, -1])
        for i in range(len(last_column)):
            if last_column[i] == self.label:
                Indices_of_current_label.append(i)
        # ----------------------------------------------------------------------------------------------------

        covered_indices = []
        all_covered_falses = []
        Intermediate_Pattern_Set = []
        Intermediate_Pattern_Set_Info = []
        counter = 1
        un_covered = self.Diff(Indices_of_current_label, covered_indices)
        # while True continue, when it becomes False, break.
        while len(un_covered) != 0:
            # the covered indices will contain the evaluation of the best rule w.r.t the whole dataset:
            best_rule = self.best_pattern_CD(self.P, self.df)
            obj = Calculate(dataset=self.df_Const, root=self.root, pattern=best_rule, label=self.label,
                            mValue=self.m_estimate)
            extent = obj.extentCD()[0]
            intersection = list(set(extent).intersection(Indices_of_current_label))
            U_TP = self.Diff(intersection, covered_indices)
            # ----------------------- The false examples ----------------------------------------------------------
            falses = self.Diff(extent, Indices_of_current_label)
            U_FP = self.Diff(falses, all_covered_falses)
            all_covered_falses.extend(U_FP)
            all_covered_falses = [i for j, i in enumerate(all_covered_falses) if i not in all_covered_falses[:j]]
            # -----------------------------------------------------------------------------------------------------
            # remove the redundant elements
            covered_indices.extend(extent)
            covered_indices = [i for j, i in enumerate(covered_indices) if i not in covered_indices[:j]]
            # ----------------------------------------------------------------------------------------------------
            # Intermediate.append([best_rule, U_TP, U_FP])
            Intermediate_Pattern_Set.append(best_rule)
            Intermediate_Pattern_Set_Info.append([U_TP, U_FP])
            # the second part, to handle the dropping phase:
            obj2 = Calculate(dataset=self.df, root=self.root, pattern=best_rule, label=self.label,
                             mValue=self.m_estimate)
            extent2 = obj2.extentCD()[0]
            self.df = self.df.drop(labels=extent2, axis=0)
            self.df = self.df.reset_index(drop=True)
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
