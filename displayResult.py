
class Show_Pattern_set:
    def __init__(self, df, classifier, path):
        self.df = df
        self.root = self.Root_Pattern()
        self.classifier = classifier
        self.path = path

    def Root_Pattern(self):
        Root_node = []
        for i in range(len(self.df.columns) - 1):
            column = self.df.iloc[:, i].tolist()
            interval = [min(sorted(column)), max(sorted(column))]
            Root_node.append(interval)

        return Root_node

    def Diff(self, li1, li2):
        return list(set(li1) - set(li2))

    def show_ND(self):
        header = [i for i in self.df]
        covered_labels = []
        P = 0
        N = 0

        with open(self.path, 'w') as fle:
            for Set in self.classifier:
                label = Set[0]
                covered_labels.append(label)
                pattern_desc = ''
                x = ''
                for k in Set[1][0]:
                    pattern_info = Set[1][0].index(k)
                    index = 0
                    for f, b in zip(self.root, k):
                        if f == b:
                            pass
                        else:
                            if index != len(k) - 1:
                                if f[0] != b[0] and f[1] != b[1]:
                                    Att = header[index]
                                    x = str(f'({b[0]} <= {Att} <= {b[1]}) and ')
                                elif f[0] != b[0]:
                                    Att = header[index]
                                    x = str(f'({b[0]} <= {Att}) and ')
                                elif f[1] != b[1]:
                                    Att = header[index]
                                    x = str(f'({Att} <= {b[1]}) and ')

                            elif index == len(k) - 1:
                                if f[0] != b[0] and f[1] != b[1]:
                                    Att = header[index]
                                    x = str(f'({b[0]} <= {Att} <= {b[1]})')
                                elif f[0] != b[0]:
                                    Att = header[index]
                                    x = str(f'({b[0]} <= {Att})')
                                elif f[1] != b[1]:
                                    Att = header[index]
                                    x = str(f'({Att} <= {b[1]})')
                            pattern_desc = pattern_desc + x
                        index += 1
                    # here the pattern needs to be printed:
                    word_list = pattern_desc.split()
                    if word_list[-1] == 'and':
                        word_list.pop()
                    else:
                        pass
                    new_w = ' '.join(word_list)
                    fle.write(f'{new_w}\n')
                    print(f'{new_w}')
                    all_info = Set[1][1]
                    pattern_desc = ''
                    a = ''
                    b = ''
                    for l in range(len(all_info[pattern_info])):
                        if l == 0:
                            a = str(f'({len(all_info[pattern_info][0])}|')
                            P += len(all_info[pattern_info][0])
                        else:
                            b = str(f'{len(all_info[pattern_info][1])})')
                            N += len(all_info[pattern_info][1])

                    c = a + b
                    fle.write(f'Class= {label} {c}\n')
                    print(f'Class= {label} {c}')
                    fle.write('\n')
                    print()

            last_column = self.df.iloc[:, -1].tolist()
            my_dict = {i: last_column.count(i) for i in last_column}
            supposed_covered = 0
            for i in covered_labels:
                supposed_covered += my_dict.get(i)
            missing = supposed_covered - P
            keys_as_list = list(my_dict.keys())
            un_covered_label = self.Diff(keys_as_list, covered_labels)
            P_of_Uncovered_label = my_dict.get(un_covered_label[0])
            fle.write(f'=>Class= {un_covered_label} ({P_of_Uncovered_label + missing}|{missing})\n')
            print(f'=>Class= {un_covered_label} ({P_of_Uncovered_label + missing}|{missing})')
            fle.write('\n')
            print()
            fle.close()

    # To print the Categorical datasets:

    def show_CD(self):
        header = [i for i in self.df]
        covered_labels = []
        P = 0
        N = 0

        with open(self.path, 'w') as fle:
            for Set in self.classifier:
                label = Set[0]
                covered_labels.append(label)
                pattern_desc = ''
                x = ''
                for k in Set[1][0]:
                    pattern_info = Set[1][0].index(k)
                    index = 0
                    for f, b in zip(self.root, k):
                        if f == b:
                            pass
                        else:
                            if index != len(k) - 1:
                                if f[0] != b[0] and f[1] != b[1]:
                                    Att = header[index]
                                    original_att = Att.rpartition('_')[0]
                                    original_value = Att.rpartition('_')[2]
                                    x = str(f'({original_att} = {original_value}) and ')

                                elif f[0] != b[0]:
                                    Att = header[index]
                                    original_att = Att.rpartition('_')[0]
                                    original_value = Att.rpartition('_')[2]
                                    x = str(f'({original_att} = {original_value}) and ')
                                elif f[1] != b[1]:
                                    Att = header[index]
                                    original_att = Att.rpartition('_')[0]
                                    original_value = Att.rpartition('_')[2]
                                    x = str(f'({original_att} = {original_value}) and ')

                            elif index == len(k) - 1:
                                if f[0] != b[0] and f[1] != b[1]:
                                    Att = header[index]
                                    original_att = Att.rpartition('_')[0]
                                    original_value = Att.rpartition('_')[2]
                                    x = str(f'({original_att} = {original_value}) and ')
                                elif f[0] != b[0]:
                                    Att = header[index]
                                    original_att = Att.rpartition('_')[0]
                                    original_value = Att.rpartition('_')[2]
                                    x = str(f'({original_att} = {original_value}) and ')
                                elif f[1] != b[1]:
                                    Att = header[index]
                                    original_att = Att.rpartition('_')[0]
                                    original_value = Att.rpartition('_')[2]
                                    x = str(f'({original_att} = {original_value}) and ')
                            pattern_desc = pattern_desc + x
                        index += 1
                    # here the pattern needs to be printed:
                    word_list = pattern_desc.split()
                    if word_list[-1] == 'and':
                        word_list.pop()
                    else:
                        pass
                    new_w = ' '.join(word_list)
                    fle.write(f'{new_w}\n')
                    print(f'{new_w}')
                    all_info = Set[1][1]
                    pattern_desc = ''
                    a = ''
                    b = ''
                    for l in range(len(all_info[pattern_info])):
                        if l == 0:
                            a = str(f'({len(all_info[pattern_info][0])}|')
                            P += len(all_info[pattern_info][0])
                        else:
                            b = str(f'{len(all_info[pattern_info][1])})')
                            N += len(all_info[pattern_info][1])

                    c = a + b
                    fle.write(f'Class= {label} {c}\n')
                    print(f'Class= {label} {c}')
                    fle.write('\n')
                    print()

            last_column = self.df.iloc[:, -1].tolist()
            my_dict = {i: last_column.count(i) for i in last_column}
            supposed_covered = 0
            for i in covered_labels:
                supposed_covered += my_dict.get(i)
            missing = supposed_covered - P
            keys_as_list = list(my_dict.keys())
            un_covered_label = self.Diff(keys_as_list, covered_labels)
            P_of_Uncovered_label = my_dict.get(un_covered_label[0])
            fle.write(f'=>Class= {un_covered_label} ({P_of_Uncovered_label + missing}|{missing})\n')
            print(f'=>Class= {un_covered_label} ({P_of_Uncovered_label + missing}|{missing})')
            fle.write('\n')
            print()
            fle.close()
