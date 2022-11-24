from Extent import Calculate
import pandas as pd


class ChildrenCD:
    def __init__(self, dataset, pattern, Root, min_support):
        self.df = dataset
        self.pattern = pattern
        self.constant = pd.DataFrame(pattern)
        self.Min_Support = min_support
        self.Root = Root

    def Direct_ChildrenCD(self):
        Direct_Children = []

        # the header except the class:
        Header = [k for k in self.df.iloc[:, :-1]]

        # the currently restricted features:
        Current_Restrictions = [i for i, x in enumerate(self.pattern) if x == [1, 1]]
        Restricted_Attributes = []
        # Here, I will have the attributes that are already restricted, like A1
        for i in Current_Restrictions:
            Restricted_Attributes.append(Header[i].rpartition('_')[0].lower())

        # Allowed indices to be restricted:
        Allowed_Indices = []
        for col in Header:
            if col.rpartition('_')[0].lower() in Restricted_Attributes:
                pass
            else:
                Allowed_Indices.append(Header.index(col))

        # In each iteration, add a restriction:
        for index in Allowed_Indices:
            self.pattern = self.constant.values.tolist()
            self.pattern[index] = [1, 1]
            # Check if it's frequent:
            obj = Calculate(dataset=self.df, pattern=self.pattern, root=self.Root, mValue=None, label=None)
            if obj.extentCD()[1] >= self.Min_Support:
                Direct_Children.append(self.pattern)
            else:
                pass

        return Direct_Children

    def Direct_ChildrenCD_Simulation(self):
        Direct_Children = []

        # the header except the class:
        Header = [k for k in self.df.iloc[:, :-1]]

        # the currently restricted features:
        Current_Restrictions = [i for i, x in enumerate(self.pattern) if x == [1, 1]]
        Restricted_Attributes = []
        # Here, I will have the attributes that are already restricted, like A1
        for i in Current_Restrictions:
            Restricted_Attributes.append(Header[i].rpartition('_')[0].lower())

        # Allowed indices to be restricted:
        Allowed_Indices = []
        for col in Header:
            if col.rpartition('_')[0].lower() in Restricted_Attributes:
                pass
            else:
                Allowed_Indices.append(Header.index(col))

        # In each iteration, add a restriction:
        for index in Allowed_Indices:
            self.pattern = self.constant.values.tolist()
            self.pattern[index] = [1, 1]
            Direct_Children.append(self.pattern)

        return Direct_Children
