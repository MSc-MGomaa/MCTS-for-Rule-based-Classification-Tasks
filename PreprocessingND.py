import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


class Preprocessing:
    def __init__(self, df):
        self.df = df

    def isnumber(self, x):
        try:
            float(x)
            return True
        except:
            return False

    def isstring(self, x):
        return x.islower()

    def encode_df(self, df):
        cols = df.columns[:-1]
        num_cols = df._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        cat_cols = sorted(cat_cols)

        for i in range(len(cat_cols)):
            dummies = pd.get_dummies(df[cat_cols[i]], prefix=f'{cat_cols[i]}')
            df = pd.concat([df, dummies], axis='columns')
            df = df.drop([cat_cols[i]], axis='columns')

        for i in df.columns:
            # Make sure the class column is the last column:
            if i.lower() == 'class':
                if df.columns.get_loc(i) == df.columns.size - 1:
                    pass
                else:
                    column = df[[i]]
                    df.drop(columns=df.columns[df.columns.get_loc(i)], axis=1, inplace=True)
                    df = df.join(column)
        return df

    def run(self):

        # Over the Columns:
        for i in self.df.columns:
            # remove the id column if exists:
            if i.lower() == 'id':
                self.df.drop(columns=self.df.columns[self.df.columns.get_loc(i)], axis=1, inplace=True)

            # Make sure the class column is the last column:
            if i.lower() == 'class':
                if self.df.columns.get_loc(i) == self.df.columns.size - 1:
                    pass
                else:
                    column = self.df[[i]]
                    self.df.drop(columns=self.df.columns[self.df.columns.get_loc(i)], axis=1, inplace=True)
                    self.df = self.df.join(column)

        # in case one column has one value for all samples:
        for i in self.df.columns:
            List = self.df[i].to_list()
            if all(element == List[0] for element in List):
                self.df.drop(columns=self.df.columns[self.df.columns.get_loc(i)], axis=1, inplace=True)
            else:
                pass

        # here, I can make a check to know whether the dataset is numeric or categorical:
        # x = list(elem == self.df.dtypes.tolist()[0] for elem in self.df.dtypes.tolist())

        x = list(elem == 'float64' or elem == 'int64' for elem in self.df.dtypes.tolist())
        #if all(x[:-1]):
        if True :
            # replace all non-numeric entries with NaN in a pandas dataframe:
            self.df.iloc[:, :-1] = self.df.iloc[:, :-1][self.df.iloc[:, :-1].applymap(self.isnumber)]

            # replace nan values with average of columns:
            self.df.fillna(self.df.mean(), inplace=True)

            # as a confirmation, convert the dataset to the type float:
            self.df.iloc[:, :-1] = self.df.iloc[:, :-1].apply(pd.to_numeric)

            return self.df, None

        else:
            # incase the dataset is categorical:

            # 1: replace all non-categorical entries with NaN in a pandas dataframe:
            try:
                self.df.iloc[:, :-1] = self.df.iloc[:, :-1][self.df.iloc[:, :-1].applymap(self.isstring)]
            except AttributeError:
                pass

            # replace nan values with the most value of columns:
            self.df.fillna(self.df.mode(), inplace=True)

            # One-Hot Encoding:
            original = self.df
            self.df = self.encode_df(self.df)

            return self.df, original

