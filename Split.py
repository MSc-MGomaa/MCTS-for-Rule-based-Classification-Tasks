from sklearn.model_selection import train_test_split


class TTS:
    def __init__(self, df, test_size):
        self.df = df
        self.ts = test_size

    def Split(self):
        last_column = self.df.columns[-1]
        X = self.df.loc[:, self.df.columns != last_column]
        y = self.df.iloc[:, -1]

        # keep the probability of examples constant in the training and testing: enable the stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.ts, random_state=42,
                                                            shuffle=True, stratify=y)

        # X_train, X_test, y_train, y_test.tolist() -> To compare the indexes.
        trainData = X_train.assign(Class=y_train.tolist())
        trainData = trainData.reset_index(drop=True)

        return trainData, X_test, y_test
