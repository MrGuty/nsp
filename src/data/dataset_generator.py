import sklearn.model_selection
import pandas as pd
import imblearn.over_sampling

class Splitter:
    def __init__(self,data,label="NSP"):
        self.data = pd.read_csv(data)
        self.label = self.data[label]
        self.features = self.data.drop(columns=[label], axis=1)
        self.data = None
    def split(self,location):
        self.features_train, self.features_test, self.label_train, self.label_test = sklearn.model_selection.train_test_split(self.features, self.label, test_size=0.33, random_state=11, stratify=self.label)
        sm = imblearn.over_sampling.SMOTE(random_state=11)
        self.features_train_resampled, self.label_train_resampled = sm.fit_resample(self.features_train, self.label_train)
        self.features_train.to_csv(location+"features_train.csv",index=False)
        self.label_train.to_csv(location+"label_train.csv",index=False,header=True)
        self.features_test.to_csv(location+"features_test.csv",index=False)
        self.label_test.to_csv(location+"label_test.csv",index=False,header=True)
        self.features_train_resampled.to_csv(location+"features_train_resampled.csv",index=False)
        self.label_train_resampled.to_csv(location+"label_train_resampled.csv",index=False,header=True)