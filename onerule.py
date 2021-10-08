# One Rule
import numpy as np
from operator import itemgetter
from collections import defaultdict
class one_rule:
    def __init__(self,train_x,train_y,test_x,test_y):
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
    
    def train_ov(self,featur_i):
        pre_dict = {}
        errors = []
        values = set(self.train_x[:, featur_i])
        for v in values:
            mfc, error = self.train_fv(featur_i, v)
            pre_dict[v] = mfc
            errors.append(error)    
        total_error = sum(errors)
        return pre_dict, total_error
        
    def train_fv(self, featur_i, value):
        c_counts = defaultdict(int)
        for sample, y in zip(self.train_x,self.train_y):
            if sample[featur_i] == value:
                c_counts[y] += 1
        
        sorted_class_counts = sorted(c_counts.items(), key=itemgetter(1), reverse=True)
        mfc = sorted_class_counts[0][0]
        
        error_predictions = [cc for cv, cc in c_counts.items() if cv != mfc]
        error = sum(error_predictions)
        return mfc, error
        
    def predict(self, or_model):
        feature = or_model["feature"]
        predictor = or_model["predictor"]
        y_predicted = np.array([predictor[int(sample[feature])] for sample in self.test_x])
        return y_predicted
    
    def predict_proba(self, or_model):
        feature = or_model["feature"]
        predictor = or_model["predictor"]
        y_train_predicted = np.array([predictor[int(sample[feature])] for sample in self.train_x])
        y_label=np.unique(self.train_y)
        predictor_proba={i:np.zeros(len(y_label)) for i in list(predictor.keys())}
        for i in range(len(predictor)):
            predictor_proba[i][predictor[i]]=sum(y_train_predicted[y_train_predicted==predictor[i]]==self.train_y[y_train_predicted==predictor[i]])/len(self.train_y)
            for j in range(len(y_label)):
                if j!=predictor[i]:
                    predictor_proba[i][j]=(1-predictor_proba[i][predictor[i]])/(len(y_label)-1)    
        y_test_prob_predicted = np.array([predictor_proba[int(sample[feature])] for sample in self.test_x])    
        return y_test_prob_predicted
    
    def fit(self):
        predictors_dict = {}
        errors = {}
        for featur_i in range(self.train_x.shape[1]):
            predictor, error = self.train_ov(featur_i)
            predictors_dict[featur_i] = predictor
            errors[featur_i] = error
        best_feature, best_error = sorted(errors.items(), key=itemgetter(1))[0]
        or_model = {"feature": best_feature, 'predictor': predictors_dict[best_feature]}
        return or_model
    
