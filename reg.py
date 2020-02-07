#https://subscription.packtpub.com/video/data/9781838987671/p6/video6_10/underfitting-and-overfitting


#https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781787286382/12/ch12lvl1sec107/create-a-simple-estimator

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.base import BaseEstimator, ClassifierMixin


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge

class GEEClassifier(BaseEstimator, ClassifierMixin):
    
    """A Classifier made from statsmodels' Generalized Estimating Equations
    
    documentation available at: http://www.statsmodels.org/dev/gee.html
       
    """
    
    def __init__(self,group_by_feature):
        self.group_by_feature = group_by_feature
          
    def fit(self, X, y = None):
        #Same settings as the documentation's example: 
        self.fam = sm.families.Poisson()
        self.ind = sm.cov_struct.Exchangeable()
        
        #Auxiliary function: only used in this method within the class
        def expand_X(X, y, desired_group): 
            X_plus = X.copy()
            X_plus['y'] = y
    
            #roughly make ten groups
            X_plus[desired_group + '_group'] = (X_plus[desired_group] * 10)//10
    
            return X_plus
        
        #save the seen class labels
        self.class_labels = np.unique(y)
        
        dataframe_feature_names = X.columns
        not_group_by_features = [x for x in dataframe_feature_names if x != self.group_by_feature]
        
        formula_in = 'y ~ ' + ' + '.join(not_group_by_features)
        
        data = expand_X(X,y,self.group_by_feature)
        self.mod = smf.gee(formula_in, 
                           self.group_by_feature + "_group", 
                           data, 
                           cov_struct=self.ind, 
                           family=self.fam)
        
        self.res = self.mod.fit()
        
        return self
    
    def predict(self, X_test):
        #store the results of the internal GEE regressor estimator
        results = self.res.predict(X_test)
        
        #find the nearest class label
        return np.array([self.class_labels[np.abs(self.class_labels - x).argmin()] for x in results])
        
    def print_fit_summary(self):
        print res.summary()
        return self
gee_classifier = GEEClassifier('mean_concavity')     
gee_classifier.fit(X_train, y_train)
gee_classifier.score(X_test, y_test)
