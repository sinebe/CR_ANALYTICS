#https://stats.stackexchange.com/questions/108834/calculate-coefficients-in-a-ordinal-logistic-regression-with-r
import pandas as pd
import numpy as np
X = pd.read_csv("C:\\Users\\enisbe\\Documents\\jupyter\\data\\X.csv")
y = pd.read_csv("C:\\Users\\enisbe\\Documents\\jupyter\\data\\y.csv")
start = pd.read_csv("C:\\Users\\enisbe\\Documents\\jupyter\\data\\start.csv")
X = X.iloc[:,1:].to_numpy()
y  = np.ravel(  y.iloc[:,1:].to_numpy())
s0 =start.iloc[:,1:].to_numpy()
np.set_printoptions(precision=4)
#
# X['t'] = y
# y.shape
# X.shape
# X.head()
# X["y"] = y
#
# X.head()
# X.to_csv("C:\\Users\\enisbe\\Documents\\jupyter\\data\\X_all.csv")
#
X_new = X[:,[0,2]]





def cdf1(X):
    X = np.asarray(X)
    return 1 / (1 + np.exp(-X))


def cdf2(X):
    X = np.asarray(X)
    return 1 / (1 + np.exp(X))



def loglike2(theta, x, y):

    a1 = theta[0]
    a2 = theta[1]
    a3 = theta[2]
    b = theta[3:]

    cdf = cdf1

    XB = np.dot(x, b)

    y_eq_1 = np.log(cdf(a1 + XB))
    y_eq_2 = np.log(cdf(a2 + XB) - cdf(a1 + XB))
    y_eq_3 = np.log(cdf(a3 + XB) - cdf(a2 + XB))
    y_eq_4 = np.log(1 - cdf(a3 + XB))

    ll = np.sum(y_eq_1[y == 1]) + np.sum(y_eq_2[y == 2]) + np.sum(y_eq_3[y == 3]) + np.sum(y_eq_4[y == 4])
    cnt =+ 1
    # print(cnt, theta)

    return -ll
theta0 = np.array([-2,1,4,1,-1])

print(loglike2(theta0, X_new,y))

def cdf1(X):
    X = np.asarray(X)
    return 1 / (1 + np.exp(-X))



exog = pd.DataFrame(X_new, columns=['A','B' ])
endog = pd.DataFrame (y, columns=['y'])

from statsmodels.base.model import GenericLikelihoodModel


class MyOlr(GenericLikelihoodModel):

    def __init__(self, endog, exog, descending=True):

        self.descending = descending
        self.c = np.unique(endog)
        self.c.sort()
        if self.descending==False:
            self.c[::-1].sort()

        self.q = self.c.shape[0]
        # self.exog = exog

        import copy
        exog = exog.copy() # for debuging I want to copy and keep two dataframe separate so that I can rerun the data. This is not memory efficient as I am holding 2 dataset in memory
        for i in range(self.q - 1):
            exog.insert(0, 'Intercept {}'.format(str(self.c[i+1])), 1)

        super(MyOlr, self).__init__(endog, exog)

        self.exog = self.exog[:,self.q-1:]

    def fit(self):

        #intitialize thetas
        coeff = self.exog.shape[1] + self.q-1
        theta0 = [0.1]*coeff
        i =theta0[:self.q-1]
        i =list(np.cumsum(i))
        theta0= i + theta0[self.q-1:]
        theta0 = np.array([-2, 1, 4, 1, -1])
        #TODO:
        # Add constraints. It fits better
        return super(MyOlr, self).fit(start_params=theta0, method='minimize', min_method='slsqp',options = {'maxiter': 500, 'disp': True})


    def loglike(self, theta):

        c = self.c
        q = self.q

        a = theta[:q - 1]
        b = theta[q - 1:]

        x = self.exog
        y = self.endog

        XB = np.dot(x, b)

        c_ll = np.zeros(c.shape[0])

        a1 = a[0]
        ak = a[q - 2]
        c_ll[0] = np.sum(np.log(self.cdf(a1 + XB))[y == c[0]])
        c_ll[q - 1] = np.sum(np.log(1 - self.cdf(ak + XB))[y == c[q - 1]])

        for cls in range(q - 2):
            c_ll[cls + 1] = np.sum(np.log(self.cdf(a[cls + 1] + XB) - self.cdf(a[cls] + XB))[y == c[cls + 1]])

        return np.sum(c_ll)

    def cdf(self, X):
        X = np.asarray(X)
        return 1 / (1 + np.exp(-X))


    def wald(self,res):

        from scipy.stats import chi2

        params = res.params
        std_err = res.bse
        wald_chisquare = (params / std_err) ** 2

        walds_p = 1 - chi2.cdf(wald_chisquare, 1)

        wald_table = pd.DataFrame([params, std_err, wald_chisquare, walds_p], columns=self.exog_names).T
        wald_table.columns = ['params', 'std_err', 'Wald Statistic', 'Pr ChiSq > 0']
        return wald_table.round( 3  )






mod = MyOlr(endog, exog, descending=True) #

mod.exog_names
mod.exog
exog
res = mod.fit()#.fit(start_params=theta0, method='minimize', use_t=True, min_method='slsqp' ,constraints=cons)


print(res.summary())
print(mod.wald(res))
# theta0 = np.array([-2,1,4,1,-1])

# mod.loglike(theta0)
# type(mod.endog)
#
# exog
# print(mod.exog)
# print(mod.exog_names)
# # print(mod.exog.head())
#

# i = np.array([1,2,3,4])
# q = i.shape[0]
# b = ['A','B']
#
# i = -np.sort(-i)
# for c in range(q-1):
#     b.insert(0,"Intercept " + str((i[c])))
# print(b)
