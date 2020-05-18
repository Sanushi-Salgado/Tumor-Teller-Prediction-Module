# source - http://www.insightsbot.com/chi-square-feature-selection-in-python/

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency


class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p < alpha:
            result = "{0} is IMPORTANT for Prediction".format(colX)
        else:
            result = "{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
        print(result)

    def TestIndependence(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        self._print_chisquare_result(colX, alpha)


# df = pd.pandas.read_csv("train.csv")
# df['dummyCat'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])
# # Initialize ChiSquare Class
# cT = ChiSquare(df)
# # Feature Selection
# testColumns = ['Embarked', 'Cabin', 'Pclass', 'Age', 'Name', 'dummyCat']
# for var in testColumns:
#     cT.TestIndependence(colX=var, colY="Survived")