"""
Data cleaning module, contains wrapper functions for operations using
pandas, scipy, functools, and numpy
"""

# for data exploration
import pandas as pd

# for data filtering
from scipy.stats import iqr
from functools import reduce

# for predictions
from numpy.polynomial import Polynomial
import numpy as np

__all__ = [
    'importCSVPrettily',
    'dropIrrelevantColumns',
    'fillCatNAwithNone',
    'fillNumNAwithZero',
    'fillCatNAwithGroupedMode',
    'fillCatNAwithGroupedModeList',
    'fillCatNAwithMode',
    'fillCatNAwithModeList',
    'makeFilterIQRRule',
    'removeIQROutliersFromDF',
    'buildPolyModelsDict',
    'predictAndFillNumericalNAsGroupedByCategory',
    'fillDiscreteNAwithMode',
    'fillContNAwithMean',
    'remap',
    'remapCommonDict',
    'checkIfIn']

def importCSVasDF(filepath):
    """Import CSV stored data as pandas DataFrame"""
    return pd.read_csv(filepath)

def removeWhiteSpaceFromColumns(df):
    """ Change column names of dataframe
    to remove white spaces between words"""
    df.columns = list(map(lambda x: x.replace(" ", ""),
                          list(df.columns)))

def makeCamelCaseColumnNames(df):
    """ Change column names of dataframe
    to make first letter of each word in
    column name capitalized
    """
    df.columns = list(map(lambda x: x.title(),
                          list(df.columns)))

def removeDelimiterFromColumnNames(df, char):
    """ Change column names of dataframe
    to remove delimiter from between words
    char -- common delimiter characters,
            like "_", " ", etc.
    """
    df.columns = list(map(lambda x: x.replace(char, ""),
                         list(df.columns)))

def importCSVandRemoveWhiteSpace(filepath):
    """Combine importCSVasDF and
    removeWhiteSpaceFromColumns"""
    df = importCSVasDF(filepath)
    removeWhiteSpaceFromColumns(df)
    return df

def importCSVPrettily(filepath, delim):
    """ imports CSV at filepath to dataframe,
    and standardizes the column names"""
    df = importCSVasDF(filepath)
    # makeCamelCaseColumnNames(df)
    removeDelimiterFromColumnNames(df, delim)
    return df

def dropIrrelevantColumns(df, cols):
    """Drop specified columns from df in place"""
    df.drop(cols, axis=1, inplace=True)

def fillCatNAwithNone(df, cols):
    """fill categorical NAs with "None"

    cols -- list of categorical columns (not numerical)"""
    for col in cols:
        df[col].fillna("None", inplace = True)

def fillNumNAwithZero(df, cols):
    """fill numerical NAs with 0

    cols -- list of numerical columns (not categorical)"""
    for col in cols:
        df[col].fillna(0, inplace = True)

def fillCatNAwithGroupedMode(df,
                             colwithNAs,
                             groupingcol):
    """
    Fill the NAs of a category with the mode of
    the category when grouped by groupingcol
    """
    df[colwithNAs].fillna(
        df.groupby(groupingcol)[colwithNAs].transform(
            lambda x: x.mode().iloc[0]), inplace = True)

def fillCatNAwithGroupedModeList(df,
                                 listofcolswithNAs,
                                 groupingcol):
    """
    Run fillCatNAwithGroupedMode on a list of columns
    which have the same groupingcol
    """
    for col in listofcolswithNAs:
        fillCatNAwithGroupedMode(df, col, groupingcol)

def fillCatNAwithMode(df, col):
    """
    Fill the NAs of a category with the mode
    of the category
    """
    df[col].fillna(df[col].mode()[0],
                   inplace = True)

def fillCatNAwithModeList(df, cols):
    """
    Call fillCatNAwithMode on list of columns
    """
    for col in cols:
        fillCatNAwithMode(df, col)

def getIQR(df, col):
    """Get the IQR for a numeric column"""
    return iqr(df[col])

def getQ1Q3(df, col):
    """
    Returns Q1 and Q3 for a numeric column
    as a list, where [0] is Q1, [1] is Q3
    """
    return list(df[col].quantile([0.25, 0.75]))

def makeFilterIQRRule(df, col):
    """
    Make filter for dataframe based on
    1.5*IQR rule to filter outliers
    """
    iqr = getIQR(df, col)
    q1, q3 = getQ1Q3(df, col)
    return [col, lambda x: (x >= q1 - 1.5*iqr) & (x <= q3 + 1.5*iqr)]

def makeIQROutlierPairs(df, cols):
    """
    Create a list of pairs, which define
    a numerical column, and a lambda function that will
    filter the column based on outliers
    """
    pairs = []

    for col in cols:
        pairs.append(makeFilterIQRRule(df, col))

    return pairs

def removeIQROutliersFromDF(df,
                            cols):
    """
    Takes in a dataframe and numerical columns that need
    to be filtered based on 1.5*IQR, returns the dataframe without
    samples w/ detected outliers
    """
    return filterFrame(
        df,
        makeFilters(df, makeIQROutlierPairs(df, cols)),
        list(df.columns))

def computeDegreeNPolyModelCoef(xdata, ydata, degree):
    """
    Compute degree n polynomial model coefficients,
    s.t. p(xdata) ~ ydata
    Returns a list of n+1 items
    """
    return Polynomial.fit(xdata,
                          ydata,
                          degree).convert().coef

def filterFrame(df,
                colpreds,
                colnames):
    """
    produces filtered dataframe based on colpreds
    of data from columns in colnames

    df       -- dataframe you want to filter
    colpreds -- list of boolean filters you want
                to apply using logical AND
    colnames -- columns you want out of the filtered dataframe
    """
    return df[reduce(lambda x, y: x & y, colpreds)][colnames]

def getLevels(df, factor):
    """Returns a list of levels within a factor"""
    return list(df[factor].value_counts().index)

def buildPolyModelsDict(df,
                        factor1,
                        factor2,
                        factor3,
                        pairs,
                        degree):
    """
    Builds a dictionary of polynomial model coefficients for
    different factor1 levels (C) to use factor2 (N)
    to make an estimate for factor3 (N)

    factor1 -- has specific levels we want to separate the data into
    factor2 -- predictor for factor3, no NAs
    factor3 -- has some unknown values
    pairs   -- list of tuples where
               p[0] -- name of column to have condition applied,
               p[1] -- condition as partial function
    degree  -- degree of polynomial for your prediction
    """

    # get list of levels from factor1
    levels = getLevels(df, factor1)

    # get column names from pairs, so that we can properly
    # pass the column names to filterFrame
    colnames = []

    for p in pairs:
        colnames.append(p[0])

    # initialize dictionary to return, where
    # keys   -- levels of factor1,
    # values -- lists of coefficients used to predict
    #           factor3 from factor2
    coef_dict = {}

    for level in levels:
        # this guarantees the first filter is always separating out
        # all samples where factor1 == level
        filters = [df[factor1] == level] + makeFilters(df, pairs)

        data = filterFrame(df, filters, colnames)

        coef_dict[level] = computeDegreeNPolyModelCoef(
            data[factor2],
            data[factor3],
            degree)

    return coef_dict

def makeFilters(df, pairs):
    """ returns a list of dataframe predicates
    generated by applying a condition to a column
    of the dataframe

    requires list of pairs, where
    p[0] -- column name
    p[1] -- condition as partial function

    e.g. -- ['LotArea', functools.partial(lambda y, x: x <= y, 80000)]
    (will return predicates where sample's
    LotArea is less than or equal to 80000)
    """
    filters = []

    for p in pairs:
        filters.append(p[1](df[p[0]]))

    return filters

def evaluatePolynomial(coef, x):
    """ takes in an array of polynomial coefficients and a point x,
    returns the value of the polynomial evaluated at x
    using numpy's polyval"""
    return np.polyval(coef, x)

def predictAndFillNumericalNAsGroupedByCategory(df,
                                                coefdict,
                                                factor1,
                                                factor2,
                                                factor3):
    """
    takes in a dictionary of polynomial model coefficients,
    dataframe, factor1 (C), factor2 (N), and factor3 (N)
    factor1 -- has specific levels we want to separate the data into
    factor2 -- predictor for factor3, no NAs
    factor3 -- has some unknown values
    """
    df[factor3].fillna(
        df.apply(lambda x: evaluatePolynomial(
            coefdict[x.loc[factor1]],
            x.loc[factor2]),
                 axis = 1),
        inplace = True)

def fillDiscreteNAwithMode(df, factor1, factor2):
    """
    fill a discrete numerical variable with the mode
    when grouped by related factor2

    factor1 -- discrete numerical feature with NAs
    factor2 -- categorical feature w/ levels, no NAs
    """
    df[factor1].fillna(
        df.groupby(factor2)[factor1].transform(
            lambda x: x.mode().iloc[0]),
        inplace = True)

def fillContNAwithMean(df, factor1, factor2):
    """
    fill a continuous numerical variable with the mean
    when grouped by related factor2

    factor1 -- continuous numerical feature with NAs
    factor2 -- categorical feature w/ levels, no NAs
    """
    df[factor1].fillna(
        df.groupby(factor2)[factor1].transform(
            lambda x: x.mean()),
        inplace = True)

def remap(df, col, dictmap, dtype='int'):
    """ remaps some feature's values using dictmap
    where default type is 'int', can be specified as 'str'

    dictmap -- dictionary with keys of all values in col,
               and values of new values
    dtype   -- datatype for new values (default 'int')
    """
    df[col] = df[col].map(dictmap).astype(dtype)

def remapCommonDict(df, cols, dictmap, dtype='int'):
    """
    takes in list of columns that have common mapping scheme,
    calls remap on each column using dictmap
    all columns in cols should have same datatype
    """
    for col in cols:
        remap(df, col, dictmap, dtype)

def checkIfIn(list1, list2):
    """ takes in 2 lists, checks if
    any items in list1 are in list2"""
    for item in list1:
        if item in list2:
            return True
    return False
