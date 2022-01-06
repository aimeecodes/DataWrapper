"""
Module for wrapping sklearn functions and objects
"""

# for data handling
import pandas as pd

# for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# for feature importance
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# for model training
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# for model evaluation
from sklearn.metrics import mean_absolute_error

# for tracking
import time
import datetime

# for numerical manip
import numpy as np

__all__ = [
    'separateNumCatFeatDFs',
    'rfModel',
    'rfModelWithImportances']

def getNumCatLists(df):
    """
    takes in a dataframe,
    returns an array with 2 lists:
    num_feat -- numerical column names of df
    cat_feat -- categorical column names of df
    """
    isCat = df.dtypes == 'object'
    isBool = df.dtypes == 'bool'
    nonNum = isCat | isBool

    # get list of features
    cat_feat = list(df.dtypes[nonNum].index)
    num_feat = list(df.dtypes[~nonNum].index)

    return [num_feat, cat_feat]

def separateNumCatFeatDFs(df):
    '''
    takes in a dataframe of features,
    returns an array w/ two dataframes and two lists:
    numdf -- dataframe of numerical features
    catdf -- dataframe of categorical features
    '''
    num_feat, cat_feat = getNumCatLists(df)

    # generate the two dataframes
    numdf = df[num_feat].copy(deep=True)
    catdf = df[cat_feat].copy(deep=True)

    return [numdf, catdf]

def createRFPipeline(df):
    '''
    creates a pipeline with the following steps:
    numerical_pipe       -- imputes means and scales
    categorical_encoder  -- one hot encodes
    regressor            -- randomforestregressor,
                            chosen parameters

    returns callable pipeline object
    '''

    # get column names and column lists, and separate dataframes
    num_feat, cat_feat = getNumCatLists(df)

    # create numerical handler
    # (impute unknown means, then scale)
    numerical_pipe = Pipeline(
        [('imputer', SimpleImputer(strategy='mean')),
         ('scaler', StandardScaler())])

    # create categorical encoder
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')

    # create preprocessing transform step
    preprocessing = ColumnTransformer(
        [('cat', categorical_encoder, cat_feat),
         ('num', numerical_pipe, num_feat)])

    # create model pipeline
    rf = Pipeline(
        [('preprocess', preprocessing),
         ('regressor', RandomForestRegressor(
             # max_depth=15,
             # n_estimators=500,
             # max_features='auto',
             oob_score=True,
             random_state=42,
             n_jobs=-1))])

    return rf

def fitModel(model, Xtrain, Ytrain, verbose):
    """
    fits model to Xtrain, Ytrain,
    verbose should be between 0 and 1,
    0 >    no output,
    1 >    text output to console
    """
    start_time = time.time()
    model.fit(Xtrain, Ytrain)
    train_time = time.time() - start_time
    if verbose == 1:
        print(f'Training time on all cores: {train_time} seconds \n')

def printModelScores(model,
                     modelname,
                     Xtrain,
                     Xtest,
                     Ytrain,
                     Ytest):
    train_score = model.score(Xtrain, Ytrain)
    test_score = model.score(Xtest, Ytest)
    outofbagscore = model['regressor'].oob_score_

    # get predictions
    pred = model.predict(Xtest)
    test_mae = mean_absolute_error(pred, Ytest)

    print(f'{modelname} model results:')
    print(f'Training score  : {train_score}')
    print(f'Cross-val score : {test_score}')
    print(f'Out of bag score: {outofbagscore}')
    print('Mean absolute error: {:,.0f}\n'.format(test_mae))


def computePermutedImportances(model,
                               Xtest,
                               Ytest,
                               verbose):
    """ Computes model's permutation importance of features

    model   -- fitted estimator
    Xtest,
    Ytest   -- samples and target value, should not have been seen
               by model before (not part of training set)
    verbose -- [0,1]
               if 0, no text output printed, only save importances
               if 1, text output printed, save importances
    """
    start_time = time.time()
    result = permutation_importance(model,
                                    Xtest,
                                    Ytest,
                                    n_repeats = 20,
                                    random_state = 42,
                                    n_jobs = -1)
    elapsed_time = time.time() - start_time
    if verbose ==1:
        print(f"Elapsed time to compute importances: {elapsed_time} seconds \n")

    # create sorted index from lowest to highest importance value
    sorted_idx = result.importances_mean.argsort()

    # export data to csv for graphing file to handle
    importance_data = result.importances[sorted_idx]
    importance_labels = Xtest.columns[sorted_idx]

    importancesdf = pd.DataFrame(importance_data, index = importance_labels).T

    # save this new importancesdf to a csv
    t = datetime.datetime.now()
    csvtitle = f"permimportances{t.month}{t.day}{t.hour}{t.minute}.csv"
    importancesdf.to_csv(csvtitle)

    return importancesdf

def rfModel(X, Y, modelname):
    """ Creates RF Pipeline, performs train / test split,
    fits the model to training data, prints performance scores

    X         -- samples, unsplit
    Y         -- target variable, unsplit
    modelname -- string name of model
                 (usualy differentiated by features used)
    """

    # create model object
    rfmodel = createRFPipeline(X)

    # create train / test splits
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,
                                                    Y,
                                                    test_size = 0.4,
                                                    random_state=42)

    # fit the regressor to Xtrain and Ytrain
    fitModel(rfmodel, Xtrain, Ytrain, 1)

    # print performance of model
    printModelScores(rfmodel, modelname,
                     Xtrain, Xtest, Ytrain, Ytest)

    return rfmodel

def rfModelWithImportances(X, Y, modelname):
    """ Creates RF Pipeline, splits the data,
    fits the model to training data, prints performance scores,
    and computes permuted importances of features used by model

    Returns an array, where [0] is model object,
    and [1] is dataframe of permuted importances

    X         -- samples, unsplit
    Y         -- target variable, unsplit
    modelname -- string name of model
                 (usualy differentiated by features used)
    """
    # create model object
    rfmodel = createRFPipeline(X)

    # create train / test splits
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,
                                                    Y,
                                                    test_size = 0.4,
                                                    random_state=42)

    # fit the regressor to Xtrain and Ytrain
    fitModel(rfmodel, Xtrain, Ytrain, 1)

    # print performance of model
    printModelScores(rfmodel, modelname,
                     Xtrain, Xtest, Ytrain, Ytest)

    importancesdf = computePermutedImportances(rfmodel, Xtest, Ytest, 1)

    return [rfmodel, importancesdf]
