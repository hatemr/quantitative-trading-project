from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import functools
import seaborn as sns
import time
import h5py
import statsmodels.api as sm
import copy
import pickle
import pdb
from math import ceil
from scipy import stats
import importlib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler



def aggregate_from_daily(df, ticker):
    """
    Use in combine_all_tickers.
    Aggregates data from daily to monthly level.
    """
    
    df_temp = df.copy()
    df_temp.index.names = ['Date']
    
    ticker_price_beginning = df_temp.loc[:,['Adj_Close']].resample('M').first().rename(columns={'Adj_Close':'ticker_price_beginning'})
    ticker_price_end = df_temp.loc[:,['Adj_Close']].resample('M').last().rename(columns={'Adj_Close':'ticker_price_end'})
    mkt_price_beginning = df_temp.loc[:,['QQQ_Adj_Close']].resample('M').first().rename(columns={'QQQ_Adj_Close':'mkt_price_beginning'})
    mkt_price_end = df_temp.loc[:,['QQQ_Adj_Close']].resample('M').last().rename(columns={'QQQ_Adj_Close':'mkt_price_end'})
    df4 = ticker_price_beginning.merge(ticker_price_end, left_index=True, right_index=True, how='outer').merge(mkt_price_beginning, left_index=True, right_index=True, how='outer').merge(mkt_price_end, left_index=True, right_index=True, how='outer')
    
    custom_aggretator = lambda array_like: array_like.max()  # aggregate days to month, for sentiment features
    
    df1 = df_temp.drop(columns=['Adj_Close', 'Adj_Volume','QQQ_Adj_Close']).resample('M').apply(custom_aggretator)  # for sentiment features, take max
    df2 = df_temp.loc[:, ['Adj_Volume']].resample('M').median().merge(df1, left_index=True, right_index=True, how='outer')    # for volume, take median
    
    df2 = df2.assign(Ticker=ticker).reset_index().set_index(['Date','Ticker']).rename(columns={'Adj_Volume': 'volume'})
    df5 = df4.merge(df2, left_index=True, right_index=True, how='outer')
    return df5

def combine_all_tickers(ticker_dict):
    """
    Combines dictionary-of-dataframes into one dataframe.
    """
    
    first_key = list(ticker_dict.keys())[0]
    df1 = ticker_dict[first_key].copy()    # df for first ticker
    first_df_cols = df1.columns.tolist()
    df2 = aggregate_from_daily(df1, first_key)   # aggregate to monthly level
    j=0
    for key, value in ticker_dict.items():    # for each ticker, aggregate then concat to master df
        #print(key,j)
        #j+=1
        if key==first_key: continue
        if first_df_cols != value.columns.tolist(): print('bad columns for {}!'.format(key))
        df3 = aggregate_from_daily(value, key)
        df2 = pd.concat([df2, df3])
        
    df2 = df2.sort_index(level=[0,1])
    return df2

# used in create_weights()
def sigmoid(x):
    p = 1 / (1 + np.exp(-x))
    return p/p.sum()

# used in create_weights()
def position_weights(x):
    x = x.astype(float).values.copy()
    idx_nonzero = np.nonzero(x)
    x[idx_nonzero] = sigmoid( x[idx_nonzero] )  # change the non-zero scores to 
    return pd.Series(x)


def transform_function(x, seed=False, num_long_tickers=10, mult_extra=2.):
	"""
	Used in create_weights(). Takes in series, randomly puts one
	for random num_long_tickers tickers.
	Seed is if you don't want to choose randomly, to get reproducible
	strategy.
	"""
	#mult_extra=2

	if seed: np.random.seed(42)
	if num_long_tickers*mult_extra > x.shape[0]: print('mult_extra is too big!')
	output = np.concatenate([np.ones(num_long_tickers*mult_extra), np.zeros( x.shape[0]-mult_extra*num_long_tickers)])
	indices = np.random.choice(np.arange(mult_extra*num_long_tickers), replace=False, size=num_long_tickers)
	output[indices] = 0.
	return pd.Series(output)  # go long top fraction of tickers, no short positions


def create_weights(df, sort_column='score', frac_long_tickers=0.01, seed=False, mult_extra=2.):
    """
    For each month, find the tickers that we will go long.
    Calculate their weights accoring to the ranking column.
    Shift those weights a month forward so we can trade on them.
    """
    
    df1 = df.copy()
    num_tickers = df1.index.get_level_values(1).unique().shape[0]
    num_long_tickers = int(round(num_tickers*frac_long_tickers))
    #transform_function = lambda x: pd.Series(np.concatenate([np.ones(num_long_tickers), np.zeros(x.shape[0]-2*num_long_tickers), np.zeros(num_long_tickers)]))  # go long top fraction of tickers, no short positions. DEPRECATED
    transform_function1 = lambda x: transform_function(x, seed=seed, num_long_tickers=num_long_tickers, mult_extra=int(mult_extra))

    df2 = df1.loc[:,[sort_column]].sort_values(by=['Date',sort_column], ascending=[True, False]).groupby(level=0, as_index=False).transform(transform_function1)  # turn into position (1,0,0). transform takes in series
    
    # print(num_long_tickers, df2.loc[:,sort_column].groupby(level=0).sum())
    df2 = df2.rename(columns={sort_column:'position_current'})
    df1 = df1.loc[:,[sort_column, 'ticker_price_beginning', 'ticker_price_end', 'mkt_price_beginning', 'mkt_price_end']]
    df3 = df1.merge(df2, left_index=True, right_index=True, how='outer').sort_values(by=['Date','Ticker'], ascending=[True, True])  # merge position (1,0,-1) with other columns         
    df3 = df3.assign(position_predictive=df3.position_current.shift(num_tickers)) #.sort_values(by=['Date',sort_column], ascending=[True, False])  # predictive takes position from _previous_ month
    
    df3 = df3.assign(score_current = df3[sort_column]*df3.position_current)
    df3 = df3.assign(weight_current = df3.loc[:,['score_current']].groupby(level=0, as_index=False).transform(position_weights).iloc[:,0])  # find weights based on current rankings
    df3 = df3.assign(weight_pred = df3.weight_current.shift(num_tickers))  # find weights based on current rankings
    df3 = df3.drop(columns=['score_current'])
    
    df4 = df3[(df3.position_predictive!=0)& (df3.position_predictive.notna())].copy()
    return df4

def calculate_capital(df, initial_capital=1e6):
    """
    Start with the initial capital. For each month, spend it all on long positions.
    Then calculate the caplital left at end of the month, and re-invest that amount 
    next month.
    """
    
    i=0
    for date, new_df in df.groupby(level=0):  # iterate through dates, compute new capital held, and new positions
        if i==0:
            new_df1 = new_df.copy()
            new_df1 = new_df1.assign(total_notional_begin = initial_capital)
            new_df1 = new_df1.assign(notional_begin = new_df1.total_notional_begin *new_df1.weight_pred)  # must provide initial capital
            new_df1 = new_df1.assign(num_shares_begin = np.floor(new_df1.notional_begin/new_df1.ticker_price_beginning))  # buy the number of shares afforded by the capital to spend
            new_df1 = new_df1.assign(notional_end = new_df1.ticker_price_end*new_df1.num_shares_begin)  # exit position, calculate ending capital 
            new_df1 = new_df1.assign(cashflow= new_df1.notional_end - new_df1.notional_begin)   # create cashflows
            total_notional_end = new_df1.notional_end.groupby(level=0).sum().values[0]  # sum the ending notional accros assets
            new_df1 = new_df1.assign(total_notional_end=total_notional_end)
            i+=1
        else:
            new_df2 = new_df.copy()
            new_df2 = new_df2.assign(total_notional_begin = total_notional_end)  # use total_notional_end from previous iteration
            new_df2 = new_df2.assign(notional_begin = new_df2.total_notional_begin *new_df2.weight_pred)  # must provide initial capital
            new_df2 = new_df2.assign(num_shares_begin = np.floor(new_df2.notional_begin/new_df2.ticker_price_beginning))
            new_df2 = new_df2.assign(notional_end = new_df2.ticker_price_end*new_df2.num_shares_begin)
            new_df2 = new_df2.assign(cashflow= new_df2.notional_end - new_df2.notional_begin)   # create cashflows
            total_notional_end = new_df2.notional_end.groupby(level=0).sum().values[0]
            new_df2 = new_df2.assign(total_notional_end=total_notional_end)
            new_df1 = pd.concat([new_df1, new_df2])       
    return new_df1
    
    
def calculate_beta(df, ticker_dict, beta_cap_floor=-10.0, beta_cap_ceil=10.0):
    """
    For each of the tickers that we go long, compute beta from prior month's daily returns.
    """
    
    df1 = df.copy()
    df_temp = df1.assign(beta=0.)
    df_temp = df_temp.loc[:,['beta']]
    
    for index, row in df_temp.iterrows():
        date_end = index[0] - pd.tseries.offsets.MonthEnd(1)  # dates for prior month
        date_begin = date_end - pd.tseries.offsets.MonthBegin(1)
        ticker = index[1]

        df2 = ticker_dict[ticker][date_begin:date_end].copy()  # select prior month's data
        df2 = df2[df2.index.dayofweek < 5]  # remove weekends
        df2 = df2.loc[:,['Adj_Close','QQQ_Adj_Close']]  # only need ticker's price and QQQ price
        
        X = df2.values[:,[0]]
        y = df2.values[:,[1]]
        reg = LinearRegression().fit(X, y)  # run regression
        df_temp.at[index,'beta'] = reg.coef_  # put in beta
      
    df1 = df1.assign(beta=df_temp.beta.clip(beta_cap_floor, beta_cap_ceil))  # add beta column, with beta clipped
    return df1

 
def calculate_mkt_positions(df):
    """
    Use the betas to calculate the positions in the index. The market positions are short the same amount (except
    rounding) as the long positions times beta.
    """
    df1 = df.copy()
    df1 = df1.assign(notional_begin_mkt = - df1.beta * df1.notional_begin)  # if long x, then short (beta) * x
    df1 = df1.assign(total_notional_begin_mkt=0.)  # start with no market positions
    df1 = df1.assign(num_shares_begin_mkt = np.floor(df1.notional_begin_mkt/df1.mkt_price_beginning))  # buy value -beta*num_shares for each ticker
    df1 = df1.assign(notional_end_mkt = df1.mkt_price_end*df1.num_shares_begin_mkt)  # notional held at end of month, after exiting positions
    
    df1 = df1.assign(cashflow_mkt = df1.notional_end_mkt - df1.notional_begin_mkt)  # cashflow over month period
    df1a = df1.loc[:,['cashflow_mkt']].groupby(by=['Date']).sum().rename(columns={'cashflow_mkt':'total_notional_end_mkt'})  # ending notional summed across assets
    df1 = df1.merge(df1a, left_index=True, right_index=True, how='inner')
    
    i =df1.index.get_level_values(0)[0]  # first date
    num_tickers = df1.loc[i,:].index.values.shape[0]  # number of tickers, for shifting
    df1 = df1.assign(notional_from_prior_period = df1.total_notional_end_mkt.shift(num_tickers))  # shift, to add later
    df1.notional_from_prior_period.fillna(0, inplace=True)
    
    df1 = df1.assign(total_notional = df1.total_notional_end + df1.total_notional_end_mkt + df1.notional_from_prior_period) # add tickers, market, and leftover from prior period's market
    return df1
    
    
def calculate_pnl_sub_strategy(df, initial_capital=1e6):
    """
    Aggregates monthly positions across tickers to get monthly PnL
    """
    df1 = df.loc[:,['total_notional']].groupby(level=0, as_index=True).median()  # take median of total_notional, which are all same anyway
    
    df3 = df.loc[:,['mkt_price_beginning','mkt_price_end']].groupby(by='Date').first()  # take market prices
    df3 = df3.assign(returns_mkt = (df3.mkt_price_end - df3.mkt_price_beginning)/df3.mkt_price_beginning )  # market returns over month
   
    df1 = df1.assign(returns_mkt = df3.returns_mkt)
    
    #df1 = df1.assign(mkt_price_beginning = df3.mkt_price_beginning.values)
    #df1 = df1.assign(mkt_price_end = df3.mkt_price_end.values)
    
    #ind = np.array([np.datetime64('2013-01-31')])  # add Jan '13 data point
    #df2 = pd.DataFrame(data={'total_notional':initial_capital, 'returns_mkt':np.nan, 'mkt_price_beginning':np.nan, 'mkt_price_end':np.nan}, index=ind)
    #df2.index = df2.index.rename('Date')
    #df2 = pd.concat([df1,df2]).sort_index()
    
    df2 = df1.copy()
    df2 = df2.assign(returns_strategy=df2.total_notional.pct_change())
    
    df2 = df2[['returns_mkt','returns_strategy','total_notional']]
    df2 = df2.rename(columns={'total_notional':'pnl'})
    return df2


def run_strategy(df, ticker_dict, sort_column='score', frac_long_tickers=0.01, seed=False, mult_extra=2., beta_cap_floor=-10.0, beta_cap_ceil=10.0, plot=False):
    df2=df.copy()
    df3 = create_weights(df2, sort_column=sort_column, frac_long_tickers=frac_long_tickers, seed=seed, mult_extra=mult_extra)
    df4 = calculate_capital(df3, initial_capital=1e6)
    #print('Calculated capital for each month')
    df5 = calculate_beta(df4, ticker_dict=ticker_dict, beta_cap_floor=beta_cap_floor, beta_cap_ceil=beta_cap_ceil)
    #print('Calculated market beta of the ticker-level returns')
    df6 = calculate_mkt_positions(df5)
    #print('Calculated market positions from the betas')
    df7 = calculate_pnl_sub_strategy(df6)
    if plot: df7.loc[:,['pnl']].plot()
    return df7


def print_hi():
    print('no more')


##### Create features for machine learning #####

def aggregate_from_daily_ml(df, ticker):
    """
    Use in combine_all_tickers_ml. Aggregates data from daily to monthly level.
    Features are quantiles over previous month.
    """
    
    df_temp = df.copy()
    df_temp.index.names = ['Date']
    ticker_price_beginning = df_temp.loc[:,['Adj_Close']].resample('M').first().rename(columns={'Adj_Close':'ticker_price_beginning'})
    ticker_price_end = df_temp.loc[:,['Adj_Close']].resample('M').last().rename(columns={'Adj_Close':'ticker_price_end'})
    mkt_price_beginning = df_temp.loc[:,['QQQ_Adj_Close']].resample('M').first().rename(columns={'QQQ_Adj_Close':'mkt_price_beginning'})
    mkt_price_end = df_temp.loc[:,['QQQ_Adj_Close']].resample('M').last().rename(columns={'QQQ_Adj_Close':'mkt_price_end'})
    df4 = ticker_price_beginning.merge(ticker_price_end, left_index=True, right_index=True, how='outer').merge(mkt_price_beginning, left_index=True, right_index=True, how='outer').merge(mkt_price_end, left_index=True, right_index=True, how='outer')
    df4.columns = pd.MultiIndex.from_tuples([(col, 'NA') for col in df4.columns.tolist()])
    
    quantiles = [.5,.75,.9]
    custom_aggretator = lambda array_like: array_like.quantile(q=quantiles)  # aggregate days to month, for sentiment features
    
    df1 = df_temp.drop(columns=['Adj_Close','QQQ_Adj_Close']).resample('M').apply(custom_aggretator)  # for sentiment features, take max
    
    #df6 = df_temp.drop(columns=['Adj_Close','QQQ_Adj_Close']).rolling(30).agg(custom_aggretator) #.resample('M')
    
    df1 = df1.rename(columns={'Adj_Volume':'Volume'})
    df1 = df1.unstack()   # turn the row index into columns
    
    df2 = df1.iloc[:,0:3].drop([0.75,0.9], axis=1, level=1)  # keep only median for volume
    df1 = df1.drop('Volume', axis=1, level=0)
    df1 = df1.merge(df2, left_index=True, right_index=True)
    
    df4 = df4.merge(df1, left_index=True, right_index=True, how='inner')
    df4 = df4.assign(Ticker=ticker).reset_index().set_index(['Date','Ticker'])
    
    df4[('Returns','Next_Month')] = ((df4[('ticker_price_end', 'NA')] - df4[('ticker_price_beginning', 'NA')]) / df4[('ticker_price_beginning', 'NA')]).shift(-1)
    
    return df4


# one-time function
def create_features(overwrite=False):
    """
    Combines dictionary-of-dataframes into one dataframe.
    Creates features as it goes.
    Creates data.pkl
    """
    
    data_dict = pd.read_pickle('complete_dataset.pickle')  # upload dictionary of tickers
    ticker_dict = data_dict['Raw_Data']

    # initialize dataframe
    first_key = list(ticker_dict.keys())[0]  # find the first ticker
    df1 = ticker_dict[first_key].copy()    # df for first ticker
    first_df_cols = df1.columns.tolist()
    df2 = aggregate_from_daily_ml(df1, first_key)   # aggregate to monthly level
    j=0
    for key, value in ticker_dict.items():    # for each ticker, aggregate then concat to master df
        if key==first_key: continue
        if first_df_cols != value.columns.tolist(): print('bad columns for {}!'.format(key))
        df3 = aggregate_from_daily_ml(value, key)
        
        df2 = pd.concat([df2, df3])
        if j%(round(len(ticker_dict)/10))==0: print('Fraction done: {}'.format(round(j/len(ticker_dict),5)))
        j+=1
    df2 = df2.sort_index(level=[0,1])
    
    df2.columns = [col[0] + '_' + str(col[1]) if str(col[1])!='NA' else col[0] for col in df2.columns.tolist()]

    df3 = create_target(df2, threshold=0.0)
    df3.columns = [col[0] + '_' + str(col[1]) if str(col[1])!='NA' else col[0] for col in df3.columns.tolist()]


    if overwrite:
        print('Saving to data.pkl')
        df3.to_pickle('data.pkl')
    else:
        print('File not being saved. To save, use overwrite=True')

    return df3


def create_target(data, threshold=0.0):
    '''
    Create target variable that is binary {0,1}.
    Split into X and y.
    data: dataframe
    '''

    data1 = data.dropna().copy()
    binarizer = Binarizer(threshold=threshold)
    target = binarizer.transform(data1[('Returns','Next_Month')].values.reshape(-1,1))

    data1 = data1.join(pd.DataFrame(target,
        columns=pd.MultiIndex.from_product([['Returns'], ['Target']]),
        index=data1.index))

    return data1


def create_predictions_sklearn(data, model, name='y_pred_ml', num_cols_non_feats=1):
    """
    Runs the models on the data (dataframe) and creates a new column called score.
    Works for sklearn models, not keras models.
    """
    X = data.iloc[:,:-num_cols_non_feats].copy().values  # pick out only the predictor variables. The model picks certain columns via its pipeline.
    y_pred = model.predict_proba(X)[:,1]
    data1 = data.copy()
    data1[name] = y_pred
    return data1


# split a multivariate sequence into samples
# helper function
def split_sequences(sequences, n_steps=3):
	"""
    Takes in dateset with has (X,y) stacked together horizontally. Samples with rolling window
    of length n_steps, and outputs the results to X (n_samples, n_timesteps, n_features) and
    y (n_samples). Note that y is the one-step-ahead y, as it should for time-series prediction.
    """

	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	
	return np.array(X), np.array(y)


def build_dataset_for_rnn(df11, n_steps=3, frac_train=0.5):
    """
    Takes in dataframe. Splits out X and y. Normalizes X. Samples (X,y) with a rolling window of length n_steps,
    and saves as 3d array (n_samples, n_timesteps, n_features).

    This acts only on sentiment features!!
    """
    
    # Split by company
    X_tickers = []
    y_list = []

    scaler = MinMaxScaler(feature_range=(-1,1))
    #print('Building dataset...')
    for i, ticker in enumerate(df11.index.get_level_values(1).unique()):  # split by ticker
        X = df11.reset_index().loc[df11.reset_index().Ticker==ticker].set_index(['Date','Ticker']).iloc[:,:-1].values
        X = scaler.fit_transform(X)  # standardize the features
        y = df11.reset_index().loc[df11.reset_index().Ticker==ticker].set_index(['Date','Ticker']).iloc[:,-1].values
        dataset = np.hstack((X, y.reshape(-1,1)))
        # convert into input/output
        X, y = split_sequences(dataset, n_steps)
        X_tickers.append(X)
        y_list.append(y)
        #if i%(int(df11.index.get_level_values(1).unique().shape[0])/5)==0: print('Done {} percent'.format(round(100*i/int(df11.index.get_level_values(1).unique().shape[0]),2)))
    
    n_features = X_tickers[0].shape[2]
    
    try:
    	X = np.array(X_tickers).reshape(-1, n_steps, n_features)
    except ValueError:
    	print(len(X_tickers), X_tickers[0].shape, n_steps, n_features)
    y = np.array(y_list).reshape(-1,1)

    end_train = int(X.shape[0]*frac_train)
    x_train = X[:end_train]
    y_train = y[:end_train]
    x_test = X[end_train:]
    y_test = y[end_train:]
    
    #print('Done building dataset')
    #print('x_train shape:', x_train.shape)
    #print('x_test shape:', x_test.shape)
    #print('y_test shape:', y_train.shape)
    #print('y_test shape:', y_test.shape)
    
    return x_train, y_train, x_test, y_test


def create_predictions_keras(df_test, model, name='y_pred_nn',n_steps=3,verbose=False):
    """
    Takes in dataframe of features and makes predictions from the model.
    The number of steps (n_steps) must match the number of steps the
    model was trained on.
    """
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    
    count_rows = 0
    tickers = df_test.index.get_level_values(1).unique()
    for i, ticker in enumerate(tickers):  # split by ticker
    
        if i==0:
            df1 = df_test.reset_index().loc[df_test.reset_index().Ticker==ticker].set_index(['Date','Ticker']).copy()
            X = df1.iloc[:,:-3].values  # chop off two y's from previous two models
            X = scaler.fit_transform(X)  # standardize the features
            y = df1.iloc[:,-3].values
            dataset = np.hstack((X, y.reshape(-1,1)))
            X, y = split_sequences(dataset, n_steps)
            y_pred = model.predict(X)
            try:
                y_pred = model.predict(X)
            except ValueError: 
                print('Not enough dates to make prediction on ticker {}. Returning  current dataframe.'.format(ticker))
                return df1
            a = np.array((n_steps-1)*[np.nan])
            y_pred = np.concatenate([a, y_pred.ravel()])
            df1[name] = y_pred
            count_rows += df1.shape[0]
            
        else:
            df2 = df_test.reset_index().loc[df_test.reset_index().Ticker==ticker].set_index(['Date','Ticker']).copy()
            X = df2.iloc[:,:-3].values  # chop off two y's from previous two models
            X = scaler.fit_transform(X)  # standardize the features
            y = df2.iloc[:,-3].values
            dataset = np.hstack((X, y.reshape(-1,1)))
            X, y = split_sequences(dataset, n_steps=n_steps)
            try:
                y_pred = model.predict(X)
            except ValueError:
                print('Not enough dates to make prediction on ticker {}. Returning current dataframe.'.format(ticker))
                return df1
            a = np.array((n_steps-1)*[np.nan])
            y_pred = np.concatenate([a, y_pred.ravel()])
            df2[name] = y_pred
            df1 = pd.concat([df1, df2])
            count_rows += df2.shape[0]
            df1 = df1.sort_values(by=['Date','Ticker'])
        if verbose:
            if i%(tickers.shape[0]/10)==0: print('done {} percent'.format(100*i/tickers.shape[0]))
    
    df1 = df1.sort_values(by=['Date','Ticker'])
    return df1


############## Three ML Models ##############
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import fbeta_score, make_scorer, f1_score, precision_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

##### Logistic Regression #####
def develop_logistic_model(df11_train):
    # dataset FOR SKLEARN
    N = df11_train.shape[0]
    end = int(N*0.75)

    # select a training set of first half
    X_train = df11_train.iloc[:end, :-1].values
    y_train = df11_train.iloc[:end, -1].values
    X_val = df11_train.iloc[end:, :-1].values
    y_val = df11_train.iloc[end:, -1].values

    clf = LogisticRegression(penalty='l2',
                             C=1.0,
                             random_state=0,
                             solver='sag',
                             max_iter=10000)
    # use later
    #pca = PCA(n_components=5)

    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_function_transformer.html#sphx-glr-auto-examples-preprocessing-plot-function-transformer-py
    def select_sentiment_colums(X):  # selects the 15 sentiment features
        return X[:,-16:-1]

    pipe = Pipeline(steps=[('sentiment_cols', FunctionTransformer(select_sentiment_colums, validate=True)), 
                           ('scale', StandardScaler()), 
                           ('logreg', clf)])

    # https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
    param_grid = {
        'logreg__C': [1e-2, 5e-2, 7.5e-2, 1.0]
    }

    def my_custom_loss_func(y_true, y_pred):
        c = np.array([y_pred,y_true])
        d = np.sort(c)
        e = np.flip(d, axis=1)
        f = e[:,0:int(e.shape[1]*0.5)]
        g = np.absolute(f[0,:] - f[1,:])
        return g.mean()
    my_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)

    search = GridSearchCV(pipe, 
                          param_grid, 
                          scoring='roc_auc', #'roc_auc', # my_scorer
                          iid=False,
                          cv=TimeSeriesSplit(n_splits=3))

    search.fit(X_train, y_train.ravel());
    
    y_pred = search.predict_proba(X_train)[:,1]
    #print('\n ###### Logistic Regression ###### \n')
    #print('Best params:',search.best_params_)
    #print('Mean (CV) AUC of best estimator:',search.best_score_)
    #print('Validation AUC of best estimator:',search.best_estimator_.score(X_val, y_val)) # why is this so low?
    #print('Validation AUC of best estimator:',search.score(X_val, y_val))  # should be auc
    #print(accuracy_score(y,y_pred))
    #print('Train AUC:',roc_auc_score(y_train, y_pred))
    
    return search

##### Extreme Gradient Boosting #####
import xgboost as xgb
#from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import graphviz

def develop_xgb_model(df11_train):

	# dataset FOR SKLEARN
    N = df11_train.shape[0]
    end = int(N*0.75)

    # select a training set of first half
    X_train = df11_train.iloc[:end, :-1].values
    y_train = df11_train.iloc[:end, -1].values
    X_val = df11_train.iloc[end:, :-1].values
    y_val = df11_train.iloc[end:, -1].values

    xgb_model = xgb.XGBClassifier()

    def select_sentiment_colums(X):  # selects the 15 sentiment features
        return X[:,-18:-3]

    pipe = Pipeline(steps=[#('sentiment_cols', FunctionTransformer(select_sentiment_colums, validate=True)), 
                           ('scale', StandardScaler()), 
                           ('xgb', xgb_model)])

    test_params = {
        'xgb__eta': [0.05, 0.3, 1],
        'xgb__min_child_weight': [1],
        'xgb__max_depth': [2],#,5],
        'xgb__gamma': [0],#,0.1,0.2],
        'xgb__n_estimators': [20],#30,40],
        'xgb__reg_alpha':[1e-5]#, 1e-2, 0.1]
    }

    xgb_search = GridSearchCV(pipe,
                              test_params,
                              scoring='roc_auc', #'roc_auc', # my_scorer
                              iid=False,
                              cv=TimeSeriesSplit(n_splits=3))

    xgb_search.fit(X_train, y_train.ravel())
    y_pred = xgb_search.predict_proba(X_train)
    
    #print('\n ###### XGB ###### \n')
    #print('Best params: {}'.format(xgb_search.best_params_))
    #print('Mean cv score of best estimator: {}'.format( round(xgb_search.best_score_,2)))  # shoiuld be AUC
    #print('Training AUC: {}'.format( round(roc_auc_score(y_train, y_pred[:,1]),2)))
    #print('Test AUC: {}'.format( round(roc_auc_score(y_val, xgb_search.predict_proba(X_val)[:,1])),2) )
    
    return xgb_search


##### LSTM #####
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# turorials
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

def develop_lstm_model(df11_train):
    print('\n Building dataset for LSTM...')
    x_train, y_train, x_val, y_val = build_dataset_for_rnn(df11_train, n_steps=3, frac_train=0.75)
    batch_size = 32
    n_steps, n_features = x_train.shape[1], x_train.shape[2]

    print('Build LSTM...')
    model = Sequential()
    #model.add(Embedding(input_dim=max_features, output_dim=128)) #use only for 
    #model.add(Dense(32, input_shape=(70,)))
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2, input_shape=(n_steps, n_features)))  # 50 memory units
    model.add(Dense(1, activation='sigmoid'))  # classification, so sigmoid outcome

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',  # since we're doing binary classification
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train LSTM...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10, #15
              validation_data=(x_val, y_val),
              verbose=0)
    score, acc = model.evaluate(x_val, y_val,
                                batch_size=batch_size)
    y_pred = model.predict(x_val)

    #print('\n ###### LSTM ###### \n')
    #print('Test score:', round(score,2))
    #print('Test accuracy:', round(acc,2))  # accuracy
    #print('Train AUC:', round(roc_auc_score(y_train, model.predict(x_train)),2))
    val_auc = roc_auc_score(y_val, y_pred)
    #print('Validation AUC:', round(val_auc,2))
    
    return model, val_auc


def make_predictions(df11_test, df11_all, model1, model2, model3):
    df1 = create_predictions_sklearn(df11_test, model1, name='y_pred_log_reg', num_cols_non_feats=1) # chop off y
    #print(df1.shape)
    df2 = create_predictions_sklearn(df1, model2, name='y_pred_xgb', num_cols_non_feats=2)  # chop off 2 y's
    #print(df2.shape)
    df3 = create_predictions_keras(df2, model3, n_steps=3, verbose=False)
    #print(df3.shape)
    df4 = df3.join(df11_all.iloc[:,:4], how='left')
    cols = df4.columns.tolist()
    df5 = df4[cols[-4:]+cols[:-4]].copy()  # switch order of columns
    y_pred_avg = df5.iloc[:,-3:].mean(axis=1).values
    df6 = df5.assign(y_pred_avg=y_pred_avg)
    return df6

##### prepare dataset #####
def prepare_dataset(frac_training=0.5, use_sentiment=True):
    """
	Split dataset and prepare it for the ML/NN model fitting.
	It rounds down to the month, to avoid splitting months between
	training and test.
    """

    d2 = pd.read_pickle('complete_dataset.pickle')
    #df2 = pd.read_pickle('all_tickers_combined.pickle')
    #df2a = df2.loc[:,['ticker_price_beginning', 'ticker_price_end', 'mkt_price_beginning','mkt_price_end','Sentiment']]
    df11_all = pd.read_pickle('data.pkl')
    df11 = df11_all.iloc[:, list(range(4,73))+[-1]].copy()  # select only features and target variable
    
    if not use_sentiment: df11 = df11.iloc[:,-15:].copy()  # remove sentiment features

    # dataset FOR SKLEARN
    N = df11.shape[0]
    end = int(N*frac_training)  # worked with 0.5

    remainder = end%1130
    end = end-remainder
    
    # select a training set of first half
    df11_train = df11.iloc[:end,:]
    df11_test = df11.iloc[end:,:]
    
    return df11_train, df11_test


def develop_all_three_models(frac_training=0.6, use_sentiment=True):
    """
	Splits data, builds all three models, and returns some results.
    """

    # split data
    df11_all = pd.read_pickle('data.pkl')
    df11_train, df11_test = prepare_dataset(frac_training=frac_training, use_sentiment=use_sentiment)

    # select logistic regression
    search = develop_logistic_model(df11_train)
    print('done logistic')
    
    # select xgb
    xgb_search = develop_xgb_model(df11_train)
    print('done gradient boosting')
    
    # train (not select) LSTM
    model, val_auc = develop_lstm_model(df11_train)
    print('done LSTM')
    
    # make predictions
    print('Making predictions...')
    df17_test = make_predictions(df11_test=df11_test, df11_all=df11_all, model1=search, model2=xgb_search, model3=model)
    print('done predictions...')
    
    results = pd.DataFrame({'Validation AUC':[search.best_score_, xgb_search.best_score_, val_auc]})
    results.index = ['log','xgb','lstm']
    
    print('Mean CV AUC:')
    print(results)

    return df17_test, results

    # run strategy
#    print('running strategy...')
#    df18_test = run_strategy(df17_test, ticker_dict=d2['Raw_Data'], sort_column='y_pred_avg', seed=False, frac_long_tickers=0.01, mult_extra=2, beta_cap_floor=-10., beta_cap_ceil=10., plot=False)
    
#    results = pd.DataFrame({'Validation AUC':[search.best_score_, xgb_search.best_score_, val_auc]})
#    results.index = ['log','xgb','lstm']
#    return df18_test, results


def get_results_by_training_cutoff(save_result=False):
    """
    Splits training and test set, develops models, runs strategy,
    and returns some results for plotting and analysis.
    Creates results_by_training_cutoff.pkl.
    Takes 30+ minutes to run.
    """

    print('Warning: this code could take longer than 30 minutes.')
    res = {}
    d2 = pd.read_pickle('complete_dataset.pickle')  # complete_dataset.pkl was made by Benjamin, not Robert
    for i in np.arange(0.1,1,0.2):
        print(i)
        df18_test, results = develop_all_three_models(frac_training=i)    
        df19_test = run_strategy(df18_test, ticker_dict=d2['Raw_Data'], sort_column='y_pred_avg', seed=False, frac_long_tickers=0.01, mult_extra=2, beta_cap_floor=-10., beta_cap_ceil=10., plot=False)
        res[i]=(df18_test, results, df19_test)
    if save_result:
        print('Saving dictionary to results_by_training_cutoff.pkl')
        # Create an variable to pickle and open it in write mode
        list_pickle_path = 'results_by_training_cutoff.pkl'
        list_pickle = open(list_pickle_path, 'wb')
        pickle.dump(res, list_pickle)
        list_pickle.close()
    return res







if __name__ == '__main__':  # prints when run 
    print('You just ran this from the command line')

#if __name__ == 'final_project':  # prints when you import the package
#    print('Importing final_project')