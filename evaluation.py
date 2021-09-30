# several methods for evaluating and plotting models performane

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import math
from tensorflow import keras
from utils import sequentialize


# train model
def quick_train_eval(untrained_model,train_data,val_data,y_train,y_val,batch_size,epochs,seq_lenght):
    optimizer = keras.optimizers.Adam(learning_rate=0.0001,clipnorm=1.0)
    x_train, _ =  sequentialize(train_data,seq_lenght)
    x_val, _ = sequentialize(val_data,seq_lenght)
    y_train = y_train[seq_lenght:]
    y_val = y_val[seq_lenght:]
    model = keras.models.clone_model(untrained_model)
    model.compile(optimizer=optimizer,loss=keras.losses.MSE)
    history = model.fit(x_train,y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose = 0
    )

    return model


# train and evalute the model in 3 different holdout periods
def multiple_period_evaluation(untrained_model,data,best_alloc,strategy_daily_returns,batch_size = 5, epochs=5,seq_lenght=125,plots=False):
    """
    :type data: dataframe dateindexed
    """
    data = data.iloc[data.index >= best_alloc.index[0]]

    
    # train on 2014-2018 and evaluate in 2019:

    train_data = data.iloc[data.index < '2019-01-01'].values
    val_data = data.iloc[data.index >= '2019-01-01']
    val_dates = val_data.index
    val_data = val_data.values
    y_train = best_alloc.iloc[best_alloc.index < '2019-01-01'].values
    y_val = best_alloc.iloc[best_alloc.index >= '2019-01-01'].values
    model1 = quick_train_eval(untrained_model,train_data,val_data,y_train,y_val,batch_size,epochs,seq_lenght) 
    x_val, _ = sequentialize(val_data,seq_lenght)
    alloc = model1.predict(x_val)
    alloc = pd.DataFrame(alloc,columns=strategy_daily_returns.columns,index=val_dates[seq_lenght:])
    perf_val1 = get_comulated_returns(alloc,strategy_daily_returns)[0][1][-1]
    
    if plots:
        alloc.plot(figsize=(12,8))
        plt.show()
        plot_comulated_returns(alloc,strategy_daily_returns.loc[(strategy_daily_returns.index >= '2020-01-01') & (strategy_daily_returns.index <= '2020-12-31')],target_alloc=best_alloc,equal_allocation=True, risk_parity=True, inverse_volatility=True)
        plt.show()
        plot_comulated_returns(alloc,strategy_daily_returns.loc[(strategy_daily_returns.index <= '2019-12-31') & (strategy_daily_returns.index >= '2019-01-01')],target_alloc=best_alloc,equal_allocation=True, risk_parity=True, inverse_volatility=True)
        plt.show() 
        plot_sharpes(seq_lenght,alloc,strategy_daily_returns,target_alloc=best_alloc,equal_allocation=True,risk_parity=True, inverse_volatility=True)

    # train on 2015-2020 and evaluate in 2014
    train_data = data.iloc[data.index >= '2015-01-01'].values
    val_data = data.iloc[data.index < '2015-01-01']
    val_dates = val_data.index
    val_data = val_data.values
    y_train = best_alloc.iloc[best_alloc.index >= '2015-01-01'].values
    y_val = best_alloc.iloc[best_alloc.index < '2015-01-01'].values
    model2 = quick_train_eval(untrained_model,train_data,val_data,y_train,y_val,batch_size,epochs,seq_lenght)
    x_val, _ = sequentialize(val_data,seq_lenght)
    alloc = model2.predict(x_val)
    alloc = pd.DataFrame(alloc,columns=strategy_daily_returns.columns,index=val_dates[seq_lenght:])
    perf_val2 = get_comulated_returns(alloc,strategy_daily_returns)[0][1][-1]
    
    if plots:
        alloc.plot(figsize=(12,8))
        plt.show()
        plot_comulated_returns(alloc,strategy_daily_returns,target_alloc=best_alloc,equal_allocation=True, risk_parity=True, inverse_volatility=True)
        plt.show() 
        plot_sharpes(seq_lenght,alloc,strategy_daily_returns,target_alloc=best_alloc,equal_allocation=True,risk_parity=True, inverse_volatility=True)
    
        
    # Training on external periods 2013-2015, 2018-2020 and validation on the middle period 2016-2017
    # cant use quick_train_eval() need to explicit code
    training_period_1 = (data.index <= '2016-01-01') 
    training_period_2 = (data.index >= '2018-01-01') 
    train_data_1 = data.iloc[training_period_1==1].values
    train_data_2 = data.iloc[training_period_2==1].values
    val_data = data.iloc[(1-(training_period_1 | training_period_2))==1]
    val_dates = val_data.index
    val_data = val_data.values
    y_train_1 = best_alloc.iloc[training_period_1==1].values
    y_train_2 = best_alloc.iloc[training_period_2==1].values
    y_val = best_alloc.iloc[(1-(training_period_1 | training_period_2))==1].values
    optimizer = keras.optimizers.Adam(learning_rate=0.0001,clipnorm=1.0)
    x_train_1, _ = sequentialize(train_data_1,seq_lenght)
    x_train_2, _ = sequentialize(train_data_2,seq_lenght)
    x_train = np.concatenate((x_train_1,x_train_2),axis=0)
    x_val, _ = sequentialize(val_data,seq_lenght)
    y_train = np.concatenate((y_train_1[seq_lenght:],y_train_2[seq_lenght:]),axis=0)
    y_val = y_val[seq_lenght:]
    model3 = keras.models.clone_model(untrained_model)
    model3.compile(optimizer=optimizer,loss=keras.losses.MSE)
    history = model3.fit(x_train,y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose = 0
    )
    alloc = model3.predict(x_val)
    alloc = pd.DataFrame(alloc,columns=strategy_daily_returns.columns,index=val_dates[seq_lenght:])
    perf_val3 = get_comulated_returns(alloc,strategy_daily_returns)[0][1][-1]
    
    if plots:
        alloc.plot(figsize=(12,8))
        plt.show()
        plot_comulated_returns(alloc,strategy_daily_returns,target_alloc=best_alloc,equal_allocation=True, risk_parity=True, inverse_volatility=True)
        plt.show() 
        plot_sharpes(seq_lenght,alloc,strategy_daily_returns,target_alloc=best_alloc,equal_allocation=True,risk_parity=True, inverse_volatility=True)
    
    # perf_val1, perf_val2, perf_val3
    return (perf_val1 + perf_val2 + perf_val3) / 3

def get_comulated_return(weighted_returns):
    portfolio_daily_return = weighted_returns.sum(axis=1)
    portfolio_cumol_ret = (1 + portfolio_daily_return).cumprod() -1
    return portfolio_cumol_ret

def plot_sharpes(window,alloc,returns, target_alloc=None, equal_allocation=False,
                risk_parity=False,inverse_volatility=False):
    return_list, dates = get_sharpes(window,alloc,returns,target_alloc,equal_allocation,
                                    risk_parity,inverse_volatility)
    pd.concat([pd.DataFrame(item[1],index=dates,columns=[item[0]]) for item in return_list],axis=1).plot(figsize=(16,8),colormap='tab10',grid=True,fontsize=18)
    plt.title("Sharpe ratios with window {}".format(window),fontsize=18)
    plt.legend(prop={'size': 18})
    plt.show()

def plot_comulated_returns(alloc,returns, target_alloc=None, equal_allocation=False,
                risk_parity=False,inverse_volatility=False):
    return_list = get_comulated_returns(alloc,returns,target_alloc,equal_allocation,
                                       risk_parity, inverse_volatility)
    dates = alloc.index
    pd.concat([pd.DataFrame(item[1],index=dates,columns=[item[0]]) for item in return_list],axis=1).plot(figsize=(16,8),colormap='tab10',grid=True,fontsize=18)
    plt.title("Cumulative return",fontsize=18)
    plt.legend(prop={'size': 18})
    plt.show()
    
def plot_volatilities(window,alloc,returns, target_alloc=None, equal_allocation=False,
                risk_parity=False,inverse_volatility=False):
    return_list, dates = get_volatilities(window,alloc,returns,target_alloc,equal_allocation,
                                         risk_parity, inverse_volatility)
    pd.concat([pd.DataFrame(item[1],index=dates,columns=[item[0]]) for item in return_list],axis=1).plot(figsize=(12,8),colormap='tab10',grid=True)
    plt.title("volatility with window {}".format(window))
    plt.legend()
    plt.show()

def get_windowed_sharpe(weighted_returns,window):
    return np.array([get_sharpe(weighted_returns.iloc[(i-window):i]) for i in range(window,weighted_returns.shape[0])])
    
def get_sharpe(weighted_returns):
    sharpe = get_comulated_return(weighted_returns)[-1] / get_portfolio_volatility(weighted_returns)
    return sharpe * (math.sqrt(252) / weighted_returns.shape[0])

def get_windowed_volatility(weighted_returns,window):
    return weighted_returns.sum(axis=1).rolling(window).std().iloc[window:]
    
def get_portfolio_volatility(weighted_returns):
    portfolio_daily_return = weighted_returns.sum(axis=1)
    return portfolio_daily_return.std()

    
def align_dates(alloc,returns, target_alloc=None):
    starting_date = max(alloc.index[0],returns.index[0])
    ending_date = min(alloc.index[-1],returns.index[-1])
    

        
    if target_alloc is not None:
        starting_date = max(starting_date,target_alloc.index[0])
        ending_date = min(ending_date,target_alloc.index[-1])
        
    return starting_date, ending_date
    
def get_comulated_returns(alloc,returns, target_alloc=None, equal_allocation=False,
                risk_parity=False,inverse_volatility=False):

    starting_date, ending_date = align_dates(alloc,returns,target_alloc)

    alloc_cut = alloc.loc[starting_date:ending_date]
    returns_cut = returns.loc[starting_date:ending_date]
    
    return_list = [('new allocation',get_comulated_return(alloc_cut*returns_cut))]

        
    if risk_parity is not None and risk_parity is not False:
        risk_parity = pd.read_csv("risk_parity.csv",index_col=0,parse_dates=True)
        risk_parity_cut = risk_parity.loc[starting_date:ending_date]
        return_list.append(('risk parity',get_comulated_return(risk_parity_cut*returns_cut)))
        
    if inverse_volatility is not None and inverse_volatility is not False:
        inverse_volatility = pd.read_csv("inverse_volatility.csv",index_col=0,parse_dates=True)
        inverse_volatility_cut = inverse_volatility.loc[starting_date:ending_date]
        return_list.append(('inverse volatility',get_comulated_return(inverse_volatility_cut*returns_cut)))
        
        
    
    if target_alloc is not None:
        target_alloc_cut = target_alloc.loc[starting_date:ending_date]
        return_list.append(('target allocation',get_comulated_return(target_alloc_cut*returns_cut)))
    
    
    if equal_allocation:
        equal_alloc = pd.DataFrame(np.full((returns_cut.shape[0],4),0.25),
                                   index=returns_cut.index,
                                   columns=returns_cut.columns)
        return_list.append(('1/N allocation',get_comulated_return(equal_alloc*returns_cut)))
        
        
    return return_list

def get_sharpes(window,alloc,returns, target_alloc=None, equal_allocation=False,
                risk_parity=False,inverse_volatility=False):
    
    starting_date, ending_date = align_dates(alloc,returns,target_alloc)

    alloc_cut = alloc.loc[starting_date:ending_date]
    returns_cut = returns.loc[starting_date:ending_date]
    
    return_list = [('new allocation',get_windowed_sharpe(alloc_cut*returns_cut,window))]
    
    dates = returns_cut.index[window:]
        
    if risk_parity is not None and risk_parity is not False:
        risk_parity = pd.read_csv("risk_parity.csv",index_col=0,parse_dates=True)
        risk_parity_cut = risk_parity.loc[starting_date:ending_date]
        return_list.append(('risk parity',get_windowed_sharpe(risk_parity_cut*returns_cut,window)))
        
    if inverse_volatility is not None and inverse_volatility is not False:
        inverse_volatility = pd.read_csv("inverse_volatility.csv",index_col=0,parse_dates=True)
        inverse_volatility_cut = inverse_volatility.loc[starting_date:ending_date]
        return_list.append(('inverse volatility',get_windowed_sharpe(inverse_volatility_cut*returns_cut,window)))
    
    if target_alloc is not None:
        target_alloc_cut = target_alloc.loc[starting_date:ending_date]
        return_list.append(('target allocation',get_windowed_sharpe(target_alloc_cut*returns_cut,window)))
    
    
    if equal_allocation:
        equal_alloc = pd.DataFrame(np.full((returns_cut.shape[0],4),0.25),
                                   index=returns_cut.index,
                                   columns=returns_cut.columns)
        return_list.append(('1/N allocation',get_windowed_sharpe(equal_alloc*returns_cut,window)))
        
        
    return return_list, dates

def get_volatilities(window,alloc,returns, target_alloc=None, equal_allocation=False,
                risk_parity=False,inverse_volatility=False):
    starting_date, ending_date = align_dates(alloc,returns,target_alloc)

    alloc_cut = alloc.loc[starting_date:ending_date]
    returns_cut = returns.loc[starting_date:ending_date]
    
    return_list = [('new allocation',get_windowed_volatility(alloc_cut*returns_cut,window))]
    
    dates = returns_cut.index[window:]
        
     
    if risk_parity is not None and risk_parity is not False:
        risk_parity = pd.read_csv("risk_parity.csv",index_col=0,parse_dates=True)
        risk_parity_cut = risk_parity.loc[starting_date:ending_date]
        return_list.append(('risk parity',get_windowed_volatility(risk_parity_cut*returns_cut,window)))
        
    if inverse_volatility is not None and inverse_volatility is not False:
        inverse_volatility = pd.read_csv("inverse_volatility.csv",index_col=0,parse_dates=True)
        inverse_volatility_cut = inverse_volatility.loc[starting_date:ending_date]
        return_list.append(('inverse volatility',get_windowed_volatility(inverse_volatility_cut*returns_cut,window)))
    
    if target_alloc is not None:
        target_alloc_cut = target_alloc.loc[starting_date:ending_date]
        return_list.append(('target allocation',get_windowed_volatility(target_alloc_cut*returns_cut,window)))
    
    
    if equal_allocation:
        equal_alloc = pd.DataFrame(np.full((returns_cut.shape[0],4),0.25),
                                   index=returns_cut.index,
                                   columns=returns_cut.columns)
        return_list.append(('1/N allocation',get_windowed_volatility(equal_alloc*returns_cut,window)))
        
        
    return return_list, dates