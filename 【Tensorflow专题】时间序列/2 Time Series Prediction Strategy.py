#%% 1 Train/Validation/test sets for TS

# Naive Forcast (使用上一个发生等值当作本次等预测)，当作base line
# 训练集合划分方法
# (1) Fixed partition: 分成3部分，按照固定时间节点划分开
# (2) Roll-forward Partitioning: start with a short training period, and we gradually increase it,
#     say by one day at a time, or by one week at a time. At each iteration, we train the model on a training period.
#     And we use it to forecast the following day, or the following week, in the validation period


#%% 2 Metrics for Evaluating performance

"""
errors = forecasts - actual
mse = np.square(errors).mean()
rmse = np.sqrt(mse)
mae = np.abs(errors).mean()
mape = np.abs(errors/x_valid).mean()

# MAE
keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()
"""

#%% 3 Moving Average and differencing

# MA(t)
# Series(t) - Series(t-365) # 消除Seasonality
# 可以在此基础上，进一步消除past noise(预测的时候，我们加上过去的centre windows-ma，而非raw hist data)
# 使用trailing window/centre window for MA(t)
