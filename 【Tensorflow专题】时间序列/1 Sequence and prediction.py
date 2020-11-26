#%% 1 时间序列引入

# Time series is defined as an ordered sequence of values that usually equally spaced over time;
# Multivariate Time series: Multiple values at each time step
# Multivariate TS: useful ways of understanding the impact of related data.

# ==========================================
# Machine Learning in TS:
# -> Prediction of forecasting based on the data
# -> Imputation: 归因；根据现在的回溯历史没有的；
# -> Fill hole of data which doesn't exist
# -> Detect anomalies：异常检测
# -> Spot Patterns for TS: (如识别声波曲线->翻译成words)

#%% 2 TS的共同特性

# 1 Trend
# 2 Seasonality
# 3 Trend + Seasonality
# 4 White Noise
# 5 Auto-correlated TS: 如 V(t) = 0.99 * v(t-1) + Occasional Spike (其中V(t)和V(t-1)高度相关，Spike(难以预测) called innovations --> 无法根据历史预测Spike)
# 6 Multiple AutoCorrelations TS: 如 V(t) = 0.7 * V(t-1) + 0.2 * V(t-50) + Occasional Spike
# 7 Non-Stationary TS: 在某个中间节点，其规律发生了改变（如经济危机等）（对这种数据，我们可以只看规律改变后的数据）
# ==========================================
# 真实数据中经常有多种类型的组合TS，机器学习就是设计为发现这个规律，然后预测

#%% 3 Synthetic TS data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


#%% 3.1 Trend and Seasonality

def trend(time, slope=0):
    return slope * time

# Let's create a time series that just trends upward:
time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.1)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

#%% 3.2 Now let's generate a time series with a seasonal pattern:
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


#%% 3.3 Now let's create a time series with both trend and seasonality:
slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


#%% 3.4 Noise

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()

# 添加噪声
series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

#%% 3.5 我们尝试预测它
# All right, this looks realistic enough for now.
# Let's try to forecast it. We will split it into two periods:
# the training period and the validation period (in many cases,
# you would also want to have a test period). The split will be at time step 1000.

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


#%% 3.6 自回归项
def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ1 = 0.5
    φ2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += φ1 * ar[step - 50]
        ar[step] += φ2 * ar[step - 33]
    return ar[50:] * amplitude


def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude

series = autocorrelation(time, 10, seed=42)
plot_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + trend(time, 2)
plot_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
plot_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
series2 = autocorrelation(time, 5, seed=42) + seasonality(time, period=50, amplitude=2) + trend(time, -1) + 550
series[200:] = series2[200:]
#series += noise(time, 30)
plot_series(time[:300], series[:300])
plt.show()

#%% 3.7 随机pulse
def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series

series = impulses(time, 10, seed=42)
plot_series(time, series)
plt.show()

def autocorrelation(source, φs):
    ar = source.copy()
    max_lag = len(φs)
    for step, value in enumerate(source):
        for lag, φ in φs.items():
            if step - lag > 0:
              ar[step] += φ * ar[step - lag]
    return ar

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.99})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.70, 50: 0.2})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

series_diff1 = series[1:] - series[:-1]
plot_series(time[1:], series_diff1)


# 可以发现自回归项
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)
plt.show()
#%% 尝试使用ARIMA建模

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())