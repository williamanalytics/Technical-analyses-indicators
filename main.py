import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


snp500 = pd.read_csv("constituents.csv")
symbols = snp500['Symbol'].sort_values().tolist()

ticker = st.sidebar.selectbox(
    'Choose a S&P 500 Stock',
     symbols)


def calcMovingAverage(data, size):
    df = data.copy()
    df['sma'] = df['Adj Close'].rolling(size).mean()
    df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
    df.dropna(inplace=True)
    return df


def calc_macd(data):
    df = data.copy()
    df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
    df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df.dropna(inplace=True)
    return df

def calcRSI(data, window):
    df = data.copy()
    delta = df['Adj Close'].diff()
    delta = delta[1:]  # Exclude first NaN value
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window).mean()
    avg_loss = abs(down.rolling(window).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    df.dropna(inplace=True)
    return df


def calcBollinger(data, size):
    df = data.copy()
    df["sma"] = df['Adj Close'].rolling(size).mean()
    df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0)
    df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0)
    df["width"] = df["bolu"] - df["bold"]
    df.dropna(inplace=True)
    return df

st.title('Technical Indicators')
st.subheader('Simple and Exponential Moving Average')

coMA1, coMA2 = st.columns(2)

with coMA1:
    numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)

with coMA2:
    windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)


    start = dt.datetime.today()-dt.timedelta(numYearMA * 365)
    end = dt.datetime.today()
    dataMA = yf.download(ticker,start,end)
    df_ma = calcMovingAverage(dataMA, windowSizeMA)
    df_ma = df_ma.reset_index()

figMA = go.Figure()

figMA.add_trace(
        go.Scatter(
                x = df_ma['Date'],
                y = df_ma['Adj Close'],
                name = "Prices Over Last " + str(numYearMA) + " Year(s)"
            )
    )

figMA.add_trace(
            go.Scatter(
                    x = df_ma['Date'],
                    y = df_ma['sma'],
                    name = "SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                )
        )

figMA.add_trace(
            go.Scatter(
                    x = df_ma['Date'],
                    y = df_ma['ema'],
                    name = "EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                )
        )

figMA.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

figMA.update_layout(legend_title_text='Trend')
figMA.update_yaxes(tickprefix="$")

st.plotly_chart(figMA, use_container_width=True)


st.subheader('Moving Average Convergence Divergence (MACD)')
numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2)

startMACD = dt.datetime.today()-dt.timedelta(numYearMACD * 365)
endMACD = dt.datetime.today()
dataMACD = yf.download(ticker,startMACD,endMACD)
df_macd = calc_macd(dataMACD)
df_macd = df_macd.reset_index()

figMACD = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.01)

figMACD.add_trace(
        go.Scatter(
                x = df_macd['Date'],
                y = df_macd['Adj Close'],
                name = "Prices Over Last " + str(numYearMACD) + " Year(s)"
            ),
        row=1, col=1
    )

figMACD.add_trace(
        go.Scatter(
                x = df_macd['Date'],
                y = df_macd['ema12'],
                name = "EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
            ),
        row=1, col=1
    )

figMACD.add_trace(
        go.Scatter(
                x = df_macd['Date'],
                y = df_macd['ema26'],
                name = "EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
            ),
        row=1, col=1
    )

figMACD.add_trace(
        go.Scatter(
                x = df_macd['Date'],
                y = df_macd['macd'],
                name = "MACD Line"
            ),
        row=2, col=1
    )

figMACD.add_trace(
        go.Scatter(
                x = df_macd['Date'],
                y = df_macd['signal'],
                name = "Signal Line"
            ),
        row=2, col=1
    )

figMACD.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="left",
    x=0
))

figMACD.update_yaxes(tickprefix="$")
st.plotly_chart(figMACD, use_container_width=True)





st.subheader('Bollinger Band and RSI')
coBoll1, coBoll2 = st.columns(2)
with coBoll1:
    numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6)

with coBoll2:
    windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)

startBoll= dt.datetime.today()-dt.timedelta(numYearBoll * 365)
endBoll = dt.datetime.today()
dataBoll = yf.download(ticker,startBoll,endBoll)
df_boll_rsi = calcBollinger(dataBoll, windowSizeBoll)
df_boll_rsi = calcRSI(df_boll_rsi, 14)  # Adding RSI calculation with a window of 14
df_boll_rsi = df_boll_rsi.reset_index()

figBollRSI = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.01)

figBollRSI.add_trace(
        go.Scatter(
                x = df_boll_rsi['Date'],
                y = df_boll_rsi['bolu'],
                name = "Upper Band"
            ),
        row=1, col=1
    )


figBollRSI.add_trace(
            go.Scatter(
                    x = df_boll_rsi['Date'],
                    y = df_boll_rsi['sma'],
                    name = "SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
                ),
        row=1, col=1
        )

figBollRSI.add_trace(
            go.Scatter(
                    x = df_boll_rsi['Date'],
                    y = df_boll_rsi['bold'],
                    name = "Lower Band"
                ),
        row=1, col=1
        )

figBollRSI.add_trace(
        go.Scatter(
                x = df_boll_rsi['Date'],
                y = df_boll_rsi['RSI'],
                name = "RSI"
            ),
        row=2, col=1
    )

figBollRSI.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="left",
    x=0
))

figBollRSI.update_yaxes(tickprefix="$")
st.plotly_chart(figBollRSI)
