import yfinance as yf
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import finplot as fplt
import numpy as np
import time

companies_list = ['AMZN','GOOG','WMT','TSLA','META']
period_list = ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']
intervals_list = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']

st.set_page_config()

with st.container():

# for candelsticks


# with exception handling and class 



# uploading with github readme 

# to give title to your application
st.title("Stock Fundamental and Technical Analysis App")


# to give a subheading
st.write("Select a stock ticker to view its fundamentals and perform technical analysis")


# def registration_form():
#     user_name = st.text_input(label="Name")
#     user_email_id = st.text_input(label= 'Email')
#     user_password = st.text_input(label ="Password", type='password')
#     user_image = st.file_uploader(label="Profile Photo", accept_multiple_files=False, type='jpg')
#     return user_name,user_email_id,user_password, user_image


#create a registration or login form, using a db (mysql), sending email verfication code
# with st.form("Register/Login", clear_on_submit=True):
#     user_name, user_email, user_password,user_image  = registration_form()
#     print(user_name)
#     st.form_submit_button("Register")
    # st.rerun

# Then moving to next page 

# Creating a form, making the selection required and have to use callback function
with st.form("Select Ticker",clear_on_submit=True):
    user_ticker_selected = st.selectbox("Pick a Ticker", companies_list)
    user_sentiments = st.text_area(placeholder="Input your sentiment for this stock", max_chars=500, label="Sentiments")
    form_submit =  st.form_submit_button("Submit")
    # st.rerun

#forming two columns 
col1,col2 = st.columns(2)

# or here session_state can also be used
if form_submit:
    st.write(f"You picked {user_ticker_selected}")
    st.write(f"Your Sentiments: {user_sentiments}")


def sidebar_elements():
    st.write("About Application or other such")
    st.button("Sidebar button", key='sidebar_button ')
    st.slider("With default, no key", 1, 10, value=5)
    st.selectbox("Select the Period", period_list)

with st.sidebar:
    sidebar_elements()


with st.spinner("Loading the Fundamentals and Technical Aspects"):
    time.sleep(2)

#creating a chart 
def simple_chart():
    st.write("When you move the slider, only the chart updates")
    val = st.slider("Number of bars", 1, 20, 4)
    st.bar_chart(np.random.default_rng().random(val))

st.subheader(f"Fundamentals of {user_ticker_selected}")

#Main data fetching from yahoo finance
stock = yf.Ticker(user_ticker_selected)


#from date
# to date


# end_date = datetime.now().strftime('%Y-%m-%d')


info = stock.info
st.write(f"**Sector**: {info['sector']}")
st.write(f"**Industry**: {info['industry']}")
st.write(f"**Market Cap**: {info['marketCap']}")
st.write(f"**PE Ratio (TTM)**: {info.get('trailingPE', 'N/A')}")
st.write(f"**EPS (TTM)**: {info.get('trailingEps', 'N/A')}")
st.write(f"**Dividend Yield**: {info.get('dividendYield', 'N/A')}")

st.write("### Financials")
st.write(stock.financials)

# Show balance sheet
st.write("### Balance Sheet")
st.write(stock.balance_sheet)

# Show cash flow
st.write("### Cash Flow")
st.write(stock.cashflow)

#Show Shareholders
st.write('### Share Holders')
st.write(stock.institutional_holders)

#Show Mutual Fund Holders
st.write('### Mutual Funds Holders')
st.write(stock.mutualfund_holders)

#Recommendations
st.write('# Analyist Recommendations')
st.write(stock.recommendations)

#Getting history 
tickerDf = stock.history(period='1d', start='2010-5-31', end='2020-5-31')

st.subheader(f"Technical Analysis of {user_ticker_selected}")
hist = stock.history(period="1y")


# Plot historical price data
st.write("### Stock Price")
st.line_chart(hist['Close'])


# Moving Averages
st.write("### Moving Averages")
short_window = 20
long_window = 50
hist['MA20'] = hist['Close'].rolling(window=short_window).mean()
hist['MA50'] = hist['Close'].rolling(window=long_window).mean()


fig, ax = plt.subplots()
ax.plot(hist['Close'], label='Close Price', alpha=0.5)
ax.plot(hist['MA20'], label=f'{short_window}-Day MA', linestyle='--')
ax.plot(hist['MA50'], label=f'{long_window}-Day MA', linestyle='--')
ax.legend()
st.pyplot(fig)

# RSI Calculation
st.write("### Relative Strength Index (RSI)")
delta = hist['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))

# Plot RSI
fig2, ax2 = plt.subplots()
ax2.plot(hist['RSI'], label='RSI', color='purple')
ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
ax2.legend()
st.pyplot(fig2)

# you will get MultiLevel Index with a list so transform it to single index set of column
# tickers_hist.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)


# Volume -- Area plot 
# #Plot out the volume graph for Tesla stock as well
# tsla_volume_chart = px.area(tsla['Volume'], 
#                             title='Tesla Daily Volume', 
#                             color_discrete_map={'Volume':'red'} , 
#                             width=800, height=400)
# tsla_volume_chart.show()


# # Getting the percentage change
# prices['Returns'] = prices['Close'].pct_change()
# prices.head()



 

# #amzn_hist = amzn.history(period='max',end=end_date,interval='1m')


# # Fundamental Data Analysis 

# #get_analysis, get_actions, get_balance_sheet, get_calendar, get_cashflow, get_info, get_institutional_holders, get_news, get_recommendations, get_sustainability


# tsla = yf.Ticker('TSLA')

# # CALL THE MULTIPLE FUNCTIONS AVAILABLE AND STORE THEM IN VARIABLES.
# actions = tsla.get_actions()
# analysis = tsla.get_analysis()
# balance = tsla.get_balance_sheet()
# calendar = tsla.get_calendar()
# cf = tsla.get_cashflow()
# info = tsla.get_info()
# inst_holders = tsla.get_institutional_holders()
# news = tsla.get_news()
# recommendations = tsla.get_recommendations()
# sustainability = tsla.get_sustainability()

# # Financials

# financials = ticker.financials

# # PRINT THE RESULTS
# print(actions)
# print('*'*20)
# print(analysis)
# print('*'*20)
# print(balance)
# print('*'*20)
# print(calendar)
# print('*'*20)
# print(cf)
# print('*'*20)
# print(info)
# print('*'*20)
# print(inst_holders)
# print('*'*20)
# print(news)
# print('*'*20)
# print(recommendations)
# print('*'*20)
# print(sustainability)
# print('*'*20)


# # Options Chain Data 

# tsla = yf.Ticker('TSLA')

# # FETCH OPTIONS CHAIN DATA FOR THE COMPANY
# tsla_options = tsla.option_chain()

# # ACCESS BOTH THE CALLS AND PUTS AND STORE THEM IN THEIR RESPECTIVE VARIABLES
# tsla_puts = tsla_options.puts
# tsla_calls = tsla_options.calls


# # Candelstick graph
# df = tsla.history(interval='1d',period='1y')

# # PLOT THE OHLC CANDLE CHART
# fplt.candlestick_ochl(df[['Open','Close','High','Low']])
# fplt.show()


# import pandas as pd
# df_45min = df.groupby(pd.Grouper(freq='45Min')).agg({"Open": "first", 
#                                              "High": "max", 
#                                              "Low": "min", 
#                                              "Close": "min",
#                                              "Volume": "sum"})


# # download data in csv format