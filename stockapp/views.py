from django.shortcuts import render
from django.views import View as V
from django.conf import settings
from .forms import StockForm
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import os

plt.rcParams['font.family'] = 'MS Gothic'

df=pd.read_csv('stockapp/codes.csv')
name_to_code=dict(zip(df['銘柄名'],df['銘柄コード']))


def plot_technical_chart(company_name, company_code, start, end, save_path):

    if isinstance(start, str):
        start = datetime.strptime(start, '%Y-%m-%d')
    if isinstance(end, str):
        end = datetime.strptime(end, '%Y-%m-%d')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = yf.download(company_code, start=start, end=end)

    date = df.index
    price = df['Close']
    close = df['Close'].squeeze()
    span1, span2, span3 = 5, 25, 50
    df['sma1'] = price.rolling(window=span1).mean()
    df['sma2'] = price.rolling(window=span2).mean()
    df['sma3'] = price.rolling(window=span3).mean()

    def calc_macd(close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - signal_line
        return macd, signal_line, macd_hist

    def calc_rsi(close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calc_bollinger_bands(close, period=25, num_std=2):
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return upper, ma, lower
    
    def calc_rci(close, period=9):

        rci = []

        for i in range(len(close)):
            if i < period - 1:
                rci.append(None)  
                continue

            close_slice = close[i - period + 1:i + 1].squeeze()
            date_rank = list(range(period, 0, -1))
            price_rank = close_slice.rank(ascending=False).values.tolist()

            d = [(date_rank[j] - price_rank[j]) for j in range(period)]

            rci_value = (1 - (6 * sum([diff ** 2 for diff in d])) / (period * (period ** 2 - 1))) * 100
            rci.append(rci_value)

        return pd.Series(rci, index=close.index)

    df['macd'], df['macdsignal'], df['macdhist'] = calc_macd(close)
    df['RSI'] = calc_rsi(close, period=25)
    df['upper'], df['middle'], df['lower'] = calc_bollinger_bands(close)
    df['RCI'] = calc_rci(df['Close'], period=9)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(date, price, label=company_name)
    plt.plot(date, df['sma1'], label='SMA1')
    plt.plot(date, df['sma2'], label='SMA2')
    plt.plot(date, df['sma3'], label='SMA3')
    plt.fill_between(date, df['upper'], df['lower'], color='gray', alpha=0.3)
    plt.title(f'{company_name}の株価')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(date, df['RCI'], label='RCI(9)', color='green')
    plt.ylim(-100, 100)
    plt.axhline(80, color='red', linestyle='dashed')
    plt.axhline(-80, color='blue', linestyle='dashed')
    plt.title('RCI(9)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.fill_between(date, df['macdhist'], color='gray', alpha=0.5, label='MACD')
    plt.title('MACD')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(date, df['RSI'], label='RSI', color='black')
    plt.ylim(0, 100)
    plt.title('RSI')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#---------------------------------------------------------------------------------------------------------------

class IndexView(V):
    def get(self,request):
        return render(request,'stockapp/index.html')
    
class StockAnalysis(V):

    template_name='stockapp/analyze.html'

    def get(self,request):
        form=StockForm()
        return render(request,'stockapp/analyze.html',{'form':form})
    
    def post(self,request):
        form=StockForm(request.POST)
        result=None
        chart_path=None

        if form.is_valid():
            stock_name=form.cleaned_data['code']
            start=form.cleaned_data['start_date']
            end=form.cleaned_data['end_date']
            stock_code=name_to_code.get(stock_name)

            if stock_code:
                try:

                    start_str = start.strftime('%Y-%m-%d')
                    end_str = end.strftime('%Y-%m-%d')

                    save_path=os.path.join(settings.BASE_DIR,'stockapp','static','stockapp','graph.png')
                    plot_technical_chart(stock_name,stock_code,start_str,end_str,save_path)
                    chart_path='stockapp/graph.png'
                    result=f'{stock_name}({stock_code})のグラフを表示しています'
                except Exception as e:
                    tb = traceback.format_exc()
                    result = f"エラーが発生しました：{e}<br><pre>{tb}</pre>"
            else:
                result=f'「{stock_name}」に対する銘柄コードが見つかりませんでした'
        
        return render(request,self.template_name,{'form':form,'result':result,'chart_path':chart_path})


index=IndexView.as_view()
analyze=StockAnalysis.as_view()
