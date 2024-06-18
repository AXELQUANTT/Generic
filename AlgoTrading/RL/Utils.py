"""
Utils package containing all the helper functions
needed for the RL algo
"""

import yfinance as yf
from pytickersymbols  import PyTickerSymbols

def retrieve_loader(t_start, t_end) -> list[str]:
    pt_tickers = PyTickerSymbols()
    tickers = pt_tickers.get_dow_jones_nyc_yahoo_tickers("DOW JONES")
    data = yf.download(tickers, start=t_start, end=t_end, auto_adjust=True)
    return data

df = retrieve_loader(t_start='2022-01-01', t_end='2022-12-31')
print(df)