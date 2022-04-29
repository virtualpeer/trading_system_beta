import mlfinlab
from mlfinlab.datasets import (load_tick_sample, load_stock_prices, load_dollar_bar_sample)
tick_df = load_tick_sample()
dollar_bars_df = load_dollar_bar_sample()
stock_prices_df = load_stock_prices()