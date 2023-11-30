from typing import Union, List
from pytrends.request import TrendReq
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.tsatools import lagmat
from datetime import datetime, timedelta
import time


class Analysis:

    """ Choose the market """

    BENCHMARK_TICKER = '^NDX'
    PORTFOLIO_COMPOSITION = None
    GOOGLE_KEYWORDS = None

    def __init__(self, market='america'):

        if market == 'america':
            Analysis.PORTFOLIO_COMPOSITION = list(['ATVI', 'ADBE', 'AMD', 'ALGN', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'BIIB',
                                                   'ANSS', 'AAPL', 'AMAT', 'ASML', 'TEAM', 'ADSK', 'ADP', 'BIDU', 'ADI',
                                                   'CDNS', 'CDW', 'CERN', 'CHTR', 'CHKP', 'CTAS', 'CSCO', 'CTSH', 'EA',
                                                   'CPRT', 'COST', 'CSX', 'DXCM', 'DLTR', 'KDP', 'EBAY', 'ILMN', 'EXC',
                                                   'FB', 'FAST', 'FISV', 'GILD', 'IDXX', 'CMCSA', 'INCY', 'INTC', 'INTU',
                                                   'ISRG', 'JD', 'KLAC', 'KHC', 'LRCX', 'LULU', 'MAR', 'MRVL', 'MTCH',
                                                   'MELI', 'MCHP', 'MU', 'MSFT', 'MDLZ', 'MNST', 'NTES', 'NFLX', 'NVDA',
                                                   'NXPI', 'ORLY', 'OKTA', 'PCAR', 'PYPL', 'PEP', 'BKNG', 'QCOM', 'REGN',
                                                   'ROST', 'SGEN', 'SIRI', 'SWKS', 'SPLK', 'SBUX', 'SNPS', 'TSLA',
                                                   'TXN', 'TMUS', 'VRSK', 'VRTX', 'WBA', 'WDAY', 'XEL', 'PAYX', 'VRSN'])

            Analysis.GOOGLE_KEYWORDS = list(['ATVI', 'ADBE', 'AMD', 'ALGN', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'BIIB',
                                             'ANSS', 'AAPL', 'AMAT', 'ASML', 'TEAM', 'ADSK', 'ADP', 'BIDU', 'ADI',
                                             'CDNS', 'CDW', 'CERN', 'CHTR', 'CHKP', 'CTAS', 'CSCO', 'CTSH', 'EA',
                                             'CPRT', 'COST', 'CSX', 'DXCM', 'DLTR', 'KDP', 'EBAY', 'ILMN','EXC', 'FB',
                                             'FAST', 'FISV', 'GILD', 'IDXX', 'CMCSA', 'INCY', 'INTC', 'INTU', 'ISRG',
                                             'JD', 'KLAC', 'KHC', 'LRCX', 'LULU', 'MAR', 'MRVL', 'MTCH', 'MELI', 'MCHP',
                                             'MU', 'MSFT', 'MDLZ', 'MNST', 'NTES', 'NFLX', 'NVDA', 'NXPI', 'ORLY',
                                             'OKTA', 'PCAR', 'PYPL', 'PEP', 'BKNG', 'QCOM', 'REGN', 'ROST', 'SGEN',
                                             'SIRI', 'SWKS', 'SPLK', 'SBUX', 'SNPS', 'TSLA', 'TXN', 'TMUS',
                                             'VRSK', 'VRTX', 'WBA', 'WDAY', 'XEL', 'PAYX', 'VRSN'])

        else:
            raise AttributeError('not applicable market')

        self.market = market
        self.start = "2019-01-01"
        self.end = "2022-01-01"
        self.calendar = None

        self.component_data = None
        self.component_prices = None
        self.component_volumes = None
        self.component_log_returns = None
        self.component_volatilities = None

        self.benchmark_data = None
        self.benchmark_prices = None
        self.benchmark_volumes = None
        self.benchmark_log_returns = None
        self.benchmark_volatility = None

        self.pytrends = None

    """ Calendar """

    def __compute_calendar(self, series: List[Union[pd.Series, pd.DataFrame]]):
        """
        Parameters
        ----------
        series : List[Union[pd.Series, pd.DataFrame]]
            series is a list of pandas Series or DataFrame
        Returns
        -------
            a list of sorted timestamp
        """
        return sorted(set.intersection(*list(map(lambda x: set(x.index.tolist()), series))))

    def set_calendar(self, series: List[Union[pd.Series, pd.DataFrame]]):
        self.calendar = self.__compute_calendar(series)

    """ Component Information """

    def test_to_date(self):
        for stock in Analysis.PORTFOLIO_COMPOSITION:
            data = yf.download(stock, start=self.start, end=self.end)
            if data.first_valid_index() > datetime.strptime("2018-01-02", '%Y-%m-%d'):
                print(stock, data.first_valid_index())
        # DOCU, FOX, FOXA, MRNA, PTON, PDD will be removed as they are too young for our analysis.

    def get_component_data(self):
        all_stocks = ""
        for i in range(len(Analysis.PORTFOLIO_COMPOSITION)):
            all_stocks = str(all_stocks) + " " + str(Analysis.PORTFOLIO_COMPOSITION[i])
        temp_data = yf.download(all_stocks, interval='1wk', start=self.start, end=self.end)
        self.component_data = temp_data.ffill().dropna().drop('Adj Close', axis=1)

    def get_component_prices(self):
        self.component_prices = self.component_data["Close"].reindex(columns=self.PORTFOLIO_COMPOSITION)

    def get_component_volumes(self):
        self.component_volumes = self.component_data["Volume"].reindex(columns=self.PORTFOLIO_COMPOSITION)

    def get_component_logreturns(self):
        if self.component_prices is not None:
            temp_log_returns = np.log((self.component_prices / (self.component_prices.shift(1)))).dropna()
            self.component_log_returns = temp_log_returns[temp_log_returns.index.isin(self.calendar)]
        else:
            raise TypeError(f'self.component_prices cannot be NoneType, it must be an instance of {type(pd.DataFrame)}')

    def get_component_volatilities(self):
        self.component_volatilities = np.sqrt((self.component_log_returns - self.component_log_returns.mean()) ** 2)

    """ Index Information """

    def get_benchmark_data(self):
        self.benchmark_data = yf.download(Analysis.BENCHMARK_TICKER, interval='1wk', start=self.start,
                                          end=self.end).drop('Adj Close', axis=1)

    def get_benchmark_level(self):
        self.benchmark_prices = self.benchmark_data["Close"]

    def get_benchmark_volumes(self):
        self.benchmark_volumes = self.benchmark_data["Volume"]

    def get_benchmark_logreturns(self):
        if self.benchmark_prices is not None:
            temp_log_returns = np.log((self.benchmark_prices / (self.benchmark_prices.shift(1)))).dropna()
            self.benchmark_log_returns = temp_log_returns[temp_log_returns.index.isin(self.calendar)]
        else:
            raise TypeError(f'self.benchmark_prices cannot be NoneType, it must be an instance of {type(pd.DataFrame)}')

    def get_benchmark_volatility(self):
        self.benchmark_volatility = (self.benchmark_log_returns - self.benchmark_log_returns.mean()) ** 0.5

    """ Google Search Information """

    def google_pytrends(self):
        pytrends = TrendReq(hl='en-US', tz=360)
        trend_df = pd.DataFrame()

        # build payload
        for word in Analysis.GOOGLE_KEYWORDS:
            pytrends.build_payload(kw_list=[word], cat=0, timeframe='today 5-y')
            # interest over Time
            trend_df[word] = pytrends.interest_over_time().drop(columns='isPartial')  # .reset_index()
            time.sleep(10)

        self.pytrends = trend_df.reindex(columns=Analysis.GOOGLE_KEYWORDS).truncate(before=self.start, after=self.end)

    # add other proxies data if possible e.g. reddit, twitter, yahoo


class PortfolioConstruction(Analysis):
    BASIS = 100

    def __init__(self):
        super().__init__()
        self.get_component_data()
        self.get_component_prices()
        self.get_component_volumes()

        self.get_benchmark_data()
        self.get_benchmark_level()
        self.get_benchmark_volumes()

        self.set_calendar([self.benchmark_prices, self.component_prices])

        self.get_component_logreturns()
        self.get_component_volatilities()
        self.get_benchmark_logreturns()
        self.get_benchmark_volatility()

        self.google_pytrends()

        self.weights = None
        self.portfolio_returns = None
        self.portfolio_volumes = None
        self.portfolio_volatility = None
        self.portfolio_basis_value = None
        self.benchmark_basis_value = None

    def benchmark_basis_calculation(self):
        bench_basis = (self.benchmark_prices / self.benchmark_prices[0]) * PortfolioConstruction.BASIS
        self.benchmark_basis_value = bench_basis

    def comp_weights(self):
        raise NotImplementedError

    def portfolio_ret(self):
        raise NotImplementedError

    def portfolio_basis_calculation(self):
        raise NotImplementedError

    def portfolio_v(self):
        raise NotImplementedError

    def portfolio_vol(self):
        raise NotImplementedError

    def bench_vs_index(self):
        raise NotImplementedError

    def ggl_plotter(self):
        raise NotImplementedError

    def compute_levels(self):
        self.comp_weights()
        self.portfolio_ret()
        self.portfolio_v()
        self.portfolio_vol()
        # self.portfolio_basis_calculation()
        # self.benchmark_basis_calculation()
        # self.bench_vs_index()
        # self.ggl_plotter()


class EquallyWeighted(PortfolioConstruction):

    def __init__(self):
        super().__init__()

    def comp_weights(self):
        self.weights = np.ones((len(self.component_log_returns), len(self.PORTFOLIO_COMPOSITION))) * 1 / len(
            self.PORTFOLIO_COMPOSITION)

    def portfolio_ret(self):
        self.portfolio_returns = (self.weights * self.component_log_returns).sum(axis=1)  # .values

    def portfolio_v(self):
        self.portfolio_volumes = self.component_volumes.sum(axis=1)

    def portfolio_vol(self):
        self.portfolio_volatility = (self.portfolio_returns - self.portfolio_returns.mean()) ** 0.5

    def portfolio_basis_calculation(self):
        final_calendar = self.calendar
        temp_final_mat = np.c_[
            self.portfolio_returns, lagmat(self.portfolio_returns, maxlag=len(self.portfolio_returns) - 1)]
        pf_basis_value = np.where(temp_final_mat == 0, 1, temp_final_mat).prod(axis=1) * self.BASIS
        pf_basis_value = pd.DataFrame(np.r_[self.BASIS, pf_basis_value], index=final_calendar,
                                      columns=["Index Value"])
        self.portfolio_basis_value = pf_basis_value

    def bench_vs_index(self):
        comp_levels = self.component_prices[self.component_prices.index.isin(self.calendar)]
        pf_level = (self.weights * comp_levels[1:]).sum(axis=1)
        pf_basis = pd.DataFrame((pf_level / pf_level[0]) * self.BASIS)

        plt.plot(self.benchmark_basis_value, color='orange', label='Bench Value')
        plt.plot(pf_basis, color='blue', label='Portfolio Value')
        plt.legend(loc='upper left', fontsize=12)
        plt.title('Bench vs Portfolio')
        plt.show()

    def ggl_plotter(self):
        ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)

        colors = ['b', 'r', 'g', 'c', 'm']
        for stock, stock_price, color in zip(Analysis.GOOGLE_KEYWORDS, self.component_prices, colors):
            ax1.plot(self.component_prices[stock_price], linewidth=1, color=color, label=stock)
            ax2.plot(self.pytrends[stock], linewidth=1, color=color, label=stock + ' trend')  # self.gsvi

        ax1.set_title('Prices', fontsize=14)
        ax1.legend(loc='upper left', fontsize=10)
        ax2.set_title('Web Search Interest Over Time', fontsize=14)
        ax2.legend(loc='upper left', fontsize=10)
        plt.show()


if __name__ == '__main__':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        ew = EquallyWeighted()
        ew.compute_levels()
