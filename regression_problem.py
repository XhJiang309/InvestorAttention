import pandas as pd
import numpy as np

from stock_info import PortfolioConstruction, EquallyWeighted
from dataclasses import dataclass
from statsmodels.api import OLS
from statsmodels.regression.linear_model import RegressionResults, RegressionResultsWrapper


class FactorBuilder(PortfolioConstruction):
    """
    factors are :
    - gsvi t-1
    - returns of t-1
    - returns**2
    - volatility
    - volumes
    - fama french factors
    """

    def __init__(self):
        super().__init__()
        self.all_factors = None

    def get_famafrench_factors(self):
        _raw = pd.read_csv("North_America_5_Factors_Daily.csv",
                           index_col=0, parse_dates=True, skiprows=5).truncate(before=self.start, after=self.end)

        if self.component_log_returns is None:
            raise TypeError("self.component_log_returns is None")
        index_ = _raw[_raw.index.isin(self.component_log_returns.index)]
        _ = index_.pop("RF")
        return index_

    def factors(self):
        if self.component_log_returns is None:
            raise TypeError("self.component_log_returns is None")

        gsvi = self.pytrends.set_index(self.component_log_returns.index).shift(1)
        print(gsvi)
        agsvi = (np.log(gsvi) - np.log(gsvi.rolling(window=7).median()).fillna(0))

        # re-indexing for panel regression
        date_index = np.repeat(list(self.component_log_returns.index), len(self.component_log_returns.columns))
        company_index = list(self.component_log_returns.columns)*len(self.component_log_returns.index)
        index_data = pd.DataFrame({'Date': date_index, 'Company': company_index}, columns=['Date', 'Company'])
        # multi_index = pd.MultiIndex.from_frame(index_data, names=['Date', 'Company'])

        # reformatting for panel regression
        price = self.component_prices.T.melt().drop('Date', axis=1).rename({'value': 'price'}, axis=1)
        gsvi = gsvi.T.melt().drop('Date', axis=1).rename({'value': 'GSVI t-1'}, axis=1)
        agsvi = agsvi.T.melt().drop('Date', axis=1).rename({'value': 'A_GSVI'}, axis=1)
        log_returns = self.component_log_returns.T.melt().drop('Date', axis=1).rename({'value': 'returns'}, axis=1)
        shifted_log_returns = self.component_log_returns.shift(1).T.melt().drop(
            'Date', axis=1).rename({'value': 'returns t-1'}, axis=1)
        sqrd_log_returns = (self.component_log_returns**2).T.melt().drop(
            'Date', axis=1).rename({'value': 'returns^2'}, axis=1)
        volatilities = self.component_volatilities.T.melt().drop(
            'Date', axis=1).rename({'value': 'volatilities'}, axis=1)
        volumes = self.component_volumes.T.melt().drop('Date', axis=1).rename({'value': 'volumes'}, axis=1)

        famafrench = pd.DataFrame()
        for factor in self.get_famafrench_factors().columns:
            famafrench[factor] = np.repeat(list(self.get_famafrench_factors()[factor]), len(self.component_log_returns.columns))

        reformatted_data = pd.concat([index_data, price, gsvi, agsvi, log_returns, shifted_log_returns,
                                      sqrd_log_returns, volatilities, volumes, famafrench], join='inner', axis=1)

        # add multi-index
        self.all_factors = reformatted_data.set_index(['Date', 'Company']).fillna(0)

        path = "/Users/carlpaulus/OneDrive - EDHEC/Documents/travail/FS/Master/S2/Financial Management/"
        self.all_factors.to_csv(path + 'full_data 11 new' + ".csv")


class LinearEstimator(FactorBuilder):
    _bias = "Bias"

    @dataclass(init=False)
    class LinearRegressionResult:
        t_stats: object = None
        p_values: object = None
        coefficients: object = None
        residuals: object = None
        fitted: object = None

        @classmethod
        def from_model(cls, result: RegressionResults):

            _obj = cls()
            object.__setattr__(_obj, "result", result)
            _obj.coefficients = result.params
            _obj.t_stats = result.tvalues
            _obj.p_values = result.pvalues
            _obj.residuals = result.resid
            _obj.fitted = result.fittedvalues
            return _obj

        def to_csv(self, file_name: str):
            path = "/Users/carlpaulus/OneDrive - EDHEC/Documents/travail/FS/Master/S2/Financial Management/regression results"
            to_export = {key: value for key, value in vars(self).items() if key != "result"}
            pd.DataFrame(to_export).to_csv(path + file_name + ".csv")

        def print(self):
            try:
                res = self.__getattribute__("result")
                if isinstance(res, RegressionResultsWrapper):
                    print(res.summary())
            except AttributeError:
                raise TypeError("No result in LinearResultRegression")

    def __init__(self, hasconst: bool = True):
        super().__init__()
        self._hasconst = hasconst
        self._model = None
        self._result = None

    @property
    def model(self) -> OLS:
        return self._model

    @property
    def result(self) -> LinearRegressionResult:
        return self._result

    def __call__(self, endog: np.ndarray, exog: np.ndarray):
        if len(endog) == len(exog):

            if self._hasconst:
                exog[self._bias] = 1

            self._model = OLS(endog, exog)
            self._result = self.LinearRegressionResult.from_model(self.model.fit())

        else:
            raise TypeError("endog length: {len(endog)} und exog length: {len(exog)}")

        return self.result


class Calculator(LinearEstimator):

    def __init__(self):
        super().__init__()
        self.ew = EquallyWeighted()
        self.ew.compute_levels()
        self.factors()
        self.estimator = LinearEstimator()

    def pool_compute_results(self):

        for stock_index in range(0, len(self.PORTFOLIO_COMPOSITION)):
            factors_data = self.all_factors.iloc[:, stock_index::5]
            # print(self.component_log_returns.iloc[:, stock_index].index == factors_data.index)

            # how attention affects returns
            self.estimator(self.component_log_returns.iloc[:, stock_index], factors_data).print()

            # how attention affects volumes
            # volumes = self.component_volumes.loc[self.component_log_returns.index]
            # self.estimator(volumes.iloc[:, stock_index], factors_data).print()

            # how attention affects volatility
            # volatilities = self.component_volatilities.fillna(0)
            # self.estimator(volatilities.iloc[:, stock_index], factors_data).print()


if __name__ == '__main__':
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    calculator = Calculator()
    print(calculator.all_factors)
    # calculator.all_factors.to_csv(
    #     "/Users/carlpaulus/OneDrive - EDHEC/Documents/travail/FS/Master/S2/Financial Management/regression results" +
    #     " ALL FACTORS.CSV")



