from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import yfinance as yf


class stock_reader(ABC):

    @abstractmethod
    def get_web_stock_data_period(
        self,
        stock_symbol: str,
        period: str,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_web_stock_data(
        self,
        web_data_source: str,
        stock_symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_stock_data(
        self, stock_symbol: str, start_date: datetime, end_date: datetime, file_path=str
    ) -> None:
        pass

    @abstractmethod
    def read_local_stock_data(self, file_path):
        pass

    def read_csv(self, file_path: str):
        return pd.read_csv(file_path)


class snp500_reader(stock_reader):

    def get_web_stock_data_period(self, stock_symbol, period):
        return yf.download(stock_symbol, period=period)

    def get_web_stock_data(self, stock_symbol, start_date, end_date):
        return yf.download(stock_symbol, start=start_date, end=end_date)

    def save_stock_data(self, stock_symbol, start_date, end_date, output_file_path):
        data = self.get_stock_data(stock_symbol, start_date, end_date)
        data.to_csv(output_file_path)

    def read_local_stock_data(self, file_path):
        return self.read_csv(file_path)

    def correlation_analysis(self, tickers, period):
        data = yf.download(tickers, period=period)
        return data['Adj Close'].corr()
