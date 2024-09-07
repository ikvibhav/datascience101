import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from pandas import DataFrame

from ..tests.test_snp500reader import snp500_reader


class TestSnp500Reader(unittest.TestCase):
    @patch("pandas_datareader.data.DataReader", return_value=DataFrame())
    def test_get_web_stock_data(self, mock_data_reader):
        reader = snp500_reader()
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)
        reader.get_web_stock_data("yahoo", "AAPL", start_date, end_date)
        mock_data_reader.assert_called_once_with(
            "AAPL", "yahoo", start=start_date, end=end_date
        )
