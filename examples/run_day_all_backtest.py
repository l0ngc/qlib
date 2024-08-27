from pprint import pprint

from qlib.constant import REG_US
import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from tqdm import tqdm

provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_US)
market = "SP500"
benchmark = "^gspc"


PERFORMANCE_FILE = 'port_analysis_1day.pkl'
REPORT_FILE = 'report_normal_1day.pkl'
DIR_AFTERFIX = 'portfolio_analysis'

# Result directories
RESULT_PATH = '/16t-2/cl/project/qlib/examples/us_day_res'

TRAINING_TYPES = ['ST', 'RR', 'DDGDA']
MODEL_LIST = ['LightGBM', 'Transformer', 'Linear', 'LSTM', 'XGBoost', 'MLP']
# MODEL_LIST = ['LightGBM', 'Transformer', 'Linear', 'LSTM', 'DoubleEnsemble', 'XGBoost', 'CatBoost', 'MLP']


for tt in tqdm(TRAINING_TYPES):
    for md in MODEL_LIST:
        try:
            pred_score = pd.read_pickle(f'{RESULT_PATH}/{tt}/{md}/pred.pkl')

            FREQ = "day"
            STRATEGY_CONFIG = {
                "topk": 50,
                "n_drop": 5,
                # pred_score, pd.Series
                "signal": pred_score,
            }

            EXECUTOR_CONFIG = {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            }

            backtest_config = {
                "start_time": "2017-01-01",
                "end_time": "2020-01-01",
                "account": 100000000,
                "benchmark": benchmark,
                "exchange_kwargs": {
                    "freq": FREQ,
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            }
            # strategy object
            strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
            # executor object
            executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
            # backtest
            portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
            analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
            # backtest info
            report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

            analysis = dict()
            analysis["return_without_cost"] = risk_analysis(
                report_normal["return"], freq=analysis_freq
            )
            analysis["return_with_cost"] = risk_analysis(
                report_normal["return"] - report_normal["cost"], freq=analysis_freq
            )
            analysis["excess_return_without_cost"] = risk_analysis(
                report_normal["return"] - report_normal["bench"], freq=analysis_freq
            )
            analysis["excess_return_with_cost"] = risk_analysis(
                report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
            )

            analysis_df = pd.concat(analysis)  # type: pd.DataFrame

            report_normal.to_pickle(f'{RESULT_PATH}/{tt}/{md}/{REPORT_FILE}')
            analysis_df.to_pickle(f'{RESULT_PATH}/{tt}/{md}/{PERFORMANCE_FILE}')

        except:
            print(tt, md)
            continue