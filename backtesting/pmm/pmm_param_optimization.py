import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))

import time
from datetime import datetime
now_time = datetime.fromtimestamp(time.time()).strftime("%y%m%d%H%M%S")

import logging
logging.basicConfig(
    level=logging.WARNING,
    filename=f'{current_dir}/run/{now_time}_optimization.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from backtesting import backtest_engine

from controllers.market_making import pmm_common
pmm_common.BACKTESTING = True


if __name__ == '__main__':
    # start_date = datetime(2025, 6, 14, 0, 0)
    # end_date = datetime(2025, 6, 15, 14, 0)
    
    start_date = datetime(2025, 6, 18, 0, 0)
    end_date = datetime(2025, 6, 22, 17, 0)

    config_file = 'pmm_param_optimization.yml'
    backtest_resolution = '1m'
    trade_cost = 0.0002
    slippage = 0.0001
    enable_trades = False

    space_level = 0
    param_optimization = backtest_engine.ParamOptimization()
    param_optimization.run(current_dir, config_file, start_date, end_date, space_level, backtest_resolution, trade_cost, slippage, enable_trades)
