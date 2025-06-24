import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import warnings
warnings.filterwarnings("ignore")
import time
import glob
from decimal import Decimal
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import socket
import socks
from backtesting import backtest_utils
socks.set_default_proxy(socks.SOCKS5, backtest_utils.SOCKS_PROXY_IP, backtest_utils.SOCKS_PROXY_PORT)
socket.socket = socks.socksocket

from backtesting.backtest_market_data_provider import CacheableBacktestMarketDataProvider
from backtesting.base.executor_simulator_base import ExecutorSimulation, ExecutorSimulatorBase
from backtesting.base.backtesting_engine_base import BacktestingEngineBase

import asyncio
from concurrent.futures import ProcessPoolExecutor

from hummingbot.core.data_type.common import TradeType
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.controllers import ControllerConfigBase
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.controllers.market_making_controller_base import MarketMakingControllerConfigBase
from backtesting.executor.mock_position_executor import MockPositionExecutor

local_timezone_offset_hours = 8

class BacktestResult:
    
    close_type_info_map = {
            CloseType.TIME_LIMIT: 'TimeLimit', CloseType.STOP_LOSS: 'StopLoss', CloseType.TAKE_PROFIT: 'TakeProfit', 
            CloseType.EXPIRED: 'Expired', CloseType.EARLY_STOP: 'EarlyStop', CloseType.TRAILING_STOP: 'TrailingStop', 
            CloseType.INSUFFICIENT_BALANCE: 'InsufficientBalance', CloseType.FAILED: 'Failed', 
            CloseType.COMPLETED: 'Completed', CloseType.POSITION_HOLD: 'PositionHold'}
    
    def __init__(self, backtesting_result: Dict, controller_config: ControllerConfigBase, backtest_resolution, 
                 start_date: datetime, end_date: datetime, trade_cost: float, slippage: float):
        self.processed_data = backtesting_result["processed_data"]["features"]
        self.results = backtesting_result["results"]
        self.results['trade_cost'] = trade_cost
        self.results['slippage'] = slippage
        self.executors = backtesting_result["executors"]
        self.controller_config = controller_config
        self.backtest_resolution = backtest_resolution
        self.start_date = start_date
        self.end_date = end_date
        self.trade_cost = trade_cost
        self.slippage = slippage

    def get_results_summary(self, results: Optional[Dict] = None, msg: str = ''):
        if results is None:
            results = self.results
        net_pnl_quote = results["net_pnl_quote"]
        net_pnl_pct = results["net_pnl"]
        max_drawdown = results["max_drawdown_usd"]
        max_drawdown_pct = results["max_drawdown_pct"]
        total_volume = results["total_volume"]
        cum_fees_quote = results["cum_fees_quote"]
        sharpe_ratio = results["sharpe_ratio"]
        profit_factor = results["profit_factor"]
        total_executors = results["total_executors"]
        total_executors_with_position = results["total_executors_with_position"]
        accuracy = results["accuracy"]
        total_long = results["total_long"]
        accuracy_long = results["accuracy_long"]
        total_short = results["total_short"]
        accuracy_short = results["accuracy_short"]
        take_profit = results["close_types"].get("TAKE_PROFIT", 0)
        stop_loss = results["close_types"].get("STOP_LOSS", 0)
        time_limit = results["close_types"].get("TIME_LIMIT", 0)
        trailing_stop = results["close_types"].get("TRAILING_STOP", 0)
        early_stop = results["close_types"].get("EARLY_STOP", 0)
        expired = results["close_types"].get("EXPIRED", 0)
        trading_pair = self.controller_config.dict().get('trading_pair')
        return f"""
=====================================================================================================================================    
{msg}Backtest result for {trading_pair}({self.backtest_resolution}) From: {self.start_date} to: {self.end_date} with trade cost: {self.trade_cost:.2%} and slippage:{self.slippage:.2%}
Net PNL: ${net_pnl_quote:.2f} ({net_pnl_pct*100:.2f}%) | Max Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct*100:.2f}%) | Sharpe Ratio: {sharpe_ratio:.2f} | Profit Factor: {profit_factor:.2f}
Total Volume ($): {total_volume:.2f} (fees: ${cum_fees_quote:.2f}) | Total Executors: {total_executors} (with Position: {total_executors_with_position}) | Accuracy: {accuracy:.2%}
Total Long: {total_long} | Accuracy Long: {accuracy_long:.2%} | Total Short: {total_short} | Accuracy Short: {accuracy_short:.2%}
Close Types: Take Profit: {take_profit} | Trailing Stop: {trailing_stop} | Stop Loss: {stop_loss} | Time Limit: {time_limit} | Expired: {expired} | Early Stop: {early_stop}
=====================================================================================================================================
"""

    @property
    def executors_df(self):
        executors_df = pd.DataFrame([e.dict() for e in self.executors])
        executors_df["side"] = executors_df["config"].apply(lambda x: x["side"].name)
        return executors_df

    def _get_bt_candlestick_trace(self):
        self.processed_data.index = pd.to_datetime(self.processed_data.timestamp, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
        
        return go.Candlestick(
            x=self.processed_data.index,
            open=self.processed_data['open'],
            high=self.processed_data['high'],
            low=self.processed_data['low'],
            close=self.processed_data['close'],
            increasing_line_color='#2ECC71',  # 上涨K线颜色（绿色）
            decreasing_line_color='#E74C3C',  # 下跌K线颜色（红色）
            name='K-Lines',
        )
        

    @staticmethod
    def _get_pnl_trace(executors, line_style: str = "solid"):
        pnl = [e.net_pnl_quote for e in executors]
        cum_pnl = np.cumsum(pnl)
        return go.Scatter(
            x=pd.to_datetime([e.close_timestamp for e in executors], unit="s") + pd.Timedelta(hours=local_timezone_offset_hours),
            y=cum_pnl,
            mode='lines',
            line=dict(color='gold', width=2, dash=line_style if line_style == "dash" else None),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',
            name='Cumulative PNL'
        )

    @staticmethod
    def _get_default_layout(title=None, height=800, width=1750):
        layout = {
            "template": "plotly_dark",
            "plot_bgcolor": 'rgba(0, 0, 0, 0)',  # Transparent background
            "paper_bgcolor": 'rgba(0, 0, 0, 0.1)',  # Lighter shade for the paper
            "font": {"color": 'white', "size": 15},  # Consistent font color and size
            "height": height,
            "width": width,
            "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
            "xaxis_rangeslider_visible": False,
            "hovermode": "x unified",
            "showlegend": False,
        }
        if title:
            layout["title"] = title
        return layout
    
    @staticmethod
    def _map_close_type(close_type: CloseType):
        return BacktestResult.close_type_info_map.get(close_type, 'None')

    @staticmethod
    def _add_executors_trace(fig, executors: List[ExecutorInfo], row=1, col=1, line_style="solid"):
        for executor in executors:
            start_time = pd.to_datetime(executor.timestamp, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
            
            entry_time = executor.custom_info.get("entry_timestamp", None)
            if entry_time is not None:
                entry_time = pd.to_datetime(entry_time, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
                
            entry_price = executor.custom_info["current_position_average_price"]
            end_time = pd.to_datetime(executor.close_timestamp, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
            exit_price = executor.custom_info["close_price"]
            close_type = BacktestResult._map_close_type(executor.close_type)
            name = f"Buy-{close_type}" if executor.config.side == TradeType.BUY else f"Sell-{close_type}"

            if executor.filled_amount_quote == 0:
                fig.add_trace(
                    go.Scatter(x=[start_time, end_time], y=[entry_price, entry_price], mode='lines', showlegend=False,
                               line=dict(color='grey', width=3, dash=line_style if line_style == "dash" else None),
                               name=name), row=row, col=col)
            else:
                if executor.net_pnl_quote > Decimal(0):
                    if entry_time is not None:
                        fig.add_trace(go.Scatter(x=[start_time, entry_time], y=[entry_price, entry_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color='blue', width=4,
                                                       dash=line_style if line_style == "dash" else None), name=name), 
                                      row=row, col=col)
                        start_time = entry_time
                    
                    color = 'green'
                    if executor.close_type == CloseType.TAKE_PROFIT:
                        color = 'lime'
                    elif executor.close_type == CloseType.EARLY_STOP:
                        color = 'olive'
                        
                    fig.add_trace(go.Scatter(x=[start_time, end_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color=color, width=4,
                                                       dash=line_style if line_style == "dash" else None), name=name), 
                                  row=row, col=col)
                else:
                    if entry_time is not None:
                        fig.add_trace(go.Scatter(x=[start_time, entry_time], y=[entry_price, entry_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color='blue', width=4,
                                                       dash=line_style if line_style == "dash" else None), name=name), 
                                      row=row, col=col)
                        start_time = entry_time
                    
                    color = 'red' # stop loss
                    if executor.close_type == CloseType.TIME_LIMIT:
                        color = 'darkred'
                        
                    fig.add_trace(go.Scatter(x=[start_time, end_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color=color, width=4,
                                                       dash=line_style if line_style == "dash" else None), name=name),
                                  row=row, col=col)

        return fig

    def get_backtesting_figure(self):
        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, subplot_titles=('Candlestick', 'PNL Quote'),
                            row_heights=[0.7, 0.3])

        # Add candlestick trace
        fig.add_trace(self._get_bt_candlestick_trace(), row=1, col=1)

        # Add executors trace
        fig = self._add_executors_trace(fig, self.executors, row=1, col=1)

        # Add PNL trace
        fig.add_trace(self._get_pnl_trace(self.executors), row=2, col=1)

        # Apply the theme layout
        layout_settings = self._get_default_layout(f"Trading Pair: {self.controller_config.dict().get('trading_pair')}")
        layout_settings["showlegend"] = False
        fig.update_layout(**layout_settings)

        # Update axis properties
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="PNL", row=2, col=1)
        return fig


class MyPositionExecutorSimulator(ExecutorSimulatorBase):
    
    def simulate(self, df: pd.DataFrame, config: PositionExecutorConfig, executor_refresh_time: int, trade_cost: float, slippage: float) -> ExecutorSimulation:
        if config.triple_barrier_config.open_order_type.is_limit_type():
            entry_condition = (df['low'] <= config.entry_price) if config.side == TradeType.BUY else (df['high'] >= config.entry_price)
            start_timestamp = df[entry_condition].index.min()
        else:
            start_timestamp = df.index.min()
        last_timestamp = df.index.max()

        # Set up barriers
        take_profit = float(config.triple_barrier_config.take_profit) if config.triple_barrier_config.take_profit else None
        stop_loss = float(config.triple_barrier_config.stop_loss)
        trailing_sl_trigger_pct = None
        trailing_sl_delta_pct = None
        if config.triple_barrier_config.trailing_stop:
            trailing_sl_trigger_pct = float(config.triple_barrier_config.trailing_stop.activation_price)
            trailing_sl_delta_pct = float(config.triple_barrier_config.trailing_stop.trailing_delta)
        time_limit = config.triple_barrier_config.time_limit if config.triple_barrier_config.time_limit else None
        time_limit_timestamp = config.timestamp + time_limit if time_limit else last_timestamp
        early_stop_timestamp = min(last_timestamp, config.timestamp + executor_refresh_time) if executor_refresh_time > 0 else last_timestamp

        # Filter dataframe based on the conditions
        executor_simulation = df[df.index <= time_limit_timestamp].copy()
        executor_simulation['net_pnl_pct'] = 0.0
        executor_simulation['net_pnl_quote'] = 0.0
        executor_simulation['cum_fees_quote'] = 0.0
        executor_simulation['filled_amount_quote'] = 0.0
        executor_simulation["current_position_average_price"] = float(config.entry_price)

        if pd.isna(start_timestamp):
            return ExecutorSimulation(config=config, executor_simulation=executor_simulation, close_type=CloseType.TIME_LIMIT)

        if start_timestamp > early_stop_timestamp:
            executor_simulation = executor_simulation[executor_simulation.index <= early_stop_timestamp].copy()
            return ExecutorSimulation(config=config, executor_simulation=executor_simulation, close_type=CloseType.EARLY_STOP)

        simulation_filtered = executor_simulation[executor_simulation.index >= start_timestamp]
        if simulation_filtered.empty:
            return ExecutorSimulation(config=config, executor_simulation=executor_simulation, close_type=CloseType.TIME_LIMIT)
        
        entry_time = datetime.fromtimestamp(start_timestamp).strftime("%m%d:%H%M")
        last_timestamp = executor_simulation.index.max()
        
        entry_price = float(config.entry_price)
        
        timestamps = simulation_filtered.index.values
        lows = simulation_filtered['low'].values
        highs = simulation_filtered['high'].values
        closes = simulation_filtered['close'].values
        
        close_type = CloseType.TIME_LIMIT
        close_timestamp = None
        max_profit_ratio = -1e6
        trailing_stop_activated = False
        net_pnl_pct = 0
        count = len(simulation_filtered)
        
        if config.side == TradeType.BUY:
            side_multiplier = 1
            take_profit_trigger_price = entry_price * (1 + side_multiplier * (take_profit + trade_cost))
            stop_loss_trigger_price = entry_price * (1 - side_multiplier * stop_loss)
            
            for i in range(count):
                timestamp = timestamps[i]
                next_timestamp = timestamps[i+1] if i < count-1 else timestamp
                low = lows[i]
                
                if low <= stop_loss_trigger_price:
                    close_timestamp = next_timestamp
                    close_price = stop_loss_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.STOP_LOSS
                    break

                high = highs[i]
                if high >= take_profit_trigger_price:
                    close_timestamp = next_timestamp
                    close_price = take_profit_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.TAKE_PROFIT
                    break
                
                if not trailing_stop_activated:
                    if (high/entry_price - 1) >= trailing_sl_trigger_pct:
                        trailing_stop_activated = True
                        
                if trailing_stop_activated:
                    max_profit_ratio = max(max_profit_ratio, high/entry_price - 1)
                    if (max_profit_ratio - (low/entry_price - 1)) > trailing_sl_delta_pct:
                        close_timestamp = next_timestamp
                        close_price = entry_price * (1 + max_profit_ratio - trailing_sl_delta_pct) * (1 - side_multiplier*slippage)
                        close_type = CloseType.TRAILING_STOP
                        break
                
                close_price = closes[i]
                
            net_pnl_pct = (close_price - entry_price) / entry_price - trade_cost
        else:
            side_multiplier = -1
            take_profit_trigger_price = entry_price * (1 + side_multiplier * (take_profit + trade_cost))
            stop_loss_trigger_price = entry_price * (1 - side_multiplier * stop_loss)
            
            for i in range(count):
                timestamp = timestamps[i]
                next_timestamp = timestamps[i+1] if i < count-1 else timestamp
                high = highs[i]
                
                if stop_loss_trigger_price <= high:
                    close_timestamp = next_timestamp
                    close_price = stop_loss_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.STOP_LOSS
                    break

                low = lows[i]
                if low <= take_profit_trigger_price:
                    close_timestamp = next_timestamp
                    close_price = take_profit_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.TAKE_PROFIT
                    break
                
                if not trailing_stop_activated:
                    if (1 - low/entry_price) >= trailing_sl_trigger_pct:
                        trailing_stop_activated = True
                        
                if trailing_stop_activated:
                    max_profit_ratio = max(max_profit_ratio, 1 - low/entry_price)
                    if (max_profit_ratio - (1 - high/entry_price)) > trailing_sl_delta_pct:
                        close_timestamp = next_timestamp
                        close_price = entry_price * (1 - (max_profit_ratio - trailing_sl_delta_pct)) * (1 - side_multiplier*slippage)
                        close_type = CloseType.TRAILING_STOP
                        break
                
                close_price = closes[i]
                
            net_pnl_pct = (entry_price - close_price) / entry_price - trade_cost
            
        close_time = "End"
        if close_timestamp is not None:
            close_time = datetime.fromtimestamp(close_timestamp).strftime("%m%d:%H%M")
        else:
            close_timestamp = last_timestamp
        
        executor_simulation = executor_simulation[executor_simulation.index <= close_timestamp]
        
        filled_amount_quote = float(config.amount) * entry_price
        net_pnl_quote = filled_amount_quote * net_pnl_pct
        cum_fees_quote = filled_amount_quote * trade_cost
        
        executor_simulation.loc[executor_simulation.index >= start_timestamp, ['filled_amount_quote', 'cum_fees_quote']] = [filled_amount_quote, cum_fees_quote]
        
        last_loc = executor_simulation.index[-1]
        executor_simulation.loc[last_loc, "net_pnl_pct"] = net_pnl_pct
        executor_simulation.loc[last_loc, "net_pnl_quote"] = net_pnl_quote
        executor_simulation.loc[last_loc, "filled_amount_quote"] = filled_amount_quote * 2
        executor_simulation.loc[last_loc, "cum_fees_quote"] = cum_fees_quote * 2
        
        # info = f'Pnl:{net_pnl_quote:+.2f}, entry:{entry_price:.7f}, close:{close_price:.7f}, amount:{config.amount:.2f}, quote:{filled_amount_quote:.2f}, ' \
        #         f'[{config.level_id}] {close_type}({entry_time}-{close_time})[{int(close_timestamp-start_timestamp)}s], id:{config.id}'
        # print(f'\033[92m{info}\033[0m' if net_pnl_quote > 0 else f'\033[91m{info}\033[0m')
        
        simulation = ExecutorSimulation(
            config=config,
            executor_simulation=executor_simulation,
            close_type=close_type
        )
        return simulation


class BacktestEngine(BacktestingEngineBase):
    
    def __init__(self, batch: int = 1, base_dir: str = '.', print_detail: bool = False, enable_trades: bool = False):
        super().__init__()
        self.base_dir = base_dir
        self.batch = batch
        self.backtesting_data_provider = CacheableBacktestMarketDataProvider(connectors={}, base_dir=base_dir, enable_trades=enable_trades)
        self.print_detail = print_detail
        
    def run_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, 
                     backtest_resolution: str = '3m', trade_cost: float = 0.0005, slippage: float = 0.0001, enable_trades: bool = False):
        return asyncio.run(self.async_backtest(config_dir, config_path, start_date, end_date, backtest_resolution, trade_cost, slippage, enable_trades))
    
    def get_controller_config(self, config_dir: str, config_path: str):
        return self.get_controller_config_instance_from_yml(controllers_conf_dir_path=config_dir, config_path=config_path)
    
    async def async_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, 
                             backtest_resolution: str = '3m', trade_cost: float = 0.0005, slippage: float = 0.0001, enable_trades: bool = False):
        controller_config = self.get_controller_config(config_dir, config_path)
        return await self.async_backtest_with_config(controller_config, start_date, end_date, backtest_resolution, trade_cost, slippage, enable_trades)
    
    async def async_backtest_with_config(self, controller_config: ControllerConfigBase, start_date: datetime, end_date: datetime, 
                             backtest_resolution: str, trade_cost: float, slippage: float, enable_trades: bool = False) -> BacktestResult:
        backtest_result = None
        try:
            t = time.time()
            executor_refresh_time = 0
            if isinstance(controller_config, MarketMakingControllerConfigBase):
                executor_refresh_time = int(controller_config.executor_refresh_time)
            
            start = int(start_date.timestamp())
            end = int(end_date.timestamp())
            result = await self.do_backtest(controller_config, start, end, executor_refresh_time, backtest_resolution, trade_cost, slippage, enable_trades)
            
            backtest_result = BacktestResult(result, controller_config, backtest_resolution, start_date, end_date, trade_cost, slippage)
            print(backtest_result.get_results_summary(None, f"[Batch-{self.batch}(time:{int(time.time() - t)}s)]. "))
        except Exception as e:
            print(f"Error during backtest {self.batch}: {e}")
            # if self.print_detail:
            import traceback
            traceback.print_exc()
        return backtest_result

    async def do_backtest(self,
                          controller_config: ControllerConfigBase,
                          start: int, end: int, executor_refresh_time: int, backtesting_resolution: str = "1m",
                          trade_cost=0.0006, slippage: float=0.001, enable_trades: bool = False):
        t = time.time()
        self.backtesting_data_provider.update_backtesting_time(start, end)
        await self.backtesting_data_provider.initialize_trading_rules(controller_config.connector_name)
        
        controller_class = controller_config.get_controller_class()
        self.controller = controller_class(config=controller_config, market_data_provider=self.backtesting_data_provider, actions_queue=None)
        
        self.backtesting_resolution = backtesting_resolution
        await self.initialize_backtesting_data_provider()
        
        candles_df = self.prepare_market_data()
        # processed_data will be recreated by MarketMakingControllerBase.update_processed_data() if not implemented by sub-class, so we keep it for backtest result
        processed_data_features = self.controller.processed_data["features"]
        
        if enable_trades:
            connector_name = controller_config.connector_name
            trading_pair = controller_config.trading_pair
            trades_df = self.backtesting_data_provider.get_trades(connector_name, trading_pair, start, end)

        market_data_df = trades_df if enable_trades else candles_df
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Prepare market data:{int(time.time() - t)}s')
        t = time.time()
        
        active_executors: Dict[str, MockPositionExecutor] = {}
        stopped_executors_info: List[ExecutorInfo] = []
        
        for _, row in market_data_df.iterrows():
            current_timestamp = int(row["timestamp"])
            self.backtesting_data_provider.update_time(current_timestamp)
            
            # stopped_level_ids: List[str] = []
            
            # 1. stop actions
            # stop the executors if needed, e.g. time to refresh
            self.controller.executors_info = [e.executor_info for e in active_executors.values()] + stopped_executors_info
            stop_actions = self.controller.stop_actions_proposal()
            for stop_action in stop_actions:
                if not isinstance(stop_action, StopExecutorAction):
                    continue
                
                # inactive executor id will be deleted in dict so we need to make a copy of items
                for executor_id, executor in list(active_executors.items()):
                    if not executor_id == stop_action.executor_id:
                        continue
                    executor.early_stop()
                    del active_executors[executor_id]
                    stopped_executors_info.append(executor.executor_info)

            # 2. simulate the control task in position executor
            for executor_id, executor in list(active_executors.items()):
                if enable_trades:
                    active = executor.on_trade(row)
                else:
                    active = executor.on_candle(row)
                
                if not active:
                    # stopped_level_ids.append(executor.config.level_id)
                    del active_executors[executor_id]
                    stopped_executors_info.append(executor.executor_info)
            
            # update the executors info for controller to determine the actions
            self.controller.executors_info = [e.executor_info for e in active_executors.values()] + stopped_executors_info

            # 3. simulate the control task in controller
            # 3.1 update the processed data
            if len(controller_config.candles_config) > 0:
                candle_seconds = CandlesBase.interval_to_seconds[controller_config.candle_interval]
                current_start = start - (100 * candle_seconds)
                
                reference_price_seconds = CandlesBase.interval_to_seconds[self.backtesting_resolution]
                current_end = current_timestamp - reference_price_seconds
                
                self.backtesting_data_provider.update_start_end_time(current_start, current_end)
            
            await self.controller.update_processed_data()
            
            # 3.2 determine the actions
            actions = self.controller.determine_executor_actions()
            
            # 3.2.1 process create actions
            for executor_action in actions:
                if not isinstance(executor_action, CreateExecutorAction):
                    continue
                
                # # prevent create actions for current stopped level. continue determination on the next market data.
                # if action.executor_config.level_id in stopped_level_ids:
                #     continue
                
                executor_id = executor_action.executor_config.id
                executor = MockPositionExecutor(
                    config=executor_action.executor_config,
                    market_data_provider=self.backtesting_data_provider,
                    trade_cost=trade_cost,
                    slippage=slippage
                )
                if enable_trades:
                    active = executor.on_trade(row)
                else:
                    active = executor.on_candle(row)
                
                if active:
                    active_executors[executor_id] = executor
                else:
                    stopped_executors_info.append(executor.executor_info)
                    
            # 3.2.2 process stop actions
            for executor_action in actions:
                if not isinstance(executor_action, StopExecutorAction):
                    continue
                
                # inactive executor id will be deleted in dict so we need to make a copy of items
                for executor_id, executor in list(active_executors.items()):
                    if not executor_id == executor_action.executor_id:
                        continue
                    executor.early_stop()
                    del active_executors[executor_id]
                    stopped_executors_info.append(executor.executor_info)
        
        # stop all active executors
        for executor_id, executor in list(active_executors.items()):
            executor.expired_stop()
            stopped_executors_info.append(executor.executor_info)
        
        self.controller.executors_info = stopped_executors_info
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Simulation:{int(time.time() - t)}s')
        t = time.time()
        
        results = self.summarize_results(self.controller.executors_info, controller_config.total_amount_quote)
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Summarize results:{int(time.time() - t)}s')
        
        self.controller.processed_data['features'] = processed_data_features
        
        return {
            "executors": self.controller.executors_info,
            "results": results,
            "processed_data": self.controller.processed_data,
        }
    
    async def do_backtest_v1(self,
                          controller_config: ControllerConfigBase,
                          start: int, end: int,
                          executor_refresh_time: int,
                          backtesting_resolution: str = "1m",
                          trade_cost=0.0006, slippage: float=0.001):
        self.active_executor_simulations: List[ExecutorSimulation] = []
        self.stopped_executors_info: List[ExecutorInfo] = []
        self.backtesting_resolution = backtesting_resolution
        
        t = time.time()
        self.backtesting_data_provider.update_backtesting_time(start, end)
        await self.backtesting_data_provider.initialize_trading_rules(controller_config.connector_name)
        
        # TODO: 新增的Executor需要注册
        controller_class = controller_config.get_controller_class()
        self.controller = controller_class(config=controller_config, market_data_provider=self.backtesting_data_provider, actions_queue=None)
        
        await self.initialize_backtesting_data_provider()
        
        market_data = self.prepare_market_data()
        # processed_data will be recreated by MarketMakingControllerBase.update_processed_data() if not implemented by sub-class, so we keep it for backtest result
        market_data_features = self.controller.processed_data["features"]
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Prepare market data:{int(time.time() - t)}s')
        t = time.time()
        
        for i, row in market_data.iterrows():
            if len(controller_config.candles_config) > 0:
                current_start = start - (450 * CandlesBase.interval_to_seconds[controller_config.candles_config[0].interval])
                current_end = int(row["timestamp_bt"])
                self.backtesting_data_provider.update_start_end_time(current_start, current_end)
                
            await self.controller.update_processed_data()
            await self.update_state(row)
            active_executor_simulation_ids = set([e.config.id for e in self.active_executor_simulations])
            
            for action in self.controller.determine_executor_actions():
                if isinstance(action, CreateExecutorAction):
                    market_data_from_start = market_data.loc[i:]
                    simulation_result = MyPositionExecutorSimulator().simulate(market_data_from_start, action.executor_config, executor_refresh_time, trade_cost, slippage)
                    if simulation_result.executor_simulation.empty or simulation_result.config.id in active_executor_simulation_ids:
                        continue
                    
                    if simulation_result.close_type != CloseType.FAILED:
                        self.active_executor_simulations.append(simulation_result)
                elif isinstance(action, StopExecutorAction):
                    self.handle_stop_action(action, row["timestamp"])
        
        executors_info = self.controller.executors_info
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Simulation:{int(time.time() - t)}s')
        t = time.time()
        
        results = self.summarize_results(executors_info, controller_config.total_amount_quote)
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Summarize results:{int(time.time() - t)}s')
        
        self.controller.processed_data['features'] = market_data_features
        
        return {
            "executors": executors_info,
            "results": results,
            "processed_data": self.controller.processed_data,
        }  

    async def do_backtest_v2(self,
                          controller_config: ControllerConfigBase,
                          start: int, end: int,
                          executor_refresh_time: int,
                          backtesting_resolution: str = "1m",
                          trade_cost=0.0006, slippage: float=0.001):
        self.backtesting_resolution = backtesting_resolution
        
        t = time.time()
        self.backtesting_data_provider.update_backtesting_time(start, end)
        await self.backtesting_data_provider.initialize_trading_rules(controller_config.connector_name)
        
        controller_class = controller_config.get_controller_class()
        self.controller = controller_class(config=controller_config, market_data_provider=self.backtesting_data_provider, actions_queue=None)
        
        await self.initialize_backtesting_data_provider()
        
        market_data = self.prepare_market_data()
        # processed_data will be recreated by MarketMakingControllerBase.update_processed_data() if not implemented by sub-class, so we keep it for backtest result
        market_data_features = self.controller.processed_data["features"]
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Prepare market data:{int(time.time() - t)}s')
        t = time.time()
        
        active_executors: Dict[str, MockPositionExecutor] = {}
        stopped_executors_info: List[ExecutorInfo] = []
        
        for _, row in market_data.iterrows():
            current_timestamp = int(row["timestamp"])
            self.backtesting_data_provider.update_time(current_timestamp)
            
            # stopped_level_ids: List[str] = []
            
            # 0. refresh executors
            # update the executors info for controller to refresh
            self.controller.executors_info = [e.executor_info for e in active_executors.values()] + stopped_executors_info
            refresh_actions = self.controller.executors_to_refresh()
            for refresh_action in refresh_actions:
                if not isinstance(refresh_action, StopExecutorAction):
                    continue
                
                # inactive executor id will be deleted in dict so we need to make a copy of items
                for executor_id, executor in list(active_executors.items()):
                    if not executor_id == refresh_action.executor_id:
                        continue
                    executor.early_stop()
                    del active_executors[executor_id]
                    stopped_executors_info.append(executor.executor_info)

            # 1. simulate the control task in position executor
            for executor_id, executor in list(active_executors.items()):
                active = executor.on_candle(row)
                if not active:
                    # stopped_level_ids.append(executor.config.level_id)
                    del active_executors[executor_id]
                    stopped_executors_info.append(executor.executor_info)
            
            # update the executors info for controller to determine the actions
            self.controller.executors_info = [e.executor_info for e in active_executors.values()] + stopped_executors_info

            # 2. simulate the control task in controller
            # 2.1 update the processed data
            if len(controller_config.candles_config) > 0:
                candle_seconds = CandlesBase.interval_to_seconds[controller_config.candles_config[0].interval]
                current_start = start - (100 * candle_seconds)
                current_end = current_timestamp - candle_seconds
                self.backtesting_data_provider.update_start_end_time(current_start, current_end)
            
            self.controller.processed_data.update(row.to_dict())
            mid_price = (row['high'] + row['low']) / 2
            self.backtesting_data_provider.update_price(self.controller.config.connector_name, self.controller.config.trading_pair, mid_price)
            await self.controller.update_processed_data()
            
            # 2.2 determine the actions
            actions = self.controller.determine_executor_actions()
            
            # 2.2.1 process create actions
            for refresh_action in actions:
                if not isinstance(refresh_action, CreateExecutorAction):
                    continue
                
                # # prevent create actions for current stopped level. continue determination on the next market data.
                # if action.executor_config.level_id in stopped_level_ids:
                #     continue
                
                executor_id = refresh_action.executor_config.id
                executor = MockPositionExecutor(
                    config=refresh_action.executor_config,
                    market_data_provider=self.backtesting_data_provider,
                    trade_cost=trade_cost,
                    slippage=slippage
                )
                active = executor.on_candle(row)
                
                if active:
                    active_executors[executor_id] = executor
                else:
                    stopped_executors_info.append(executor.executor_info)
                    
            # 2.2.2 process stop actions
            for refresh_action in actions:
                if not isinstance(refresh_action, StopExecutorAction):
                    continue
                
                # inactive executor id will be deleted in dict so we need to make a copy of items
                for executor_id, executor in list(active_executors.items()):
                    if not executor_id == refresh_action.executor_id:
                        continue
                    executor.early_stop()
                    del active_executors[executor_id]
                    stopped_executors_info.append(executor.executor_info)
        
        # stop all active executors
        for executor_id, executor in list(active_executors.items()):
            executor.expired_stop()
            stopped_executors_info.append(executor.executor_info)
        
        self.controller.executors_info = stopped_executors_info
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Simulation:{int(time.time() - t)}s')
        t = time.time()
        
        results = self.summarize_results(self.controller.executors_info, controller_config.total_amount_quote)
        
        if self.print_detail:
            print(f'[Batch-{self.batch}] Summarize results:{int(time.time() - t)}s')
        
        self.controller.processed_data['features'] = market_data_features
        
        return {
            "executors": self.controller.executors_info,
            "results": results,
            "processed_data": self.controller.processed_data,
        }


@dataclass
class BacktestParam:
    
    batch: int
    base_dir: str
    config_dict: Dict
    start_date: datetime
    end_date: datetime
    backtest_resolution: str
    trade_cost: float
    slippage: float
    enable_trades: bool


class ParamSpace:
    
    def __init__(self, level: int = 100):
        self.level = level
    
    def generate(self, base_backtest_param: BacktestParam):
        if self.level == 0:
            return [base_backtest_param]
        elif self.level == 1:
            return self.generate_1(base_backtest_param)
        elif self.level == 100:
            return self.generate_100(base_backtest_param)
    
    def generate_1(self, base_backtest_param: BacktestParam):
        backtest_params = []
        batch = 1
        
        executor_refresh_time_space = [60, 120, 180]
        
        for executor_refresh_time in executor_refresh_time_space:
            backtest_param = copy.deepcopy(base_backtest_param)
            
            config_dict = backtest_param.config_dict
            config_dict['executor_refresh_time'] = executor_refresh_time

            backtest_param.batch = batch
            backtest_params.append(backtest_param)
            batch += 1

        return backtest_params

    def generate_100(self, base_backtest_param: BacktestParam):
        backtest_params = []
        batch = 1
        
        executor_refresh_time_space = [910, 1810]
        take_profit_space = np.arange(1, 6.1, 1)
        stop_loss_space = np.arange(2, 10.1, 1)
        # cooldown_time_space = [600, 900, 1800]
        spread_space = [[0.5], [0.75], [1]]
        reference_price_type_space = ['close']
        # trailing_stop_space = np.arange(0.015, 0.026, 0.005)
        # cci_threshold_space = [80]
        # length_space = np.arange(20, 41, 10)
        natr_length_space = [15, 21, 41]
        widen_space = [8]
        narrow_space = [1]
        max_stop_loss_space = [0.04, 0.05, 0.06]
        # max_stop_loss_space = np.arange(0.02, 0.031, 0.01)
        
        for executor_refresh_time in executor_refresh_time_space:
            for take_profit in take_profit_space:
                for stop_loss in stop_loss_space:
                    # for cooldown_time in cooldown_time_space:
                    for spread in spread_space:
                        for reference_price_type in reference_price_type_space:
                            # for trailing_stop in trailing_stop_space:
                            # for cci_threshold in cci_threshold_space:
                                # for length in length_space:
                            for natr_length in natr_length_space:
                                for widen in widen_space:
                                    for narrow in narrow_space:
                                        for max_stop_loss in max_stop_loss_space:
                                            backtest_param = copy.deepcopy(base_backtest_param)
                                
                                            config_dict = backtest_param.config_dict
                                            config_dict['executor_refresh_time'] = executor_refresh_time
                                            config_dict['take_profit'] = take_profit
                                            config_dict['stop_loss'] = stop_loss
                                            # config_dict['cooldown_time'] = cooldown_time
                                            config_dict['buy_spreads'] = spread
                                            config_dict['sell_spreads'] = spread
                                            config_dict['reference_price_type'] = reference_price_type
                                            # config_dict['trailing_stop']['activation_price'] = trailing_stop
                                            # config_dict['cci_threshold'] = cci_threshold
                                            # config_dict['sma_length'] = length
                                            # config_dict['cci_length'] = length
                                            config_dict['natr_length'] = natr_length
                                            config_dict['widen_spread_multiplier'] = widen
                                            config_dict['narrow_spread_multiplier'] = narrow
                                            config_dict['max_stop_loss'] = max_stop_loss
                                            
                                            backtest_param.batch = batch
                                            backtest_params.append(backtest_param)
                                            batch += 1

        return backtest_params


class ParamOptimization:
    
    def run(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, 
            space_level: int = 0, backtest_resolution: str = '3m', trade_cost: float = 0.0005, slippage: float = 0.0001, enable_trades: bool = False):
        t = time.time()
        base_config_dict = BacktestEngine.load_controller_config(config_path=config_path, controllers_conf_dir_path=config_dir)
        trading_pair = base_config_dict.get('trading_pair')
        candles_config = CandlesConfig(
            connector=base_config_dict.get('connector_name'),
            trading_pair=trading_pair,
            interval=backtest_resolution
        )
        data_provider = CacheableBacktestMarketDataProvider(connectors={}, base_dir=config_dir, enable_trades=enable_trades)
        
        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()
        asyncio.run(data_provider.init_data(int(start_timestamp), int(end_timestamp), candles_config))
        
        if 'candle_interval' in base_config_dict.keys():
            candles_config.interval = base_config_dict['candle_interval']
            asyncio.run(data_provider.init_data(int(start_timestamp), int(end_timestamp), candles_config))
        
        base_backtest_param = BacktestParam(0, config_dir, base_config_dict, start_date, end_date, backtest_resolution, trade_cost, slippage, enable_trades)
        backtest_params = ParamSpace(space_level).generate(base_backtest_param)
        print(f'Total param count:{len(backtest_params)}')
        
        result_dir = os.path.join(config_dir, 'result')
        start_time = datetime.fromtimestamp(start_date.timestamp()).strftime("%y%m%d%H%M%S")
        end_time = datetime.fromtimestamp(end_date.timestamp()).strftime("%y%m%d%H%M%S")
        now_time = datetime.fromtimestamp(time.time()).strftime("%y%m%d%H%M%S")
        result_file = f'{trading_pair}-{backtest_resolution}-{start_time}-{end_time}-tested-{now_time}.csv'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        print(f'Start running param optimization.')
        cpus = min(os.cpu_count()-4, round(0.8 * os.cpu_count()))
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            results = list(pool.map(self.run_one, backtest_params))
            
            rows = []
            for backtest_param, backtest_result in results:
                if backtest_result is None:
                    continue
                
                row = backtest_param.config_dict
                row.update(backtest_result.results)
                del row["close_types"]
                rows.append(row)
            
            if len(rows) == 0:
                print('Empty results')
                return
            
            pd.set_option('display.max_columns', None)
            result_df = pd.DataFrame(rows).sort_values("net_pnl", ascending=False)
            result_df_cols = ['net_pnl','net_pnl_quote','total_volume','cum_fees_quote','sharpe_ratio','profit_factor','total_executors_with_position','accuracy','accuracy_long','accuracy_short','max_drawdown_usd','max_drawdown_pct','buy_spreads','sell_spreads','executor_refresh_time','reference_price_type','stop_loss','max_stop_loss','take_profit','trailing_stop','sma_short_length','sma_length','cci_length','cci_threshold','natr_length','widen_spread_multiplier','narrow_spread_multiplier','cooldown_time','trade_cost','slippage','total_amount_quote','total_executors','total_long','total_short','total_positions','win_signals','loss_signals','buy_amounts_pct','sell_amounts_pct','sleep_interval','time_limit','update_interval','candle_interval','candles_config','connector_name','controller_name','trading_pair','controller_type','id','leverage','manual_kill_switch','position_mode','take_profit_order_type']
            result_df = result_df[result_df_cols]
            result_df.to_csv(os.path.join(result_dir, result_file), index=False)
            
            top_ratio_list = [0.05, 0.1, 0.2, 0.3, 0.5]
            total_count = len(result_df)
            for top_ratio in top_ratio_list:
                top_count = round(total_count * top_ratio)
                if top_count > 0 and total_count >= top_count:
                    description = result_df.head(top_count)[['buy_spreads','sell_spreads','executor_refresh_time','stop_loss','take_profit','trailing_stop','sma_short_length','sma_length','cci_length','cci_threshold','natr_length','widen_spread_multiplier','narrow_spread_multiplier']].describe(include='all')
                    print(f'Top {top_ratio:.2%}({top_count}/{total_count})\n:{description}\n')
            
        print(f'Total time:{int(time.time() - t)}s. Saved result to:{result_file}')
        exit(0)

    def run_one(self, backtest_param: BacktestParam):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        backtest_result = None
        try:
            backtest_engine = BacktestEngine(batch=backtest_param.batch, base_dir=backtest_param.base_dir, enable_trades=backtest_param.enable_trades)
            controller_config = backtest_engine.get_controller_config_instance_from_dict(backtest_param.config_dict)
            backtest_result = loop.run_until_complete(backtest_engine.async_backtest_with_config(controller_config, backtest_param.start_date, backtest_param.end_date, 
                                                    backtest_param.backtest_resolution, backtest_param.trade_cost, backtest_param.slippage, backtest_param.enable_trades))
            
            for task in asyncio.all_tasks(loop):
                task.cancel()
            
            return (backtest_param, backtest_result)
        except Exception as e:
            print(f'{backtest_param} Exception: {e}')
            return (backtest_param, backtest_result)
        finally:
            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
            loop.close()
