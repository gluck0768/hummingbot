import os
import pickle
import glob
from decimal import Decimal
from typing import Dict

import pandas as pd

from backtesting.base.backtesting_data_provider import BacktestingDataProvider

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig


class CacheableBacktestMarketDataProvider(BacktestingDataProvider):
    
    def __init__(self, connectors: Dict[str, ConnectorBase], base_dir: str = '.'):
        super().__init__(connectors)
        self.data_dir = os.path.join(base_dir, 'data')
    
    async def init_data(self, start_time: int, end_time: int, candles_config: CandlesConfig):
        self.update_start_end_time(start_time, end_time)
        await self.initialize_trading_rules(candles_config.connector)
        await self.get_candles_feed(candles_config)
    
    async def initialize_trading_rules(self, connector: str):
        if len(self.trading_rules.get(connector, {})) == 0:
            file_path = os.path.join(self.data_dir, self._get_local_trading_rules_file(connector))
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    # print(f'Loaded {connector} trading rules from {file_path}')
                    self.trading_rules[connector] = pickle.load(f)
                    return
        else:
            return
        
        await super().initialize_trading_rules(connector)
        
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            
        with open(file_path, "wb") as f:
            pickle.dump(self.trading_rules[connector], f)
            print(f'Saved {connector} trading rules to {file_path}')
    
    def update_start_end_time(self, start_time: int, end_time: int):
        self.start_time = start_time
        self.end_time = end_time
        
    def update_time(self, time: int):
        self._time = time
    
    def _get_local_trading_rules_file(self, key: str):
        return f'{key}-trading-rules.pkl'
    
    def _get_local_market_data_file(self, key: str):
        return f'{key}-{self.start_time}-{self.end_time}.parquet'
    
    def _load_market_data_from_local(self, key: str):
        file_path = os.path.join(self.data_dir, self._get_local_market_data_file(key))
        if not os.path.exists(file_path):
            files = glob.glob(os.path.join(self.data_dir, f"{key}-*.parquet"))
            
            existing = False
            for f in files:
                if int(f.split('-')[-2]) <= self.start_time and self.end_time <= int(f.split('-')[-1].split('.')[0]):
                    file_path = f
                    existing = True
                    break
                
            if not existing:
                return pd.DataFrame()
        
        # print(f'Loaded market data from {file_path}')
        return self.index_and_sort_by_timestamp(pd.read_parquet(file_path, engine="pyarrow"))
    
    def _save_to_local(self, df: pd.DataFrame, key: str):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        file_path = os.path.join(self.data_dir, self._get_local_market_data_file(key))
        df.to_parquet(file_path, engine="pyarrow")
        print(f'Saved market data to {file_path}')
        
    def _filter_existing_feed(self, existing_feed: pd.DataFrame):
        if existing_feed.empty:
            return existing_feed
        
        if 'timestamp' in existing_feed.columns:
            existing_feed = self.index_and_sort_by_timestamp(existing_feed)
        
        existing_feed_start_time = existing_feed.index.min()
        existing_feed_end_time = existing_feed.index.max()
        if existing_feed_start_time <= self.start_time and existing_feed_end_time >= self.end_time:
            return self.index_and_sort_by_timestamp(existing_feed[existing_feed.index < self.end_time])
        else:
            return pd.DataFrame()

    async def get_candles_feed(self, config: CandlesConfig):
        key = self._generate_candle_feed_key(config)
        existing_feed = self._filter_existing_feed(self.candles_feeds.get(key, pd.DataFrame()))
        if not existing_feed.empty:
            return existing_feed
        
        candles_df = self._load_market_data_from_local(key)
        if candles_df.empty:
            candles_df = await super().get_candles_feed(config)
            self._save_to_local(candles_df, key)
        else:
            self.candles_feeds[key] = candles_df
        return self._filter_existing_feed(candles_df)
    
    @staticmethod
    def index_and_sort_by_timestamp(df: pd.DataFrame):
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('int64')
            df = df.set_index('timestamp').sort_index(ascending=True)
            df['timestamp'] = df.index
        return df
