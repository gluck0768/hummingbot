import os
import glob
import ccxt
from backtesting import backtest_utils
import pandas as pd
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CCXTDownloader:
    
    def __init__(self, exchange_id: str, base_dir = '.'):
        if '_' in exchange_id:
            exchange_id, perpetual = exchange_id.split('_')
            self.is_perpetual = perpetual.lower() == 'perpetual'
        else:
            self.exchange_id = exchange_id
            self.is_perpetual = False
            
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class()
        self.exchange.enableRateLimit = True
        self.exchange.socksProxy = backtest_utils.SOCKS_PROXY_URL
        self.data_dir = os.path.join(base_dir, 'data')
    
    def fetch_trades(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        symbol = self._to_ccxt_symbol(symbol, self.is_perpetual)
        existing_trades = self._load_trades_from_local(symbol, start_time, end_time)
        if not existing_trades.empty:
            return existing_trades
        
        if not self.exchange.has['fetchTrades']:
            logger.error(f"Exchange {self.exchange_id} does not support fetching trades")
            return pd.DataFrame()
        
        logger.info(f"Fetching {symbol} trades from {datetime.fromtimestamp(start_time).isoformat()} "
            f"to {datetime.fromtimestamp(end_time).isoformat()}")
        
        seen_ids = set()
        
        start_time_ms = start_time * 1000
        end_time_ms = end_time * 1000
        current_since_ms = start_time_ms
        last_timestamp = start_time_ms
        last_trade_id = None
        
        all_trades = []
        try:
            while current_since_ms < end_time_ms:
                trades = self.exchange.fetch_trades(symbol, since=current_since_ms, limit=1000)
                
                if not trades:
                    break
                    
                new_trades = []
                for trade in trades:
                    trade_id = int(trade['id'])
                    
                    if trade_id in seen_ids:
                        continue
                        
                    if trade['timestamp'] < start_time_ms:
                        continue
                        
                    if trade['timestamp'] >= end_time_ms:
                        break
                        
                    if last_trade_id is not None:
                        if trade_id <= last_trade_id:
                            logger.error(f"Current trade id:{trade_id} is less than last trade id:{last_trade_id}")
                        elif trade_id != last_trade_id + 1:
                            logger.error(f"Trade id:{trade_id} is not sequential, last trade id:{last_trade_id}")
                    
                    last_trade_id = trade_id
                    
                    new_trades.append({"timestamp": int(trade['timestamp'] / 1000), "price": trade['price'], "volume": trade['amount']})
                    seen_ids.add(trade_id)
                
                if new_trades:
                    all_trades.extend(new_trades)
                    last_timestamp = new_trades[-1]['timestamp']
                    progress = min(1, (last_timestamp - start_time) / (end_time - start_time))
                    logger.info(f"Fetched {len(all_trades)} trades, progress: {progress:.2%}")
                
                current_since_ms = trades[-1]['timestamp']
                if current_since_ms > end_time_ms:
                    break
                
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
        except Exception as e:
            logger.error(f"Unknown error: {e}")
        
        result_df = pd.DataFrame(all_trades)
        self._save_to_local(result_df, symbol, start_time, end_time)
        return result_df
        
    def _to_ccxt_symbol(self, symbol: str, is_perpetual: bool):
        quote, asset = symbol.upper().split("-")
        if is_perpetual:
            return f"{quote}/{asset}:{asset}"
        else:
            return f"{quote}/{asset}"

    def _get_local_trades_file_path(self, symbol: str, start_time: int, end_time: int) -> str:
        key = self._generate_key(symbol)
        return f"{key}-{start_time}-{end_time}.parquet"
    
    def _load_trades_from_local(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, self._get_local_trades_file_path(symbol, start_time, end_time))
        if not os.path.exists(file_path):
            key = self._generate_key(symbol)
            files = glob.glob(os.path.join(self.data_dir, f"{key}-*.parquet"))
            
            existing = False
            for f in files:
                if int(f.split('-')[-2]) <= start_time and end_time <= int(f.split('-')[-1].split('.')[0]):
                    file_path = f
                    existing = True
                    break
                
            if not existing:
                return pd.DataFrame()
            
        return pd.read_parquet(file_path, engine="pyarrow")
    
    def _save_to_local(self, df: pd.DataFrame, symbol: str, start_time: int, end_time: int):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        file_path = os.path.join(self.data_dir, self._get_local_trades_file_path(symbol, start_time, end_time))
        df.to_parquet(file_path, engine="pyarrow")
        print(f'Saved trades to {file_path}')
    
    def _generate_key(self, symbol: str) -> str:
        symbol = symbol.replace("/", "-")
        symbol = symbol.replace(":", "_")
        return f"{symbol}-trades"


# if __name__ == "__main__":
#     ccxt_downloader = CCXTDownloader('binance_perpetual')
    
#     start_date = datetime(2025, 6, 14, 16, 52, 43)
#     end_date = datetime(2025, 6, 14, 17, 30, 0)
    
#     trades = ccxt_downloader.fetch_trades('TRUMP-USDT', int(start_date.timestamp()), int(end_date.timestamp()))
    