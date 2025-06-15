from decimal import Decimal
from typing import List, Tuple
import time

import pandas_ta as ta
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TrailingStop, TripleBarrierConfig


class PMMTrendingAdaptiveV2ControllerConfig(MarketMakingControllerConfigBase):
    controller_name: str = "pmm_trending_adaptive_v2"
    candles_config: List[CandlesConfig] = []
    buy_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of buy spreads measured in units of volatility(e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sell_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of sell spreads measured in units of volatility(e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sleep_interval: int = Field(
        default="3",
        json_schema_extra={
            "prompt": "Enter the sleep interval seconds(3): ",
            "prompt_on_new": True})
    update_interval: int = Field(
        default="180",
        json_schema_extra={
            "prompt": "Enter the update data interval seconds(180): ",
            "prompt_on_new": True})
    candle_interval: str = Field(
        default="15m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    sma_short_length: int = Field(
        default=10,
        json_schema_extra={"prompt": "Enter the SMA short length(10): ", "prompt_on_new": True})
    sma_length: int = Field(
        default=30,
        json_schema_extra={"prompt": "Enter the SMA length(30): ", "prompt_on_new": True})
    cci_length: int = Field(
        default=30,
        json_schema_extra={"prompt": "Enter the CCI length(30): ", "prompt_on_new": True})
    cci_threshold: int = Field(
        default=60,
        json_schema_extra={"prompt": "Enter the CCI threshold(60): ", "prompt_on_new": True})
    natr_length: int = Field(
        default=14,
        json_schema_extra={"prompt": "Enter the NATR length(14): ", "prompt_on_new": True})
    widen_spread_multiplier: float = Field(
        default=3,
        json_schema_extra={"prompt": "Enter the widen spread multiplier(3): ", "prompt_on_new": True})
    narrow_spread_multiplier: float = Field(
        default=0.6,
        json_schema_extra={"prompt": "Enter the narrow spread multiplier(0.5): ", "prompt_on_new": True})
    backtesting: bool = Field(default=False, json_schema_extra={"is_updatable": True})


class PMMTrendingAdaptiveV2Controller(MarketMakingControllerBase):
    """
    This is a dynamic version of the PMM controller.It uses the SMA and ADX to shift the mid-price 
    and the NATR to make the spreads dynamic. It also uses the Triple Barrier Strategy to manage the risk.
    """
    def __init__(self, config: PMMTrendingAdaptiveV2ControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.sma_length, config.cci_length, config.natr_length) + 100
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.connector_name,
                trading_pair=config.trading_pair,
                interval=config.candle_interval,
                max_records=self.max_records
            )]
        super().__init__(config, update_interval=self.config.sleep_interval, *args, **kwargs)
        self.update_data_interval = self.config.update_interval
        self.last_update_data_time = time.time() - self.update_data_interval - 10
        self.last_candle_timestamp = 0

    async def update_processed_data(self):
        current_time = time.time()
        if current_time - self.last_update_data_time < self.update_data_interval:
            return
        
        self.logger().info(f"Updating candles for {self.config.trading_pair} {self.config.candle_interval}")
        candles = self.market_data_provider.get_candles_df(connector_name=self.config.connector_name,
                                                           trading_pair=self.config.trading_pair,
                                                           interval=self.config.candle_interval,
                                                           max_records=self.max_records)
        last_candle_timestamp = candles["timestamp"].max()
        if last_candle_timestamp > self.last_candle_timestamp:
            self.last_candle_timestamp = last_candle_timestamp
            msg = f'There are new candles, updating processed data up to {int(self.last_candle_timestamp)} at {int(self.market_data_provider.time())}'
            self.logger().info(msg)
            # print(msg)
        else:
            if not self.config.backtesting:
                self.last_update_data_time = current_time
            return
        
        sma_short = ta.sma(candles["close"], length=self.config.sma_short_length, talib=False)
        sma = ta.sma(candles["close"], length=self.config.sma_length, talib=False)
        cci = ta.cci(candles["high"], candles["low"], candles["close"], length=self.config.cci_length, talib=False)
        natr = ta.natr(candles["high"], candles["low"], candles["close"], length=self.config.natr_length, scalar=1, talib=False)
        # adx = ta.adx(candles["high"], candles["low"], candles["close"], length=self.config.adx_length)
        
        candle_close = candles["close"].iloc[-1]
        candle_sma_short = sma_short.iloc[-1]
        candle_sma = sma.iloc[-1]
        candle_cci = cci.iloc[-1]
        
        trend = ""
        if candle_close > candle_sma_short and candle_close > candle_sma and candle_cci > self.config.cci_threshold:
            trend = "up"
            self.logger().info(f"Up trend => candle_close:{candle_close:.5f} > candle_sma_short:{candle_sma_short:.5f} and candle_sma:{candle_sma:.5f}, " \
                               f"candle_cci:{candle_cci:.1f} > threshold:{self.config.cci_threshold:.1f}")
        elif candle_close < candle_sma_short and candle_close < candle_sma and candle_cci < -self.config.cci_threshold:
            trend = "down"
            self.logger().info(f"Down trend => candle_close:{candle_close:.5f} < candle_sma_short:{candle_sma_short:.5f} and candle_sma:{candle_sma:.5f}, " \
                               f"candle_cci:{candle_cci:.1f} < threshold:{-self.config.cci_threshold:.1f}")
        
        reference_price = self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
        
        self.processed_data.update(
            {
                "reference_price": Decimal(reference_price),
                "natr": Decimal(natr.iloc[-1]),
                "trend": trend
            }
        )
        
        if not self.config.backtesting:
            self.last_update_data_time = current_time

    def get_price_and_amount(self, level_id: str) -> Tuple[Decimal, Decimal]:
        """
        Get the spread and amount in quote for a given level id.
        """
        level = self.get_level_from_level_id(level_id)
        trade_type = self.get_trade_type_from_level_id(level_id)
        spreads, amounts_quote = self.config.get_spreads_and_amounts_in_quote(trade_type)
        reference_price = Decimal(self.processed_data["reference_price"])
        base_spread_multiplier = Decimal(self.processed_data["natr"]) / 2
        spread_in_pct = Decimal(spreads[int(level)]) * base_spread_multiplier
        
        trend = self.processed_data["trend"]
        
        if trend == "up":
            if trade_type == TradeType.BUY:
                spread_in_pct *= Decimal(self.config.narrow_spread_multiplier)
                self.logger().info(f"Up trend [Narrow] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
            elif trade_type == TradeType.SELL:
                spread_in_pct *= Decimal(self.config.widen_spread_multiplier)
                self.logger().info(f"Up trend [Widen] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
        elif trend == "down":
            if trade_type == TradeType.BUY:
                spread_in_pct *= Decimal(self.config.widen_spread_multiplier)
                self.logger().info(f"Down trend [Widen] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
            elif trade_type == TradeType.SELL:
                spread_in_pct *= Decimal(self.config.narrow_spread_multiplier)
                self.logger().info(f"Down trend [Narrow] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
        
        side_multiplier = Decimal("-1") if trade_type == TradeType.BUY else Decimal("1")
        order_price = reference_price * (1 + side_multiplier * spread_in_pct)
        if order_price <= 0:
            return 0, 0
        
        return order_price, round(Decimal(amounts_quote[int(level)]) / order_price)

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        if price == 0:
            return None
        if amount == 0:
            return None
        
        level = self.get_level_from_level_id(level_id)
        trade_type = self.get_trade_type_from_level_id(level_id)
        spreads, amounts_quote = self.config.get_spreads_and_amounts_in_quote(trade_type)
        
        natr = Decimal(self.processed_data["natr"])
        base_spread_multiplier = natr / 2
        
        initial_config = self.config.triple_barrier_config
        
        stop_loss = initial_config.stop_loss
        take_profit = max(initial_config.take_profit, base_spread_multiplier * 4)
        time_limit = initial_config.time_limit
        trailing_stop_activation_price = min(Decimal(0.8) * initial_config.take_profit, max(initial_config.trailing_stop.activation_price, base_spread_multiplier * 2))
        trailing_stop_delta = min(2 * initial_config.trailing_stop.trailing_delta, max(initial_config.trailing_stop.trailing_delta, base_spread_multiplier))
        
        trailing_stop = TrailingStop(activation_price=trailing_stop_activation_price, trailing_delta=trailing_stop_delta)
        
        triple_barrier_config = TripleBarrierConfig(
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_limit=time_limit,
            trailing_stop=trailing_stop,
            open_order_type=OrderType.LIMIT,  # Defaulting to LIMIT as is a Maker Controller
            take_profit_order_type=self.config.triple_barrier_config.take_profit_order_type,
            stop_loss_order_type=OrderType.MARKET,  # Defaulting to MARKET as per requirement
            time_limit_order_type=OrderType.MARKET  # Defaulting to MARKET as per requirement
        )
        
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )
