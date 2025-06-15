from decimal import Decimal
from typing import List, Tuple

import pandas_ta as ta
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig


class PMMTrendingAdaptiveControllerConfig(MarketMakingControllerConfigBase):
    controller_name: str = "pmm_trending_adaptive"
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
    update_interval: int = Field(
        default="60",
        json_schema_extra={
            "prompt": "Enter the update interval seconds: ",
            "prompt_on_new": True})
    candle_interval: str = Field(
        default="15m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    sma_length: int = Field(
        default=30,
        json_schema_extra={"prompt": "Enter the SMA length: ", "prompt_on_new": True})
    cci_length: int = Field(
        default=30,
        json_schema_extra={"prompt": "Enter the CCI length: ", "prompt_on_new": True})
    natr_length: int = Field(
        default=14,
        json_schema_extra={"prompt": "Enter the NATR length: ", "prompt_on_new": True})
    backtesting: bool = Field(default=False, json_schema_extra={"is_updatable": True})


class PMMTrendingAdaptiveController(MarketMakingControllerBase):
    """
    This is a dynamic version of the PMM controller.It uses the SMA and ADX to shift the mid-price 
    and the NATR to make the spreads dynamic. It also uses the Triple Barrier Strategy to manage the risk.
    """
    def __init__(self, config: PMMTrendingAdaptiveControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.sma_length, config.cci_length, config.natr_length) + 100
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.connector_name,
                trading_pair=config.trading_pair,
                interval=config.candle_interval,
                max_records=self.max_records
            )]
        super().__init__(config, update_interval=self.config.update_interval, *args, **kwargs)

    async def update_processed_data(self):
        candles = self.market_data_provider.get_candles_df(connector_name=self.config.connector_name,
                                                           trading_pair=self.config.trading_pair,
                                                           interval=self.config.candle_interval,
                                                           max_records=self.max_records)
        sma = ta.sma(candles["close"], length=self.config.sma_length, talib=False)
        cci = ta.cci(candles["high"], candles["low"], candles["close"], length=self.config.cci_length, talib=False)
        natr = ta.natr(candles["high"], candles["low"], candles["close"], length=self.config.natr_length, scalar=1, talib=False)
        # adx = ta.adx(candles["high"], candles["low"], candles["close"], length=self.config.adx_length)
        
        reference_price = self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)

        self.processed_data.update(
            {
                "reference_price": Decimal(reference_price),
                "natr": Decimal(natr.iloc[-1]),
                "candle_close": candles["close"].iloc[-1],
                "candle_sma": sma.iloc[-1],
                "candle_cci": cci.iloc[-1]
            }
        )

    def get_price_and_amount(self, level_id: str) -> Tuple[Decimal, Decimal]:
        """
        Get the spread and amount in quote for a given level id.
        """
        level = self.get_level_from_level_id(level_id)
        trade_type = self.get_trade_type_from_level_id(level_id)
        spreads, amounts_quote = self.config.get_spreads_and_amounts_in_quote(trade_type)
        reference_price = Decimal(self.processed_data["reference_price"])
        spread_multiplier = Decimal(self.processed_data["natr"]) / 2
        spread_in_pct = Decimal(spreads[int(level)]) * spread_multiplier
        
        candle_close = self.processed_data["candle_close"]
        candle_sma = self.processed_data["candle_sma"]
        candle_cci = self.processed_data["candle_cci"]
        
        if trade_type == TradeType.BUY:
            if candle_close < candle_sma and candle_cci < -60:
                spread_in_pct *= 3
                msg = f"Widen {level_id} spread:{spread_in_pct:.2%}, spread_multiplier:{spread_multiplier:.2%}, reference_price:{reference_price:.5f}, " \
                        f"candle_close:{candle_close:.5f}, candle_sma:{candle_sma:.5f}, candle_cci:{candle_cci:.1f}"
                self.logger().info(msg)
        else:
            if candle_close > candle_sma and candle_cci > 60:
                spread_in_pct *= 3
                msg = f"Widen {level_id} spread:{spread_in_pct:.2%}, spread_multiplier:{spread_multiplier:.2%}, reference_price:{reference_price:.5f}, " \
                        f"candle_close:{candle_close:.5f}, candle_sma:{candle_sma:.5f}, candle_cci:{candle_cci:.1f}"
                self.logger().info(msg)
        
        side_multiplier = Decimal("-1") if trade_type == TradeType.BUY else Decimal("1")
        order_price = reference_price * (1 + side_multiplier * spread_in_pct)
        # print(f"order_price:{order_price}, spread_in_pct:{spread_in_pct:.2%}, spread_multiplier:{spread_multiplier:.2%}, reference_price:{reference_price:.5f}, candle_close:{candle_close:.5f}, candle_sma:{candle_sma:.5f}, candle_cci:{candle_cci:.1f}")
        if order_price <= 0:
            return 0, 0
        
        return order_price, Decimal(amounts_quote[int(level)]) / order_price

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        if price == 0:
            return None
        if amount == 0:
            return None
        
        trade_type = self.get_trade_type_from_level_id(level_id)
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )
