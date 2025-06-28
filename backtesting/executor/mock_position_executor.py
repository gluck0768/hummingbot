from datetime import datetime
import os
import sys

from hummingbot.core.data_type.trade_fee import TradeFeeBase, TradeFeeSchema

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))

import uuid
from typing import Dict
from decimal import Decimal

from backtesting.base.backtesting_data_provider import BacktestingDataProvider

from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate 
from hummingbot.core.event.events import BuyOrderCompletedEvent, BuyOrderCreatedEvent, OrderCancelledEvent, OrderFilledEvent, SellOrderCompletedEvent, SellOrderCreatedEvent
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor


class MockConnectorStrategy(ScriptStrategyBase):
    
    def __init__(self, market_data_provider: BacktestingDataProvider, connectors={}, config = None):
        super().__init__(connectors, config)
        self.ready_to_trade = True
        self.market_data_provider = market_data_provider
        
    def buy(self, connector_name, trading_pair, amount, order_type, price=Decimal("NaN"), position_action=PositionAction.OPEN):
        return uuid.uuid4()
    
    def sell(self, connector_name, trading_pair, amount, order_type, price=Decimal("NaN"), position_action=PositionAction.OPEN):
        return uuid.uuid4()
    
    def cancel(self, connector_name, trading_pair, order_id):
        return
    
    @property
    def current_timestamp(self) -> Decimal:
        return self.market_data_provider.time()
    

class MockPositionExecutor(PositionExecutor):
    
    def __init__(self, config: PositionExecutorConfig, market_data_provider: BacktestingDataProvider, trade_cost: float, slippage: float, strategy: MockConnectorStrategy = None, update_interval = 1, max_retries = 10):
        self.market_data_provider = market_data_provider
        self.trade_cost = Decimal(trade_cost)
        self.slippage = Decimal(slippage)
        self.in_flight_orders: Dict[str, InFlightOrder] = {}
        self.entry_timestamp = None
        
        if strategy is None:
            strategy = MockConnectorStrategy(market_data_provider=market_data_provider, connectors={})
        
        super().__init__(strategy, config, update_interval, max_retries)
        
        self._status = RunnableStatus.RUNNING
    
    def current_time(self):
        return self.market_data_provider.time()
    
    def get_custom_info(self) -> Dict:
        custom_info = super().get_custom_info()
        custom_info['entry_timestamp'] = self.entry_timestamp
        return custom_info
    
    def get_trading_rules(self, connector_name, trading_pair):
        return self.market_data_provider.get_trading_rules(connector_name, trading_pair)
    
    def register_events(self):
        return
    
    def unregister_events(self):
        return

    def early_stop(self, keep_position: bool = False):
        if self.is_trading:
            self.place_close_order_and_cancel_open_orders(CloseType.EARLY_STOP)

        super().early_stop(keep_position)
        super().stop()
        
    def expired_stop(self):
        close_type = CloseType.EXPIRED
        if self.is_trading:
            self.place_close_order_and_cancel_open_orders(close_type)
        else:
            self.close_type = close_type
        super().stop()
    
    def quantize_order_price(self, price) -> Decimal:
        return self.market_data_provider.quantize_order_price(self.config.connector_name, self.config.trading_pair, price)
    
    def quantize_order_amount(self, amount) -> Decimal:
        return self.market_data_provider.quantize_order_amount(self.config.connector_name, self.config.trading_pair, amount)

    @property
    def open_filled_amount(self) -> Decimal:
        if self._open_order:
            if self._open_order.fee_asset == self.config.trading_pair.split("-")[0]:
                open_filled_amount = self._open_order.executed_amount_base - self._open_order.cum_fees_base
            else:
                open_filled_amount = self._open_order.executed_amount_base
            return self.quantize_order_amount(open_filled_amount)
        else:
            return Decimal("0")
    
    def get_in_flight_order(self, connector_name, order_id: str) -> InFlightOrder:
        return self.in_flight_orders.get(order_id, None)
    
    def update_in_flight_order(self, order_id: str, order: InFlightOrder):
        self.in_flight_orders[order_id] = order
        
    def remove_in_flight_order(self, order_id: str):
        if order_id and order_id in self.in_flight_orders.keys():
            del self.in_flight_orders[order_id]
    
    def can_fill(self, order_type: OrderType, side: TradeType, price: Decimal):
        if order_type == OrderType.MARKET:
            return True
        
        market_price = self.get_market_price()
        is_buy = side == TradeType.BUY
        if (is_buy and market_price > price) or (not is_buy and market_price < price):
            return False
        return True
    
    def update_order_state(self, order: InFlightOrder, order_state: OrderState):
        if not order:
            return
        
        order.update_with_order_update(OrderUpdate(
            trading_pair=order.trading_pair,
            update_timestamp=self.current_time(),
            new_state=order_state,
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id
        ))
    
    def calc_fee(self, order: InFlightOrder) -> TradeFeeBase:
        fee_schema = TradeFeeSchema()
        if self.is_perpetual:
            trade_fee = TradeFeeBase.new_perpetual_fee(
                fee_schema=fee_schema,
                position_action=order.position,
                percent=self.trade_cost,
                percent_token=None,
                flat_fees=[]
            )
        else:
            trade_fee = TradeFeeBase.new_spot_fee(
                fee_schema=fee_schema,
                trade_type=order.trade_type,
                percent=self.trade_cost,
                percent_token=None,
                flat_fees=[]
            )
        return trade_fee
    
    def update_order_trade(self, order: InFlightOrder):
        if not order:
            return
        
        fill_price = self.calc_order_filled_price(order)
        order.update_with_trade_update(TradeUpdate(
            trade_id=uuid.uuid4(),
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            trading_pair=order.trading_pair,
            fill_timestamp=self.current_time(),
            fill_price=fill_price,
            fill_base_amount=order.amount,
            fill_quote_amount=order.amount*fill_price,
            fee=self.calc_fee(order)
        ))
    
    def place_order(self, connector_name, trading_pair, order_type, side, amount, position_action = PositionAction.NIL, price=Decimal("NaN")):
        price = self.process_nan_price(price)
        order_id = super().place_order(connector_name, trading_pair, order_type, side, amount, position_action, price)
        
        order = InFlightOrder(
            client_order_id=order_id,
            trading_pair=trading_pair,
            order_type=order_type,
            trade_type=side,
            amount=amount,
            creation_timestamp=self.current_time(),
            price=price,
            exchange_order_id=uuid.uuid4(),
            leverage=self.config.leverage,
            position=position_action,
        )
        
        self.update_in_flight_order(order_id, order)
        return order_id
    
    def place_open_order(self):
        super().place_open_order()
        
        order_id = self._open_order.order_id
        order: InFlightOrder = self.get_in_flight_order(self.config.connector_name, order_id)
        
        event_class = BuyOrderCreatedEvent if self.is_buy() else SellOrderCreatedEvent
        created_event = event_class(
            timestamp=self.current_time(),
            type=order.order_type,
            trading_pair=order.trading_pair,
            amount=order.amount,
            price=order.price,
            order_id=order_id,
            creation_timestamp=order.creation_timestamp,
            exchange_order_id=order.exchange_order_id,
            leverage=order.leverage,
            position=order.position
        )
        
        self.update_order_state(order, OrderState.OPEN)
        self.update_in_flight_order(order_id, order)
        self.process_order_created_event(None, None, created_event)

    def process_nan_price(self, price: Decimal):
        if price.is_nan():
            return self.current_market_price
        return price
    
    def calc_close_price_for_market_order(self, price: Decimal, close_type: CloseType) -> Decimal:
        if not self._open_order or not self._open_order.order:
            return price
        
        open_order = self._open_order.order

        if close_type == CloseType.TAKE_PROFIT:
            close_price = self.take_profit_price
            return self.calc_filled_price_with_slippage(close_price, OrderType.MARKET, self.close_order_side)
        elif close_type == CloseType.STOP_LOSS:
            factor = 1 if open_order.trade_type == TradeType.SELL else -1
            close_price = self.entry_price * (1 + factor * self.config.triple_barrier_config.stop_loss)
            return self.calc_filled_price_with_slippage(close_price, OrderType.MARKET, self.close_order_side)
        elif close_type == CloseType.TRAILING_STOP or close_type == CloseType.TIME_LIMIT \
            or close_type == CloseType.EARLY_STOP or close_type == CloseType.EXPIRED:
            close_price = self.get_market_price()
            return self.calc_filled_price_with_slippage(close_price, OrderType.MARKET, self.close_order_side)
    
        return price
        
    def place_close_order_and_cancel_open_orders(self, close_type: CloseType, price=Decimal("NaN")):
        price = self.process_nan_price(price)
        price = self.calc_close_price_for_market_order(price, close_type)
        super().place_close_order_and_cancel_open_orders(close_type, price)
        
        if not self._close_order:
            return
        
        order_id = self._close_order.order_id
        close_order = self.get_in_flight_order(self.config.connector_name, order_id)
        if not close_order:
            return
        
        if self._open_order:
            close_order.creation_timestamp = self._open_order.creation_timestamp
            self.update_in_flight_order(order_id, close_order)
        
        if self.can_fill(close_order.order_type, close_order.trade_type, price):
            self.update_order_state(close_order, OrderState.FILLED)
            self.update_order_trade(close_order)
            self.build_and_process_filled_event(close_order, close_type)

    def place_take_profit_limit_order(self):
        if self.config.triple_barrier_config.take_profit_order_type == OrderType.LIMIT:
            # Actually when placing take profit limit order, the take_profit_price does not include trade_cost.
            # It differs from the take_profit_price calculated in market order type, which will calculate pnl including fees.
            # Here just show the difference between the two prices. The backtest result will also be different but it's fine.
            factor = -1 if self._open_order.order.trade_type == TradeType.SELL else 1
            entry_price = self.take_profit_price * (1 + factor * self.trade_cost)
        
        order_price = self.quantize_order_price(self.take_profit_price)
        order_id = self.place_order(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            amount=self.amount_to_close,
            price=order_price,
            order_type=self.config.triple_barrier_config.take_profit_order_type,
            position_action=PositionAction.CLOSE,
            side=self.close_order_side,
        )
        self._take_profit_limit_order = TrackedOrder(order_id=order_id)
        self.logger().debug(f"Executor ID: {self.config.id} - Placing take profit order {order_id}")

    def format_timestamp(self, current_timestamp):
        return datetime.fromtimestamp(current_timestamp).strftime('%m%d/%H:%M:%S')
    
    def calc_filled_price_with_slippage(self, price: Decimal, order_type: OrderType, trade_type: TradeType) -> Decimal:
        if order_type == OrderType.LIMIT:
            return price
        
        factor = -1 if trade_type == TradeType.SELL else 1
        return self.quantize_order_price(price * (1 + factor * self.slippage))
    
    def calc_order_filled_price(self, order: InFlightOrder) -> Decimal:
        return self.calc_filled_price_with_slippage(order.price, order.order_type, order.trade_type)
    
    def build_and_process_filled_event(self, order: InFlightOrder, close_type: CloseType = None):
        trade_fee = self.calc_fee(order)
        current_timestamp = self.current_time()
        filled_event = OrderFilledEvent(
            timestamp=current_timestamp,
            order_id=order.client_order_id,
            trading_pair=order.trading_pair,
            trade_type=order.trade_type,
            order_type=order.order_type,
            price=self.calc_order_filled_price(order),
            amount=order.amount,
            trade_fee=trade_fee,
            exchange_order_id=order.exchange_order_id,
            leverage=order.leverage,
            position=order.position
        )
        self.process_order_filled_event(None, None, filled_event)
        
        fee = self.get_cum_fees_quote()
        if order.position == PositionAction.CLOSE:
            open_order = self._open_order.order
            msg = f"[{self.format_timestamp(open_order.creation_timestamp)}-{self.format_timestamp(open_order.last_update_timestamp)}-{self.format_timestamp(current_timestamp)}] Close {close_type.name}: " \
                f"pnl={self.net_pnl_quote:.5f}, pct={self.net_pnl_pct:.4%}, entry@{open_order.price:.7f}, filled@{filled_event.price:.7f}, amount={filled_event.amount:.2f}, " \
                f"quote={filled_event.amount*filled_event.price:.7f}, fee={fee:.7f}, " \
                f"{filled_event.trade_type.name}-{filled_event.order_type.name}-{filled_event.position.name}, executor:{self.config.id}, status:{self._status.name}"
            self.logger().warning(f'\033[92m{msg}\033[0m' if self.net_pnl_quote > 0 else f'\033[91m{msg}\033[0m')
        else:
            self.logger().warning(f"[{self.format_timestamp(order.creation_timestamp)}-{self.format_timestamp(filled_event.timestamp)}] Open filled: fill price={filled_event.price:.7f}, " 
                                  f"amount={filled_event.amount:.2f}, quote={filled_event.amount*filled_event.price:.7f}, {filled_event.trade_type.name}-{filled_event.order_type.name}-{filled_event.position.name}, executor:{self.config.id}, status:{self._status.name}")
            
    def control_take_profit(self):
        super().control_take_profit()
        
        if not self._take_profit_limit_order:
            return
        
        open_order = self._open_order.order
        
        order_id = self._take_profit_limit_order.order_id
        take_profit_order = self.get_in_flight_order(self.config.connector_name, order_id)
        
        if take_profit_order.is_open \
            and not take_profit_order.is_filled \
            and self.can_fill(take_profit_order.order_type, take_profit_order.trade_type, take_profit_order.price):
            self.update_order_state(take_profit_order, OrderState.FILLED)
            self.update_order_trade(take_profit_order)
            # will set close order to take profit order
            self.build_and_process_completed_event(take_profit_order)
            
            fee = self.get_cum_fees_quote()
            msg = f"[{self.format_timestamp(open_order.creation_timestamp)}-{self.format_timestamp(take_profit_order.creation_timestamp)}-{self.format_timestamp(take_profit_order.last_update_timestamp)}] Close TAKE_PROFIT limit: " \
                f"pnl={self.net_pnl_quote:.5f}, pct={self.net_pnl_pct:.4%}, entry@{open_order.price:.7f}, filled@{take_profit_order.price:.7f}, amount={take_profit_order.amount:.2f}, " \
                f"quote={take_profit_order.amount*take_profit_order.price:.7f}, fee={fee:.7f}, " \
                f"{take_profit_order.trade_type.name}-{take_profit_order.order_type.name}-{take_profit_order.position.name}, executor:{self.config.id}, status:{self._status.name}"
            self.logger().warning(f'\033[92m{msg}\033[0m' if self.net_pnl_quote > 0 else f'\033[91m{msg}\033[0m')

    def build_and_process_completed_event(self, order: InFlightOrder):
        event_class = BuyOrderCompletedEvent if self.is_buy() else SellOrderCompletedEvent
        if order.order_type == OrderType.MARKET:
            price = self.get_market_price()
        else:
            price = order.price
            
        event = event_class(
            timestamp=self.current_time(),
            order_id=order.client_order_id,
            base_asset=order.base_asset,
            quote_asset=order.quote_asset,
            base_asset_amount=order.amount,
            quote_asset_amount=order.amount*price,
            order_type=order.order_type
        )
        self.process_order_completed_event(None, None, event)
        
    def cancel_order(self, order: TrackedOrder):
        if not order:
            return
        
        order_id = order.order_id
        order = self.get_in_flight_order(self.config.connector_name, order_id)
        if not order:
            return
        
        self.update_order_state(order, OrderState.CANCELED)
        # self.remove_in_flight_order(order_id)
        
        event = OrderCancelledEvent(
            timestamp=self.current_time(),
            order_id=order_id,
            exchange_order_id=order.exchange_order_id
        )
        self.process_order_canceled_event(None, None, event)
    
    def cancel_open_order(self):
        super().cancel_open_order()
        self.cancel_order(self._open_order)
        
    def cancel_take_profit(self):
        super().cancel_take_profit()
        self.cancel_order(self._take_profit_limit_order)
    
    def get_side(self):
        return self.config.side
    
    def is_buy(self):
        return self.get_side() == TradeType.BUY
    
    def get_price(self, connector_name, trading_pair, price_type = PriceType.MidPrice):
        return self.get_market_price(price_type)
    
    def get_market_price(self, price_type = PriceType.MidPrice):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)
    
    def on_trade(self, trade_data) -> bool:
        self.market_data_provider.update_price(self.config.connector_name, self.config.trading_pair, trade_data['price'])
        self.market_data_provider.update_volume(self.config.connector_name, self.config.trading_pair, trade_data['volume'])
        
        self.control_open_order()
        if self.determine_filled(self._open_order):
            self.entry_timestamp = self.current_time()
        self.control_barriers()
        
        if self.status == RunnableStatus.SHUTTING_DOWN:
            super().stop()
            return False
        
        return True
    
    def on_candle(self, candle_data) -> bool:
        """ test on market data

        Args:
            market_data (_type_): _description_

        Returns:
            bool: whether the executor is active
        """
        prices = []
        
        if candle_data['open'] < candle_data['close']:
            prices.extend([candle_data['low'], candle_data['high']])
        else:
            prices.extend([candle_data['high'], candle_data['low']])
        
        prices.append(candle_data['close'])
        volume = candle_data['volume'] / len(prices)

        for price in prices:
            self.market_data_provider.update_price(self.config.connector_name, self.config.trading_pair, price)
            self.market_data_provider.update_volume(self.config.connector_name, self.config.trading_pair, volume)
            
            self.control_open_order()
            if self.determine_filled(self._open_order):
                self.entry_timestamp = self.current_time()
            self.control_barriers()
            
            if self.status == RunnableStatus.SHUTTING_DOWN:
                super().stop()
                return False
        
        return True
    
    def determine_filled(self, tracked_order: TrackedOrder) -> bool:
        if not tracked_order:
            return False
        
        order = tracked_order.order
        if not order or order.is_filled:
            return False
        
        if not self.can_fill(order.order_type, order.trade_type, order.price):
            return False
        
        self.update_order_state(order, OrderState.FILLED)
        self.update_order_trade(order)
        self.build_and_process_filled_event(order, None)
        return True
    