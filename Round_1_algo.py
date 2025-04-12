from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics
import jsonpickle

# set all parameters on Bollinger Bands (rolling window and multiplier)
WINDOW = 15
STD_DEV_MULTIPLIER = 1.3

# resin position limit and fair price
resin_position_limit: int = 50
resin_fair_price: int = 10000

# kelp position limit
kelp_position_limit: int = 50

# ink position limit
ink_position_limit: int = 50

# algo to compute Bollinger Bands
def compute_bollinger_bands(prices: List[float], window: int = WINDOW, multiplier: float = STD_DEV_MULTIPLIER):
    # if the window is not long enough
    if len(prices) < window:
        return None, None
    # get the rolling window historical price
    window_prices = prices[-window:]
    # calculate the mean and std
    mean_price = statistics.mean(window_prices)
    std_dev = statistics.stdev(window_prices)
    # calculate the Bollinger Bands (determine the range; assume normal distribution)
    upper_band = mean_price + multiplier * std_dev
    lower_band = mean_price - multiplier * std_dev
    return upper_band, lower_band

def kelp_algo(state: TradingState, product, kelp_position_limit, historical_prices):

    # retrieve all relevant information about kelp
    orders: list[Order] = []
    order_depth: OrderDepth = state.order_depths[product]
    positions = state.position
    cur_position = positions.get(product, 0)

    # Mid-price estimation
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    # skip the mid_price calculation if one of the side is closed
    if best_ask is None or best_bid is None:
        return orders

    # calculate the mid_price and store in the historical data
    mid_price = (best_ask + best_bid) / 2
    historical_prices.append(mid_price)

    # Calculate Bollinger Bands
    upper, lower = compute_bollinger_bands(historical_prices)
    # Not enough data in the rolling window
    if upper is None or lower is None:
        return orders

    # if the mid_price is lower, buy and check the limit
    if mid_price < lower and cur_position < kelp_position_limit:
        buy_price = best_ask  # try to buy at the current ask
        buy_volume = min(order_depth.sell_orders[buy_price], kelp_position_limit - cur_position)
        orders.append(Order(product, buy_price, buy_volume))

    # if the mid_price is higher, sell and check the limit
    elif mid_price > upper and cur_position > -kelp_position_limit:
        sell_price = best_bid  # try to sell at the current bid
        sell_volume = min(-order_depth.buy_orders[sell_price], cur_position + kelp_position_limit)
        orders.append(Order(product, sell_price, sell_volume))

    return orders

def ink_algo(state: TradingState, product, ink_position_limit, historical_prices):

    # retrieve all relevant information about kelp
    orders: list[Order] = []
    order_depth: OrderDepth = state.order_depths[product]
    positions = state.position
    cur_position = positions.get(product, 0)

    # Mid-price estimation
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    # skip the mid_price calculation if one of the side is closed
    if best_ask is None or best_bid is None:
        return orders

    # calculate the mid_price and store in the historical data
    mid_price = (best_ask + best_bid) / 2
    historical_prices.append(mid_price)

    # Calculate Bollinger Bands
    upper, lower = compute_bollinger_bands(historical_prices)
    # Not enough data in the rolling window
    if upper is None or lower is None:
        return orders

    # if the mid_price is lower, buy and check the limit
    if mid_price < lower and cur_position < ink_position_limit:
        buy_price = best_ask  # try to buy at the current ask
        buy_volume = min(order_depth.sell_orders[buy_price], ink_position_limit - cur_position)
        orders.append(Order(product, buy_price, buy_volume))

    # if the mid_price is higher, sell and check the limit
    elif mid_price > upper and cur_position > -ink_position_limit:
        sell_price = best_bid  # try to sell at the current bid
        sell_volume = min(-order_depth.buy_orders[sell_price], cur_position + ink_position_limit)
        orders.append(Order(product, sell_price, sell_volume))

    return orders

def resin_algo(state: TradingState, product, resin_fair_price, resin_position_limit):

    # define new variables to record the possible order volume
    possible_buy_volume: int = 0
    possible_sell_volume: int = 0

    # find the corresponding market (resin) quote (depth)
    order_depth: OrderDepth = state.order_depths[product]

    # create the empty order book
    orders: list[Order] = []

    # retrieve the position information
    positions = state.position

    # check if the buy side market open
    if order_depth.buy_orders:

        # get the best bid price and the corresponding volume available
        best_bid_price = max(order_depth.buy_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid_price]

        # -----if possible to market make-----
        if (
                best_bid_price > resin_fair_price
                and -best_bid_volume + positions.get(product, 0) >= -resin_position_limit
        ):
            if best_bid_price > resin_fair_price + 2:
                orders.append(Order(product, best_bid_price, -best_bid_volume))
                possible_sell_volume = -best_bid_volume
            else:
                if -best_bid_volume + positions.get(product, 0) >= -resin_position_limit / 2:
                    orders.append(Order(product, best_bid_price, -best_bid_volume))
                    possible_sell_volume = -best_bid_volume

    # check if the sell side market open
    if order_depth.sell_orders:

        # get the best ask price and the corresponding volume available
        best_ask_price = min(order_depth.sell_orders.keys())
        best_ask_volume = order_depth.sell_orders[best_ask_price]

        # -----if possible to market make-----
        if (
                best_ask_price < resin_fair_price
                and -best_ask_volume + positions.get(product, 0) <= resin_position_limit
        ):
            if best_ask_price < resin_fair_price - 2:
                orders.append(Order(product, best_ask_price, -best_ask_volume))
                possible_buy_volume = -best_ask_volume
            else:
                if -best_ask_volume + positions.get(product, 0) <= resin_position_limit / 2:
                    orders.append(Order(product, best_ask_price, -best_ask_volume))
                    possible_buy_volume = -best_ask_volume

    # exploit the opportunity to do market making (be careful about the position limit)
    cur_position = positions.get(product, 0)
    later_position = max(abs(cur_position + possible_buy_volume), abs(cur_position + possible_sell_volume))
    if later_position < resin_position_limit:
        rest_volume = resin_position_limit - later_position
        orders.append(Order(product, resin_fair_price - 2, rest_volume))
        orders.append(Order(product, resin_fair_price + 2, -rest_volume))

    return orders

class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            # this is for RAINFOREST_RESIN
            if product == "RAINFOREST_RESIN":
                # run the resin algo
                resin_orders = resin_algo(state, product, resin_fair_price, resin_position_limit)
                result[product] = resin_orders

            # this is for KELP
            if product == "KELP":
                # get the historical data
                try:
                    data = jsonpickle.decode(state.traderData)
                    kelp_historical_prices = data.get("kelp_prices", [])
                except:
                    kelp_historical_prices = []

                # run the kelp algo
                kelp_orders = kelp_algo(state, product, kelp_position_limit, kelp_historical_prices)
                result[product] = kelp_orders

            # this is for ink
            if product == "SQUID_INK":
                try:
                    data = jsonpickle.decode(state.traderData)
                    ink_historical_prices = data.get("ink_prices", [])
                except:
                    ink_historical_prices = []

                # run the ink algo
                ink_orders = ink_algo(state, product, kelp_position_limit, ink_historical_prices)
                result[product] = ink_orders

        traderData = jsonpickle.encode(
            {
                "kelp_prices": kelp_historical_prices[-15:],
                "ink_prices": ink_historical_prices[-15:]
            }
        )

        conversions = 1

        # Return the dict of orders
        # These possibly contain buy or sell orders
        # Depending on the logic above

        return result, conversions, traderData

