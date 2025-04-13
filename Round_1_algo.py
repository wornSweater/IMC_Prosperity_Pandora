from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics
import jsonpickle

# KELP parameters (Mean-Reversion with Bollinger Bands)
KELP_WINDOW = 10
KELP_STD_DEV_MULTIPLIER = 1.0
KELP_POSITION_LIMIT = 50

# SQUID_INK parameters (Momentum with SMA Crossover)
INK_SMA_SHORT = 5
INK_SMA_LONG = 15
INK_POSITION_LIMIT = 50

# Resin parameters (unchanged)
RESIN_POSITION_LIMIT = 50
RESIN_FAIR_PRICE = 10000

# Pairs trading parameters
PAIRS_SPREAD_WINDOW = 15
PAIRS_THRESHOLD = 1.5  # 1.5 standard deviations


def picnic_basket_arbitrage(state: TradingState) -> Dict[str, List[Order]]:
    """
    Performs delta hedge arbitrage between picnic baskets and their components.
    """
    result = {
        "CROISSANTS": [],
        "JAMS": [],
        "DJEMBES": [],
        "PICNIC_BASKET1": [],
        "PICNIC_BASKET2": []
    }

    # Get current positions
    positions = {
        "CROISSANTS": state.position.get("CROISSANTS", 0),
        "JAMS": state.position.get("JAMS", 0),
        "DJEMBES": state.position.get("DJEMBES", 0),
        "PICNIC_BASKET1": state.position.get("PICNIC_BASKET1", 0),
        "PICNIC_BASKET2": state.position.get("PICNIC_BASKET2", 0)
    }

    # Position limits
    limits = {
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100
    }

    # Basket compositions
    basket1_composition = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
    basket2_composition = {"CROISSANTS": 4, "JAMS": 2, "DJEMBES": 0}

    # Get best prices for each product
    best_prices = {}
    for product in result.keys():
        if product not in state.order_depths:
            continue

        order_depth = state.order_depths[product]
        if order_depth.sell_orders:
            best_prices[f"{product}_ask"] = min(order_depth.sell_orders.keys())
            best_prices[f"{product}_ask_vol"] = order_depth.sell_orders[best_prices[f"{product}_ask"]]
        if order_depth.buy_orders:
            best_prices[f"{product}_bid"] = max(order_depth.buy_orders.keys())
            best_prices[f"{product}_bid_vol"] = order_depth.buy_orders[best_prices[f"{product}_bid"]]

    # Check if we have enough price information
    required_products = ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
    if not all(f"{product}_bid" in best_prices and f"{product}_ask" in best_prices
               for product in required_products if product in state.order_depths):
        return result

    # Calculate theoretical values for baskets
    if all(k in best_prices for k in ["CROISSANTS_mid", "JAMS_mid", "DJEMBES_mid"]):
        best_prices["CROISSANTS_mid"] = (best_prices.get("CROISSANTS_bid", 0) + best_prices.get("CROISSANTS_ask",
                                                                                                0)) / 2
        best_prices["JAMS_mid"] = (best_prices.get("JAMS_bid", 0) + best_prices.get("JAMS_ask", 0)) / 2
        best_prices["DJEMBES_mid"] = (best_prices.get("DJEMBES_bid", 0) + best_prices.get("DJEMBES_ask", 0)) / 2

        basket1_theoretical = (basket1_composition["CROISSANTS"] * best_prices["CROISSANTS_mid"] +
                               basket1_composition["JAMS"] * best_prices["JAMS_mid"] +
                               basket1_composition["DJEMBES"] * best_prices["DJEMBES_mid"])

        basket2_theoretical = (basket2_composition["CROISSANTS"] * best_prices["CROISSANTS_mid"] +
                               basket2_composition["JAMS"] * best_prices["JAMS_mid"])

    # Arbitrage 1: Buy PICNIC_BASKET1, sell components
    if "PICNIC_BASKET1_ask" in best_prices and "CROISSANTS_bid" in best_prices and "JAMS_bid" in best_prices and "DJEMBES_bid" in best_prices:
        basket1_buy_price = best_prices["PICNIC_BASKET1_ask"]
        components_sell_price = (basket1_composition["CROISSANTS"] * best_prices["CROISSANTS_bid"] +
                                 basket1_composition["JAMS"] * best_prices["JAMS_bid"] +
                                 basket1_composition["DJEMBES"] * best_prices["DJEMBES_bid"])

        if basket1_buy_price < components_sell_price:
            # Determine max volume based on position limits
            max_basket1 = limits["PICNIC_BASKET1"] - positions["PICNIC_BASKET1"]
            max_croissants = (limits["CROISSANTS"] + positions["CROISSANTS"]) // basket1_composition["CROISSANTS"]
            max_jams = (limits["JAMS"] + positions["JAMS"]) // basket1_composition["JAMS"]
            max_djembe = (limits["DJEMBES"] + positions["DJEMBES"]) // basket1_composition["DJEMBES"]

            # Also consider available volume in order book
            max_basket1_market = abs(best_prices["PICNIC_BASKET1_ask_vol"])
            max_croissants_market = abs(best_prices["CROISSANTS_bid_vol"]) // basket1_composition["CROISSANTS"]
            max_jams_market = abs(best_prices["JAMS_bid_vol"]) // basket1_composition["JAMS"]
            max_djembe_market = abs(best_prices["DJEMBES_bid_vol"]) // basket1_composition["DJEMBES"]

            arb_volume = min(max_basket1, max_croissants, max_jams, max_djembe,
                             max_basket1_market, max_croissants_market, max_jams_market, max_djembe_market)

            if arb_volume > 0:
                result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", best_prices["PICNIC_BASKET1_ask"], arb_volume))
                result["CROISSANTS"].append(Order("CROISSANTS", best_prices["CROISSANTS_bid"],
                                                  -arb_volume * basket1_composition["CROISSANTS"]))
                result["JAMS"].append(Order("JAMS", best_prices["JAMS_bid"],
                                            -arb_volume * basket1_composition["JAMS"]))
                result["DJEMBES"].append(Order("DJEMBES", best_prices["DJEMBES_bid"],
                                              -arb_volume * basket1_composition["DJEMBES"]))

    # Arbitrage 2: Sell PICNIC_BASKET1, buy components
    if "PICNIC_BASKET1_bid" in best_prices and "CROISSANTS_ask" in best_prices and "JAMS_ask" in best_prices and "DJEMBES_ask" in best_prices:
        basket1_sell_price = best_prices["PICNIC_BASKET1_bid"]
        components_buy_price = (basket1_composition["CROISSANTS"] * best_prices["CROISSANTS_ask"] +
                                basket1_composition["JAMS"] * best_prices["JAMS_ask"] +
                                basket1_composition["DJEMBES"] * best_prices["DJEMBES_ask"])

        if basket1_sell_price > components_buy_price:
            # Determine max volume based on position limits
            max_basket1 = positions["PICNIC_BASKET1"] + limits["PICNIC_BASKET1"]
            max_croissants = (limits["CROISSANTS"] - positions["CROISSANTS"]) // basket1_composition["CROISSANTS"]
            max_jams = (limits["JAMS"] - positions["JAMS"]) // basket1_composition["JAMS"]
            max_djembe = (limits["DJEMBES"] - positions["DJEMBES"]) // basket1_composition["DJEMBES"]

            # Also consider available volume in order book
            max_basket1_market = abs(best_prices["PICNIC_BASKET1_bid_vol"])
            max_croissants_market = abs(best_prices["CROISSANTS_ask_vol"]) // basket1_composition["CROISSANTS"]
            max_jams_market = abs(best_prices["JAMS_ask_vol"]) // basket1_composition["JAMS"]
            max_djembe_market = abs(best_prices["DJEMBES_ask_vol"]) // basket1_composition["DJEMBES"]

            arb_volume = min(max_basket1, max_croissants, max_jams, max_djembe,
                             max_basket1_market, max_croissants_market, max_jams_market, max_djembe_market)

            if arb_volume > 0:
                result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", best_prices["PICNIC_BASKET1_bid"], -arb_volume))
                result["CROISSANTS"].append(Order("CROISSANTS", best_prices["CROISSANTS_ask"],
                                                  arb_volume * basket1_composition["CROISSANTS"]))
                result["JAMS"].append(Order("JAMS", best_prices["JAMS_ask"],
                                            arb_volume * basket1_composition["JAMS"]))
                result["DJEMBES"].append(Order("DJEMBES", best_prices["DJEMBES_ask"],
                                              arb_volume * basket1_composition["DJEMBES"]))

    # Arbitrage 3: Buy PICNIC_BASKET2, sell components
    if "PICNIC_BASKET2_ask" in best_prices and "CROISSANTS_bid" in best_prices and "JAMS_bid" in best_prices:
        basket2_buy_price = best_prices["PICNIC_BASKET2_ask"]
        components_sell_price = (basket2_composition["CROISSANTS"] * best_prices["CROISSANTS_bid"] +
                                 basket2_composition["JAMS"] * best_prices["JAMS_bid"])

        if basket2_buy_price < components_sell_price:
            # Determine max volume based on position limits
            max_basket2 = limits["PICNIC_BASKET2"] - positions["PICNIC_BASKET2"]
            max_croissants = (limits["CROISSANTS"] + positions["CROISSANTS"]) // basket2_composition["CROISSANTS"]
            max_jams = (limits["JAMS"] + positions["JAMS"]) // basket2_composition["JAMS"]

            # Also consider available volume in order book
            max_basket2_market = abs(best_prices["PICNIC_BASKET2_ask_vol"])
            max_croissants_market = abs(best_prices["CROISSANTS_bid_vol"]) // basket2_composition["CROISSANTS"]
            max_jams_market = abs(best_prices["JAMS_bid_vol"]) // basket2_composition["JAMS"]

            arb_volume = min(max_basket2, max_croissants, max_jams,
                             max_basket2_market, max_croissants_market, max_jams_market)

            if arb_volume > 0:
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_prices["PICNIC_BASKET2_ask"], arb_volume))
                result["CROISSANTS"].append(Order("CROISSANTS", best_prices["CROISSANTS_bid"],
                                                  -arb_volume * basket2_composition["CROISSANTS"]))
                result["JAMS"].append(Order("JAMS", best_prices["JAMS_bid"],
                                            -arb_volume * basket2_composition["JAMS"]))

    # Arbitrage 4: Sell PICNIC_BASKET2, buy components
    if "PICNIC_BASKET2_bid" in best_prices and "CROISSANTS_ask" in best_prices and "JAMS_ask" in best_prices:
        basket2_sell_price = best_prices["PICNIC_BASKET2_bid"]
        components_buy_price = (basket2_composition["CROISSANTS"] * best_prices["CROISSANTS_ask"] +
                                basket2_composition["JAMS"] * best_prices["JAMS_ask"])

        if basket2_sell_price > components_buy_price:
            # Determine max volume based on position limits
            max_basket2 = positions["PICNIC_BASKET2"] + limits["PICNIC_BASKET2"]
            max_croissants = (limits["CROISSANTS"] - positions["CROISSANTS"]) // basket2_composition["CROISSANTS"]
            max_jams = (limits["JAMS"] - positions["JAMS"]) // basket2_composition["JAMS"]

            # Also consider available volume in order book
            max_basket2_market = abs(best_prices["PICNIC_BASKET2_bid_vol"])
            max_croissants_market = abs(best_prices["CROISSANTS_ask_vol"]) // basket2_composition["CROISSANTS"]
            max_jams_market = abs(best_prices["JAMS_ask_vol"]) // basket2_composition["JAMS"]

            arb_volume = min(max_basket2, max_croissants, max_jams,
                             max_basket2_market, max_croissants_market, max_jams_market)

            if arb_volume > 0:
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_prices["PICNIC_BASKET2_bid"], -arb_volume))
                result["CROISSANTS"].append(Order("CROISSANTS", best_prices["CROISSANTS_ask"],
                                                  arb_volume * basket2_composition["CROISSANTS"]))
                result["JAMS"].append(Order("JAMS", best_prices["JAMS_ask"],
                                            arb_volume * basket2_composition["JAMS"]))

    # Arbitrage 5: Buy PICNIC_BASKET1, sell PICNIC_BASKET2 + DJEMBES
    if "PICNIC_BASKET1_ask" in best_prices and "PICNIC_BASKET2_bid" in best_prices and "DJEMBES_bid" in best_prices:
        basket1_buy_price = best_prices["PICNIC_BASKET1_ask"]
        basket2_plus_djembe_sell_price = best_prices["PICNIC_BASKET2_bid"] + best_prices["DJEMBES_bid"]

        # Check if there's a profitable spread between the two baskets (accounting for the extra DJEMBES and 2 CROISSANTS + 1 JAM)
        croissants_diff = basket1_composition["CROISSANTS"] - basket2_composition["CROISSANTS"]
        jams_diff = basket1_composition["JAMS"] - basket2_composition["JAMS"]

        if "CROISSANTS_bid" in best_prices and "JAMS_bid" in best_prices:
            extra_components_sell_price = (croissants_diff * best_prices["CROISSANTS_bid"] +
                                           jams_diff * best_prices["JAMS_bid"])

            if basket1_buy_price < basket2_plus_djembe_sell_price + extra_components_sell_price:
                # Determine max volume based on position limits
                max_basket1 = limits["PICNIC_BASKET1"] - positions["PICNIC_BASKET1"]
                max_basket2 = positions["PICNIC_BASKET2"] + limits["PICNIC_BASKET2"]
                max_djembe = positions["DJEMBES"] + limits["DJEMBES"]
                max_croissants = (limits["CROISSANTS"] + positions["CROISSANTS"]) // croissants_diff
                max_jams = (limits["JAMS"] + positions["JAMS"]) // jams_diff

                # Also consider available volume in order book
                max_basket1_market = abs(best_prices["PICNIC_BASKET1_ask_vol"])
                max_basket2_market = abs(best_prices["PICNIC_BASKET2_bid_vol"])
                max_djembe_market = abs(best_prices["DJEMBES_bid_vol"])
                max_croissants_market = abs(best_prices["CROISSANTS_bid_vol"]) // croissants_diff
                max_jams_market = abs(best_prices["JAMS_bid_vol"]) // jams_diff

                arb_volume = min(max_basket1, max_basket2, max_djembe, max_croissants, max_jams,
                                 max_basket1_market, max_basket2_market, max_djembe_market,
                                 max_croissants_market, max_jams_market)

                if arb_volume > 0:
                    result["PICNIC_BASKET1"].append(
                        Order("PICNIC_BASKET1", best_prices["PICNIC_BASKET1_ask"], arb_volume))
                    result["PICNIC_BASKET2"].append(
                        Order("PICNIC_BASKET2", best_prices["PICNIC_BASKET2_bid"], -arb_volume))
                    result["DJEMBES"].append(Order("DJEMBES", best_prices["DJEMBES_bid"], -arb_volume))
                    result["CROISSANTS"].append(Order("CROISSANTS", best_prices["CROISSANTS_bid"],
                                                      -arb_volume * croissants_diff))
                    result["JAMS"].append(Order("JAMS", best_prices["JAMS_bid"],
                                                -arb_volume * jams_diff))

    # Arbitrage 6: Sell PICNIC_BASKET1, buy PICNIC_BASKET2 + DJEMBES
    if "PICNIC_BASKET1_bid" in best_prices and "PICNIC_BASKET2_ask" in best_prices and "DJEMBES_ask" in best_prices:
        basket1_sell_price = best_prices["PICNIC_BASKET1_bid"]
        basket2_plus_djembe_buy_price = best_prices["PICNIC_BASKET2_ask"] + best_prices["DJEMBES_ask"]

        # Check if there's a profitable spread (accounting for the extra 2 CROISSANTS + 1 JAM)
        croissants_diff = basket1_composition["CROISSANTS"] - basket2_composition["CROISSANTS"]
        jams_diff = basket1_composition["JAMS"] - basket2_composition["JAMS"]

        if "CROISSANTS_ask" in best_prices and "JAMS_ask" in best_prices:
            extra_components_buy_price = (croissants_diff * best_prices["CROISSANTS_ask"] +
                                          jams_diff * best_prices["JAMS_ask"])

            if basket1_sell_price > basket2_plus_djembe_buy_price + extra_components_buy_price:
                # Determine max volume based on position limits
                max_basket1 = positions["PICNIC_BASKET1"] + limits["PICNIC_BASKET1"]
                max_basket2 = limits["PICNIC_BASKET2"] - positions["PICNIC_BASKET2"]
                max_djembe = limits["DJEMBES"] - positions["DJEMBES"]
                max_croissants = (limits["CROISSANTS"] - positions["CROISSANTS"]) // croissants_diff
                max_jams = (limits["JAMS"] - positions["JAMS"]) // jams_diff

                # Also consider available volume in order book
                max_basket1_market = abs(best_prices["PICNIC_BASKET1_bid_vol"])
                max_basket2_market = abs(best_prices["PICNIC_BASKET2_ask_vol"])
                max_djembe_market = abs(best_prices["DJEMBES_ask_vol"])
                max_croissants_market = abs(best_prices["CROISSANTS_ask_vol"]) // croissants_diff
                max_jams_market = abs(best_prices["JAMS_ask_vol"]) // jams_diff

                arb_volume = min(max_basket1, max_basket2, max_djembe, max_croissants, max_jams,
                                 max_basket1_market, max_basket2_market, max_djembe_market,
                                 max_croissants_market, max_jams_market)

                if arb_volume > 0:
                    result["PICNIC_BASKET1"].append(
                        Order("PICNIC_BASKET1", best_prices["PICNIC_BASKET1_bid"], -arb_volume))
                    result["PICNIC_BASKET2"].append(
                        Order("PICNIC_BASKET2", best_prices["PICNIC_BASKET2_ask"], arb_volume))
                    result["DJEMBES"].append(Order("DJEMBES", best_prices["DJEMBES_ask"], arb_volume))
                    result["CROISSANTS"].append(Order("CROISSANTS", best_prices["CROISSANTS_ask"],
                                                      arb_volume * croissants_diff))
                    result["JAMS"].append(Order("JAMS", best_prices["JAMS_ask"],
                                                arb_volume * jams_diff))

    # Clean up empty product order lists
    for product in list(result.keys()):
        if not result[product]:
            del result[product]

    return result

def compute_bollinger_bands(prices: List[float], window: int = KELP_WINDOW,
                            multiplier: float = KELP_STD_DEV_MULTIPLIER):
    if len(prices) < window:
        return None, None
    window_prices = prices[-window:]
    mean_price = statistics.mean(window_prices)
    std_dev = statistics.stdev(window_prices)
    upper_band = mean_price + multiplier * std_dev
    lower_band = mean_price - multiplier * std_dev
    return upper_band, lower_band


def compute_sma(prices: List[float], period: int):
    if len(prices) < period:
        return None
    return statistics.mean(prices[-period:])


def kelp_algo(state: TradingState, product: str, historical_prices: List[float]) -> List[Order]:
    orders: List[Order] = []
    order_depth: OrderDepth = state.order_depths[product]
    cur_position = state.position.get(product, 0)

    # Mid-price calculation
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    if not best_ask or not best_bid:
        return orders

    mid_price = (best_ask + best_bid) / 2
    historical_prices.append(mid_price)

    # Bollinger Bands for mean-reversion
    upper, lower = compute_bollinger_bands(historical_prices)
    if upper is None or lower is None:
        return orders

    # Trading logic
    if mid_price < lower and cur_position < KELP_POSITION_LIMIT:
        buy_price = best_ask
        buy_volume = min(order_depth.sell_orders[buy_price], KELP_POSITION_LIMIT - cur_position)
        orders.append(Order(product, buy_price, buy_volume))
    elif mid_price > upper and cur_position > -KELP_POSITION_LIMIT:
        sell_price = best_bid
        sell_volume = min(-order_depth.buy_orders[sell_price], cur_position + KELP_POSITION_LIMIT)
        orders.append(Order(product, sell_price, -sell_volume))

    return orders


def ink_algo(state: TradingState, product: str, historical_prices: List[float]) -> List[Order]:
    orders: List[Order] = []
    order_depth: OrderDepth = state.order_depths[product]
    cur_position = state.position.get(product, 0)

    # Mid-price calculation
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    if not best_ask or not best_bid:
        return orders

    mid_price = (best_ask + best_bid) / 2
    historical_prices.append(mid_price)

    # SMA Crossover for momentum
    sma_short = compute_sma(historical_prices, INK_SMA_SHORT)
    sma_long = compute_sma(historical_prices, INK_SMA_LONG)
    if sma_short is None or sma_long is None:
        return orders

    prev_sma_short = compute_sma(historical_prices[:-1], INK_SMA_SHORT) if len(
        historical_prices) > INK_SMA_SHORT else None
    prev_sma_long = compute_sma(historical_prices[:-1], INK_SMA_LONG) if len(historical_prices) > INK_SMA_LONG else None
    if prev_sma_short is None or prev_sma_long is None:
        return orders

    # Trading logic
    if sma_short > sma_long and prev_sma_short <= prev_sma_long and cur_position < INK_POSITION_LIMIT:
        buy_price = best_ask
        buy_volume = min(order_depth.sell_orders[buy_price], INK_POSITION_LIMIT - cur_position)
        orders.append(Order(product, buy_price, buy_volume))
    elif sma_short < sma_long and prev_sma_short >= prev_sma_long and cur_position > -INK_POSITION_LIMIT:
        sell_price = best_bid
        sell_volume = min(-order_depth.buy_orders[sell_price], cur_position + INK_POSITION_LIMIT)
        orders.append(Order(product, sell_price, -sell_volume))

    return orders


def pairs_trading(state: TradingState, kelp_prices: List[float], ink_prices: List[float]) -> Dict[str, List[Order]]:
    result = {"KELP": [], "SQUID_INK": []}
    if len(kelp_prices) < PAIRS_SPREAD_WINDOW or len(ink_prices) < PAIRS_SPREAD_WINDOW:
        return result

    # Calculate spread
    spreads = [k - i for k, i in zip(kelp_prices[-PAIRS_SPREAD_WINDOW:], ink_prices[-PAIRS_SPREAD_WINDOW:])]
    spread_mean = statistics.mean(spreads)
    spread_std = statistics.stdev(spreads)
    current_spread = kelp_prices[-1] - ink_prices[-1]

    kelp_position = state.position.get("KELP", 0)
    ink_position = state.position.get("SQUID_INK", 0)

    # Spread widens (KELP outperforms)
    if current_spread > spread_mean + PAIRS_THRESHOLD * spread_std:
        if kelp_position > -KELP_POSITION_LIMIT and ink_position < INK_POSITION_LIMIT:
            best_bid_kelp = max(state.order_depths["KELP"].buy_orders.keys())
            best_ask_ink = min(state.order_depths["SQUID_INK"].sell_orders.keys())
            sell_vol_kelp = min(-state.order_depths["KELP"].buy_orders[best_bid_kelp],
                                kelp_position + KELP_POSITION_LIMIT)
            buy_vol_ink = min(state.order_depths["SQUID_INK"].sell_orders[best_ask_ink],
                              INK_POSITION_LIMIT - ink_position)
            result["KELP"].append(Order("KELP", best_bid_kelp, -sell_vol_kelp))
            result["SQUID_INK"].append(Order("SQUID_INK", best_ask_ink, buy_vol_ink))
    # Spread narrows (SQUID_INK outperforms)
    elif current_spread < spread_mean - PAIRS_THRESHOLD * spread_std:
        if kelp_position < KELP_POSITION_LIMIT and ink_position > -INK_POSITION_LIMIT:
            best_ask_kelp = min(state.order_depths["KELP"].sell_orders.keys())
            best_bid_ink = max(state.order_depths["SQUID_INK"].buy_orders.keys())
            buy_vol_kelp = min(state.order_depths["KELP"].sell_orders[best_ask_kelp],
                               KELP_POSITION_LIMIT - kelp_position)
            sell_vol_ink = min(-state.order_depths["SQUID_INK"].buy_orders[best_bid_ink],
                               ink_position + INK_POSITION_LIMIT)
            result["KELP"].append(Order("KELP", best_ask_kelp, buy_vol_kelp))
            result["SQUID_INK"].append(Order("SQUID_INK", best_bid_ink, -sell_vol_ink))

    return result


def resin_algo(state: TradingState, product: str, resin_fair_price: int, resin_position_limit: int) -> List[Order]:
    # Unchanged from original code
    possible_buy_volume = 0
    possible_sell_volume = 0
    order_depth: OrderDepth = state.order_depths[product]
    orders: List[Order] = []
    positions = state.position
    cur_position = positions.get(product, 0)

    if order_depth.buy_orders:
        best_bid_price = max(order_depth.buy_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid_price]
        if (best_bid_price > resin_fair_price and
                -best_bid_volume + cur_position >= -resin_position_limit):
            if best_bid_price > resin_fair_price + 2:
                orders.append(Order(product, best_bid_price, -best_bid_volume))
                possible_sell_volume = -best_bid_volume
            else:
                if -best_bid_volume + cur_position >= -resin_position_limit / 2:
                    orders.append(Order(product, best_bid_price, -best_bid_volume))
                    possible_sell_volume = -best_bid_volume

    if order_depth.sell_orders:
        best_ask_price = min(order_depth.sell_orders.keys())
        best_ask_volume = order_depth.sell_orders[best_ask_price]
        if (best_ask_price < resin_fair_price and
                -best_ask_volume + cur_position <= resin_position_limit):
            if best_ask_price < resin_fair_price - 2:
                orders.append(Order(product, best_ask_price, -best_ask_volume))
                possible_buy_volume = -best_ask_volume
            else:
                if -best_ask_volume + cur_position <= resin_position_limit / 2:
                    orders.append(Order(product, best_ask_price, -best_ask_volume))
                    possible_buy_volume = -best_ask_volume

    later_position = max(abs(cur_position + possible_buy_volume), abs(cur_position + possible_sell_volume))
    if later_position < resin_position_limit:
        rest_volume = resin_position_limit - later_position
        orders.append(Order(product, resin_fair_price - 2, rest_volume))
        orders.append(Order(product, resin_fair_price + 2, -rest_volume))

    return orders


class Trader:
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}

        # Load historical data
        try:
            data = jsonpickle.decode(state.traderData)
            kelp_historical_prices = data.get("kelp_prices", [])
            ink_historical_prices = data.get("ink_prices", [])
        except:
            kelp_historical_prices = []
            ink_historical_prices = []

        # Process each product
        for product in state.order_depths.keys():
            if product == "RAINFOREST_RESIN":
                result[product] = resin_algo(state, product, RESIN_FAIR_PRICE, RESIN_POSITION_LIMIT)
            elif product == "KELP":
                result[product] = kelp_algo(state, product, kelp_historical_prices)
            elif product == "SQUID_INK":
                result[product] = ink_algo(state, product, ink_historical_prices)

        # Pairs trading for KELP and SQUID_INK
        pairs_orders = pairs_trading(state, kelp_historical_prices, ink_historical_prices)
        for product, orders in pairs_orders.items():
            if orders:
                result[product] = result.get(product, []) + orders

        # Check if picnic basket products are in the market
        picnic_products = ["CROISSANTS", "JAMS", "DJEMBE", "PICNIC_BASKET1", "PICNIC_BASKET2"]
        picnic_in_market = any(product in state.order_depths for product in picnic_products)

        # If picnic products are available, run the delta hedge arbitrage
        if picnic_in_market:
            picnic_orders = picnic_basket_arbitrage(state)
            # Merge picnic orders with existing orders
            for product, orders in picnic_orders.items():
                if orders:
                    result[product] = result.get(product, []) + orders

        # Update trader data
        traderData = jsonpickle.encode({
            "kelp_prices": kelp_historical_prices[-max(KELP_WINDOW, PAIRS_SPREAD_WINDOW):],
            "ink_prices": ink_historical_prices[-max(INK_SMA_LONG, PAIRS_SPREAD_WINDOW):]
        })

        conversions = 1
        return result, conversions, traderData