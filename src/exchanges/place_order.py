#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Limit Order Placement Utility

This script provides a command-line interface for placing limit orders on Bitget,
optimized for maker fees. It implements intelligent order placement strategies and
offers both single command execution and interactive mode.

Usage:
    ./place_order.py --symbol BTCUSDT --side buy --amount 0.001 --strategy optimal
    ./place_order.py --symbol BTCUSDT --interactive
"""

import argparse
import logging
import sys
import time
import os
from datetime import datetime
import json
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Union, Any
from bitget_exchange import BitgetClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"limit_orders_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("limit_order_placer")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Limit Order Placement Utility")
    
    # Common arguments
    parser.add_argument("--symbol", type=str, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("--side", type=str, choices=["buy", "sell"], help="Order side")
    parser.add_argument("--amount", type=float, help="Order amount in base currency")
    parser.add_argument("--price", type=float, help="Limit price (optional, will calculate if not provided)")
    parser.add_argument("--strategy", type=str, choices=["optimal", "smart", "passive"], 
                        default="optimal", help="Order placement strategy")
    
    # Strategy-specific arguments
    parser.add_argument("--offset", type=float, default=0.0012, 
                        help="Price offset percentage for optimal strategy (default 0.12%)")
    parser.add_argument("--order-count", type=int, default=3, 
                        help="Number of orders for smart strategy (default 3)")
    parser.add_argument("--spread", type=float, default=0.03, 
                        help="Price spread percentage for smart strategy (default 3%)")
    
    # Mode arguments
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--monitor", action="store_true", help="Monitor orders after placement")
    parser.add_argument("--testnet", action="store_true", help="Use testnet API")
    
    return parser.parse_args()

def format_currency(value, decimal_places=8):
    """Format currency value with appropriate precision."""
    return str(Decimal(str(value)).quantize(Decimal('0.' + '0' * decimal_places), rounding=ROUND_DOWN))

def get_user_confirmation(message="Proceed with this order?"):
    """Get user confirmation for an action."""
    response = input(f"{message} (y/n): ").lower().strip()
    return response in ['y', 'yes']

def display_order_preview(client, symbol, side, amount, price=None, strategy="optimal", **kwargs):
    """
    Display a preview of the order to be placed.
    
    Args:
        client: BitgetClient instance
        symbol: Trading symbol
        side: Order side (buy/sell)
        amount: Order amount
        price: Limit price (optional)
        strategy: Order placement strategy
        **kwargs: Additional strategy-specific parameters
    
    Returns:
        dict: Order preview information
    """
    # Get current market data
    ticker = client.get_ticker(symbol)
    if 'data' not in ticker:
        logger.error(f"Failed to get ticker data for {symbol}")
        return None
    
    ticker_data = ticker['data']
    last_price = float(ticker_data.get('last', 0))
    
    # Get instrument details
    try:
        instrument = client.get_instrument(symbol)
        min_size = float(instrument.get('minTradeAmount', 0))
        price_precision = int(instrument.get('pricePrecision', 8))
        size_precision = int(instrument.get('quantityPrecision', 8))
    except Exception as e:
        logger.error(f"Error getting instrument details: {e}")
        min_size = 0
        price_precision = 8
        size_precision = 8
    
    # Check minimum order size
    if amount < min_size:
        logger.warning(f"Amount {amount} is below minimum order size {min_size} for {symbol}")
    
    # Calculate or validate price
    calculated_price = None
    if price is None:
        if strategy == "optimal":
            offset = kwargs.get('offset', 0.0012)
            calculated_price = client.calculate_maker_price(symbol, side, offset)
        elif strategy == "passive":
            # More passive pricing (further from current price)
            offset = kwargs.get('offset', 0.0025)  # 0.25% by default
            calculated_price = client.calculate_maker_price(symbol, side, offset)
        else:
            # Default for "smart" strategy - just get the reference price
            orderbook = client.get_order_book(symbol, limit=5)
            if 'data' in orderbook:
                if side.lower() == 'buy':
                    calculated_price = float(orderbook['data']['bids'][0][0])
                else:
                    calculated_price = float(orderbook['data']['asks'][0][0])
            else:
                calculated_price = last_price
    else:
        # User provided a price, format it correctly
        calculated_price = client.format_price(symbol, price)
    
    # Format amount
    formatted_amount = client.format_size(symbol, amount)
    
    # Calculate total value
    price_float = float(calculated_price)
    total_value = price_float * float(formatted_amount)
    
    # Prepare preview
    preview = {
        "symbol": symbol,
        "side": side,
        "strategy": strategy,
        "amount": formatted_amount,
        "price": calculated_price,
        "total_value": format_currency(total_value, price_precision),
        "current_price": ticker_data.get('last', 'N/A'),
        "price_precision": price_precision,
        "size_precision": size_precision,
        "min_size": min_size
    }
    
    # Add strategy-specific info
    if strategy == "smart":
        order_count = kwargs.get('order_count', 3)
        spread = kwargs.get('spread', 0.03)
        
        # Calculate per-order size
        size_per_order = float(formatted_amount) / order_count
        
        # Add to preview
        preview["order_count"] = order_count
        preview["spread_percentage"] = f"{spread * 100:.2f}%"
        preview["size_per_order"] = format_currency(size_per_order, size_precision)
    
    return preview

def display_orderbook_analysis(client, symbol, side):
    """
    Display an analysis of the current orderbook to help with order placement.
    
    Args:
        client: BitgetClient instance
        symbol: Trading symbol
        side: Order side (buy/sell)
        
    Returns:
        dict: Orderbook analysis
    """
    # Get order book
    orderbook = client.get_order_book(symbol, limit=20)
    if 'data' not in orderbook:
        logger.error(f"Failed to get orderbook for {symbol}")
        return None
    
    # Extract data
    book_data = orderbook['data']
    bids = book_data.get('bids', [])
    asks = book_data.get('asks', [])
    
    if not bids or not asks:
        logger.error(f"Empty orderbook for {symbol}")
        return None
    
    # Get top levels
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid
    spread_pct = (spread / best_bid) * 100
    
    # Analyze liquidity
    bid_liquidity = sum(float(bid[1]) for bid in bids[:5])
    ask_liquidity = sum(float(ask[1]) for ask in asks[:5])
    
    # Calculate potential maker prices at different offsets
    maker_prices = {}
    for offset in [0.0005, 0.0010, 0.0020, 0.0050]:
        if side.lower() == 'buy':
            maker_prices[f"{offset*100:.2f}%"] = format_currency(best_bid * (1 - offset), 8)
        else:
            maker_prices[f"{offset*100:.2f}%"] = format_currency(best_ask * (1 + offset), 8)
    
    # Prepare analysis
    analysis = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_pct": f"{spread_pct:.4f}%",
        "bid_liquidity": bid_liquidity,
        "ask_liquidity": ask_liquidity,
        "maker_prices": maker_prices,
        "imbalance": f"{(bid_liquidity/ask_liquidity if ask_liquidity else float('inf')):.2f}"
    }
    
    return analysis

def place_order(client, symbol, side, amount, price=None, strategy="optimal", monitor=False, **kwargs):
    """
    Place a limit order using the specified strategy.
    
    Args:
        client: BitgetClient instance
        symbol: Trading symbol
        side: Order side (buy/sell)
        amount: Order amount
        price: Limit price (optional)
        strategy: Order placement strategy
        monitor: Whether to monitor the order after placement
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        dict: Order response
    """
    order_result = None
    
    # Place order based on strategy
    try:
        if strategy == "optimal":
            offset = kwargs.get('offset', 0.0012)
            logger.info(f"Placing optimal limit order with {offset*100:.2f}% offset")
            
            order_result = client.place_optimal_maker_order(
                symbol=symbol,
                side=side,
                size=amount,
                offset_percentage=offset
            )
        elif strategy == "smart":
            order_count = kwargs.get('order_count', 3)
            spread = kwargs.get('spread', 0.03)
            
            logger.info(f"Placing smart limit orders: {order_count} orders with {spread*100:.2f}% spread")
            
            order_result = client.place_smart_limit_order(
                symbol=symbol,
                side=side,
                total_amount=amount,
                min_order_count=1,
                max_order_count=order_count,
                spread_range=spread,
                client_order_id_prefix=f"SMART_{symbol}_{int(time.time())}"
            )
        elif strategy == "passive":
            # Similar to optimal but with larger offset
            offset = kwargs.get('offset', 0.0025)
            logger.info(f"Placing passive limit order with {offset*100:.2f}% offset")
            
            order_result = client.place_optimal_maker_order(
                symbol=symbol,
                side=side,
                size=amount,
                offset_percentage=offset
            )
        else:
            # Standard limit order with specified price
            if price is None:
                price = client.calculate_maker_price(symbol, side)
                
            logger.info(f"Placing standard limit order at price {price}")
            
            formatted_size = client.format_size(symbol, amount)
            order_result = client.place_limit_order(
                symbol=symbol,
                side=side,
                price=price,
                size=formatted_size,
                time_in_force="post_only"
            )
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None
    
    # Handle order result
    if order_result:
        if isinstance(order_result, list):
            # Multiple orders for smart strategy
            logger.info(f"Placed {len(order_result)} orders")
            for i, order in enumerate(order_result):
                if 'data' in order:
                    order_id = order['data'].get('orderId')
                    price = order['data'].get('price')
                    size = order['data'].get('size')
                    logger.info(f"Order {i+1}: ID {order_id}, Price {price}, Size {size}")
        else:
            # Single order
            if 'data' in order_result:
                order_id = order_result['data'].get('orderId')
                price = order_result['data'].get('price')
                size = order_result['data'].get('size')
                logger.info(f"Order placed: ID {order_id}, Price {price}, Size {size}")
    
    # Monitor order if requested
    if monitor and order_result:
        if isinstance(order_result, list):
            # For multiple orders, we'll monitor them individually
            orders_to_monitor = []
            for order in order_result:
                if 'data' in order:
                    order_id = order['data'].get('orderId')
                    if order_id:
                        orders_to_monitor.append((symbol, order_id))
            
            monitor_multiple_orders(client, orders_to_monitor)
        else:
            # For single order
            if 'data' in order_result:
                order_id = order_result['data'].get('orderId')
                if order_id:
                    monitor_order(client, symbol, order_id)
    
    return order_result

def monitor_order(client, symbol, order_id, poll_interval=2.0, max_duration=3600):
    """
    Monitor an order until it's filled, canceled, or the duration expires.
    
    Args:
        client: BitgetClient instance
        symbol: Trading symbol
        order_id: Order ID to monitor
        poll_interval: Time between order status checks in seconds
        max_duration: Maximum monitoring time in seconds
        
    Returns:
        dict: Final order information
    """
    logger.info(f"Monitoring order {order_id} for {symbol}...")
    
    start_time = time.time()
    last_status = None
    
    try:
        while time.time() - start_time < max_duration:
            # Get order status
            order_info = client.get_order(symbol, order_id)
            
            if 'data' not in order_info:
                logger.error(f"Failed to get order info for {order_id}")
                time.sleep(poll_interval)
                continue
            
            order_data = order_info['data']
            status = order_data.get('status')
            
            # Log status changes
            if status != last_status:
                filled_size = order_data.get('filledSize', '0')
                size = order_data.get('size', '0')
                fill_pct = float(filled_size) / float(size) * 100 if float(size) > 0 else 0
                
                logger.info(f"Order {order_id} status: {status}, filled: {filled_size}/{size} ({fill_pct:.1f}%)")
                last_status = status
            
            # Check if order is done
            if status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                # If filled, show execution details
                if status == 'FILLED':
                    avg_price = order_data.get('avgPrice', 'N/A')
                    fee = order_data.get('fee', 'N/A')
                    logger.info(f"Order filled at avg price {avg_price}, fee: {fee}")
                
                return order_info
            
            time.sleep(poll_interval)
        
        # Timed out
        logger.warning(f"Monitoring timed out for order {order_id}")
        return None
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Error monitoring order: {e}")
        return None

def monitor_multiple_orders(client, order_list, poll_interval=3.0, max_duration=3600):
    """
    Monitor multiple orders simultaneously.
    
    Args:
        client: BitgetClient instance
        order_list: List of (symbol, order_id) tuples
        poll_interval: Time between status checks in seconds
        max_duration: Maximum monitoring time in seconds
        
    Returns:
        dict: Map of order IDs to final order information
    """
    logger.info(f"Monitoring {len(order_list)} orders...")
    
    start_time = time.time()
    order_statuses = {order_id: None for _, order_id in order_list}
    results = {}
    
    try:
        while order_list and time.time() - start_time < max_duration:
            # Make a copy so we can remove from the original
            current_orders = order_list.copy()
            
            for symbol, order_id in current_orders:
                # Get order status
                order_info = client.get_order(symbol, order_id)
                
                if 'data' not in order_info:
                    continue
                
                order_data = order_info['data']
                status = order_data.get('status')
                
                # Log status changes
                if status != order_statuses.get(order_id):
                    filled_size = order_data.get('filledSize', '0')
                    size = order_data.get('size', '0')
                    fill_pct = float(filled_size) / float(size) * 100 if float(size) > 0 else 0
                    
                    logger.info(f"Order {order_id} status: {status}, filled: {filled_size}/{size} ({fill_pct:.1f}%)")
                    order_statuses[order_id] = status
                
                # Check if order is done
                if status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                    # If filled, show execution details
                    if status == 'FILLED':
                        avg_price = order_data.get('avgPrice', 'N/A')
                        fee = order_data.get('fee', 'N/A')
                        logger.info(f"Order {order_id} filled at avg price {avg_price}, fee: {fee}")
                    
                    # Store result and remove from monitoring list
                    results[order_id] = order_info
                    order_list.remove((symbol, order_id))
            
            # If all orders are done, stop monitoring
            if not order_list:
                break
                
            time.sleep(poll_interval)
        
        # Check if any orders are still being monitored
        if order_list:
            logger.warning(f"Monitoring timed out for {len(order_list)} orders")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        return results
    except Exception as e:
        logger.error(f"Error monitoring orders: {e}")
        return results

def show_account_balance(client, quote_currency='USDT'):
    """
    Show account balance information.
    
    Args:
        client: BitgetClient instance
        quote_currency: Currency to use for valuation
        
    Returns:
        dict: Balance information
    """
    try:
        assets_info = client.get_account_assets()
        
        if 'data' not in assets_info:
            logger.error("Failed to get account assets")
            return None
        
        assets = assets_info['data']
        
        # Extract relevant balances
        balances = {}
        total_value = 0.0
        
        for asset in assets:
            coin = asset.get('coin', '')
            available = float(asset.get('available', '0'))
            frozen = float(asset.get('frozen', '0'))
            total = available + frozen
            
            if total > 0:
                # Get asset value in quote currency if possible
                try:
                    value = total
                    if coin != quote_currency:
                        symbol = f"{coin}{quote_currency}"
                        ticker = client.get_ticker(symbol)
                        if 'data' in ticker and 'last' in ticker['data']:
                            price = float(ticker['data']['last'])
                            value = total * price
                    
                    balances[coin] = {
                        'available': available,
                        'frozen': frozen,
                        'total': total,
                        'value': value
                    }
                    
                    total_value += value
                except Exception:
                    # Fallback if we can't get price
                    balances[coin] = {
                        'available': available,
                        'frozen': frozen,
                        'total': total,
                        'value': 'N/A'
                    }
        
        # Sort by value
        sorted_balances = dict(sorted(balances.items(), 
                               key=lambda x: x[1]['value'] if isinstance(x[1]['value'], (int, float)) else 0, 
                               reverse=True))
        
        return {
            'balances': sorted_balances,
            'total_value': total_value,
            'quote_currency': quote_currency
        }
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return None

def show_active_orders(client, symbol=None):
    """
    Show active orders for a symbol or all symbols.
    
    Args:
        client: BitgetClient instance
        symbol: Trading symbol (optional)
        
    Returns:
        list: Active orders
    """
    try:
        open_orders = client.get_open_orders(symbol)
        
        if 'data' not in open_orders:
            logger.error("Failed to get open orders")
            return None
        
        orders = open_orders['data']
        
        if not orders:
            return []
        
        # Extract relevant fields and sort by symbol and time
        formatted_orders = []
        for order in orders:
            formatted_orders.append({
                'symbol': order.get('symbol', 'N/A'),
                'side': order.get('side', 'N/A'),
                'price': order.get('price', 'N/A'),
                'size': order.get('size', 'N/A'),
                'filled_size': order.get('filledSize', '0'),
                'order_id': order.get('orderId', 'N/A'),
                'status': order.get('status', 'N/A'),
                'created_time': order.get('cTime', 'N/A')
            })
        
        # Sort by symbol and creation time
        formatted_orders.sort(key=lambda x: (x['symbol'], x.get('created_time', 0)))
        
        return formatted_orders
    except Exception as e:
        logger.error(f"Error getting active orders: {e}")
        return None

def run_interactive_mode(client):
    """
    Run the utility in interactive mode with a simple menu system.
    
    Args:
        client: BitgetClient instance
    """
    logger.info("Starting interactive mode")
    
    while True:
        print("\n===== Limit Order Placement Utility - Interactive Mode =====")
        print("1. Place a limit order")
        print("2. View open orders")
        print("3. Cancel an order")
        print("4. Cancel all orders")
        print("5. View account balance")
        print("6. View order book for a symbol")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            # Collect order details
            symbol = input("Enter symbol (e.g., BTCUSDT): ").strip().upper()
            side = input("Enter side (buy/sell): ").strip().lower()
            
            if side not in ['buy', 'sell']:
                print("Invalid side. Must be 'buy' or 'sell'.")
                continue
            
            try:
                amount = float(input("Enter amount: ").strip())
            except ValueError:
                print("Invalid amount. Must be a number.")
                continue
            
            print("\nSelect order strategy:")
            print("1. Optimal (best maker price)")
            print("2. Smart (multiple orders)")
            print("3. Passive (larger price offset)")
            print("4. Custom price")
            
            strategy_choice = input("Enter choice (1-4): ").strip()
            
            strategy = "optimal"
            params = {}
            
            if strategy_choice == '1':
                strategy = "optimal"
                offset = input("Enter price offset percentage [0.12]: ").strip() or "0.12"
                try:
                    params['offset'] = float(offset) / 100
                except ValueError:
                    print("Invalid offset. Using default 0.12%.")
                    params['offset'] = 0.0012
            
            elif strategy_choice == '2':
                strategy = "smart"
                order_count = input("Enter number of orders [3]: ").strip() or "3"
                spread = input("Enter price spread percentage [3.0]: ").strip() or "3.0"
                
                try:
                    params['order_count'] = int(order_count)
                    params['spread'] = float(spread) / 100
                except ValueError:
                    print("Invalid input. Using defaults.")
                    params['order_count'] = 3
                    params['spread'] = 0.03
            
            elif strategy_choice == '3':
                strategy = "passive"
                offset = input("Enter price offset percentage [0.25]: ").strip() or "0.25"
                try:
                    params['offset'] = float(offset) / 100
                except ValueError:
                    print("Invalid offset. Using default 0.25%.")
                    params['offset'] = 0.0025
            
            elif strategy_choice == '4':
                strategy = "custom"
                try:
                    price = float(input("Enter custom price: ").strip())
                    params['price'] = price
                except ValueError:
                    print("Invalid price. Must be a number.")
                    continue
            
            # Show order preview
            print("\n--- Order Preview ---")
            preview = display_order_preview(client, symbol, side, amount, price=params.get('price'), 
                                          strategy=strategy, **params)
            
            if not preview:
                print("Failed to generate order preview.")
                continue
            
            print(f"Symbol: {preview['symbol']}")
            print(f"Side: {preview['side']}")
            print(f"Strategy: {preview['strategy']}")
            print(f"Amount: {preview['amount']}")
            print(f"Price: {preview['price']}")
            print(f"Total value: {preview['total_value']}")
            print(f"Current market price: {preview['current_price']}")
            
            if strategy == "smart":
                print(f"Order count: {preview['order_count']}")
                print(f"Spread: {preview['spread_percentage']}")
                print(f"Size per order: {preview['size_per_order']}")
            
            # Show orderbook analysis
            analysis = display_orderbook_analysis(client, symbol, side)
            if analysis:
                print("\n--- Order Book Analysis ---")
                print(f"Best bid: {analysis['best_bid']}")
                print(f"Best ask: {analysis['best_ask']}")
                print(f"Spread: {analysis['spread']} ({analysis['spread_pct']})")
                print(f"Bid/ask imbalance: {analysis['imbalance']}")
                print("\nMaker prices at different offsets:")
                for offset, price in analysis['maker_prices'].items():
                    print(f"  {offset}: {price}")
            
            # Confirm order
            if get_user_confirmation("\nDo you want to place this order?"):
                monitor = get_user_confirmation("Do you want to monitor the order after placement?")
                
                order_result = place_order(client, symbol, side, amount, 
                                       price=params.get('price'), 
                                       strategy=strategy, 
                                       monitor=monitor, 
                                       **params)
                
                if order_result:
                    print("Order placed successfully.")
                else:
                    print("Failed to place order.")
            else:
                print("Order cancelled.")
        
        elif choice == '2':
            # View open orders
            symbol = input("Enter symbol (or leave empty for all): ").strip().upper() or None
            
            orders = show_active_orders(client, symbol)
            
            if orders is None:
                print("Failed to get open orders.")
            elif not orders:
                print("No open orders.")
            else:
                print(f"\n--- Open Orders ({len(orders)}) ---")
                for i, order in enumerate(orders):
                    filled_pct = float(order['filled_size']) / float(order['size']) * 100 if float(order['size']) > 0 else 0
                    print(f"{i+1}. {order['symbol']} {order['side']} {order['size']} @ {order['price']}")
                    print(f"   ID: {order['order_id']}, Status: {order['status']}")
                    print(f"   Filled: {order['filled_size']}/{order['size']} ({filled_pct:.1f}%)")
        
        elif choice == '3':
            # Cancel an order
            symbol = input("Enter symbol: ").strip().upper()
            order_id = input("Enter order ID: ").strip()
            
            if not symbol or not order_id:
                print("Symbol and order ID are required.")
                continue
            
            if get_user_confirmation(f"Cancel order {order_id} for {symbol}?"):
                try:
                    result = client.cancel_order(symbol, order_id)
                    if 'code' in result and result['code'] == '00000':
                        print("Order cancelled successfully.")
                    else:
                        print("Failed to cancel order.")
                except Exception as e:
                    print(f"Error cancelling order: {e}")
        
        elif choice == '4':
            # Cancel all orders
            symbol = input("Enter symbol (or leave empty for all): ").strip().upper() or None
            
            confirm_msg = f"Cancel all orders{f' for {symbol}' if symbol else ''}?"
            if get_user_confirmation(confirm_msg):
                try:
                    if symbol:
                        result = client.cancel_all_orders(symbol)
                    else:
                        result = client.cancel_all_orders()
                        
                    if 'code' in result and result['code'] == '00000':
                        print("All orders cancelled successfully.")
                    else:
                        print("Failed to cancel orders.")
                except Exception as e:
                    print(f"Error cancelling orders: {e}")
        
        elif choice == '5':
            # View account balance
            quote = input("Enter quote currency [USDT]: ").strip().upper() or "USDT"
            
            balance_info = show_account_balance(client, quote)
            
            if not balance_info:
                print("Failed to get account balance.")
            else:
                balances = balance_info['balances']
                total_value = balance_info['total_value']
                quote = balance_info['quote_currency']
                
                print(f"\n--- Account Balance (Total: {total_value:.2f} {quote}) ---")
                for coin, data in balances.items():
                    value_str = f"{data['value']:.2f} {quote}" if isinstance(data['value'], (int, float)) else data['value']
                    print(f"{coin}: {data['total']} (Available: {data['available']}, Frozen: {data['frozen']}) - Value: {value_str}")
        
        elif choice == '6':
            # View order book
            symbol = input("Enter symbol: ").strip().upper()
            
            if not symbol:
                print("Symbol is required.")
                continue
            
            try:
                order_book = client.get_order_book(symbol, limit=10)
                
                if 'data' not in order_book:
                    print("Failed to get order book.")
                    continue
                
                book = order_book['data']
                asks = book.get('asks', [])
                bids = book.get('bids', [])
                
                print(f"\n--- Order Book for {symbol} ---")
                
                print("\nAsks (Sell Orders):")
                for i, ask in enumerate(reversed(asks[:10])):
                    price, size = ask
                    print(f"{i+1}. Price: {price}, Size: {size}")
                
                print("\nBids (Buy Orders):")
                for i, bid in enumerate(bids[:10]):
                    price, size = bid
                    print(f"{i+1}. Price: {price}, Size: {size}")
                
                # Calculate spread
                if asks and bids:
                    best_ask = float(asks[0][0])
                    best_bid = float(bids[0][0])
                    spread = best_ask - best_bid
                    spread_pct = (spread / best_bid) * 100
                    
                    print(f"\nBest ask: {best_ask}")
                    print(f"Best bid: {best_bid}")
                    print(f"Spread: {spread} ({spread_pct:.4f}%)")
                
            except Exception as e:
                print(f"Error getting order book: {e}")
        
        elif choice == '7':
            # Exit
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter a number from 1 to 7.")
        
        # Pause before returning to menu
        input("\nPress Enter to continue...")

def main():
    """Main function to run the utility."""
    args = parse_args()
    
    try:
        # Create API client
        client = BitgetClient(use_testnet=args.testnet)
        
        # Test API connection
        try:
            ticker = client.get_ticker("BTCUSDT")
            if 'data' not in ticker:
                logger.error("Failed to connect to Bitget API. Please check your credentials.")
                return
            logger.info("Connected to Bitget API successfully.")
        except Exception as e:
            logger.error(f"Error connecting to Bitget API: {e}")
            return
        
        # Run in interactive mode or process command-line arguments
        if args.interactive:
            run_interactive_mode(client)
        elif args.symbol and args.side and args.amount:
            # Process command-line order
            logger.info(f"Processing command-line order: {args.symbol} {args.side} {args.amount}")
            
            # Get price if specified
            price = args.price
            
            # Get strategy-specific parameters
            params = {
                'offset': args.offset,
                'order_count': args.order_count,
                'spread': args.spread
            }
            
            # Show order preview
            preview = display_order_preview(client, args.symbol, args.side, args.amount, 
                                         price, args.strategy, **params)
            
            if not preview:
                logger.error("Failed to generate order preview.")
                return
            
            # Display preview
            logger.info(f"Order preview: {args.symbol} {args.side} {preview['amount']} @ {preview['price']}")
            
            # Place order
            order_result = place_order(client, args.symbol, args.side, args.amount, 
                                    price, args.strategy, args.monitor, **params)
            
            if not order_result:
                logger.error("Failed to place order.")
        else:
            logger.error("Missing required arguments. Use --interactive or provide --symbol, --side, and --amount.")
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()