import ccxt
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Configuration
MIN_DAILY_VOLUME = 5000000  # $5M daily volume minimum
PIVOT_PERIOD = 5
MAX_BARS_TO_CHECK = 100
MIN_DIVERGENCES = 3  # Minimum number of REGULAR divergences required
MAX_DIVERGENCE_AGE = 2  # Only alert on divergences confirmed within last 2 candles
DISCORD_WEBHOOK = os.environ.get('DISCORD_WEBHOOK')
DONT_CONFIRM = True  # New parameter matching PineScript's dontconfirm

# At the top of your script
print(f"Script started at {datetime.now()}")
sys.stdout.flush()  # Ensure output is visible in logs

class TechnicalIndicators:
    """Calculate technical indicators matching the PineScript logic"""
    
    @staticmethod
    def rsi(close, period=14):
        """Calculate RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        """Calculate MACD, Signal, and Histogram"""
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high, low, close, period=14, smooth=3):
        """Calculate Stochastic"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=smooth).mean()
        return d_percent
    
    @staticmethod
    def cci(high, low, close, period=10):
        """Calculate CCI"""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - ma) / (0.015 * mad)
        return cci
    
    @staticmethod
    def momentum(close, period=10):
        """Calculate Momentum"""
        return close - close.shift(period)
    
    @staticmethod
    def obv(close, volume):
        """Calculate OBV"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def vwmacd(close, volume, fast=12, slow=26):
        """Calculate Volume Weighted MACD"""
        vwma_fast = (close * volume).rolling(window=fast).sum() / volume.rolling(window=fast).sum()
        vwma_slow = (close * volume).rolling(window=slow).sum() / volume.rolling(window=slow).sum()
        return vwma_fast - vwma_slow
    
    @staticmethod
    def cmf(high, low, close, volume, period=21):
        """Calculate Chaikin Money Flow"""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    @staticmethod
    def mfi(high, low, close, volume, period=14):
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        positive_flow = pd.Series(0.0, index=typical_price.index)
        negative_flow = pd.Series(0.0, index=typical_price.index)
        
        price_diff = typical_price.diff()
        
        positive_flow[price_diff > 0] = raw_money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = raw_money_flow[price_diff < 0]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

class DivergenceDetector:
    """Detect divergences matching the PineScript logic"""
    
    def __init__(self, df, pivot_period=5, max_bars=100, max_pivot_points=10, dont_confirm=True):
        self.df = df
        self.pivot_period = pivot_period
        self.max_bars = max_bars
        self.max_pivot_points = max_pivot_points
        self.dont_confirm = dont_confirm
        
    def find_pivot_highs_realtime(self):
        """Find pivot high points with real-time detection like PineScript"""
        highs = self.df['high'].values
        pivot_highs = []
        
        # We can identify a pivot high once we have pivot_period bars after it
        # This matches PineScript's behavior
        for i in range(self.pivot_period, len(highs) - self.pivot_period):
            # Check left side
            if all(highs[i] > highs[i-j] for j in range(1, self.pivot_period + 1)):
                # Check right side
                if all(highs[i] > highs[i+j] for j in range(1, self.pivot_period + 1)):
                    pivot_highs.append(i)
        
        # Check if we might have a developing pivot at the end
        # This allows detection similar to PineScript's real-time behavior
        if self.dont_confirm and len(highs) > self.pivot_period:
            # Check positions where we have enough left side but not full right side
            for i in range(max(self.pivot_period, len(highs) - self.pivot_period), len(highs)):
                if i >= len(highs):
                    break
                # Need at least pivot_period bars on the left
                if i >= self.pivot_period:
                    # Check left side
                    if all(highs[i] > highs[i-j] for j in range(1, min(self.pivot_period + 1, i + 1))):
                        # Check available right side
                        right_bars = len(highs) - i - 1
                        if right_bars > 0:
                            if all(highs[i] > highs[i+j] for j in range(1, min(right_bars + 1, self.pivot_period + 1))):
                                # Potential pivot, but need to verify it's not already in our list
                                if i not in pivot_highs:
                                    pivot_highs.append(i)
        
        return sorted(pivot_highs, reverse=True)  # Most recent first
    
    def find_pivot_lows_realtime(self):
        """Find pivot low points with real-time detection like PineScript"""
        lows = self.df['low'].values
        pivot_lows = []
        
        # Standard pivot detection
        for i in range(self.pivot_period, len(lows) - self.pivot_period):
            if all(lows[i] < lows[i-j] for j in range(1, self.pivot_period + 1)):
                if all(lows[i] < lows[i+j] for j in range(1, self.pivot_period + 1)):
                    pivot_lows.append(i)
        
        # Check developing pivots at the end
        if self.dont_confirm and len(lows) > self.pivot_period:
            for i in range(max(self.pivot_period, len(lows) - self.pivot_period), len(lows)):
                if i >= len(lows):
                    break
                if i >= self.pivot_period:
                    if all(lows[i] < lows[i-j] for j in range(1, min(self.pivot_period + 1, i + 1))):
                        right_bars = len(lows) - i - 1
                        if right_bars > 0:
                            if all(lows[i] < lows[i+j] for j in range(1, min(right_bars + 1, self.pivot_period + 1))):
                                if i not in pivot_lows:
                                    pivot_lows.append(i)
        
        return sorted(pivot_lows, reverse=True)  # Most recent first
    
    def validate_divergence_line(self, indicator, start_idx, end_idx, price_start, price_end, is_bullish):
        """Validate that intermediate values don't break the divergence pattern"""
        if end_idx - start_idx <= 5:  # Minimum distance check
            return False
            
        # Calculate slopes for virtual lines
        indicator_slope = (indicator.iloc[end_idx] - indicator.iloc[start_idx]) / (end_idx - start_idx)
        price_slope = (price_end - price_start) / (end_idx - start_idx)
        
        # Check all intermediate points
        for i in range(start_idx + 1, end_idx):
            # Calculate expected values on the virtual lines
            indicator_line_value = indicator.iloc[start_idx] + indicator_slope * (i - start_idx)
            price_line_value = price_start + price_slope * (i - start_idx)
            
            if is_bullish:
                # For bullish divergence, indicator and price should not go below their respective lines
                if indicator.iloc[i] < indicator_line_value or self.df['low'].iloc[i] < price_line_value:
                    return False
            else:
                # For bearish divergence, indicator and price should not go above their respective lines
                if indicator.iloc[i] > indicator_line_value or self.df['high'].iloc[i] > price_line_value:
                    return False
                    
        return True
    
    def check_positive_regular_divergence(self, indicator, pivot_lows):
        """Check for positive regular divergence (bullish) - matching Pine Script logic"""
        if len(pivot_lows) < 2:
            return False, -1
        
        current_bar = len(self.df) - 1
        startpoint = 0 if self.dont_confirm else 1
        
        # If dont_confirm is True and indicators/price are not confirming, check from current bar
        if not self.dont_confirm:
            if not (indicator.iloc[-1] > indicator.iloc[-2] or self.df['close'].iloc[-1] > self.df['close'].iloc[-2]):
                return False, -1
        
        # Search through multiple pivot points (not just last 2)
        for i in range(min(self.max_pivot_points, len(pivot_lows))):
            curr_pivot_idx = pivot_lows[i]
            
            # Skip if pivot is too far from current bar
            if current_bar - curr_pivot_idx > self.max_bars:
                break
                
            # Check against previous pivots
            for j in range(i + 1, min(i + self.max_pivot_points, len(pivot_lows))):
                prev_pivot_idx = pivot_lows[j]
                
                # Skip if distance is less than minimum
                if curr_pivot_idx - prev_pivot_idx <= 5:
                    continue
                
                # For recent divergences, we might check against current bar
                check_idx = current_bar - startpoint if current_bar - curr_pivot_idx < self.pivot_period else curr_pivot_idx
                
                # Check divergence condition: price lower low, indicator higher low
                price_condition = self.df['low'].iloc[check_idx] < self.df['low'].iloc[prev_pivot_idx]
                indicator_condition = indicator.iloc[check_idx] > indicator.iloc[prev_pivot_idx]
                
                if price_condition and indicator_condition:
                    # Validate intermediate values
                    if self.validate_divergence_line(
                        indicator, 
                        prev_pivot_idx, 
                        check_idx,
                        self.df['low'].iloc[prev_pivot_idx],
                        self.df['low'].iloc[check_idx],
                        is_bullish=True
                    ):
                        # Return True and the bar index where divergence was confirmed
                        return True, check_idx
                        
        return False, -1
    
    def check_negative_regular_divergence(self, indicator, pivot_highs):
        """Check for negative regular divergence (bearish) - matching Pine Script logic"""
        if len(pivot_highs) < 2:
            return False, -1
        
        current_bar = len(self.df) - 1
        startpoint = 0 if self.dont_confirm else 1
        
        # If dont_confirm is False, check confirmation
        if not self.dont_confirm:
            if not (indicator.iloc[-1] < indicator.iloc[-2] or self.df['close'].iloc[-1] < self.df['close'].iloc[-2]):
                return False, -1
        
        for i in range(min(self.max_pivot_points, len(pivot_highs))):
            curr_pivot_idx = pivot_highs[i]
            
            if current_bar - curr_pivot_idx > self.max_bars:
                break
                
            for j in range(i + 1, min(i + self.max_pivot_points, len(pivot_highs))):
                prev_pivot_idx = pivot_highs[j]
                
                if curr_pivot_idx - prev_pivot_idx <= 5:
                    continue
                
                check_idx = current_bar - startpoint if current_bar - curr_pivot_idx < self.pivot_period else curr_pivot_idx
                
                # Check divergence condition: price higher high, indicator lower high
                price_condition = self.df['high'].iloc[check_idx] > self.df['high'].iloc[prev_pivot_idx]
                indicator_condition = indicator.iloc[check_idx] < indicator.iloc[prev_pivot_idx]
                
                if price_condition and indicator_condition:
                    if self.validate_divergence_line(
                        indicator,
                        prev_pivot_idx,
                        check_idx,
                        self.df['high'].iloc[prev_pivot_idx],
                        self.df['high'].iloc[check_idx],
                        is_bullish=False
                    ):
                        # Return True and the bar index where divergence was confirmed
                        return True, check_idx
                        
        return False, -1
    
    def check_positive_hidden_divergence(self, indicator, pivot_lows):
        """Check for positive hidden divergence (bullish continuation) - matching Pine Script logic"""
        if len(pivot_lows) < 2:
            return False
        
        current_bar = len(self.df) - 1
        startpoint = 0 if self.dont_confirm else 1
        
        if not self.dont_confirm:
            if not (indicator.iloc[-1] > indicator.iloc[-2] or self.df['close'].iloc[-1] > self.df['close'].iloc[-2]):
                return False
        
        for i in range(min(self.max_pivot_points, len(pivot_lows))):
            curr_pivot_idx = pivot_lows[i]
            
            if current_bar - curr_pivot_idx > self.max_bars:
                break
                
            for j in range(i + 1, min(i + self.max_pivot_points, len(pivot_lows))):
                prev_pivot_idx = pivot_lows[j]
                
                if curr_pivot_idx - prev_pivot_idx <= 5:
                    continue
                
                check_idx = current_bar - startpoint if current_bar - curr_pivot_idx < self.pivot_period else curr_pivot_idx
                
                # Check divergence condition: price higher low, indicator lower low
                price_condition = self.df['low'].iloc[check_idx] > self.df['low'].iloc[prev_pivot_idx]
                indicator_condition = indicator.iloc[check_idx] < indicator.iloc[prev_pivot_idx]
                
                if price_condition and indicator_condition:
                    if self.validate_divergence_line(
                        indicator,
                        prev_pivot_idx,
                        check_idx,
                        self.df['low'].iloc[prev_pivot_idx],
                        self.df['low'].iloc[check_idx],
                        is_bullish=True
                    ):
                        return True
                        
        return False
    
    def check_negative_hidden_divergence(self, indicator, pivot_highs):
        """Check for negative hidden divergence (bearish continuation) - matching Pine Script logic"""
        if len(pivot_highs) < 2:
            return False
        
        current_bar = len(self.df) - 1
        startpoint = 0 if self.dont_confirm else 1
        
        if not self.dont_confirm:
            if not (indicator.iloc[-1] < indicator.iloc[-2] or self.df['close'].iloc[-1] < self.df['close'].iloc[-2]):
                return False
        
        for i in range(min(self.max_pivot_points, len(pivot_highs))):
            curr_pivot_idx = pivot_highs[i]
            
            if current_bar - curr_pivot_idx > self.max_bars:
                break
                
            for j in range(i + 1, min(i + self.max_pivot_points, len(pivot_highs))):
                prev_pivot_idx = pivot_highs[j]
                
                if curr_pivot_idx - prev_pivot_idx <= 5:
                    continue
                
                check_idx = current_bar - startpoint if current_bar - curr_pivot_idx < self.pivot_period else curr_pivot_idx
                
                # Check divergence condition: price lower high, indicator higher high
                price_condition = self.df['high'].iloc[check_idx] < self.df['high'].iloc[prev_pivot_idx]
                indicator_condition = indicator.iloc[check_idx] > indicator.iloc[prev_pivot_idx]
                
                if price_condition and indicator_condition:
                    if self.validate_divergence_line(
                        indicator,
                        prev_pivot_idx,
                        check_idx,
                        self.df['high'].iloc[prev_pivot_idx],
                        self.df['high'].iloc[check_idx],
                        is_bullish=False
                    ):
                        return True
                        
        return False

class BybitDivergenceScanner:
    """Main scanner class"""
    
    def __init__(self):
        self.exchange = ccxt.bybit()
        self.indicators = TechnicalIndicators()
        # Use environment variable if available
        self.discord_webhook = os.environ.get('DISCORD_WEBHOOK', DISCORD_WEBHOOK)
        
    def fetch_perps_with_volume(self):
        """Fetch perpetual pairs with volume > threshold"""
        try:
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            perps = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT:USDT') and 'info' in ticker:
                    volume_24h = float(ticker.get('quoteVolume', 0))
                    if volume_24h > MIN_DAILY_VOLUME:
                        perps.append({
                            'symbol': symbol,
                            'volume': volume_24h
                        })
            
            return sorted(perps, key=lambda x: x['volume'], reverse=True)
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=500):  # Increased from 200 to 500
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return None
    
    def calculate_all_indicators(self, df):
        """Calculate all indicators"""
        indicators = {}
        
        # Calculate indicators
        indicators['macd'], indicators['signal'], indicators['histogram'] = self.indicators.macd(df['close'])
        indicators['rsi'] = self.indicators.rsi(df['close'])
        indicators['stoch'] = self.indicators.stochastic(df['high'], df['low'], df['close'])
        indicators['cci'] = self.indicators.cci(df['high'], df['low'], df['close'])
        indicators['momentum'] = self.indicators.momentum(df['close'])
        indicators['obv'] = self.indicators.obv(df['close'], df['volume'])
        indicators['vwmacd'] = self.indicators.vwmacd(df['close'], df['volume'])
        indicators['cmf'] = self.indicators.cmf(df['high'], df['low'], df['close'], df['volume'])
        indicators['mfi'] = self.indicators.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        return indicators
    
    def detect_divergences(self, df, indicators):
        """Detect all divergences for all indicators"""
        detector = DivergenceDetector(df, PIVOT_PERIOD, MAX_BARS_TO_CHECK, 
                                      max_pivot_points=10, dont_confirm=DONT_CONFIRM)
        
        pivot_highs = detector.find_pivot_highs_realtime()
        pivot_lows = detector.find_pivot_lows_realtime()
        
        current_bar = len(df) - 1
        max_age = MAX_DIVERGENCE_AGE  # Only consider divergences confirmed within this many candles
        
        divergences = {
            'positive_regular': [],
            'negative_regular': []
        }
        
        # Track the most recent confirmation bar for any divergence
        most_recent_confirmation = -1
        
        indicator_names = ['macd', 'histogram', 'rsi', 'stoch', 'cci', 'momentum', 'obv', 'vwmacd', 'cmf', 'mfi']
        
        for ind_name in indicator_names:
            if ind_name in indicators:
                indicator = indicators[ind_name]
                
                # Check positive regular divergence
                has_div, confirmation_bar = detector.check_positive_regular_divergence(indicator, pivot_lows)
                if has_div and (current_bar - confirmation_bar) <= max_age:
                    divergences['positive_regular'].append(ind_name)
                    most_recent_confirmation = max(most_recent_confirmation, confirmation_bar)
                    
                # Check negative regular divergence
                has_div, confirmation_bar = detector.check_negative_regular_divergence(indicator, pivot_highs)
                if has_div and (current_bar - confirmation_bar) <= max_age:
                    divergences['negative_regular'].append(ind_name)
                    most_recent_confirmation = max(most_recent_confirmation, confirmation_bar)
        
        # Add information about when the divergence was confirmed
        if most_recent_confirmation > -1:
            bars_ago = current_bar - most_recent_confirmation
            divergences['confirmation_info'] = {
                'bars_ago': bars_ago,
                'timestamp': df['timestamp'].iloc[most_recent_confirmation]
            }
        
        return divergences
    
    def send_discord_alert(self, message):
        """Send alert to Discord with automatic message splitting for long content"""
        try:
            # Discord has a 2000 character limit
            max_length = 1900  # Leave some buffer
            
            if len(message) <= max_length:
                # Message fits in one post
                data = {
                    "content": message,
                    "username": "Divergence Scanner"
                }
                response = requests.post(self.discord_webhook, json=data)
                if response.status_code != 204:
                    print(f"Discord webhook error: {response.status_code}")
            else:
                # Split message into multiple parts
                parts = []
                current_part = ""
                lines = message.split('\n')
                
                for line in lines:
                    # If adding this line would exceed limit, start a new part
                    if len(current_part) + len(line) + 1 > max_length:
                        if current_part:
                            parts.append(current_part)
                        current_part = line + '\n'
                    else:
                        current_part += line + '\n'
                
                # Don't forget the last part
                if current_part:
                    parts.append(current_part)
                
                # Send each part
                for i, part in enumerate(parts):
                    data = {
                        "content": part,
                        "username": f"Divergence Scanner ({i+1}/{len(parts)})"
                    }
                    response = requests.post(self.discord_webhook, json=data)
                    if response.status_code != 204:
                        print(f"Discord webhook error on part {i+1}: {response.status_code}")
                    time.sleep(0.5)  # Small delay between messages to avoid rate limiting
                    
        except Exception as e:
            print(f"Error sending Discord alert: {e}")
    
    def scan(self):
        """Main scanning function"""
        print(f"\n🔍 Starting Bybit Recent Divergence Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Min Volume: ${MIN_DAILY_VOLUME:,.0f} | Min Regular Divergences: {MIN_DIVERGENCES}")
        print(f"Max Age: {MAX_DIVERGENCE_AGE} candles | Don't Confirm Mode: {'ON' if DONT_CONFIRM else 'OFF'}")
        print("=" * 80)
        
        # Get perpetual pairs with volume filter
        perps = self.fetch_perps_with_volume()
        print(f"\nFound {len(perps)} perpetual pairs with volume > ${MIN_DAILY_VOLUME:,.0f}")
        
        alerts = []
        
        for i, perp in enumerate(perps[:50], 1):  # Limit to top 50 by volume to avoid rate limits
            symbol = perp['symbol']
            volume = perp['volume']
            
            print(f"\n[{i}/{min(50, len(perps))}] Analyzing {symbol} (24h Vol: ${volume:,.0f})...", end='')
            
            # Fetch OHLCV data
            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 200:
                print(" ❌ Insufficient data")
                continue
            
            # Calculate indicators
            indicators = self.calculate_all_indicators(df)
            
            # Detect divergences
            divergences = self.detect_divergences(df, indicators)
            
            # Count total divergences
            total_divs = sum(len(divs) for divs_type, divs in divergences.items() if divs_type != 'confirmation_info')
            
            # Check if RSI and OBV are included
            all_divergence_indicators = []
            for div_type, indicators_list in divergences.items():
                if div_type != 'confirmation_info':
                    all_divergence_indicators.extend(indicators_list)
            
            has_rsi = 'rsi' in all_divergence_indicators
            has_obv = 'obv' in all_divergence_indicators
            
            # Only process if we have recent divergences (confirmed within last 2 candles)
            has_recent_divergence = 'confirmation_info' in divergences
            
            if total_divs >= MIN_DIVERGENCES and has_recent_divergence: # and has_rsi and has_obv 
                print(f" ✅ ALERT! {total_divs} divergences found (confirmed {divergences['confirmation_info']['bars_ago']} bars ago)!")
                
                alert_data = {
                    'symbol': symbol,
                    'volume': volume,
                    'total_divergences': total_divs,
                    'divergences': divergences,
                    'current_price': df['close'].iloc[-1],
                    'rsi': indicators['rsi'].iloc[-1],
                    'timestamp': datetime.now(),
                    'confirmation_bars_ago': divergences['confirmation_info']['bars_ago']
                }
                alerts.append(alert_data)
                
                # Print detailed info
                print(f"\n  📊 {symbol}")
                print(f"  💰 Price: ${alert_data['current_price']:.4f}")
                print(f"  📈 RSI: {alert_data['rsi']:.2f}")
                print(f"  🔄 Total Divergences: {total_divs} (confirmed {divergences['confirmation_info']['bars_ago']} bars ago)")
                
                for div_type, ind_list in divergences.items():
                    if ind_list and div_type != 'confirmation_info':
                        div_type_formatted = "Bullish" if div_type == "positive_regular" else "Bearish"
                        print(f"  • {div_type_formatted}: {', '.join(ind_list)}")
            else:
                print(f" · {total_divs} divs", end='')
                if total_divs > 0 and not has_recent_divergence:
                    print(f" (too old)", end='')
                elif total_divs > 0 and (not has_rsi or not has_obv):
                    print(f" (missing {'RSI' if not has_rsi else ''}{' & ' if not has_rsi and not has_obv else ''}{'OBV' if not has_obv else ''})", end='')
            
            # Rate limiting
            time.sleep(0.5)
        
        # Send Discord alerts
        if alerts:
            print(f"\n\n🚨 FOUND {len(alerts)} ALERTS 🚨")
            
            # Sort alerts by total divergences (descending) to prioritize strongest signals
            alerts_sorted = sorted(alerts, key=lambda x: x['total_divergences'], reverse=True)
            
            # Option 1: Send all alerts (will be split automatically if too long)
            discord_message = f"**🔥 RECENT DIVERGENCE ALERTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🔥**\n\n"
            discord_message += f"*Found {len(alerts)} symbols with divergences confirmed in last 2 candles*\n"
            discord_message += f"*Mode: {'Early Detection' if DONT_CONFIRM else 'Confirmed'}*\n\n"
            
            for alert in alerts_sorted:
                # Determine divergence type
                bullish_count = len(alert['divergences'].get('positive_regular', []))
                bearish_count = len(alert['divergences'].get('negative_regular', []))
                div_type = ""
                if bullish_count > 0 and bearish_count > 0:
                    div_type = f"🔄 Mixed ({bullish_count}↑/{bearish_count}↓)"
                elif bullish_count > 0:
                    div_type = f"🟢 Bullish ({bullish_count})"
                else:
                    div_type = f"🔴 Bearish ({bearish_count})"
                
                # Show how recent the confirmation was
                bars_ago = alert.get('confirmation_bars_ago', 0)
                recency = "📍 JUST NOW" if bars_ago == 0 else f"📍 {bars_ago} bar{'s' if bars_ago > 1 else ''} ago"
                
                discord_message += f"**{alert['symbol']}** - {div_type} {recency} | ${alert['current_price']:.4f} | RSI: {alert['rsi']:.0f}\n"
            
            discord_message += f"\n```Requirements: ≥{MIN_DIVERGENCES} regular divergences in last 2 candles```"
            
            self.send_discord_alert(discord_message)
            
            # Option 2: Send detailed info only for top alerts
            # Uncomment the section below to send detailed info for top 10 alerts only
            """
            top_alerts = alerts_sorted[:10]
            
            discord_message = f"**🔥 TOP DIVERGENCE ALERTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🔥**\n\n"
            discord_message += f"*Total found: {len(alerts)} | Showing top {len(top_alerts)}*\n"
            discord_message += f"*Mode: {'Early Detection' if DONT_CONFIRM else 'Confirmed'}*\n\n"
            
            for alert in top_alerts:
                discord_message += f"**{alert['symbol']}**\n"
                discord_message += f"• Price: ${alert['current_price']:.4f} | RSI: {alert['rsi']:.2f}\n"
                discord_message += f"• Total Divergences: {alert['total_divergences']}\n"
                
                div_summary = []
                for div_type, indicators in alert['divergences'].items():
                    if indicators:
                        div_summary.append(f"{div_type.replace('_', ' ').title()}: {len(indicators)}")
                
                discord_message += f"• {' | '.join(div_summary)}\n\n"
            
            if len(alerts) > 10:
                discord_message += f"\n*... and {len(alerts) - 10} more symbols with divergences*\n"
            
            discord_message += f"\n```Requirements met: ≥{MIN_DIVERGENCES} divergences```"
            
            self.send_discord_alert(discord_message)
            """
            
        else:
            print("\n\n✅ Scan complete. No alerts triggered.")
        
        return alerts

if __name__ == "__main__":
    scanner = BybitDivergenceScanner()
    
    # Calculate when to run for optimal timing
    current_time = datetime.now()
    minutes = current_time.minute
    seconds = current_time.second
    
    # Calculate seconds until next 15-minute mark + 30 seconds buffer
    next_15_min = ((minutes // 15) + 1) * 15
    if next_15_min >= 60:
        next_15_min = 0
    
    seconds_to_wait = ((next_15_min - minutes) * 60) - seconds + 30  # 30 second buffer after candle close
    
    # if seconds_to_wait > 0 and seconds_to_wait < 900:  # Don't wait more than 15 minutes
    #     print(f"Waiting {seconds_to_wait} seconds until next 15-minute candle closes...")
    #     time.sleep(seconds_to_wait)
    
    # Run the scanner
    alerts = scanner.scan()
    
    # Optionally, you can run this in a loop synchronized with 15-minute candles
    # while True:
    #     alerts = scanner.scan()
    #     
    #     # Calculate time until next 15-minute mark + 30 seconds
    #     current_time = datetime.now()
    #     minutes = current_time.minute
    #     seconds = current_time.second
    #     next_15_min = ((minutes // 15) + 1) * 15
    #     if next_15_min >= 60:
    #         next_15_min = 0
    #     seconds_to_wait = ((next_15_min - minutes) * 60) - seconds + 30
    #     
    #     if seconds_to_wait > 0:
    #         print(f"\nNext scan in {seconds_to_wait} seconds...")
    #         time.sleep(seconds_to_wait)