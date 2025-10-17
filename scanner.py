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
MIN_DIVERGENCES = 4
DISCORD_WEBHOOK = os.environ.get('DISCORD_WEBHOOK')

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
    
    def __init__(self, df, pivot_period=5, max_bars=100):
        self.df = df
        self.pivot_period = pivot_period
        self.max_bars = max_bars
        
    def find_pivot_highs(self):
        """Find pivot high points"""
        highs = self.df['high'].values
        pivot_highs = []
        
        for i in range(self.pivot_period, len(highs) - self.pivot_period):
            if all(highs[i] > highs[i-j] for j in range(1, self.pivot_period + 1)) and \
               all(highs[i] > highs[i+j] for j in range(1, self.pivot_period + 1)):
                pivot_highs.append(i)
        
        return pivot_highs
    
    def find_pivot_lows(self):
        """Find pivot low points"""
        lows = self.df['low'].values
        pivot_lows = []
        
        for i in range(self.pivot_period, len(lows) - self.pivot_period):
            if all(lows[i] < lows[i-j] for j in range(1, self.pivot_period + 1)) and \
               all(lows[i] < lows[i+j] for j in range(1, self.pivot_period + 1)):
                pivot_lows.append(i)
        
        return pivot_lows
    
    def check_positive_regular_divergence(self, indicator, pivot_lows):
        """Check for positive regular divergence (bullish)"""
        if len(pivot_lows) < 2:
            return False
            
        # Get the last two pivot lows within max_bars
        recent_pivots = [p for p in pivot_lows if len(self.df) - p <= self.max_bars]
        if len(recent_pivots) < 2:
            return False
            
        curr_idx = recent_pivots[-1]
        prev_idx = recent_pivots[-2]
        
        # Price makes lower low, indicator makes higher low
        if self.df['low'].iloc[curr_idx] < self.df['low'].iloc[prev_idx] and \
           indicator.iloc[curr_idx] > indicator.iloc[prev_idx]:
            return True
        
        return False
    
    def check_negative_regular_divergence(self, indicator, pivot_highs):
        """Check for negative regular divergence (bearish)"""
        if len(pivot_highs) < 2:
            return False
            
        # Get the last two pivot highs within max_bars
        recent_pivots = [p for p in pivot_highs if len(self.df) - p <= self.max_bars]
        if len(recent_pivots) < 2:
            return False
            
        curr_idx = recent_pivots[-1]
        prev_idx = recent_pivots[-2]
        
        # Price makes higher high, indicator makes lower high
        if self.df['high'].iloc[curr_idx] > self.df['high'].iloc[prev_idx] and \
           indicator.iloc[curr_idx] < indicator.iloc[prev_idx]:
            return True
        
        return False
    
    def check_positive_hidden_divergence(self, indicator, pivot_lows):
        """Check for positive hidden divergence (bullish continuation)"""
        if len(pivot_lows) < 2:
            return False
            
        recent_pivots = [p for p in pivot_lows if len(self.df) - p <= self.max_bars]
        if len(recent_pivots) < 2:
            return False
            
        curr_idx = recent_pivots[-1]
        prev_idx = recent_pivots[-2]
        
        # Price makes higher low, indicator makes lower low
        if self.df['low'].iloc[curr_idx] > self.df['low'].iloc[prev_idx] and \
           indicator.iloc[curr_idx] < indicator.iloc[prev_idx]:
            return True
        
        return False
    
    def check_negative_hidden_divergence(self, indicator, pivot_highs):
        """Check for negative hidden divergence (bearish continuation)"""
        if len(pivot_highs) < 2:
            return False
            
        recent_pivots = [p for p in pivot_highs if len(self.df) - p <= self.max_bars]
        if len(recent_pivots) < 2:
            return False
            
        curr_idx = recent_pivots[-1]
        prev_idx = recent_pivots[-2]
        
        # Price makes lower high, indicator makes higher high
        if self.df['high'].iloc[curr_idx] < self.df['high'].iloc[prev_idx] and \
           indicator.iloc[curr_idx] > indicator.iloc[prev_idx]:
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
    
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=200):
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
        detector = DivergenceDetector(df, PIVOT_PERIOD, MAX_BARS_TO_CHECK)
        
        pivot_highs = detector.find_pivot_highs()
        pivot_lows = detector.find_pivot_lows()
        
        divergences = {
            'positive_regular': [],
            'negative_regular': [],
            'positive_hidden': [],
            'negative_hidden': []
        }
        
        indicator_names = ['macd', 'histogram', 'rsi', 'stoch', 'cci', 'momentum', 'obv', 'vwmacd', 'cmf', 'mfi']
        
        for ind_name in indicator_names:
            if ind_name in indicators:
                indicator = indicators[ind_name]
                
                if detector.check_positive_regular_divergence(indicator, pivot_lows):
                    divergences['positive_regular'].append(ind_name)
                    
                if detector.check_negative_regular_divergence(indicator, pivot_highs):
                    divergences['negative_regular'].append(ind_name)
                    
                if detector.check_positive_hidden_divergence(indicator, pivot_lows):
                    divergences['positive_hidden'].append(ind_name)
                    
                if detector.check_negative_hidden_divergence(indicator, pivot_highs):
                    divergences['negative_hidden'].append(ind_name)
        
        return divergences
    
    def send_discord_alert(self, message):
        """Send alert to Discord"""
        try:
            data = {
                "content": message,
                "username": "Divergence Scanner"
            }
            response = requests.post(DISCORD_WEBHOOK, json=data)
            if response.status_code != 204:
                print(f"Discord webhook error: {response.status_code}")
        except Exception as e:
            print(f"Error sending Discord alert: {e}")
    
    def scan(self):
        """Main scanning function"""
        print(f"\n🔍 Starting Bybit Divergence Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Min Volume: ${MIN_DAILY_VOLUME:,.0f} | Min Divergences: {MIN_DIVERGENCES}")
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
            if df is None or len(df) < 100:
                print(" ❌ Insufficient data")
                continue
            
            # Calculate indicators
            indicators = self.calculate_all_indicators(df)
            
            # Detect divergences
            divergences = self.detect_divergences(df, indicators)
            
            # Count total divergences
            total_divs = sum(len(divs) for divs in divergences.values())
            
            # Check if RSI and OBV are included
            all_divergence_indicators = []
            for div_type, indicators_list in divergences.items():
                all_divergence_indicators.extend(indicators_list)
            
            has_rsi = 'rsi' in all_divergence_indicators
            has_obv = 'obv' in all_divergence_indicators
            
            if total_divs >= MIN_DIVERGENCES and has_rsi and has_obv:
                print(f" ✅ ALERT! {total_divs} divergences found!")
                
                alert_data = {
                    'symbol': symbol,
                    'volume': volume,
                    'total_divergences': total_divs,
                    'divergences': divergences,
                    'current_price': df['close'].iloc[-1],
                    'rsi': indicators['rsi'].iloc[-1],
                    'timestamp': datetime.now()
                }
                alerts.append(alert_data)
                
                # Print detailed info
                print(f"\n  📊 {symbol}")
                print(f"  💰 Price: ${alert_data['current_price']:.4f}")
                print(f"  📈 RSI: {alert_data['rsi']:.2f}")
                print(f"  🔄 Total Divergences: {total_divs}")
                
                for div_type, ind_list in divergences.items():
                    if ind_list:
                        print(f"  • {div_type.replace('_', ' ').title()}: {', '.join(ind_list)}")
            else:
                print(f" · {total_divs} divs", end='')
                if total_divs > 0 and (not has_rsi or not has_obv):
                    print(f" (missing {'RSI' if not has_rsi else ''}{' & ' if not has_rsi and not has_obv else ''}{'OBV' if not has_obv else ''})", end='')
            
            # Rate limiting
            time.sleep(0.5)
        
        # Send Discord alerts
        if alerts:
            print(f"\n\n🚨 SENDING {len(alerts)} ALERTS TO DISCORD 🚨")
            
            discord_message = f"**🔥 DIVERGENCE ALERTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🔥**\n\n"
            
            for alert in alerts:
                discord_message += f"**{alert['symbol']}**\n"
                discord_message += f"• Price: ${alert['current_price']:.4f} | RSI: {alert['rsi']:.2f}\n"
                discord_message += f"• Total Divergences: {alert['total_divergences']}\n"
                
                div_summary = []
                for div_type, indicators in alert['divergences'].items():
                    if indicators:
                        div_summary.append(f"{div_type.replace('_', ' ').title()}: {len(indicators)}")
                
                discord_message += f"• {' | '.join(div_summary)}\n\n"
            
            discord_message += "```Requirements met: ≥4 divergences with RSI & OBV included```"
            
            self.send_discord_alert(discord_message)
            
        else:
            print("\n\n✅ Scan complete. No alerts triggered.")
        
        return alerts

if __name__ == "__main__":
    scanner = BybitDivergenceScanner()
    
    # Run the scanner
    # alerts = scanner.scan()
    
    # Optionally, you can run this in a loop
    while True:
        alerts = scanner.scan()
        time.sleep(900)  # Wait 15 minutes before next scan