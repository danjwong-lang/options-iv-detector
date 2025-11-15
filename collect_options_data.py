"""
Options IV Mispricing Detector - Production Version (Supabase Client)
Dual API support: Yahoo Finance (primary) + Alpha Vantage (backup)
Database: Supabase via REST API (more reliable than direct PostgreSQL)
"""

import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client

class OptionsIVCollector:
    def __init__(self):
        """Initialize with environment variables"""
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_KEY')
        
        # Supabase connection via REST API
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")
        
    def add_stock(self, ticker, name=None, sector=None):
        """Add stock to tracking list"""
        try:
            data = {
                'ticker': ticker.upper(),
                'name': name,
                'sector': sector
            }
            self.supabase.table('stocks').upsert(data, on_conflict='ticker').execute()
        except Exception as e:
            print(f"⚠️  Error adding {ticker}: {e}")
    
    def fetch_with_yahoo(self, ticker):
        """Attempt to fetch data using Yahoo Finance"""
        try:
            time.sleep(1)  # Rate limiting
            
            stock = yf.Ticker(ticker)
            
            # Get price
            hist = stock.history(period='5d')
            if hist.empty:
                return None
            current_price = hist['Close'].iloc[-1]
            
            # Get options
            expirations = stock.options
            if len(expirations) == 0:
                return None
            
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            
            if opt_chain.calls.empty or opt_chain.puts.empty:
                return None
            
            # Calculate average IV from ATM options
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            atm_calls = calls[calls['strike'].between(current_price * 0.95, current_price * 1.05)]
            atm_puts = puts[puts['strike'].between(current_price * 0.95, current_price * 1.05)]
            
            if atm_calls.empty and atm_puts.empty:
                return None
            
            avg_iv = pd.concat([
                atm_calls['impliedVolatility'], 
                atm_puts['impliedVolatility']
            ]).mean()
            
            print(f"✅ Yahoo: {ticker} Price=${current_price:.2f}, IV={avg_iv:.2%}")
            
            return {
                'ticker': ticker,
                'price': current_price,
                'avg_iv': avg_iv,
                'expiration': nearest_exp,
                'calls': calls,
                'puts': puts,
                'source': 'yahoo'
            }
            
        except Exception as e:
            print(f"⚠️  Yahoo failed for {ticker}: {str(e)[:50]}")
            return None
    
    def fetch_with_alphavantage(self, ticker):
        """Fallback to Alpha Vantage API"""
        try:
            time.sleep(13)  # Alpha Vantage rate limit
            
            base_url = "https://www.alphavantage.co/query"
            
            # Get price
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if 'Global Quote' not in data or not data['Global Quote']:
                return None
            
            price = float(data['Global Quote']['05. price'])
            
            time.sleep(13)
            
            # Get options
            params = {
                'function': 'HISTORICAL_OPTIONS',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return None
            
            options_df = pd.DataFrame(data['data'])
            calls = options_df[options_df['type'] == 'call'].copy()
            puts = options_df[options_df['type'] == 'put'].copy()
            
            if calls.empty and puts.empty:
                return None
            
            # Calculate IV
            calls['strike'] = calls['strike'].astype(float)
            puts['strike'] = puts['strike'].astype(float)
            
            atm_calls = calls[calls['strike'].between(price * 0.95, price * 1.05)]
            atm_puts = puts[puts['strike'].between(price * 0.95, price * 1.05)]
            
            iv_values = []
            if not atm_calls.empty:
                iv_values.extend(atm_calls['implied_volatility'].astype(float).tolist())
            if not atm_puts.empty:
                iv_values.extend(atm_puts['implied_volatility'].astype(float).tolist())
            
            avg_iv = np.mean(iv_values) if iv_values else None
            
            if avg_iv is None:
                return None
            
            expiration = calls.iloc[0]['expiration'] if not calls.empty else puts.iloc[0]['expiration']
            
            print(f"✅ AlphaV: {ticker} Price=${price:.2f}, IV={avg_iv:.2%}")
            
            return {
                'ticker': ticker,
                'price': price,
                'avg_iv': avg_iv,
                'expiration': expiration,
                'calls': calls,
                'puts': puts,
                'source': 'alphavantage'
            }
            
        except Exception as e:
            print(f"❌ AlphaVantage failed for {ticker}: {str(e)[:50]}")
            return None
    
    def fetch_options_data(self, ticker):
        """Fetch options data with fallback logic"""
        print(f"\n📊 Fetching {ticker}...")
        
        # Try Yahoo first
        data = self.fetch_with_yahoo(ticker)
        
        # Fallback to Alpha Vantage if Yahoo fails
        if data is None and self.alpha_vantage_key:
            print(f"   Trying Alpha Vantage fallback...")
            data = self.fetch_with_alphavantage(ticker)
        
        return data
    
    def save_options_data(self, ticker, options_data):
        """Save options data to Supabase"""
        if options_data is None:
            return
        
        try:
            today = datetime.now().date().isoformat()
            
            # Prepare options records
            records = []
            
            # Process calls (limit to 50 to avoid timeouts)
            for _, row in options_data['calls'].head(50).iterrows():
                record = {
                    'ticker': ticker,
                    'date': today,
                    'strike': float(row.get('strike', 0)),
                    'expiration': str(options_data['expiration']),
                    'option_type': 'call',
                    'bid': float(row.get('bid', 0)),
                    'ask': float(row.get('ask', 0)),
                    'last': float(row.get('lastPrice', row.get('last', 0))),
                    'volume': int(row.get('volume', 0)),
                    'open_interest': int(row.get('openInterest', row.get('open_interest', 0))),
                    'implied_volatility': float(row.get('impliedVolatility', row.get('implied_volatility', 0)))
                }
                records.append(record)
            
            # Process puts
            for _, row in options_data['puts'].head(50).iterrows():
                record = {
                    'ticker': ticker,
                    'date': today,
                    'strike': float(row.get('strike', 0)),
                    'expiration': str(options_data['expiration']),
                    'option_type': 'put',
                    'bid': float(row.get('bid', 0)),
                    'ask': float(row.get('ask', 0)),
                    'last': float(row.get('lastPrice', row.get('last', 0))),
                    'volume': int(row.get('volume', 0)),
                    'open_interest': int(row.get('openInterest', row.get('open_interest', 0))),
                    'implied_volatility': float(row.get('impliedVolatility', row.get('implied_volatility', 0)))
                }
                records.append(record)
            
            # Upsert to Supabase
            if records:
                self.supabase.table('options_data').upsert(records, on_conflict='ticker,date,strike,expiration,option_type').execute()
                print(f"✅ Saved {len(records)} option contracts for {ticker}")
            
        except Exception as e:
            print(f"❌ Error saving {ticker}: {e}")
    
    def calculate_historical_volatility(self, ticker, days=30):
        """Calculate HV using Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f'{days+10}d')
            
            if len(hist) < days * 0.7:
                return None
            
            returns = hist['Close'].pct_change().dropna()
            daily_std = returns.std()
            annual_hv = daily_std * np.sqrt(252)
            
            return annual_hv
        except:
            return None
    
    def calculate_iv_rank(self, ticker):
        """Calculate IV Rank from historical data"""
        try:
            # Get historical IV data from Supabase
            response = self.supabase.table('options_data')\
                .select('implied_volatility')\
                .eq('ticker', ticker)\
                .eq('option_type', 'call')\
                .gte('date', (datetime.now().date() - pd.Timedelta(days=365)).isoformat())\
                .order('date', desc=True)\
                .execute()
            
            if not response.data or len(response.data) < 10:
                return None
            
            iv_values = [row['implied_volatility'] for row in response.data if row['implied_volatility']]
            
            if not iv_values:
                return None
            
            current_iv = iv_values[0]
            iv_52w_high = max(iv_values)
            iv_52w_low = min(iv_values)
            
            if iv_52w_high == iv_52w_low:
                iv_rank = 50.0
            else:
                iv_rank = ((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100
            
            return {
                'current_iv': current_iv,
                'iv_rank': iv_rank,
                'iv_52w_high': iv_52w_high,
                'iv_52w_low': iv_52w_low
            }
        except Exception as e:
            print(f"⚠️  Error calculating IV Rank for {ticker}: {e}")
            return None
    
    def analyze_stock(self, ticker):
        """Complete analysis for one stock"""
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print(f"{'='*60}")
        
        # Fetch options data
        options_data = self.fetch_options_data(ticker)
        if options_data is None:
            print(f"❌ Could not fetch data for {ticker}")
            return None
        
        # Save to database
        self.save_options_data(ticker, options_data)
        
        # Calculate HV
        hv_30d = self.calculate_historical_volatility(ticker, 30)
        
        # Calculate IV Rank
        iv_metrics = self.calculate_iv_rank(ticker)
        
        current_iv = options_data['avg_iv']
        iv_hv_diff = current_iv - hv_30d if hv_30d else None
        
        # Save IV analysis
        if iv_metrics or hv_30d:
            try:
                today = datetime.now().date().isoformat()
                
                record = {
                    'ticker': ticker,
                    'date': today,
                    'current_iv': float(current_iv),
                    'iv_rank': float(iv_metrics['iv_rank']) if iv_metrics else None,
                    'hv_30d': float(hv_30d) if hv_30d else None,
                    'iv_hv_diff': float(iv_hv_diff) if iv_hv_diff else None,
                    'iv_52w_high': float(iv_metrics['iv_52w_high']) if iv_metrics else None,
                    'iv_52w_low': float(iv_metrics['iv_52w_low']) if iv_metrics else None
                }
                
                self.supabase.table('historical_iv').upsert(record, on_conflict='ticker,date').execute()
                
            except Exception as e:
                print(f"⚠️  Error saving IV analysis: {e}")
        
        # Print summary
        print(f"\n✅ {ticker} Analysis Complete:")
        print(f"   Current IV: {current_iv:.2%}")
        if iv_metrics:
            print(f"   IV Rank: {iv_metrics['iv_rank']:.1f}%")
        else:
            print(f"   IV Rank: N/A (need more data)")
        if hv_30d:
            print(f"   HV (30d): {hv_30d:.2%}")
            if iv_hv_diff:
                print(f"   IV - HV: {iv_hv_diff:.2%}")
        
        return {
            'ticker': ticker,
            'current_iv': current_iv,
            'iv_rank': iv_metrics['iv_rank'] if iv_metrics else None,
            'hv_30d': hv_30d,
            'iv_hv_diff': iv_hv_diff
        }
    
    def run_daily_update(self, tickers):
        """Run daily update for all stocks"""
        print(f"\n{'='*60}")
        print(f"OPTIONS IV MISPRICING DETECTOR")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S PST')}")
        print(f"📊 Analyzing {len(tickers)} stocks")
        print(f"{'='*60}\n")
        
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            result = self.analyze_stock(ticker)
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"✅ Daily update complete!")
        print(f"   Successfully analyzed: {len(results)}/{len(tickers)} stocks")
        print(f"{'='*60}\n")
        
        return results


if __name__ == "__main__":
    # Watchlist - 25 stocks
    watchlist = [
        # Mag 7
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
        # Tech/Semiconductors
        'AMD', 'AVGO', 'TSM', 'ORCL', 'CRM',
        # Leveraged ETFs
        'SOXL', 'TSLL', 'NVDL', 'TQQQ',
        # ETFs
        'SPY', 'QQQ',
        # Others
        'PLTR', 'NAIL', 'METU', 'AMZU', 'GGLL', 'TEM', 'CRWV'
    
    ]
    
    collector = OptionsIVCollector()
    
    # Add stocks to database
    print("\nAdding stocks to database...")
    for ticker in watchlist:
        collector.add_stock(ticker)
    
    # Run daily update
    collector.run_daily_update(watchlist)
