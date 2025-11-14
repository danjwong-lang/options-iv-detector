"""
Options IV Mispricing Detector - Production Version
Dual API support: Yahoo Finance (primary) + Alpha Vantage (backup)
Database: Supabase PostgreSQL
"""

import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values

class OptionsIVCollector:
    def __init__(self):
        """Initialize with environment variables"""
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_KEY')
        
        # Supabase connection
        self.db_host = os.environ.get('SUPABASE_HOST')
        self.db_name = os.environ.get('SUPABASE_DB', 'postgres')
        self.db_user = os.environ.get('SUPABASE_USER', 'postgres')
        self.db_password = os.environ.get('SUPABASE_PASSWORD')
        self.db_port = os.environ.get('SUPABASE_PORT', '5432')
        
        self.conn = None
        
    def connect_db(self):
        """Connect to Supabase PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=self.db_port
            )
            print("✅ Connected to Supabase database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def add_stock(self, ticker, name=None, sector=None):
        """Add stock to tracking list"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO stocks (ticker, name, sector)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticker) DO NOTHING
            """, (ticker.upper(), name, sector))
            self.conn.commit()
            cursor.close()
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
        """Save options data to database"""
        if options_data is None:
            return
        
        try:
            cursor = self.conn.cursor()
            today = datetime.now().date()
            
            # Save calls
            for _, row in options_data['calls'].iterrows():
                cursor.execute("""
                    INSERT INTO options_data 
                    (ticker, date, strike, expiration, option_type, bid, ask, last, 
                     volume, open_interest, implied_volatility)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date, strike, expiration, option_type) DO UPDATE
                    SET bid = EXCLUDED.bid, ask = EXCLUDED.ask, last = EXCLUDED.last,
                        volume = EXCLUDED.volume, open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility
                """, (
                    ticker, today,
                    float(row.get('strike', 0)),
                    options_data['expiration'],
                    'call',
                    float(row.get('bid', 0)),
                    float(row.get('ask', 0)),
                    float(row.get('lastPrice', row.get('last', 0))),
                    int(row.get('volume', 0)),
                    int(row.get('openInterest', row.get('open_interest', 0))),
                    float(row.get('impliedVolatility', row.get('implied_volatility', 0)))
                ))
            
            # Save puts
            for _, row in options_data['puts'].iterrows():
                cursor.execute("""
                    INSERT INTO options_data 
                    (ticker, date, strike, expiration, option_type, bid, ask, last, 
                     volume, open_interest, implied_volatility)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date, strike, expiration, option_type) DO UPDATE
                    SET bid = EXCLUDED.bid, ask = EXCLUDED.ask, last = EXCLUDED.last,
                        volume = EXCLUDED.volume, open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility
                """, (
                    ticker, today,
                    float(row.get('strike', 0)),
                    options_data['expiration'],
                    'put',
                    float(row.get('bid', 0)),
                    float(row.get('ask', 0)),
                    float(row.get('lastPrice', row.get('last', 0))),
                    int(row.get('volume', 0)),
                    int(row.get('openInterest', row.get('open_interest', 0))),
                    float(row.get('impliedVolatility', row.get('implied_volatility', 0)))
                ))
            
            self.conn.commit()
            cursor.close()
            print(f"✅ Saved {ticker} to database")
            
        except Exception as e:
            print(f"❌ Error saving {ticker}: {e}")
            self.conn.rollback()
    
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
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT implied_volatility
                FROM options_data
                WHERE ticker = %s
                AND date >= CURRENT_DATE - INTERVAL '365 days'
                AND option_type = 'call'
                ORDER BY date DESC
            """, (ticker,))
            
            rows = cursor.fetchall()
            cursor.close()
            
            if len(rows) < 10:
                return None
            
            iv_values = [row[0] for row in rows if row[0] is not None]
            
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
        if iv_metrics:
            try:
                cursor = self.conn.cursor()
                today = datetime.now().date()
                
                cursor.execute("""
                    INSERT INTO historical_iv
                    (ticker, date, current_iv, iv_rank, hv_30d, iv_hv_diff, 
                     iv_52w_high, iv_52w_low)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE
                    SET current_iv = EXCLUDED.current_iv,
                        iv_rank = EXCLUDED.iv_rank,
                        hv_30d = EXCLUDED.hv_30d,
                        iv_hv_diff = EXCLUDED.iv_hv_diff,
                        iv_52w_high = EXCLUDED.iv_52w_high,
                        iv_52w_low = EXCLUDED.iv_52w_low
                """, (
                    ticker, today, current_iv, iv_metrics['iv_rank'],
                    hv_30d, iv_hv_diff,
                    iv_metrics['iv_52w_high'], iv_metrics['iv_52w_low']
                ))
                
                self.conn.commit()
                cursor.close()
            except Exception as e:
                print(f"⚠️  Error saving IV analysis: {e}")
                self.conn.rollback()
        
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
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")


if __name__ == "__main__":
    # Watchlist - all 18 stocks
    watchlist = [
        # Mag 7
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
        # ETFs and others
        'SPY', 'QQQ', 'AMD', 'PLTR', 'SOXL'
        # Note: Removed METU, TEM, CRWV, NAIL, AMZU, NVDL (invalid tickers or no options)
    ]
    
    collector = OptionsIVCollector()
    
    # Connect to database
    if not collector.connect_db():
        print("❌ Failed to connect to database. Exiting.")
        exit(1)
    
    # Add stocks to database
    print("\nAdding stocks to database...")
    for ticker in watchlist:
        collector.add_stock(ticker)
    
    # Run daily update
    collector.run_daily_update(watchlist)
    
    # Close connection
    collector.close()
