# Options IV Mispricing Detector

Automated tool for analyzing implied volatility (IV) on options to optimize covered call and cash-secured put timing.

## What It Does

- ✅ Fetches daily options data for your watchlist
- ✅ Calculates IV Rank (where current IV sits vs 52-week range)
- ✅ Compares IV to Historical Volatility (HV)
- ✅ Stores everything in cloud database (Supabase)
- ✅ Runs automatically Monday-Friday at 1:30 PM PST

## Key Metrics

**IV Rank:**
- 60-100% = High IV → Good time to SELL premium (covered calls, CSPs)
- 40-60% = Medium IV → Marginal
- 0-40% = Low IV → WAIT for better premium (or consider buying options)

**IV - HV:**
- Positive = Options overpriced → Good for selling
- Negative = Options underpriced → Good for buying

## Current Watchlist

**Mag 7:**
- AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA

**ETFs & Others:**
- SPY, QQQ, AMD, PLTR, SOXL

## How It Works

**Dual API Strategy:**
1. **Yahoo Finance (Primary):** Fast, free, no API key needed
2. **Alpha Vantage (Backup):** Reliable fallback if Yahoo rate-limits

**Automated Schedule:**
- Runs via GitHub Actions
- Monday-Friday at 1:30 PM PST (4:30 PM ET)
- Data stored in Supabase PostgreSQL

## Setup Complete ✅

This repository is configured with:
- ✅ Supabase database connection
- ✅ GitHub Actions workflow
- ✅ API keys stored as secrets
- ✅ Automated daily runs

## Manual Trigger

To run data collection manually:
1. Go to **Actions** tab
2. Click **"Daily Options Data Collection"**
3. Click **"Run workflow"**
4. Select branch and run

## Viewing Data

**Option 1: Supabase Dashboard**
- Go to your Supabase project
- Click "Table Editor"
- View tables: `stocks`, `options_data`, `historical_iv`

**Option 2: SQL Queries**
```sql
-- See latest IV Rank for all stocks
SELECT ticker, date, current_iv, iv_rank, hv_30d, iv_hv_diff
FROM historical_iv
WHERE date = CURRENT_DATE
ORDER BY iv_rank DESC;

-- Find high IV opportunities (60%+)
SELECT ticker, current_iv, iv_rank
FROM historical_iv
WHERE date = CURRENT_DATE
AND iv_rank >= 60
ORDER BY iv_rank DESC;
```

## Next Steps

**Week 2-3: After collecting 2+ weeks of data**
- IV Rank becomes accurate
- Start using for live trading decisions

**Week 4: Dashboard**
- Build Next.js frontend
- Visual display of opportunities
- Charts showing IV trends

## Trading Strategy

**When IV Rank > 60%:**
- ✅ Sell covered calls
- ✅ Sell cash-secured puts
- ✅ Collect premium aggressively

**When IV Rank < 40%:**
- ⏸️ Wait for better premium
- 🟢 Consider buying options (if IV < HV)

## Monitoring

**Check GitHub Actions:**
- Go to Actions tab
- See if daily runs are succeeding
- View logs if errors occur

**Check Supabase:**
- Table Editor → historical_iv
- See latest data entries
- Verify daily updates

## Support

Built with Claude AI
Started: November 14, 2024
Goal: Optimize options premium selling with data-driven timing

---

**Remember:** This tool tells you WHEN to sell premium, not WHICH stocks to trade. Use it to time your existing covered call and CSP strategies.
