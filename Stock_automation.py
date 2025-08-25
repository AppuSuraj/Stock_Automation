#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved stock notifier: fetches richer info per ticker and formats a concise,
useful Telegram-friendly plain-text message.

Dependencies:
  - yfinance
  - pandas
  - requests

Usage: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as repository Secrets
and run the script.
"""
import os
import sys
import logging
from datetime import datetime, timezone, timedelta

import requests
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Default tickers (change as needed)
TICKERS = ["TATAMOTORS.NS", "M&M.NS", "ADANIGREEN.NS"]

# Helpers
def _fmt(x):
    try:
        if x is None:
            return "—"
        if isinstance(x, int):
            return f"{x:,}"
        if isinstance(x, float):
            s = f"{x:,.2f}"
            if s.endswith(".00"):
                s = s[:-3]
            return s
        return str(x)
    except Exception:
        return str(x)


def _percent(curr, prev):
    try:
        if curr is None or prev is None or prev == 0:
            return None
        return (curr - prev) / prev * 100.0
    except Exception:
        return None


def _trend_marker(pct):
    if pct is None:
        return ""
    if pct >= 5:
        return "🚀"
    if pct >= 1:
        return "🔼"
    if pct <= -5:
        return "📉"
    if pct <= -1:
        return "🔽"
    return "➖"


def fetch_stock_data(tickers):
    """
    Fetch richer stock details for each ticker using yfinance.
    Returns list of dicts with useful fields for formatting.
    """
    results = []
    for ticker in tickers:
        logging.info("Fetching %s", ticker)
        try:
            t = yf.Ticker(ticker)
            info = {}
            try:
                info = t.info or {}
            except Exception:
                logging.debug("Could not fetch .info for %s", ticker)

            # Basic fields: prefer info, fallback to history
            current = info.get("regularMarketPrice")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

            day_high = info.get("dayHigh")
            day_low = info.get("dayLow")
            w52h = info.get("fiftyTwoWeekHigh") or info.get("52WeekHigh")
            w52l = info.get("fiftyTwoWeekLow") or info.get("52WeekLow")
            volume = info.get("volume")
            avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day") or info.get("averageVolume10days")

            market_cap = info.get("marketCap")
            pe = info.get("trailingPE") or info.get("forwardPE") or info.get("pe")
            div = info.get("dividendYield")

            sma50 = info.get("fiftyDayAverage") or info.get("50DayAverage")
            sma200 = info.get("twoHundredDayAverage") or info.get("200DayAverage")

            name = info.get("shortName") or info.get("longName") or ticker

            # If critical fields missing, use history fallback
            need_history = any(x is None for x in (current, prev_close, day_high, day_low, volume))
            hist = None
            if need_history:
                try:
                    hist = t.history(period="365d")
                    if not isinstance(hist, pd.DataFrame) or hist.empty:
                        hist = None
                except Exception:
                    hist = None

            if current is None and hist is not None:
                try:
                    current = float(hist["Close"].iloc[-1])
                except Exception:
                    current = None
            if prev_close is None and hist is not None and len(hist) >= 2:
                try:
                    prev_close = float(hist["Close"].iloc[-2])
                except Exception:
                    prev_close = None
            if (day_high is None or day_low is None) and hist is not None:
                try:
                    day_high = float(hist["High"].iloc[-1])
                    day_low = float(hist["Low"].iloc[-1])
                except Exception:
                    pass
            if volume is None and hist is not None:
                try:
                    volume = int(hist["Volume"].iloc[-1])
                except Exception:
                    pass

            # Compute SMA if missing and history available
            if (sma50 is None or sma200 is None) and hist is not None:
                try:
                    sma50 = float(hist["Close"].rolling(window=50).mean().iloc[-1]) if len(hist) >= 50 else sma50
                    sma200 = float(hist["Close"].rolling(window=200).mean().iloc[-1]) if len(hist) >= 200 else sma200
                except Exception:
                    pass

            stock = {
                "symbol": ticker,
                "name": name,
                "current": float(current) if current is not None else None,
                "prev_close": float(prev_close) if prev_close is not None else None,
                "day_high": float(day_high) if day_high is not None else None,
                "day_low": float(day_low) if day_low is not None else None,
                "week52_high": float(w52h) if w52h is not None else None,
                "week52_low": float(w52l) if w52l is not None else None,
                "volume": int(volume) if volume is not None else None,
                "avg_volume": int(avg_vol) if avg_vol is not None else None,
                "market_cap": int(market_cap) if market_cap is not None else None,
                "pe": float(pe) if pe is not None else None,
                "dividend_yield": float(div) if div is not None else None,
                "sma50": float(sma50) if sma50 is not None else None,
                "sma200": float(sma200) if sma200 is not None else None,
                "url": f"https://finance.yahoo.com/quote/{ticker}"
            }
            results.append(stock)
        except Exception:
            logging.exception("Error fetching %s", ticker)
    return results


def format_message(stocks):
    """
    Build a compact, information-rich plain-text message suitable for Telegram.
    Summary at top, then one-line header per stock + compact details below.
    """
    if not stocks:
        return "No stock data available."

    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30)))
    header_time = now.strftime("%Y-%m-%d %H:%M:%S IST")

    gainers = []
    losers = []
    movers = []
    for s in stocks:
        pct = _percent(s.get("current"), s.get("prev_close"))
        if pct is not None:
            movers.append((pct, s))
            if pct > 0:
                gainers.append(s)
            elif pct < 0:
                losers.append(s)

    top_gainer = max(movers, key=lambda x: x[0]) if movers else (None, None)
    top_loser = min(movers, key=lambda x: x[0]) if movers else (None, None)

    lines = []
    lines.append(f"📈 Stock Snapshot • {len(stocks)} tracked • {header_time}")
    if movers:
        lines.append(f"Summary: {len(gainers)} ▲ | {len(losers)} ▼    Top ▲: {top_gainer[1].get('symbol') if top_gainer[1] else '—'} {top_gainer[0]:+.2f}%    Top ▼: {top_loser[1].get('symbol') if top_loser[1] else '—'} {top_loser[0]:+.2f}%")
    lines.append("")  # blank

    for s in stocks:
        sym = s.get("symbol")
        name = s.get("name")
        curr = s.get("current")
        prev = s.get("prev_close")
        pct = _percent(curr, prev)
        trend = _trend_marker(pct)

        # Header line: trend, symbol, current price and % change
        header = f"{trend} {sym} — ₹{_fmt(curr)}" if curr is not None else f"{sym} — price N/A"
        if pct is not None:
            header += f" ({pct:+.2f}%)"
        lines.append(header)

        # Compact detail line with core stats
        parts = []
        parts.append(f"Prev: ₹{_fmt(prev)}" if prev is not None else "Prev: —")
        parts.append(f"Day: ₹{_fmt(s.get('day_low'))}–₹{_fmt(s.get('day_high'))}" if (s.get('day_low') is not None or s.get('day_high') is not None) else "Day: —")
        parts.append(f"52W: ₹{_fmt(s.get('week52_low'))}–₹{_fmt(s.get('week52_high'))}" if (s.get('week52_low') is not None or s.get('week52_high') is not None) else "52W: —")
        parts.append(f"Vol: {_fmt(s.get('volume'))}" if s.get('volume') is not None else "Vol: —")
        if s.get('avg_volume') is not None:
            parts[-1] += f" (Avg {_fmt(s.get('avg_volume'))})"

        # Valuation / indicators
        val = []
        if s.get('market_cap') is not None:
            mc = s.get('market_cap')
            val.append(f"Mkt cap: {_fmt(mc)}")
        if s.get('pe') is not None:
            val.append(f"P/E: {_fmt(s.get('pe'))}")
        if s.get('dividend_yield') is not None:
            try:
                dv = float(s.get('dividend_yield'))
                if abs(dv) < 1:
                    dv = dv * 100.0
                val.append(f"Div: {_fmt(dv)}%")
            except Exception:
                val.append(f"Div: {_fmt(s.get('dividend_yield'))}")
        if s.get('sma50') is not None and s.get('sma200') is not None:
            pos = "SMA50>SMA200" if s.get('sma50') > s.get('sma200') else "SMA50<SMA200"
            val.append(pos)
        elif s.get('sma50') is not None:
            val.append(f"SMA50: {_fmt(s.get('sma50'))}")

        # Append stats and valuation as separate lines to keep header concise
        lines.append(" • ".join(parts))
        if val:
            lines.append(" • ".join(val))

        # optional chart/url line
        if s.get('url'):
            lines.append(f"Chart: {s.get('url')}")

        lines.append("")  # blank between stocks

    # Footer notes
    lines.append("Notes:")
    lines.append("- Data from Yahoo Finance via yfinance")
    lines.append("- This is informational only; not financial advice")

    return "\n".join(lines)


def _split_chunks(text, size=3900):
    """Split large messages into Telegram-safe chunks trying to respect paragraph breaks."""
    if len(text) <= size:
        return [text]
    parts = []
    remaining = text
    while remaining:
        if len(remaining) <= size:
            parts.append(remaining)
            break
        idx = remaining.rfind("\n\n", 0, size)
        if idx == -1:
            idx = remaining.rfind("\n", 0, size)
        if idx == -1:
            idx = size
        parts.append(remaining[:idx].rstrip())
        remaining = remaining[idx:].lstrip()
    return parts


def send_telegram(token, chat_id, text):
    """
    Send text to Telegram; splits into multiple messages if necessary.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    parts = _split_chunks(text, size=3900)
    last_resp = None
    for part in parts:
        payload = {"chat_id": chat_id, "text": part, "disable_web_page_preview": "true"}
        r = requests.post(url, data=payload, timeout=15)
        r.raise_for_status()
        last_resp = r.json()
    return last_resp


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set as environment variables.")
        sys.exit(1)

    stocks = fetch_stock_data(TICKERS)
    if not stocks:
        logging.error("No stock data fetched; aborting.")
        sys.exit(1)

    message = format_message(stocks)
    logging.info("Message length: %d characters", len(message))
    logging.info("Message preview:\n%s", message[:1000])

    try:
        res = send_telegram(token, chat_id, message)
        logging.info("Telegram response: %s", res)
    except Exception:
        logging.exception("Failed to send Telegram message")
        sys.exit(1)


if __name__ == "__main__":
    main()
