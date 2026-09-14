import os
import time
import logging
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:

    def __init__(self, config: dict):
  
        self.config = config
        self.market_cfg = config["market_data"]
        self.path_cfg   = config["paths"]

        logger.info("Initializing DataFetcher...")
        self.exchanges = {
            "binance": ccxt.binance({
                "apiKey":  os.environ.get("BINANCE_API_KEY", ""),
                "secret":  os.environ.get("BINANCE_SECRET", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "future"}
            }),
            "bybit": ccxt.bybit({               # ← changed
                "apiKey":  os.environ.get("BYBIT_API_KEY", ""),
                "secret":  os.environ.get("BYBIT_SECRET", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "linear"}  # ← Bybit uses "linear"
            })
        }

        logger.info(
            f"Exchanges initialized: {list(self.exchanges.keys())}"
        )

    def _get_exchange(self, exchange_name: str) -> ccxt.Exchange:
 
        exchange = self.exchanges.get(exchange_name)

        if exchange is None:
            available = list(self.exchanges.keys())
            raise ValueError(
                f"Exchange '{exchange_name}' not found. "
                f"Available exchanges: {available}"
            )

        return exchange

    def fetch_ohlcv(
        self,
        exchange_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1m",
    ) -> pd.DataFrame:
        
        exchange = self._get_exchange(exchange_name)

        since  = exchange.parse8601(f"{start_date}T00:00:00Z")
        end_ts = exchange.parse8601(f"{end_date}T00:00:00Z")

        all_candles = []
        request_count = 0

        logger.info(
            f"Fetching OHLCV | {exchange_name} | {symbol} | "
            f"{start_date} → {end_date} | interval={interval}"
        )

        while since < end_ts:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=interval,
                    since=since,
                    limit=1000       # max candles per request
                )

                if not candles:
                    logger.warning(
                        f"Empty response from {exchange_name} for {symbol}. "
                        f"Stopping pagination."
                    )
                    break

                all_candles.extend(candles)
                request_count += 1

                last_ts = candles[-1][0]
                since   = last_ts + 1

                logger.debug(
                    f"Request {request_count}: fetched {len(candles)} candles, "
                    f"last timestamp={last_ts}"
                )

                time.sleep(0.5)

            except ccxt.NetworkError as e:
                logger.error(f"Network error on {exchange_name}: {e}")
                break

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error on {exchange_name}: {e}")
                break

            except Exception as e:
                logger.error(f"Unexpected error fetching OHLCV: {e}")
                break

        if not all_candles:
            logger.warning(f"No OHLCV data fetched for {symbol} on {exchange_name}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        end_dt = pd.Timestamp(end_date, tz="UTC")
        df = df[df.index <= end_dt]
        df = df[~df.index.duplicated(keep="first")]

        df.sort_index(inplace=True)

        logger.info(
            f"OHLCV fetch complete | {symbol} | {exchange_name} | "
            f"{len(df)} rows | {request_count} requests made"
        )

        return df

    def fetch_funding_rates(
        self,
        exchange_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:

        exchange = self._get_exchange(exchange_name)

        since  = exchange.parse8601(f"{start_date}T00:00:00Z")
        end_ts = exchange.parse8601(f"{end_date}T00:00:00Z")

        all_records = []
        request_count = 0

        logger.info(
            f"Fetching funding rates | {exchange_name} | {symbol} | "
            f"{start_date} → {end_date}"
        )

        while since < end_ts:
            try:
                funding_data = exchange.fetch_funding_rate_history(
                    symbol=symbol,
                    since=since,
                    limit=1000
                )

                if not funding_data:
                    logger.warning(
                        f"No more funding rate data for {symbol} on {exchange_name}"
                    )
                    break

                for record in funding_data:
                    all_records.append({
                        "timestamp":    record["timestamp"],
                        "funding_rate": record["fundingRate"]
                    })

                request_count += 1
                last_ts = funding_data[-1]["timestamp"]
                since   = last_ts + 1

                logger.debug(
                    f"Request {request_count}: fetched {len(funding_data)} "
                    f"funding records"
                )

                time.sleep(0.5)

            except ccxt.NetworkError as e:
                logger.error(f"Network error fetching funding rates: {e}")
                break

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error fetching funding rates: {e}")
                break

            except Exception as e:
                logger.error(f"Unexpected error fetching funding rates: {e}")
                break

        if not all_records:
            logger.warning(
                f"No funding rate data fetched for {symbol} on {exchange_name}"
            )
            return pd.DataFrame(columns=["funding_rate"])

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        end_dt = pd.Timestamp(end_date, tz="UTC")
        df = df[df.index <= end_dt]

        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        logger.info(
            f"Funding rates complete | {symbol} | {exchange_name} | "
            f"{len(df)} rows"
        )

        return df

    def fetch_all(self) -> None:
       
        raw_dir    = self.path_cfg["raw_data_dir"]
        assets     = self.market_cfg["assets"]
        exchanges  = self.market_cfg["exchanges"]
        start_date = self.market_cfg["start_date"]
        end_date   = self.market_cfg["end_date"]
        interval   = self.market_cfg["interval"]

        os.makedirs(raw_dir, exist_ok=True)

        logger.info(
            f"Starting fetch_all | "
            f"assets={assets} | exchanges={exchanges} | "
            f"{start_date} → {end_date}"
        )

        success_count = 0
        failure_count = 0

        for exchange_name in exchanges:
            for asset in assets:

                symbol = asset.replace("-", "/") + ":USDT"
                safe_asset = asset.replace("-", "")  # for filenames

                logger.info(
                    f"── {exchange_name.upper()} | {symbol} ──"
                )

                try:
                    df_ohlcv = self.fetch_ohlcv(
                        exchange_name, symbol,
                        start_date, end_date, interval
                    )

                    filename = (
                        f"{exchange_name}_{safe_asset}"
                        f"_{interval}_ohlcv.csv"
                    )
                    filepath = os.path.join(raw_dir, filename)
                    df_ohlcv.to_csv(filepath)

                    logger.info(f"Saved OHLCV → {filepath}")
                    success_count += 1

                except Exception as e:
                    logger.error(
                        f"OHLCV failed | {exchange_name} | {symbol} | {e}"
                    )
                    failure_count += 1

                try:
                    df_funding = self.fetch_funding_rates(
                        exchange_name, symbol,
                        start_date, end_date
                    )

                    filename = (
                        f"{exchange_name}_{safe_asset}"
                        f"_funding_rate.csv"
                    )
                    filepath = os.path.join(raw_dir, filename)
                    df_funding.to_csv(filepath)

                    logger.info(f"Saved funding rates → {filepath}")
                    success_count += 1

                except Exception as e:
                    logger.error(
                        f"Funding rates failed | {exchange_name} | "
                        f"{symbol} | {e}"
                    )
                    failure_count += 1

        logger.info(
            f"fetch_all complete | "
            f"success={success_count} | failures={failure_count}"
        )
