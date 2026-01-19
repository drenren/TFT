"""
End-of-Day (EOD) Bank Account Balance Prediction Example with Hourly Granularity.

This script demonstrates using the Temporal Fusion Transformer for near-real time
prediction of end-of-day bank account balances. Key features:
1. Handles sparse transaction data (irregular, some days with zero transactions)
2. Creates HOURLY time series for near-real-time prediction capability
3. At any hour, predicts what the EOD balance will be
4. Uses rolling statistics and lag features for historical patterns

Near-Real-Time Capability:
- Each hour of each day has its own row
- Model learns: "Given it's hour X with current balance Y, what will EOD balance be?"
- At inference time, you can predict EOD balance at any hour during the day

The workflow:
1. Generate synthetic sparse transaction data
2. Convert transactions to complete HOURLY time series
3. Add calendar and hour features (known future)
4. Add lag and rolling features (observed only)
5. Train TFT model
6. Evaluate and visualize predictions
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

from tft.models import TemporalFusionTransformer
from tft.data import create_dataloaders
from tft.training import TFTTrainer, EarlyStopping, ModelCheckpoint
from tft.utils import TFTConfig, get_device, print_device_info
from tft.utils.visualization import plot_predictions, plot_training_history


def generate_synthetic_transactions(
    num_accounts: int = 10,
    num_days: int = 550,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic sparse transaction data for bank accounts.

    Creates transaction patterns including:
    - Monthly salary deposits (around 1st or 15th)
    - Irregular expenses (utilities, shopping, food)
    - Account-specific activity levels (daily, weekly, monthly)
    - Random gaps of 1-14 days with no transactions

    Args:
        num_accounts: Number of different bank accounts
        num_days: Total number of days (~18 months = 550 days)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: account_num, tran_timestamp, amount
    """
    np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    transactions = []

    for account_id in range(num_accounts):
        # Account characteristics
        account_type = account_id % 3  # 0: daily, 1: weekly, 2: monthly activity
        base_balance = np.random.uniform(5000, 50000)
        salary = np.random.uniform(3000, 8000)
        salary_day = np.random.choice([1, 15])  # Salary on 1st or 15th

        current_date = start_date

        for day_idx in range(num_days):
            current_date = start_date + timedelta(days=day_idx)
            day_of_month = current_date.day
            day_of_week = current_date.weekday()

            # Salary deposit (around salary day)
            if abs(day_of_month - salary_day) <= 1:
                if np.random.random() < 0.9:  # 90% chance of salary
                    trans_time = current_date.replace(
                        hour=np.random.randint(8, 12),
                        minute=np.random.randint(0, 60)
                    )
                    transactions.append({
                        'account_num': account_id,
                        'tran_timestamp': trans_time,
                        'amount': salary * np.random.uniform(0.95, 1.05),
                    })

            # Activity based on account type
            if account_type == 0:  # Daily activity
                # Multiple transactions per day
                num_trans = np.random.poisson(3)
                for _ in range(num_trans):
                    trans_time = current_date.replace(
                        hour=np.random.randint(8, 22),
                        minute=np.random.randint(0, 60)
                    )
                    # Mix of small expenses and occasional larger ones
                    if np.random.random() < 0.1:
                        amount = -np.random.uniform(100, 500)  # Larger expense
                    else:
                        amount = -np.random.uniform(5, 50)  # Small expense
                    transactions.append({
                        'account_num': account_id,
                        'tran_timestamp': trans_time,
                        'amount': amount,
                    })

            elif account_type == 1:  # Weekly activity
                # Transactions mainly on weekends
                if day_of_week >= 5 or np.random.random() < 0.2:
                    num_trans = np.random.poisson(2)
                    for _ in range(num_trans):
                        trans_time = current_date.replace(
                            hour=np.random.randint(10, 20),
                            minute=np.random.randint(0, 60)
                        )
                        amount = -np.random.uniform(20, 200)
                        transactions.append({
                            'account_num': account_id,
                            'tran_timestamp': trans_time,
                            'amount': amount,
                        })

            else:  # Monthly activity (type 2)
                # Very sparse - mainly bills
                if day_of_month in [1, 5, 10, 15, 20, 25]:
                    if np.random.random() < 0.5:
                        trans_time = current_date.replace(
                            hour=np.random.randint(9, 17),
                            minute=np.random.randint(0, 60)
                        )
                        # Monthly bills
                        amount = -np.random.uniform(50, 300)
                        transactions.append({
                            'account_num': account_id,
                            'tran_timestamp': trans_time,
                            'amount': amount,
                        })

            # Random occasional deposits (refunds, transfers, etc.)
            if np.random.random() < 0.02:
                trans_time = current_date.replace(
                    hour=np.random.randint(9, 18),
                    minute=np.random.randint(0, 60)
                )
                amount = np.random.uniform(50, 500)
                transactions.append({
                    'account_num': account_id,
                    'tran_timestamp': trans_time,
                    'amount': amount,
                })

    df = pd.DataFrame(transactions)
    df = df.sort_values(['account_num', 'tran_timestamp']).reset_index(drop=True)

    return df


def transactions_to_hourly_series(
    transactions_df: pd.DataFrame,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Convert sparse transactions to complete HOURLY time series.

    This enables near-real-time prediction:
    - Each hour of each day has its own row
    - At any hour, you can predict the EOD balance
    - Model learns patterns like "balance at 2pm → EOD balance"

    For each hour, we track:
    - current_balance: Balance at the END of this hour
    - inflow_today_so_far: Cumulative inflows today up to this hour
    - outflow_today_so_far: Cumulative outflows today up to this hour
    - transactions_today_so_far: Count of transactions today up to this hour
    - eod_balance: Target - the final balance at end of this day (same for all hours)

    Args:
        transactions_df: DataFrame with account_num, tran_timestamp, amount
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        DataFrame with complete hourly time series per account
    """
    transactions_df = transactions_df.copy()
    transactions_df['tran_timestamp'] = pd.to_datetime(transactions_df['tran_timestamp'])
    transactions_df['date'] = transactions_df['tran_timestamp'].dt.date
    transactions_df['hour'] = transactions_df['tran_timestamp'].dt.hour

    if start_date is None:
        start_date = transactions_df['date'].min()
    if end_date is None:
        end_date = transactions_df['date'].max()

    # Create hourly datetime range
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + timedelta(hours=23)
    all_hours = pd.date_range(start=start_dt, end=end_dt, freq='H')

    hourly_data = []

    for account_id in transactions_df['account_num'].unique():
        account_trans = transactions_df[transactions_df['account_num'] == account_id].copy()

        # Create base hourly dataframe
        account_hourly = pd.DataFrame({'datetime': all_hours})
        account_hourly['account_num'] = account_id
        account_hourly['date'] = account_hourly['datetime'].dt.date
        account_hourly['hour'] = account_hourly['datetime'].dt.hour

        # Aggregate transactions by hour
        hourly_agg = account_trans.groupby(['date', 'hour']).agg({
            'amount': ['sum', lambda x: x[x > 0].sum(), lambda x: x[x < 0].sum(), 'count'],
        }).reset_index()
        hourly_agg.columns = ['date', 'hour', 'hourly_net', 'hourly_inflow', 'hourly_outflow', 'hourly_trans_count']
        hourly_agg['hourly_outflow'] = hourly_agg['hourly_outflow'].abs()

        # Merge hourly transactions
        account_hourly = account_hourly.merge(hourly_agg, on=['date', 'hour'], how='left')

        # Fill missing hours with zeros
        account_hourly['hourly_net'] = account_hourly['hourly_net'].fillna(0)
        account_hourly['hourly_inflow'] = account_hourly['hourly_inflow'].fillna(0)
        account_hourly['hourly_outflow'] = account_hourly['hourly_outflow'].fillna(0)
        account_hourly['hourly_trans_count'] = account_hourly['hourly_trans_count'].fillna(0)

        # Calculate current_balance (cumulative balance at each hour)
        initial_balance = 10000 + account_id * 5000
        account_hourly['current_balance'] = initial_balance + account_hourly['hourly_net'].cumsum()

        # Calculate daily aggregates for EOD balance
        daily_net = account_hourly.groupby('date')['hourly_net'].transform('sum')

        # Calculate inflow/outflow/transactions "so far today" for each hour
        account_hourly['inflow_today_so_far'] = account_hourly.groupby('date')['hourly_inflow'].cumsum()
        account_hourly['outflow_today_so_far'] = account_hourly.groupby('date')['hourly_outflow'].cumsum()
        account_hourly['transactions_today_so_far'] = account_hourly.groupby('date')['hourly_trans_count'].cumsum()
        account_hourly['net_today_so_far'] = account_hourly.groupby('date')['hourly_net'].cumsum()

        # Calculate EOD balance for each day (target)
        # EOD balance = balance at hour 23 of each day
        eod_balances = account_hourly[account_hourly['hour'] == 23][['date', 'current_balance']].copy()
        eod_balances.columns = ['date', 'eod_balance']
        account_hourly = account_hourly.merge(eod_balances, on='date', how='left')

        # Calculate hours remaining in day
        account_hourly['hours_remaining'] = 23 - account_hourly['hour']

        # Previous day's EOD balance
        prev_eod = eod_balances.copy()
        prev_eod['date'] = prev_eod['date'] + pd.Timedelta(days=1)
        prev_eod.columns = ['date', 'prev_eod_balance']
        account_hourly = account_hourly.merge(prev_eod, on='date', how='left')

        # Fill first day's prev_eod_balance with initial balance
        account_hourly['prev_eod_balance'] = account_hourly['prev_eod_balance'].fillna(initial_balance)

        # Balance at start of day (hour 0)
        account_hourly['balance_start_of_day'] = account_hourly['prev_eod_balance']

        # CRITICAL: eod_balance_known - prevents data leakage!
        # For hours 0-22: we only know PREVIOUS day's EOD (not today's)
        # For hour 23: day is complete, we know today's EOD
        # This is what the model sees in the encoder (historical data)
        account_hourly['eod_balance_known'] = np.where(
            account_hourly['hour'] == 23,
            account_hourly['eod_balance'],      # Day complete - we know it
            account_hourly['prev_eod_balance']  # Intraday - only know yesterday's
        )

        # Days since last transaction (at daily level, propagated to hours)
        daily_trans = account_hourly.groupby('date')['hourly_trans_count'].sum().reset_index()
        daily_trans['had_transaction'] = (daily_trans['hourly_trans_count'] > 0).astype(int)

        days_since = []
        counter = 0
        for had_trans in daily_trans['had_transaction']:
            if had_trans:
                counter = 0
            days_since.append(counter)
            counter += 1
        daily_trans['days_since_last_transaction'] = days_since

        account_hourly = account_hourly.merge(
            daily_trans[['date', 'days_since_last_transaction']],
            on='date',
            how='left'
        )

        hourly_data.append(account_hourly)

    result_df = pd.concat(hourly_data, ignore_index=True)

    # Clean up columns
    result_df = result_df.drop(columns=['hourly_net', 'hourly_inflow', 'hourly_outflow', 'hourly_trans_count'], errors='ignore')

    return result_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features (known future variables).

    Features:
    - Hour of day: cyclical encoding (sin/cos) - KEY for near-real-time
    - Day of week: cyclical encoding (sin/cos)
    - Day of month: cyclical encoding (sin/cos)
    - Month: cyclical encoding (sin/cos)
    - Binary flags: is_weekend, is_month_end, is_month_start
    - hours_remaining: hours left in the day (known at inference)

    Args:
        df: DataFrame with 'datetime' column

    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Hour of day (0-23) - CRITICAL for near-real-time prediction
    hour = df['hour']
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day of week (0-6)
    day_of_week = df['datetime'].dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # Day of month (1-31)
    day_of_month = df['datetime'].dt.day
    df['day_of_month_sin'] = np.sin(2 * np.pi * day_of_month / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * day_of_month / 31)

    # Month (1-12)
    month = df['datetime'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Binary flags
    df['is_weekend'] = (day_of_week >= 5).astype(float)
    df['is_month_end'] = (df['datetime'].dt.is_month_end).astype(float)
    df['is_month_start'] = (df['datetime'].dt.is_month_start).astype(float)

    # Normalize hours_remaining to 0-1 range
    df['hours_remaining_norm'] = df['hours_remaining'] / 23.0

    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features (observed only variables).

    Features for near-real-time prediction:
    - prev_eod_balance: Yesterday's EOD balance (key baseline)
    - balance_start_of_day: Balance at start of current day
    - current_balance: Balance right now (at this hour)
    - net_today_so_far: Net change today up to this hour
    - inflow_today_so_far, outflow_today_so_far
    - transactions_today_so_far
    - Rolling averages of daily EOD balances (7-day, 30-day)
    - Rolling averages of daily inflows/outflows

    Args:
        df: DataFrame with hourly time series data

    Returns:
        DataFrame with added lag and rolling features
    """
    df = df.copy()
    df = df.sort_values(['account_num', 'datetime']).reset_index(drop=True)

    # Get daily EOD balances for rolling calculations
    daily_eod = df[df['hour'] == 23][['account_num', 'date', 'eod_balance']].copy()

    # Rolling averages of EOD balance (7-day, 30-day) - shifted by 1 to avoid leakage
    for window in [7, 30]:
        rolling_col = f'eod_rolling_{window}'
        daily_eod[rolling_col] = daily_eod.groupby('account_num')['eod_balance'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    # Daily inflows/outflows for rolling
    daily_flows = df.groupby(['account_num', 'date']).agg({
        'inflow_today_so_far': 'max',  # Max = total for day
        'outflow_today_so_far': 'max',
    }).reset_index()
    daily_flows.columns = ['account_num', 'date', 'daily_inflow', 'daily_outflow']

    # Rolling inflow/outflow (7-day)
    daily_flows['inflow_rolling_7'] = daily_flows.groupby('account_num')['daily_inflow'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    daily_flows['outflow_rolling_7'] = daily_flows.groupby('account_num')['daily_outflow'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )

    # Merge rolling features back to hourly data
    df = df.merge(
        daily_eod[['account_num', 'date', 'eod_rolling_7', 'eod_rolling_30']],
        on=['account_num', 'date'],
        how='left'
    )
    df = df.merge(
        daily_flows[['account_num', 'date', 'inflow_rolling_7', 'outflow_rolling_7']],
        on=['account_num', 'date'],
        how='left'
    )

    # Fill NaN values
    df['eod_rolling_7'] = df['eod_rolling_7'].fillna(df['current_balance'])
    df['eod_rolling_30'] = df['eod_rolling_30'].fillna(df['current_balance'])
    df['inflow_rolling_7'] = df['inflow_rolling_7'].fillna(0)
    df['outflow_rolling_7'] = df['outflow_rolling_7'].fillna(0)

    # Difference from rolling average (captures trend)
    df['balance_vs_rolling_7'] = df['current_balance'] - df['eod_rolling_7']
    df['balance_vs_rolling_30'] = df['current_balance'] - df['eod_rolling_30']

    return df


def prepare_data_for_tft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for TFT model by adding time_idx and cleaning up.

    Args:
        df: DataFrame with all features

    Returns:
        DataFrame ready for TFT training
    """
    df = df.copy()

    # Create time_idx per account (continuous integer index for each hour)
    df = df.sort_values(['account_num', 'datetime']).reset_index(drop=True)
    df['time_idx'] = df.groupby('account_num').cumcount()

    # Ensure account_num is numeric
    df['account_num'] = df['account_num'].astype(int)

    # Select and order columns
    columns = [
        'account_num',
        'time_idx',
        'datetime',
        'date',
        'hour',
        # Target (what we want to predict)
        'eod_balance',
        # Known future (calendar features - known at any point)
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_month_sin', 'day_of_month_cos',
        'month_sin', 'month_cos',
        'is_weekend', 'is_month_end', 'is_month_start',
        'hours_remaining_norm',
        # Observed only (only known up to current time - NO DATA LEAKAGE)
        'eod_balance_known',  # CRITICAL: prev_eod for hours 0-22, actual eod for hour 23
        'current_balance',
        'prev_eod_balance',
        'balance_start_of_day',
        'net_today_so_far',
        'inflow_today_so_far',
        'outflow_today_so_far',
        'transactions_today_so_far',
        'days_since_last_transaction',
        'eod_rolling_7', 'eod_rolling_30',
        'inflow_rolling_7', 'outflow_rolling_7',
        'balance_vs_rolling_7', 'balance_vs_rolling_30',
    ]

    df = df[columns]

    return df


def split_data_by_time(
    df: pd.DataFrame,
    train_months: int = 12,
    val_months: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train/val/test sets.

    Args:
        df: Full dataset
        train_months: Number of months for training (~12 months)
        val_months: Number of months for validation (~3 months)
        Remaining months go to test set (~3 months)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    min_date = df['datetime'].min()

    train_end = min_date + pd.DateOffset(months=train_months)
    val_end = train_end + pd.DateOffset(months=val_months)

    train_df = df[df['datetime'] < train_end].copy()
    val_df = df[(df['datetime'] >= train_end) & (df['datetime'] < val_end)].copy()
    test_df = df[df['datetime'] >= val_end].copy()

    # Reset time_idx for each split to be continuous per account
    for split_df in [train_df, val_df, test_df]:
        split_df['time_idx'] = split_df.groupby('account_num').cumcount()

    return train_df, val_df, test_df


def main():
    """Main execution function."""

    # =========================================================================
    # CONFIGURATION - Easy to modify for testing vs production
    # =========================================================================

    # Data generation settings
    NUM_ACCOUNTS = 2          # Number of bank accounts (2 for fast testing, 10+ for production)
    NUM_DAYS = 90             # Days of data (90 = 3 months for testing, 550 = 18 months for production)

    # Train/Val/Test split (in months)
    TRAIN_MONTHS = 1          # 1 month for testing, 12 months for production
    VAL_MONTHS = 1            # 1 month for testing, 3 months for production
    # Remaining goes to test (1 month for testing, 3 months for production)

    # Model architecture
    ENCODER_DAYS = 7          # Lookback period in days (7, 14, or 30 recommended)
    DECODER_HOURS = 24        # Prediction horizon in hours (24 = 1 day ahead)
    HIDDEN_SIZE = 64          # Hidden layer size (64 for testing, 128-256 for production)

    # Training settings
    MAX_EPOCHS = 50           # Max training epochs (50 for testing, 100+ for production)
    BATCH_SIZE = 32           # Batch size
    LEARNING_RATE = 1e-3      # Learning rate
    EARLY_STOPPING_PATIENCE = 10

    # Derived values
    ENCODER_LENGTH = ENCODER_DAYS * 24  # Convert days to hours

    # =========================================================================

    print("=" * 80)
    print("TFT - EOD Bank Account Balance Prediction (HOURLY GRANULARITY)")
    print("=" * 80)
    print("\nThis model predicts EOD balance at ANY HOUR during the day.")
    print("Near-real-time capability: Run prediction every hour to get updated forecast.\n")

    print("Configuration:")
    print(f"   Accounts: {NUM_ACCOUNTS}, Days: {NUM_DAYS} (~{NUM_DAYS//30} months)")
    print(f"   Split: {TRAIN_MONTHS}m train / {VAL_MONTHS}m val / remaining test")
    print(f"   Encoder: {ENCODER_DAYS} days ({ENCODER_LENGTH} hours)")
    print()

    # 1. Generate synthetic transactions
    print("1. Generating synthetic transaction data...")
    transactions_df = generate_synthetic_transactions(
        num_accounts=NUM_ACCOUNTS,
        num_days=NUM_DAYS,
        seed=42,
    )
    print(f"   Generated {len(transactions_df)} transactions")
    print(f"   Accounts: {transactions_df['account_num'].nunique()}")
    print(f"   Date range: {transactions_df['tran_timestamp'].min().date()} to "
          f"{transactions_df['tran_timestamp'].max().date()}")

    # 2. Convert to HOURLY time series
    print("\n2. Converting transactions to HOURLY time series...")
    hourly_df = transactions_to_hourly_series(transactions_df)
    print(f"   Created {len(hourly_df)} hourly records")
    print(f"   Records per day: 24 hours × {hourly_df['account_num'].nunique()} accounts = "
          f"{24 * hourly_df['account_num'].nunique()} per day")

    # Show sample of hourly data for one day (pick a day from middle of dataset)
    unique_dates = hourly_df[hourly_df['account_num'] == 0]['date'].unique()
    sample_date = unique_dates[min(10, len(unique_dates) - 1)]  # Day 10 or last available
    sample_day = hourly_df[(hourly_df['account_num'] == 0) &
                           (hourly_df['date'] == sample_date)].copy()
    print(f"\n   Sample day (Account 0, {sample_day['date'].iloc[0]}):")
    print(f"   Hour 0:  current_balance={sample_day[sample_day['hour']==0]['current_balance'].values[0]:.0f}")
    print(f"   Hour 12: current_balance={sample_day[sample_day['hour']==12]['current_balance'].values[0]:.0f}")
    print(f"   Hour 23: current_balance={sample_day[sample_day['hour']==23]['current_balance'].values[0]:.0f}")
    print(f"   EOD target (all hours): {sample_day['eod_balance'].iloc[0]:.0f}")

    # Show data leakage prevention
    print(f"\n   DATA LEAKAGE PREVENTION:")
    print(f"   Hour 0:  eod_balance_known={sample_day[sample_day['hour']==0]['eod_balance_known'].values[0]:.0f} (= prev day's EOD)")
    print(f"   Hour 12: eod_balance_known={sample_day[sample_day['hour']==12]['eod_balance_known'].values[0]:.0f} (= prev day's EOD)")
    print(f"   Hour 23: eod_balance_known={sample_day[sample_day['hour']==23]['eod_balance_known'].values[0]:.0f} (= today's EOD, day complete)")
    print(f"   Model sees eod_balance_known in encoder, NOT today's eod_balance for hours 0-22!")

    # 3. Add time features
    print("\n3. Adding calendar features (known future)...")
    hourly_df = add_time_features(hourly_df)
    print("   Added: hour_sin/cos (KEY for near-real-time)")
    print("   Added: day_of_week_sin/cos, day_of_month_sin/cos, month_sin/cos")
    print("   Added: is_weekend, is_month_end, is_month_start, hours_remaining_norm")

    # 4. Add lag and rolling features
    print("\n4. Adding lag and rolling features (observed only)...")
    hourly_df = add_lag_rolling_features(hourly_df)
    print("   Added: current_balance, prev_eod_balance, balance_start_of_day")
    print("   Added: net/inflow/outflow_today_so_far, transactions_today_so_far")
    print("   Added: eod_rolling_7/30, inflow_rolling_7, outflow_rolling_7")

    # 5. Prepare for TFT
    print("\n5. Preparing data for TFT...")
    tft_df = prepare_data_for_tft(hourly_df)
    print(f"   Total hourly records: {len(tft_df)}")
    print(f"   Features: {len(tft_df.columns)} columns")

    # 6. Split data
    print(f"\n6. Splitting data ({TRAIN_MONTHS}m train, {VAL_MONTHS}m val, rest test)...")
    train_df, val_df, test_df = split_data_by_time(tft_df, train_months=TRAIN_MONTHS, val_months=VAL_MONTHS)
    print(f"   Train: {len(train_df)} hourly samples")
    print(f"   Val:   {len(val_df)} hourly samples")
    print(f"   Test:  {len(test_df)} hourly samples")

    # 7. Create TFT configuration
    print("\n7. Creating TFT configuration...")
    config = TFTConfig(
        # Input features
        static_variables=['account_num'],
        known_future=[
            # Hour features - CRITICAL for near-real-time
            'hour_sin', 'hour_cos',
            'hours_remaining_norm',
            # Calendar features
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'month_sin', 'month_cos',
            'is_weekend', 'is_month_end', 'is_month_start',
        ],
        observed_only=[
            # CRITICAL: eod_balance_known prevents data leakage!
            # For hours 0-22: uses prev_eod_balance (yesterday's EOD)
            # For hour 23: uses actual eod_balance (day complete)
            'eod_balance_known',
            # Current state
            'current_balance',
            'prev_eod_balance',
            'balance_start_of_day',
            # Today's activity so far
            'net_today_so_far',
            'inflow_today_so_far',
            'outflow_today_so_far',
            'transactions_today_so_far',
            # Historical patterns
            'days_since_last_transaction',
            'eod_rolling_7', 'eod_rolling_30',
            'inflow_rolling_7', 'outflow_rolling_7',
            'balance_vs_rolling_7', 'balance_vs_rolling_30',
        ],
        target='eod_balance',

        # Sequence lengths (in hours) - configured at top of main()
        encoder_length=ENCODER_LENGTH,  # ENCODER_DAYS × 24 hours lookback
        decoder_length=DECODER_HOURS,   # Prediction horizon

        # Architecture
        hidden_size=HIDDEN_SIZE,
        num_heads=4,
        num_lstm_layers=2,
        dropout=0.1,

        # Quantiles for probabilistic forecasting
        quantiles=[0.1, 0.5, 0.9],

        # Training parameters - configured at top of main()
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=1.0,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    print(f"   Encoder length: {config.encoder_length} hours ({config.encoder_length//24} days)")
    print(f"   Decoder length: {config.decoder_length} hours ({config.decoder_length//24} day)")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Quantiles: {config.quantiles}")

    # 8. Create data loaders
    print("\n8. Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        config=config,
        batch_size=config.batch_size,
        num_workers=0,
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    # 9. Create model
    print("\n9. Creating TFT model...")
    print_device_info()
    device = get_device("auto")

    model = TemporalFusionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")

    # 10. Setup trainer
    print("\n10. Setting up trainer...")
    trainer = TFTTrainer(
        model=model,
        config=config,
        device=device,
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            min_delta=1e-4,
            verbose=True,
        ),
        ModelCheckpoint(
            filepath='eod_balance_hourly_model.pth',
            monitor='val_loss',
            save_best_only=True,
            verbose=True,
        ),
    ]

    # 11. Train model
    print("\n11. Training model...")
    print("-" * 80)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.max_epochs,
        callbacks=callbacks,
    )
    print("-" * 80)

    # 12. Evaluate on test set
    print("\n12. Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print("   Test metrics:")
    for metric, value in test_metrics.items():
        print(f"   - {metric}: {value:.6f}")

    # 13. Generate predictions with interpretability
    print("\n13. Generating predictions with interpretability outputs...")
    results = trainer.predict(
        test_loader,
        return_attention=True,
        return_variable_selection=True,
    )

    predictions = results['predictions']
    targets = results['targets']
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Targets shape: {targets.shape}")

    # 14. Analyze variable importance
    if 'encoder_variable_selection' in results:
        print("\n14. Analyzing variable importance...")
        encoder_weights = results['encoder_variable_selection']
        mean_weights = encoder_weights.mean(axis=(0, 1))

        encoder_vars = config.observed_only + [config.target]

        print("   Encoder variable importance (average selection weights):")
        var_importance = sorted(zip(encoder_vars, mean_weights), key=lambda x: -x[1])
        for var, weight in var_importance[:10]:  # Top 10
            print(f"   - {var}: {weight:.4f}")

    # 15. Visualize results
    print("\n15. Visualizing results...")

    plot_training_history(
        trainer.history,
        save_path='eod_hourly_training_history.png',
    )
    print("   Saved: eod_hourly_training_history.png")

    for sample_idx in [0, 1, 2]:
        if sample_idx < len(predictions):
            plot_predictions(
                predictions=predictions[sample_idx],
                targets=targets[sample_idx],
                quantiles=config.quantiles,
                title=f"EOD Balance Predictions (Hourly) - Sample {sample_idx}",
                save_path=f'eod_hourly_predictions_sample_{sample_idx}.png',
            )
            print(f"   Saved: eod_hourly_predictions_sample_{sample_idx}.png")

    # 16. Summary
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nSummary:")
    print(f"- Total hourly records: {len(tft_df)}")
    print(f"- Encoder: {config.encoder_length} hours ({config.encoder_length//24} days)")
    print(f"- Decoder: {config.decoder_length} hours")
    print(f"- Final train loss: {trainer.history['train_loss'][-1]:.6f}")
    print(f"- Final val loss:   {trainer.history['val_loss'][-1]:.6f}")
    print(f"- Test loss:        {test_metrics['loss']:.6f}")

    print("\n" + "=" * 80)
    print("NEAR-REAL-TIME PREDICTION USAGE")
    print("=" * 80)
    print("""
At any hour during the day, you can predict the EOD balance:

1. Prepare current hour's data with:
   - current_balance: Balance right now
   - inflow_today_so_far: Sum of today's deposits up to now
   - outflow_today_so_far: Sum of today's withdrawals up to now
   - hour_sin/cos: Current hour (e.g., 14:00 = hour 14)
   - hours_remaining_norm: (23 - current_hour) / 23

2. Run inference:
   model.load_state_dict(torch.load('eod_balance_hourly_model.pth'))
   results = trainer.predict(current_hour_loader)

   # Get prediction for current hour
   predicted_eod = results['predictions'][0, 0, 1]  # Median prediction
   lower_bound = results['predictions'][0, 0, 0]    # 10th percentile
   upper_bound = results['predictions'][0, 0, 2]    # 90th percentile

3. Example outputs:
   "At 2:00 PM, predicted EOD balance: $15,234 (90% CI: $14,800 - $15,700)"
   "At 5:00 PM, predicted EOD balance: $14,950 (90% CI: $14,700 - $15,200)"

   As the day progresses, predictions become more accurate since more
   of the day's transactions are known.
""")


if __name__ == '__main__':
    main()
