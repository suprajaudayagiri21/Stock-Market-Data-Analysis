# Stock Market Data Analysis System
# Based on the DAE project PPT
# Run: python stock_analysis.py
# Expected CSV columns: Date, Open, High, Low, Close, Volume
# Optional column: Stock / Ticker

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "stock_market_data.csv"


def load_dataset(file_path):
    """Module 1: Load the stock dataset."""
    df = pd.read_csv(file_path)
    print("\n--- Dataset loaded successfully ---")
    return df


def explore_dataset(df):
    """Module 2: Explore the dataset structure."""
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Dataset shape ---")
    print(df.shape)
    print("\n--- Column names ---")
    print(df.columns.tolist())
    print("\n--- Data types ---")
    print(df.dtypes)
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    print("\n--- Summary statistics ---")
    print(df.describe(include="all"))


def clean_and_preprocess(df):
    """Modules 3 and 4: Clean missing/duplicate records and prepare dates."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nDuplicate rows removed: {before - len(df)}")

    # Normalize common column names
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace("_", " ").replace("-", " ")
        if key in {"date", "trading date"}:
            rename_map[col] = "Date"
        elif key in {"open", "opening price"}:
            rename_map[col] = "Open"
        elif key in {"high", "high price"}:
            rename_map[col] = "High"
        elif key in {"low", "low price"}:
            rename_map[col] = "Low"
        elif key in {"close", "closing price", "adj close", "adjusted close"}:
            # Prefer a column already named Close if both are present.
            if "Close" not in rename_map.values():
                rename_map[col] = "Close"
        elif key in {"volume", "trading volume"}:
            rename_map[col] = "Volume"
        elif key in {"stock", "ticker", "symbol", "company"}:
            rename_map[col] = "Stock"

    df = df.rename(columns=rename_map)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric values with each column's median.
    # This is a simple classroom example; choose a method suitable for your dataset.
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # If there is no usable Close column, stop with a clear message.
    if "Close" not in df.columns:
        raise ValueError(
            "A closing-price column was not found. Please ensure your CSV contains "
            "'Close' or 'Closing Price' and update the column mapping if needed."
        )

    # Drop rows still missing essential values
    essential = [c for c in ["Date", "Close"] if c in df.columns]
    df = df.dropna(subset=essential)

    if "Date" in df.columns:
        sort_cols = ["Date"]
        if "Stock" in df.columns:
            sort_cols = ["Stock", "Date"]
        df = df.sort_values(sort_cols)

    # Calculate daily price change and percentage change
    group_key = "Stock" if "Stock" in df.columns else None
    if group_key:
        df["Price Change"] = df.groupby(group_key)["Close"].diff()
        df["Daily Return (%)"] = df.groupby(group_key)["Close"].pct_change() * 100
    else:
        df["Price Change"] = df["Close"].diff()
        df["Daily Return (%)"] = df["Close"].pct_change() * 100

    print("\n--- Cleaned dataset ---")
    print(f"Rows after cleaning: {len(df)}")
    print(df.head())
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    return df


def analyze_data(df):
    """Modules 5 and 6: Perform exploratory and stock-market analysis."""
    print("\n--- Stock market analysis ---")
    if "Stock" in df.columns:
        print("\nRecords per stock:")
        print(df.groupby("Stock").size())
        print("\nAverage closing price by stock:")
        print(df.groupby("Stock")["Close"].mean().sort_values(ascending=False))
    else:
        print(f"Average closing price: {df['Close'].mean():.2f}")
        print(f"Highest closing price: {df['Close'].max():.2f}")
        print(f"Lowest closing price: {df['Close'].min():.2f}")

    print(f"\nLatest closing price: {df['Close'].iloc[-1]:.2f}")
    if "Volume" in df.columns:
        print(f"Total trading volume: {df['Volume'].sum():,.0f}")
        print(f"Average trading volume: {df['Volume'].mean():,.2f}")

    if "Daily Return (%)" in df.columns:
        returns = df["Daily Return (%)"].dropna()
        if not returns.empty:
            print(f"Average daily return: {returns.mean():.3f}%")
            print(f"Daily return standard deviation: {returns.std():.3f}%")


def create_visualizations(df):
    """Modules 7 and 8: Create line charts, bar charts, histograms and heatmaps."""
    sns.set_theme()

    # Closing-price line chart
    plt.figure(figsize=(11, 5))
    if "Date" in df.columns:
        if "Stock" in df.columns:
            for stock, group in df.groupby("Stock"):
                plt.plot(group["Date"], group["Close"], label=str(stock))
            plt.legend()
            plt.xlabel("Date")
        else:
            plt.plot(df["Date"], df["Close"])
            plt.xlabel("Date")
    else:
        plt.plot(df["Close"].to_numpy())
        plt.xlabel("Record")
    plt.title("Stock Closing Price Trend")
    plt.ylabel("Closing Price")
    plt.tight_layout()
    plt.savefig("closing_price_trend.png", dpi=150)
    plt.show()

    # Trading-volume chart
    if "Volume" in df.columns:
        plt.figure(figsize=(10, 5))
        if "Stock" in df.columns:
            volume_by_stock = df.groupby("Stock")["Volume"].sum().sort_values(ascending=False)
            volume_by_stock.plot(kind="bar")
            plt.xlabel("Stock")
        else:
            x = df["Date"] if "Date" in df.columns else np.arange(len(df))
            plt.bar(x, df["Volume"])
            plt.xlabel("Date" if "Date" in df.columns else "Record")
        plt.title("Trading Volume Analysis")
        plt.ylabel("Volume")
        plt.tight_layout()
        plt.savefig("trading_volume.png", dpi=150)
        plt.show()

    # Closing-price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Close"].dropna(), kde=True)
    plt.title("Distribution of Closing Prices")
    plt.xlabel("Closing Price")
    plt.tight_layout()
    plt.savefig("closing_price_histogram.png", dpi=150)
    plt.show()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=150)
        plt.show()


def main():
    try:
        df = load_dataset(DATA_FILE)
        explore_dataset(df)
        cleaned_df = clean_and_preprocess(df)
        analyze_data(cleaned_df)
        create_visualizations(cleaned_df)
        cleaned_df.to_csv("cleaned_stock_market_data.csv", index=False)
        print("\nDone! Saved cleaned data and charts in the current folder.")
        print("Note: This project analyzes historical data; it does not guarantee future stock prices.")
    except FileNotFoundError:
        print(f"File not found: {DATA_FILE}")
        print("Place your dataset CSV in the same folder as this Python file, or update DATA_FILE.")
    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
