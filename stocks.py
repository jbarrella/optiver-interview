import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.stats import norm
from datetime import datetime
import sys

USAGE = (
    "Usage: python_start.py <QUESTION> <SUB_QUESTION>\n\n"
    "QUESTION is one of: stocks, options\n"
    "for stocks, SUB_QUESTION is one of: 1, 2, 3\n"
    "for options, SUB_QUESTION is one of: 1, 2"
)

DB_PATH = "sqlite:///trading.db"


def load_data(engine):
    df_trades = pd.read_sql("SELECT * FROM trades", engine, index_col="index")
    # convert Time column to datetime type.
    df_trades["Time"] = pd.to_datetime(df_trades["Time"], dayfirst=True)
    # Ensure the data frame is sorted by time.
    df_trades = df_trades.sort_values(by=["Time"])

    df_pos = pd.read_sql("SELECT * FROM position", engine, index_col="index")

    return df_trades, df_pos


def calc_moving_averages(trades):
    # Get list of unique stocks.
    stocks = np.unique(trades["Stock"].values)

    sma = {}
    for stock in stocks:
        # Select trades matching stock in for loop
        df_stock_trades = trades.loc[trades["Stock"] == stock].copy()

        # Take the mean of each 5-min group of trades based on "Time" column and store
        # in new "SMA" column.
        df_stock_trades["SMA"] = (
            df_stock_trades[["Time", "PRICE"]]
            .rolling("5min", on="Time")
            .mean()["PRICE"]
        )

        sma[stock] = df_stock_trades

    # Return dictionary of data frames with new SMA column for each stock.
    return sma


def analyze_sma(sma):
    # Find the five rows with the highest and lowest SMA over all time.
    stats = {}

    for stock in sma.keys():
        # Sort data frame by SMA.
        sma[stock] = sma[stock].sort_values(by="SMA", ascending=False)
        # Select Time and SMA columns for first five rows.
        top_five = sma[stock].iloc[:5, :][["Time", "SMA"]]
        # Select last five rows and reverse to get lowest first.
        bottom_five = sma[stock].iloc[:-6:-1, :][["Time", "SMA"]]

        stats[stock] = {"top_five": top_five, "bottom_five": bottom_five}

    return stats


def plot_sma(sma):
    # Merge dictionary of data frames into single data frame.
    merged = pd.concat(sma.values())

    # Plot SMA vs. Time for each stock in merged data frame.
    pd.pivot_table(merged, index="Time", columns="Stock", values="SMA").plot(
        # Show all plots in one figure with 3 rows and 2 columns.
        subplots=True,
        layout=(3, 2),
        figsize=(12, 8),
        title="5 Minute Simple Moving Average (€)",
    )

    plt.show()


def query_trade_info(engine):
    query = """
    SELECT
        Stock,
        COUNT(*) as Trades,
        AVG(PRICE) as Average,
        SUM(PRICE * SIZE) / SUM(SIZE) as Weighted_Average,
        SUM(CASE WHEN BUY_SELL_FLAG = 1 THEN PRICE * SIZE ELSE 0 END) / SUM(CASE WHEN BUY_SELL_FLAG = 1 THEN SIZE ELSE 0 END) as Buy_Weighted_Average,
        SUM(CASE WHEN BUY_SELL_FLAG = 0 THEN PRICE * SIZE ELSE 0 END) / SUM(CASE WHEN BUY_SELL_FLAG = 0 THEN SIZE ELSE 0 END) as Sell_Weighted_Average
    FROM trades
    GROUP BY
        Stock
    """

    stock_stats = pd.read_sql(query, engine)

    return stock_stats


def calc_profit_loss(trades):
    stocks = np.unique(trades["Stock"].values)

    pl = {}
    for stock in stocks:
        df_stock_trades = trades.loc[trades["Stock"] == stock].copy()

        # Get the current total position cost at the point of each trade. Store
        # the cummulative sum of all previous rows in Total_Position_Cost column.
        df_stock_trades["Total_Position_Cost"] = df_stock_trades.apply(
            lambda x: x["SIZE"] * x["PRICE"]
            if x["BUY_SELL_FLAG"] == 1
            else -x["SIZE"] * x["PRICE"],
            axis=1,
        ).cumsum()

        # Get the current total position size at the point of each trade.
        df_stock_trades["Total_Position_Size"] = df_stock_trades.apply(
            lambda x: x["SIZE"] if x["BUY_SELL_FLAG"] == 1 else -x["SIZE"],
            axis=1,
        ).cumsum()

        # Vectorized calculation of new Profit/Loss (PL) column.
        df_stock_trades["PL"] = (
            df_stock_trades["Total_Position_Size"] * df_stock_trades["PRICE"]
            - df_stock_trades["Total_Position_Cost"]
        )

        # Return dictionary of data frames with new PL column for each stock.
        pl[stock] = df_stock_trades

    return pl


def analyze_profit_loss(pl):
    # Find the stock with highest total profit at latest time.
    best_pl = 0
    for stock in pl.keys():
        final_pl = pl[stock].iloc[-1]["PL"]
        if final_pl > best_pl:
            best_pl = final_pl
            best_stock = stock

    return best_stock, best_pl


def plot_profit_loss(pl):
    merged = pd.concat(pl.values())

    pd.pivot_table(merged, index="Time", columns="Stock", values="PL").plot(
        subplots=True, layout=(3, 2), title="Profit/Loss (€)", figsize=(14, 8)
    )

    plt.show()


def parse_option(option, current_date):
    # Decode option string into parts
    volatility = option["Volatility(%)"] / 100
    stock, option_code = option["Instrument"].split(" ")
    option_type = option_code[0]
    expiry_date = datetime.strptime(option_code[1:-7], "%Y%m")
    time_to_maturity = (expiry_date - current_date).days / 365
    strike = float(option_code[-5:]) / 1000

    option_data = {
        "strike": strike,
        "type": option_type,
        "volatility": volatility,
        "time_to_maturity": time_to_maturity,
    }

    return option_data


def price_option(option, current_date, current_price, interest):
    def d1(S, K, t, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    def d2(S, K, t, r, sigma):
        return d1(S, K, t, r, sigma) - sigma * np.sqrt(t)

    def price_call(S, K, t, r, sigma):
        return S * norm.cdf(d1(S, K, t, r, sigma)) - K * np.exp(-r * t) * norm.cdf(
            d2(S, K, t, r, sigma)
        )

    def price_put(S, K, t, r, sigma):
        return K * np.exp(-r * t) - S + price_call(S, K, t, r, sigma)

    option_data = parse_option(option, current_date)

    args = [
        current_price,
        option_data["strike"],
        option_data["time_to_maturity"],
        interest,
        option_data["volatility"],
    ]

    if option_data["type"] == "C":
        return price_call(*args)
    else:
        return price_put(*args)


def price_options(positions):
    current_date = datetime(2017, 6, 1)
    current_price = 16.5
    interest = 0

    # Apply the price_option() function to each entry in the data frame.
    positions["Price"] = positions.apply(
        lambda x: price_option(x, current_date, current_price, interest), axis=1
    )

    return positions


def calc_greek(greek, option, current_date, current_price, interest):
    def d1(S, K, t, r, sigma):
        return (np.log(S / K) + (r + sigma ** 2 * 0.5) * t) / (sigma * np.sqrt(t))

    def d2(S, K, t, r, sigma):
        return d1(S, K, t, r, sigma) - sigma * np.sqrt(t)

    def call_gamma(S, K, t, r, sigma):
        return norm.pdf(d1(S, K, t, r, sigma)) / (S * sigma * np.sqrt(t))

    def call_delta(S, K, t, r, sigma):
        return norm.cdf(d1(S, K, t, r, sigma))

    def call_theta(S, K, t, r, sigma):
        return 0.01 * (
            -(S * norm.pdf(d1(S, K, t, r, sigma)) * sigma) / (2 * np.sqrt(t))
            - r * K * np.exp(-r * t) * norm.cdf(d2(S, K, t, r, sigma))
        )

    def call_vega(S, K, t, r, sigma):
        return 0.01 * (S * norm.pdf(d1(S, K, t, r, sigma)) * np.sqrt(t))

    def put_gamma(S, K, t, r, sigma):
        return norm.pdf(d1(S, K, t, r, sigma)) / (S * sigma * np.sqrt(t))

    def put_delta(S, K, t, r, sigma):
        return -norm.cdf(-d1(S, K, t, r, sigma))

    def put_theta(S, K, t, r, sigma):
        return 0.01 * (
            -(S * norm.pdf(d1(S, K, t, r, sigma)) * sigma) / (2 * np.sqrt(t))
            + r * K * np.exp(-r * t) * norm.cdf(-d2(S, K, t, r, sigma))
        )

    def put_vega(S, K, t, r, sigma):
        return 0.01 * (S * norm.pdf(d1(S, K, t, r, sigma)) * np.sqrt(t))

    option_data = parse_option(option, current_date)
    args = [
        current_price,
        option_data["strike"],
        option_data["time_to_maturity"],
        interest,
        option_data["volatility"],
    ]

    if option_data["type"] == "C":
        if greek == "gamma":
            value = call_gamma(*args)
        elif greek == "delta":
            value = call_delta(*args)
        elif greek == "theta":
            value = call_theta(*args)
        elif greek == "vega":
            value = call_vega(*args)
    else:
        if greek == "gamma":
            value = put_gamma(*args)
        elif greek == "delta":
            value = put_delta(*args)
        elif greek == "theta":
            value = put_theta(*args)
        elif greek == "vega":
            value = put_vega(*args)

    value *= option["Total Position"]

    return value


def calc_greeks(positions):
    current_price = 16.5
    interest = 0
    current_date = datetime(2017, 6, 1)

    gamma = positions.apply(
        lambda x: calc_greek("gamma", x, current_date, current_price, interest), axis=1
    ).sum()
    delta = positions.apply(
        lambda x: calc_greek("delta", x, current_date, current_price, interest), axis=1
    ).sum()
    theta = positions.apply(
        lambda x: calc_greek("theta", x, current_date, current_price, interest), axis=1
    ).sum()
    vega = positions.apply(
        lambda x: calc_greek("vega", x, current_date, current_price, interest), axis=1
    ).sum()

    greeks = {
        "gamma": gamma,
        "delta": delta,
        "theta": theta,
        "vega": vega,
    }

    return greeks


def stocks(sub_question, engine, trades):
    # Interface for stocks sub-questions.
    if sub_question == "1":
        sma = calc_moving_averages(trades)
        stats = analyze_sma(sma)
        # Display highest and lowest SMAs
        for stock in stats.keys():
            print("{}:".format(stock))
            for hi_low, values in stats[stock].items():
                print("{}:\n{}".format(hi_low, values))
            print("")
        plot_sma(sma)
    elif sub_question == "2":
        df_query_results = query_trade_info(engine)
        print(df_query_results)
    elif sub_question == "3":
        pl = calc_profit_loss(trades)
        best_stock, best_pl = analyze_profit_loss(pl)
        print(
            "Most profitable stock:\nName: {}\nPL: €{}".format(
                best_stock, best_pl.round(2)
            )
        )
        plot_profit_loss(pl)
    else:
        sys.exit(USAGE)


def options(sub_question, positions):
    # Interface for options sub-questions.
    if sub_question == "1":
        df_priced_options = price_options(positions)
        print(df_priced_options)
    elif sub_question == "2":
        greeks = calc_greeks(positions)
        print(greeks)
    else:
        sys.exit(USAGE)


def main():
    # Top-level interface to questions and sub-questions
    if len(sys.argv) != 3:
        # Selections for question and sub-question are not defined.
        sys.exit(USAGE)

    engine = create_engine(DB_PATH, echo=False)
    # Load SQL tables
    trades, positions = load_data(engine)

    question, sub_question = sys.argv[1:]
    if question == "stocks":
        stocks(sub_question, engine, trades)
    elif question == "options":
        options(sub_question, positions)
    else:
        # Argument does not match any question.
        sys.exit(USAGE)


if __name__ == "__main__":
    main()
