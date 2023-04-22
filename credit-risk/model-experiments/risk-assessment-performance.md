---
jupytext:
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: credit-risk
  language: python
  name: python3
---

# Results

```{code-cell} ipython3
import pickle
from flaml import AutoML
import pandas as pd
import numpy as np

```

## Load data

```{code-cell} ipython3
ecommerce_df = (
    pd.read_csv(
        "../../data/Ecommerce_orders_Data.csv",
        header=0,
        dtype={
            "ORDER_ID": np.dtype("int64"),
            "MAIN_SYSTEM_ID": np.dtype("int64"),
            "ORDER_PRICE": np.dtype("float64"),
            "DISCOUNT": np.dtype("float64"),
            "ORDER_PRICE_AFTER_DISCOUNT": np.dtype("float64"),
            "ORDER_CREATION_DATE": np.dtype("O"),
        },
    )
    .assign(
        ORDER_CREATION_DATE=lambda x: pd.to_datetime(
            x["ORDER_CREATION_DATE"], infer_datetime_format=True
        ),
    )
    .drop(["DISCOUNT", "ORDER_PRICE_AFTER_DISCOUNT"], axis=1)
    .sort_values("ORDER_CREATION_DATE")
)

ecommerce_df = (
    ecommerce_df.assign(
        ROLLINGMEAN_30DAYS__ORDER_PRICE__=(
            lambda x: x.groupby("MAIN_SYSTEM_ID")
            .rolling("30D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
            .mean()
            .reset_index(drop=True)
        ),
        ROLLINGSTDDEV_30DAYS__ORDER_PRICE__=(
            lambda x: x.groupby("MAIN_SYSTEM_ID")
            .rolling("30D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
            .std()
            .reset_index(drop=True)
        ),
    )[
        [
            "MAIN_SYSTEM_ID",
            "ORDER_CREATION_DATE",
            "ORDER_PRICE",
            "ROLLINGMEAN_30DAYS__ORDER_PRICE__",
            "ROLLINGSTDDEV_30DAYS__ORDER_PRICE__",
        ]
    ]
    .sort_values("ORDER_CREATION_DATE")
    .reset_index(drop=True)
)
ecommerce_df.info()

```

```{code-cell} ipython3
metadata_cols = ["LOAN_ID", "LOAN_ISSUANCE_DATE", "LOAN_AMOUNT"]
loans_df = (
    pd.read_excel("../../data/Loans_Data.xlsx")
    .drop(
        [
            # Low signal columns
            "INITIAL_COST",
            "INDEX",
            "REPAYMENT_ID",
            "FINAL_COST",
            "RETAILER_ID",
            # Columns populated after the fact, thus would lead to data leak
            "REPAYMENT_UPDATED",
            "SPENT",
            "TOTAL_FINAL_AMOUNT",
            "FIRST_TRIAL_BALANCE",
            "FIRST_TRAIL_DELAYS",
            "PAYMENT_AMOUNT",
            "LOAN_PAYMENT_DATE",
            "REPAYMENT_AMOUNT",
            "CUMMULATIVE_OUTSTANDING",
            "PAYMENT_STATUS",
        ],
        axis=1,
    )
    .assign(
        MAIN_SYSTEM_ID=lambda x: x["MAIN_SYSTEM_ID"].astype("int"),
        LOAN_ID=lambda x: x["LOAN_ID"].astype("int64"),
        LOAN_ISSUANCE_DATE=lambda x: x["LOAN_ISSUANCE_DATE"].astype("<M8[ns]"),
        LOAN_AMOUNT=lambda x: x["LOAN_AMOUNT"].astype("float64"),
        TOTAL_INITIAL_AMOUNT=lambda x: x["TOTAL_INITIAL_AMOUNT"].astype("float64"),
        INITIAL_DATE=lambda x: x["INITIAL_DATE"].astype("<M8[ns]"),
    )
)

test_df = pd.merge(
    loans_df[metadata_cols],
    pd.read_parquet("../../data/test/pandas-pca-featureframe-maxdepth2-targetSPENT.parquet").assign(MAIN_SYSTEM_ID=lambda x: x["MAIN_SYSTEM_ID"].astype("int")),
    on="LOAN_ID"
).drop_duplicates()

test_df.info()

```

## Load model and get predictions

```{code-cell} ipython3
with open("assets/sagemaker-flaml-automl-regression-maxdepth2-targetSPENT-3.pkl", "rb") as fp:
    automl = pickle.load(fp)

test_df = test_df.assign(
    y_pred=automl.predict(test_df.drop(metadata_cols + ["MAIN_SYSTEM_ID", "label"], axis=1))
).sort_values("LOAN_ISSUANCE_DATE")

```

## Join fields to generate final dataframe and report

```{code-cell} ipython3
merged_df = pd.merge_asof(
    test_df,
    ecommerce_df,
    left_on="LOAN_ISSUANCE_DATE",
    right_on="ORDER_CREATION_DATE",
    by="MAIN_SYSTEM_ID",
    direction="backward",
    tolerance=pd.Timedelta("7d"),
).query("LOAN_ISSUANCE_DATE > ORDER_CREATION_DATE")

merged_df.info()

```

## Make decision to approve or deny loans

**CONSERVATIVE_DECISION** is when:
 - model predicts `SPENT` to be within 1 stddev of Ecommerce 30-days mean
 - `LOAN_AMOUNT` does not exceed 1 stddev of Ecommerce 30-days mean

**AMBITIOUS_DECISION** is when:
 - model predicts `SPENT` to be within 2 stddev of Ecommerce 30-days mean
 - `LOAN_AMOUNT` does not exceed 2 stddev of Ecommerce 30-days mean

```{code-cell} ipython3
(
    merged_df.assign(
        CONSERVATIVE_DECISION=(
            (merged_df["y_pred"] <= merged_df["ROLLINGMEAN_30DAYS__ORDER_PRICE__"] + merged_df["ROLLINGSTDDEV_30DAYS__ORDER_PRICE__"])
            & (merged_df["LOAN_AMOUNT"] <= merged_df["ROLLINGMEAN_30DAYS__ORDER_PRICE__"] + merged_df["ROLLINGSTDDEV_30DAYS__ORDER_PRICE__"])
        ),
        AMBITIOUS_DECISION=(
            (merged_df["y_pred"] <= merged_df["ROLLINGMEAN_30DAYS__ORDER_PRICE__"] + 2 * merged_df["ROLLINGSTDDEV_30DAYS__ORDER_PRICE__"])
            & (merged_df["LOAN_AMOUNT"] <= merged_df["ROLLINGMEAN_30DAYS__ORDER_PRICE__"] + 2 * merged_df["ROLLINGSTDDEV_30DAYS__ORDER_PRICE__"])
        ),
    )
    [[
        "LOAN_ID",
        "CONSERVATIVE_DECISION",
        "AMBITIOUS_DECISION"
    ]]
    .to_csv("../../data/risk-assessment-decisions.csv", header=True, index=False)
)

```

## Result Assessment

Results generated from Test Dataset only are more reliable, see (Splitting ML Dataset)[discovery/evaluation.html#splitting-ml-dataset] for more info. However, this means our result assessment is based only on 34,418 observations out of the 73,086 provided, roughly the 50% latest observations.

**Is that a problem?**

There is no such thing as perfect method, but there are those objectively better. We are forced to choose where to place uncertainty:
 - giving models all the data, increasing chance of overfitting (models memorize the data), thus weakening confidence in results
 - training models in "past" data and assessing them on "future" data, reducing the signal available for models to learn, but increasing a lot confidence on results

The latter is objectively better. Think of it like this: is it better to have a funny friend that lies to you all the time, or the awkward one that is always there for you? For a party (i.e. boasting about astonishing performance), you want the funny friend, but what about for life?

Another point of attention is that I will consider `PAYMENT_STATUS in ('Unpaid', 'Partialy paid')` as *defaults*. I will also consider *defaults* all cases when the retailer is not able to make the payment on the first collection attempt. Let me explain. I understand retailers are not always to blame for missed collection attempt, it sometimes falls on our operations agents. This can definitely be addressed in the future with a separate track: optimizing ops agents schedules to maximize collection rates. However, for this case study, I'm going to limit the scope of our assessment, and my judgement is that there is also a component of timing in a retailer's ability to repay. This shifts the objective from being "we eventually want to collect debts" to "our forecast should also optimize for timing". Of course, this goes beyond the scope of a case study, but it is an interesting domain to explore.

With all that in mind, let's first take a look at some broad metrics:

```{code-cell} ipython3
:tags: [hide-input,remove-stdout]
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.types as spark_dtype


# Start Spark session just because I feel more comfortable with its API rather than Pandas
spark = SparkSession.builder.getOrCreate()

# Load Decisions and Loans datasets
decision_test_observations: DataFrame = (
    spark.read.option("header", "true")
    .schema(
        spark_dtype.StructType(
            [
                spark_dtype.StructField("LOAN_ID", spark_dtype.LongType(), nullable=False),
                spark_dtype.StructField("CONSERVATIVE_DECISION", spark_dtype.BooleanType(), nullable=False),
                spark_dtype.StructField("AMBITIOUS_DECISION", spark_dtype.BooleanType(), nullable=False),
            ]
        )
    )
    .csv("../../data/risk-assessment-decisions.csv")
)
loan_metadata: DataFrame = spark.createDataFrame(
    pd.read_excel("../../data/Loans_Data.xlsx")
    .drop(
        [
            # Irrelevant columns for the current assessment
            "TOTAL_INITIAL_AMOUNT", # can be inferred from other columns
            "TOTAL_FINAL_AMOUNT", # can be inferred from other columns
            "INDEX",
            "REPAYMENT_ID",
            "RETAILER_ID",
            "REPAYMENT_UPDATED",
            "PAYMENT_AMOUNT",
            "LOAN_PAYMENT_DATE",
            "REPAYMENT_AMOUNT",
            "CUMMULATIVE_OUTSTANDING",

        ],
        axis=1,
    )
    .assign(
        MAIN_SYSTEM_ID=lambda x: x["MAIN_SYSTEM_ID"].astype("int64"),
        LOAN_ID=lambda x: x["LOAN_ID"].astype("int64"),
        LOAN_ISSUANCE_DATE=lambda x: x["LOAN_ISSUANCE_DATE"].astype("<M8[ns]"),
        LOAN_DUE_DATE=lambda x: x["INITIAL_DATE"].astype("<M8[ns]"),
        # Financial info
        LOAN_AMOUNT=lambda x: x["LOAN_AMOUNT"].astype("float64"),
        INITIAL_COST=lambda x: x["INITIAL_COST"].astype("float64"),
        SPENT=lambda x: x["SPENT"].astype("float64"),
        FIRST_TRIAL_BALANCE=lambda x: x["FIRST_TRIAL_BALANCE"].astype("float64"),
        FINAL_COST=lambda x: x["FINAL_COST"].astype("float64"),
        # Label
        LABEL=lambda x: (
            (x["PAYMENT_STATUS"].str.lower != "paid") or (x["FIRST_TRIAL_BALANCE"] >= 0)
        ),
    )
)

```

Ideas, definitely no time to cover all -- choose a couple:
- daily rate of exposure (need to define exposure precisely, maybe stddev above 30-day mean predicted volume)
- weekly rate of exposure vs. realized loss
- recommend daily interest rates based on to cover previous week exposure (if time allows, account for seasonality)
- daily rate of stale capital: LOAN_AMOUNT vs. SPENT
- weekly projected growth vs. realized growth