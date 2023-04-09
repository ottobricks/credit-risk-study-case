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

**SAFE_DECISION** is when:
 - model predicts `SPENT` to be within 1 stddev of Ecommerce 30-days mean
 - `LOAN_AMOUNT` does not exceed 1 stddev of Ecommerce 30-days mean

**AMBITIOUS_DECISION** is when:
 - model predicts `SPENT` to be within 2 stddev of Ecommerce 30-days mean
 - `LOAN_AMOUNT` does not exceed 2 stddev of Ecommerce 30-days mean

```{code-cell} ipython3
(
    merged_df.assign(
        SAFE_DECISION=(
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
        "SAFE_DECISION",
        "AMBITIOUS_DECISION"
    ]]
    .to_csv("../../risk-assessment-decisions.csv", header=True, index=False)
)
```

```{code-cell} ipython3

```
