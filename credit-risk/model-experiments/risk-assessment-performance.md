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

```{warning}
This section is still under development
```

```{code-cell}
import pandas as pd

ecommerce_df = (
    pd.read_csv(
        "data/Ecommerce_orders_Data.csv",
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
            x["ORDER_CREATION_DATE"],
            infer_datetime_format=True
        ),
        ROLLINGMEAN_7DAYS__ORDER_PRICE__=lambda x: x.groupby("MAIN_SYSTEM_ID")["ORDER_PRICE"].rolling("7D", on="ORDER_CREATION_DATE").mean().reset_index(level=0, drop=True)
        ROLLINGMEAN_30DAYS__ORDER_PRICE__=lambda x: x.groupby("MAIN_SYSTEM_ID")["ORDER_PRICE"].rolling("30D", on="ORDER_CREATION_DATE").mean().reset_index(level=0, drop=True)
    )
    .drop(["DISCOUNT", "ORDER_PRICE_AFTER_DISCOUNT"], axis=1)
)

loans_df = (
    pd.read_excel("data/Loans_Data.xlsx")
    .assign(
        MAIN_SYSTEM_ID=lambda x: x["MAIN_SYSTEM_ID"].astype("int64"),
        LOAN_ID=lambda x: x["LOAN_ID"].astype("int64"),
        LOAN_ISSUANCE_DATE=lambda x: x["LOAN_ISSUANCE_DATE"].astype("<M8[ns]"),
        LOAN_AMOUNT=lambda x: x["LOAN_AMOUNT"].astype("float64"),
        TOTAL_INITIAL_AMOUNT=lambda x: x["TOTAL_INITIAL_AMOUNT"].astype("float64"),
        INITIAL_DATE=lambda x: x["INITIAL_DATE"].astype("<M8[ns]"),
    )
)    
```

```{code-cell}

```
