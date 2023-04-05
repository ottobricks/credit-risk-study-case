---
jupytext:
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

# Exploring Anecdotal Cases

```{contents}
:local:
```

+++

## Retailers who misse the first collection attempt
In the Loans dataset, there are a few cases when a retailer is not able to pay back the money they borrowed in the first collection attempt. That does not necessarily mean they will default on their payment, perhaps it's just a delay. Let's take a look at the retailer with the most delayed-but-paid loans:

```{code-cell} ipython3
loans_df
# 
```

```{code-cell} ipython3
import pandas as pd


try:
    _ = loans_df
except NameError:
    loans_df = pd.read_excel("../../data/Loans_Data.xlsx")

(
    loans_df
    # .query("FIRST_TRIAL_BALANCE < 0 and PAYMENT_STATUS == 'Paid'")
    .groupby("MAIN_SYSTEM_ID")
    .agg(
        nunique_loan_id=("LOAN_ID", "nunique"),
        nunique_loan_id_delayed_payment=(
            "LOAN_ID",
            lambda df: df[(loans_df["FIRST_TRIAL_BALANCE"] < 0) & (loans_df["PAYMENT_STATUS"] == "Paid")].nunique()
            # the above is very inefficient compute but I'm a bit rusty with Pandas
        ),
    )
    .sort_values("nunique_loan_id_delayed_payment", ascending=False)
    .head(5)
)
```

Let's investigate retailer `MAIN_SYSTEM_ID == 58316`, first looking at all 10 loans

---

## MISC

```{code-cell} ipython3
loans_df.query("MAIN_SYSTEM_ID == 58316")
```

```{code-cell} ipython3
import pandas as pd

try:
    _ = loans_df
except NameError:
    loans_df = pd.read_excel("../../data/Loans_Data.xlsx")

# Are there any LOAN_IDs that have multiple observations, i.e. first collection attempt, second, etc. ?
(
    loans_df.query("FIRST_TRIAL_BALANCE < 0")
    .groupby(["MAIN_SYSTEM_ID", "LOAN_ID"])
    [["REPAYMENT_UPDATED"]]
    .nunique()
    .query("REPAYMENT_UPDATED > 1")
)
```

?? If `FIRST_TRIAL_BALANCE < 0`, the retailer missed first collection attempt. Can we assume that retailers are given only a second collection attempt before being marked as `PAYMENT_STATUS in ('Unpaid', 'Partialy paid')`?

I'm starting to believe I was given a snapshot of the Loans dataset instead of time series as for the other 2 datasets. We can check that by looking at the following case: retailer misses the first payment but is able to fully pay on following collection attempt. We can represent that as `FIRST_TRIAL_BALANCE < 0 and PAYMENT_STATUS = 'Paid'`, then check if any `LOAN_ID` has more than one observation.

```{code-cell} ipython3
import pandas as pd

try:
    _ = loans_df
except NameError:
    loans_df = pd.read_excel("../../data/Loans_Data.xlsx")

(
    loans_df.query("FIRST_TRIAL_BALANCE < 0 and PAYMENT_STATUS == 'Paid'")
    .sort_values(by=["MAIN_SYSTEM_ID", "LOAN_ID"])
    [["MAIN_SYSTEM_ID", "LOAN_ID", "REPAYMENT_ID", "REPAYMENT_AMOUNT", "TOTAL_FINAL_AMOUNT"]]
)
```

```{code-cell} ipython3

```
