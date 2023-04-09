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

# Test Hypothesis

Before anything else, you must remeber that the objective of testing hypothesis is to force us to clearly state out beliefs (*null hypothesis*) and try our best to prove we are wrong (a.k.a reject the *null hypothesis*). So, in very simple terms, we are using evidence to try refute our own beliefs because we want only the strongest of them to last. After all, if you are your most delligent critic to yourself and you are a good critic, you are unlikely to hold misconceptions for long.

I choose to remove from the Loans dataset all observations where retailers never spent their funds. I will dismiss them as discovery of the product rather than intention to use it. Note that all positive observations (defaults) are still in the dataset after the filter.

```{code-cell} ipython3
:tags: [hide-cell]

import pandas as pd
import numpy as np


loans_df = (
    pd.read_excel("../../data/Loans_Data.xlsx")
    .assign(
        MAIN_SYSTEM_ID=lambda x: x["MAIN_SYSTEM_ID"].astype("int64"),
        LOAN_ID=lambda x: x["LOAN_ID"].astype("int64"),
        LOAN_ISSUANCE_DATE=lambda x: x["LOAN_ISSUANCE_DATE"].astype("<M8[ns]"),
        LOAN_AMOUNT=lambda x: x["LOAN_AMOUNT"].astype("float64"),
        TOTAL_INITIAL_AMOUNT=lambda x: x["TOTAL_INITIAL_AMOUNT"].astype("float64"),
        INITIAL_DATE=lambda x: x["INITIAL_DATE"].astype("<M8[ns]"),
    )
    .query("SPENT > 0")
    .sort_values("LOAN_ISSUANCE_DATE")
)
loans_df.info()
loans_df.value_counts("PAYMENT_STATUS")
```

```{code-cell} ipython3
:tags: [hide-output]

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
ecommerce_df.info()
```

Compute the rolling mean of amount retailer spent in Ecommerce over 7 and 30 days.

```{code-cell} ipython3
ecommerce_df = (
    ecommerce_df.assign(
        ROLLINGMEAN_7DAYS__ORDER_PRICE__=(
            lambda x: x.groupby("MAIN_SYSTEM_ID")
            .rolling("7D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
            .mean()
            .reset_index(drop=True)
        ),
        ROLLINGMEAN_30DAYS__ORDER_PRICE__=(
            lambda x: x.groupby("MAIN_SYSTEM_ID")
            .rolling("30D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
            .mean()
            .reset_index(drop=True)
        ),
        ROLLINGSTDDEV_7DAYS__ORDER_PRICE__=(
            lambda x: x.groupby("MAIN_SYSTEM_ID")
            .rolling("7D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
            .std()
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
            "ROLLINGMEAN_7DAYS__ORDER_PRICE__",
            "ROLLINGMEAN_30DAYS__ORDER_PRICE__",
            "ROLLINGSTDDEV_7DAYS__ORDER_PRICE__",
            "ROLLINGSTDDEV_30DAYS__ORDER_PRICE__",
        ]
    ]
    .sort_values("ORDER_CREATION_DATE")
    .reset_index(drop=True)
)
```

Perform **point-in-time correct join** between datasets, accept at max 7-day-old stale rolling_mean.

```{code-cell} ipython3
merged_df = pd.merge_asof(
    loans_df,
    ecommerce_df,
    left_on="LOAN_ISSUANCE_DATE",
    right_on="ORDER_CREATION_DATE",
    by="MAIN_SYSTEM_ID",
    direction="backward",
    tolerance=pd.Timedelta("7d"),
).query("LOAN_ISSUANCE_DATE > ORDER_CREATION_DATE")

merged_df.info()
```

(missing-observations)=
**Missing observations**

We can't retrieve rolling_mean for 25% of observations (from 57,621 to 43,225), which means that this approach is inviable as it stands.
Before pivoting, let's check the sanity of this result by looking at Ecommerce dataset for a couple of cases.

```{code-cell} ipython3
retailers_leftout = (
    pd.merge(
        loans_df[["MAIN_SYSTEM_ID"]].drop_duplicates(),
        merged_df[["MAIN_SYSTEM_ID"]].drop_duplicates(),
        on="MAIN_SYSTEM_ID",
        how="left",
        indicator=True,
    )
    .query("_merge == 'left_only'")
    .drop("_merge", axis=1)
    .drop_duplicates()
)

retailers_leftout.count()
```

```{code-cell} ipython3
pd.merge(ecommerce_df, retailers_leftout, on="MAIN_SYSTEM_ID")[
    "MAIN_SYSTEM_ID"
].drop_duplicates().count()
```

They are in fact all present. So it must mean their loan request came after 7 days of their last Ecommerce transaction. Let's expand the accepted staleness to 30 days and see how the number changes.

```{code-cell} ipython3
merged_df = (
    pd.merge_asof(
        loans_df,
        ecommerce_df,
        left_on="LOAN_ISSUANCE_DATE",
        right_on="ORDER_CREATION_DATE",
        by="MAIN_SYSTEM_ID",
        direction="backward",
        tolerance=pd.Timedelta("30d"),
    )
    .query("LOAN_ISSUANCE_DATE > ORDER_CREATION_DATE")
    .drop_duplicates()
)

merged_df.info()
```

We are still missing 20% of observations, and increasing this window means less stable estimations and more uncertainty. Let's add a longer window rolling_mean to adjust for that.

```{code-cell} ipython3
merged_df = merged_df.assign(
    ROLLINGMEAN_120DAYS__ORDER_PRICE__=(
        lambda x: x.groupby("MAIN_SYSTEM_ID")
        .rolling("120D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
        .mean()
        .reset_index(drop=True)
    ),
    ROLLINGMEAN_360DAYS__ORDER_PRICE__=(
        lambda x: x.groupby("MAIN_SYSTEM_ID")
        .rolling("360D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
        .mean()
        .reset_index(drop=True)
    ),
    ROLLINGSTDDEV_120DAYS__ORDER_PRICE__=(
        lambda x: x.groupby("MAIN_SYSTEM_ID")
        .rolling("120D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
        .std()
        .reset_index(drop=True)
    ),
    ROLLINGSTDDEV_360DAYS__ORDER_PRICE__=(
        lambda x: x.groupby("MAIN_SYSTEM_ID")
        .rolling("360D", on="ORDER_CREATION_DATE")["ORDER_PRICE"]
        .std()
        .reset_index(drop=True)
    ),
).sort_values("ORDER_CREATION_DATE")
```

```{code-cell} ipython3
merged_df.value_counts("PAYMENT_STATUS")
```

For the sake of time, I will finish this experiment with 20% of missing observations. Unfortunately, 5 out of 9 "not fully paid" observations are left out because of staleness. That would be an interesting tangent for another set of experiments. Something in the shape of "inactive-retailer" segment for credit risk.

However, this leaves us with a smaller section of an already very minor class. It doesn't mean we cannot test our hypothesis, but the confidence in results will be diminished. Not all is lost, though. We can then use Power Analysis to find out how many more observations we woudl require to achieve 95% confidence intervals. But first things first, let's test the first set of hypotheses:
 - *null hypothesis 1*: retailers who default (fully or partially) on their loan are requesting to borrow amounts **above 1 standard deviation** of their usual ecommerce volume
 - *alternate hypothesis 1*: retailers who default (fully or partially) on their loan are requesting to borrow amounts **within 1 standard deviation** of their usual ecommerce volume

To be very honest this first set of hypotheses becomes rather pointless to test if we only have 4 observations. Let's skip to the second set, which can lead into more insight.

- *null hypothesis 2*: retailers who pay their loans in full are requesting to borrow amounts **within 1 standard deviation** of their usual ecommerce volume
 - *alternate hypothesis 2*: retailers who pay their loans in full are requesting to borrow amounts **above 1 standard deviation** of their usual ecommerce volume

If our tests accept the *null hypothesis*, we will have more confidence that credit risk assessment based on ecommerce volume wouldn't jeopardize our best customers in volume, safeguarding customer experience. But the fact that we can't test set 1 of hypotheses means we would still be opened to financial exposure.

I'm going to run 1000 trials of the hypothesis test for each aggregation window (7, 30, 120, 360 days) and use random sampling with replacement at each trial. This is a technique known as Bootstrapping and it's great for estimating confidence intervals for tests.

```{code-cell} ipython3
# Define the number of bootstrap rounds
n_bootstrap_rounds = 1000

for w_size in [7, 30, 120, 360]:
    # Calculate the mean and standard deviation of the usual ecommerce volume
    mean_usual_ecommerce_volume = merged_df[
        f"ROLLINGMEAN_{w_size}DAYS__ORDER_PRICE__"
    ].mean()
    std_usual_ecommerce_volume = merged_df[
        f"ROLLINGSTDDEV_{w_size}DAYS__ORDER_PRICE__"
    ].std()

    # Calculate the 1 standard deviation limit
    one_std_limit = mean_usual_ecommerce_volume + std_usual_ecommerce_volume

    # Define function to calculate the proportion of loan_amount greater than one_std_limit
    def proportion_above_limit(data):
        return np.sum(data > one_std_limit) / len(data)

    # Initialize an empty array to store bootstrap sample proportions
    bootstrap_proportions = np.zeros(n_bootstrap_rounds)

    # Perform bootstrapping
    for i in range(n_bootstrap_rounds):
        bootstrap_sample = merged_df["LOAN_AMOUNT"].sample(frac=0.3, replace=True)
        bootstrap_proportions[i] = proportion_above_limit(bootstrap_sample)

    # Calculate the 95% confidence intervals
    lower_bound = np.percentile(bootstrap_proportions, 2.5)
    upper_bound = np.percentile(bootstrap_proportions, 97.5)

    # Check if the confidence interval contains 0.5 (equal proportions)
    print(f"Results on rolling_mean(ORDER_PRICE over {w_size} days)")
    if lower_bound > 0.2:
        print(
            "\tReject null hypothesis 2",
            f"\tRetailers are requesting to borrow amounts **above 1 standard deviation** of their mean volume in the previous {w_size} days",
            sep="\n",
        )
    else:
        print(
            "\tFail to reject null hypothesis 2",
            f"\tRetailers are requesting to borrow amounts **within 1 standard deviation** of their mean volume in the previous {w_size} days",
            sep="\n",
        )
    print(
        f"\tWe are 95% confident that between {lower_bound:.1%} and {upper_bound:.1%} of retailers "
        + "borrow above their mean ecommerce volume"
    )
    print(f"\t95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print("\n")
```

That's an encoraging but expected result. Most retailers are responsible businees people who tend to be better at managing money. However, I was not expecting the percentage of retailer who go beyond this "safety zone" to bo so low. And here I can point out that not making that statement previously -- what I expect to see -- is something that sould always be avoided in experiments. I should have stated that I expected to see it around the 10% mark before commiting to this experiment. Maybe it's the hurry that's getting the best of me.

Anyhow, there is much more to be explored and proposed, but we can confidently add the rolling features. And because there was little difference between 7 and 30 days, and because of [this limitation](missing-observations) I will take the 30-day window.
