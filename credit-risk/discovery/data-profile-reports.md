---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
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

# Data Profile Reports

In the next 3 sections, I present a full profiler report on the 3 datasets. These reports are generated using Python package [ydata-profiling](https://ydata-profiling.ydata.ai/docs/master/index.html) and aim to provide an overview on the distribution of data in each column, aside from useful statistics, correlation analysis, among other things.

## Low signal columns
These are columns that bring little-to-no information, thus can interfere with ML model training. I'll remove the following columns from any further step:
  - Loans dataset
    - `INITIAL_COST`: constant value (zero), currently not charging retailers
    - `FINAL_COST`: constant value (zero), not currently charging retailers
    - `FINAL_COST`: constant value (zero)
    - `INDEX`: another identifier field
    - `LOAN_ID`: another identifier field
    - `REPAYMENT_ID`: another identifier field
  - Ecommerce dataset
    - `ORDER_ID`: another identifier field


## Highly correlated columns
These are data points that capture the same underlying phenomenon, thus behave closely to one another. Using multiple columns with high correlation tends to bias the importance of a given phenomenon captured by the data. I won't straight away remove those columns from analysis, but the correlation group must be consolidated into 1 representative when it comes to feature engineering and model induction.
  - Loans dataset
    - Retailer identity (as expected):
      - `RETAILER_ID`
      - `MAIN_SYSTEM_ID`
    - Loan invoice details:
      - `LOAN_AMOUNT`
      - `TOTAL_INITIAL_AMOUNT`
      - `INITIAL_COST`: currently not charging retailers, but it will be part of the correlation group once we start charging
    - Retailer consumption of loan:
      - `SPENT`
      - `FINAL_COST`: currently not charging retailers, but it will be part of the correlation group once we start charging.
      - `PAYMENT_AMOUNT`
      - `TOTAL_FINAL_AMOUNT`
      - `CUMMULATIVE_OUTSTANDING`:
        - `FIRST_TRIAL_BALANCE` + `REPAYMENT_AMOUNT`: 99.9% match with 1 decimal point precision
  - Ecommerce dataset
    - `ORDER_PRICE_AFTER_DISCOUNT`:
      - `ORDER_PRICE` - `DISCOUNT`: 99.9% match with 1 decimal point precision, 99.6% with 2 decimal points precision


Although there is strong correlation between columns of "*Loan invoice details*" and "*Retailer consumption of loan*", I choose to keep them separate groups. The reasoning is based on how the credit product is implemented at the moment. We evaluate retailers request for a given amount, but allow them to spend only a fraction of that amount. This means there are 2 related but distinct behaviors to capture: the rationale when requesting funds, and the actual need to spend them.


## Time Series Data
At first glance, I believe the 3 datasets are some form of time series data plus aggregations on those series. However, not all time series data are equal. We need to find out whether these are changelog tables (contain all change events) or latest snapshot per entity. How can we check that?



---

+++

The sections above were drawn from the following data profile reports:

```{tableofcontents}
```

+++

Code to generate each report:

``````{div} full-width

`````{tab-set}

````{tab-item} Retailer Loans
```{code-block} python
import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_excel("data/Loans_Data.xlsx")
profile = ProfileReport(df, explorative=True, interactions=None)
profile.config.html.navbar_show = False
profile.config.html.full_width = True
profile.to_file("credit-risk/discovery/data-profile-reports/loans-profile.html")
```
````

````{tab-item} Fintech Transactions
```{code-block} python
import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_csv("data/Retailer_Transactions_Data.csv", header=0)
profile = ProfileReport(df, explorative=True, interactions=None)
profile.config.html.navbar_show = False
profile.config.html.full_width = True
profile.to_file("credit-risk/discovery/data-profile-reports/fintech-profile.html")
```
````

````{tab-item} Ecommerce Transactions
```{code-block} python
import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_csv("data/Ecommerce_orders_Data.csv", header=0)
profile = ProfileReport(df, explorative=True, interactions=None)
profile.config.html.navbar_show = False
profile.config.html.full_width = True
profile.to_file("credit-risk/discovery/data-profile-reports/ecommerce-profile.html")
```
````

`````

``````

````{dropdown} Code for CUMMULATIVE_OUTSTANDING percentage match
```{code-block} python
import pandas as pd
import numpy as np


df = pd.read_excel("data/Loans_Data.xlsx")

df.assign(
    match=np.isclose(
      df["FIRST_TRIAL_BALANCE"] + df["REPAYMENT_AMOUNT"], df["CUMMULATIVE_OUTSTANDING"],
      rtol=1e-1,
      atol=1e-1
    )
)["match"].value_counts()
```
````

````{dropdown} Code for ORDER_PRICE_AFTER_DISCOUNT percentage match
```{code-block} python
import pandas as pd
import numpy as np


df = pd.read_csv("../../data/Ecommerce_orders_Data.csv", header=0)

df.assign(
    match=np.isclose(
      df["ORDER_PRICE"] - df["DISCOUNT"], df["ORDER_PRICE_AFTER_DISCOUNT"],
      rtol=1e-1,
      atol=1e-1
    )
)["match"].value_counts()
```
````
