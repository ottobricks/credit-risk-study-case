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

## Low-signal Columns
These are columns that bring little-to-no information, thus can interfere with ML model training. I'll remove the following columns from any further step:
  - Loans data
    - `INITIAL_COST`: constant value (zero)
    - `FINAL_COST`: constant value (zero)
    - `INDEX`: another identifier field
    - `LOAN_ID`: another identifier field
    - `REPAYMENT_ID`: another identifier field

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


df = pd.read_csv("data/Retailer_Transactions_Data.csv", header=0)
profile = ProfileReport(df, explorative=True, interactions=None)
profile.config.html.navbar_show = False
profile.config.html.full_width = True
profile.to_file("credit-risk/discovery/data-profile-reports/ecommerce-profile.html")
```
````

`````

``````
