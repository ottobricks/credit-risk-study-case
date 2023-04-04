---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '1.3'
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Data Profile Reports

``````{div} full-width

`````{tab-set}

````{tab-item} Retailer Loans
```{raw} html
:file: data-profile-reports/loans-profile.html
```
````

````{tab-item} Fintech Transactions
```{raw} html
:file: data-profile-reports/fintech-profile.html
```
````

````{tab-item} Ecommerce Transactions
```{raw} html
:file: data-profile-reports/ecommerce-profile.html
```
````

`````

``````

Here is the code to generate each report:

``````{div} full-width

`````{tab-set}

````{tab-item} Retailer Loans
```{code-block} python
import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_excel("data/Loans_Data.xlsx")
profile = ProfileReport(df, explorative=True)
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
profile = ProfileReport(df, explorative=True)
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
profile = ProfileReport(df, explorative=True)
profile.config.html.navbar_show = False
profile.config.html.full_width = True
profile.to_file("credit-risk/discovery/data-profile-reports/ecommerce-profile.html")
```
````

`````

``````