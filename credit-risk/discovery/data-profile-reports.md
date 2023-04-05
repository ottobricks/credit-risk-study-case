# Data Profile Reports

```{tableofcontents}
```

Here is the code to generate each report:

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