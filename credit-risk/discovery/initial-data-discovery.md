---
jupytext:
  formats: ipynb,md:myst
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

# Initial Data Discovery

My understanding of the datasets is that:
- Ecommerce Data relates to inventory purchase retailers make on MaxAB
- Fintech Data relates to customer-facing sales retailers make using MaxAB payment processing
- Loans Data relates to micro-loans retailers request to purchase iventory through MaxAB

## Data points
Although Retailer Loans dataset contains definition of columns, the other 2 datasets do not. So first order of business is to inspect them and infer what their columns mean.

+++

### Loans data

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd


pd.read_excel(
    "../../data/Loans_Data.xlsx"
).iloc[0].sort_index()
```

```{code-cell} ipython3
:tags: [remove-cell]

# pd.read_excel(
#     "../../data/Loans_Data.xlsx",
#     sheet_name="column descriptions"
# ).to_markdown(
#     "./_loans-data-column-descriptions.md",
#     index=False
# )
```

````{dropdown} Column Description

| Column                  | Description  |
|:------------------------|:------------------|
| INDEX                   | Negligible number  |
| MAIN_SYSTEM_ID          | the retailer's common id between e-commerce and fintech  |
| RETAILER_ID             | the retailer's id on the fintech system only  |
| LOAN_ID                 | id of the Loan  |
| LOAN_ISSUANCE_DATE      | the datetime when the loan was requested  |
| LOAN_AMOUNT             | the pricipal amount the retailer requested in credit (value only shows intention, but amount is considered lent once the retailer uses money for transaction)  |
| INITIAL_COST            | the interest rate (from retailer's credit profile) * loan amount (value is also non-binding; is recalculated on consumed credit amounts)  |
| TOTAL_INITIAL_AMOUNT    | initial principal + initial cost  |
| INITIAL_DATE            | Due date of the loan  |
| LOAN_PAYMENT_DATE       | The finalization date of the loan's first collection order (could be different than the initial date for one of two reasons: -retailer was visited and requested a delay -the operations team was not able to fulfill the order on the correct date)  |
| PAYMENT_AMOUNT          | The paid amount during the loans's first collection order  |
| FIRST_TRAIL_DELAYS      | The number of delays received on the first collection order  |
| SPENT                   | The consumed amount of the loan = total value of transactions fulfilled from the initial credit amount  |
| FIRST_TRIAL_BALANCE     | The retailer's balance after the payment amount was logged in the wallet; -If the first trial balance is negative, it means the payment amount did not cover the spent credit amount and the negative amount is now overdue; -If the first trial balance >= 0 then it means the payment amount covered the spent credit amount and no amounts are due for the said loan following the payment date |
| FINAL_COST              | the interest rate (from retailer's credit profile) * spent amount  |
| TOTAL_FINAL_AMOUNT      | spent amount + final cost (the minimum amount required to collect from the retailer to fulfill due amounts)  |
| REPAYMENT_ID            | the id of the last cash-in used to repay the loan -if first trial balance >= 0, then it will equal the loan_id -if first trial balance < 0 and there were no attempts to recollect yet,  then it will equal the loan_id (incorrect design, loan_id does not have a functional purpose in this column)  |
| REPAYMENT_AMOUNT        | amount paid during last collection order used to repay the loan -if first trial balance >=0, then = 0 (because loan was already paid)  |
| REPAYMENT_UPDATED       | the date of the last collection trial                                                                                                                                 |
| CUMMULATIVE_OUTSTANDING | It's the cummulation of the retailer's balance from the due date until the date it is paid -If it is positive, it means no amounts are unpaid on the said loan level -If it is negative, it means the retailer's balance is still negative, and there are overdue amounts on the said loan level  |
| PAYMENT_STATUS          | the status of the loan and it can take 3 values paid - unpaid - partialy paid  |
````

+++

### Fintech data

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd


pd.read_csv(
    "../../data/Retailer_Transactions_Data.csv",
    header=0
).iloc[0].sort_index()
```

From the column definitions in Loans data, we know `MAIN_SYSTEM_ID` represents the retailer's identity between Fintech and Ecommerce.

Furthermore, we can already predict that there is strong correlation between

```{dropdown} Column Description

|  **Column**  |  **Description**  |
|--------------|------------------|
|  AMOUNT  |  Eq. `invoice_value` before tax and fees  |
|  CREATED_AT  |  Timestamp when payment processing started  |
|  FEES  |  MaxAB fee for providing payment processing service, etc  |
|  ID  |  Internal identification of payment processing  |
|  MAIN_SYSTEM_ID  |  Retailer internal identification across ecommerce and fintech, table `foreign_key`  |
|  RETAILER_CUT  |  Estimated profit margin of retailer on the sale (?)  |
|  STATUS  |  Latest status of payment processing according to `UPDATED_AT`  |
|  TOTAL_AMOUNT_INCLUDING_TAX  |  `AMOUNT` + `FEES` + `TAX`  |
|  TOTAL_AMOUNT_PAID  |  The amount paid to the retailer by their customer  |
|  UPDATED_AT  |  Timestamp of the current `STATUS` update  |
|  WALLET_BALANCE_BEFORE_TRANSACTION  |  Retailer balance before the current sale  |

```

+++

### Ecommerce data

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd


pd.read_csv(
    "../../data/Ecommerce_orders_Data.csv",
    header=0
).iloc[0].sort_index()
```

This dataset is simpler in nature if compared with the other 2, and as expected `MAIN_SYSTEM_ID` is present here too. Let's organize column definitions is a table:

```{dropdown} Column Description

|  **Column**  |  **Description**  |
|--------------|------------------|
|  DISCOUNT  |  Amount to be deducted from `ORDER_PRICE`  |
|  MAIN_SYSTEM_ID  |  Retailer internal identification across ecommerce and fintech, table `foreign_key`  |
|  ORDER_CREATION_DATE  |  Timestamp when inventory purchase order is created  |
|  ORDER_ID  | Table index, unique identifier for orders |
|  ORDER_PRICE  |  Invoice value *before* `DISCOUNT` is applied  |
|  ORDER_PRICE_AFTER_DISCOUNT   |  Invoice value *after* `DISCOUNT` is applied  |

```
