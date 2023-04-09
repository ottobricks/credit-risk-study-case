# Avoiding Data Leak
One of the most common shortcomings in Machine Learning experiments is the delta in performance between development and production. Very often you'll hear cases when models perform beautifully while being developed but then fail to reach the same performance once deployed. Why does that happen?

More often than not, the performance delta is a symptom of Data Leak. The model has access to data during training that it cannot possibly have in production, let me give a concrete example.

We can see in [Initial Data Discovery](./initial-data-discovery.md) that Loans dataset is in fact a composite of 3 stages in the lifetime of a loan: issuance, collection and balance rollover. We need to assess credit risk before issuing the loan, but we won't have information about collection and rollover until after the loan is granted or denied. This means using some data points that are present in the row will lead the model to have a "vision of the future", in other words, **Data Leak**.

In order to avoid Data Leak, these columns will be removed from feature engineering:
 - REPAYMENT_ID
 - FINAL_COST
 - REPAYMENT_UPDATED
 - SPENT
 - TOTAL_FINAL_AMOUNT
 - FIRST_TRIAL_BALANCE
 - FIRST_TRAIL_DELAYS
 - PAYMENT_AMOUNT
 - LOAN_PAYMENT_DATE
 - REPAYMENT_AMOUNT
 - CUMMULATIVE_OUTSTANDING
 
Of course, there are many techniques that can be applied to handle Data Leak better than just throwing data points out (e.g. adding lag) but I won't have the time to explore them in this case study.

**What about Data Leak from other datasets?**

We address this during [Automatic Feature Engineering](auto-feat-eng) via time-aware <a href="https://featuretools.alteryx.com/en/stable/getting_started/using_entitysets.html" target="_blank">EntitySets</a> and secondary time indexes.