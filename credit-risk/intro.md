# Credit Risk - Case Study

## Introduction

Welcome to my case study regarding credit risk for MaxAB. I must first emphasize that this is a simplistic approach to prevent retailers from contracting loans at high risk of defaulting. In this vein, the case study is framed as a binary classification problem and a final answer in the form of "yes" or "no" should dictate whether a loan is approved.

In order to conduct the case study, I was provided with 3 data sets:
- loans by retailer
- fintech transactions by retailer
- ecommerce orders by retailer

It goes without saying that this exercise can be enriched with many different data sources -- e.g. CBE interest rate, inflation, information about retailers business segment, age of business, etc. But for the sake of brevity, I'll work only with the data sets provided.

I decided to set aside some time to go over the design and components of my (proposal) for a more mature automated decision-making system and suggestions about the current loan process in chapter [Credit Risk Decision API](./decision-api/overview.md). Of course, it won't be extensive nor exhaustive of the possibilities since I must time-box it.

## Methodology
The beginner approach to the problem is to use `PAYMENT_STATUS != 'Paid'` as a target, and conduct this experiment as a simple binary classification problem. However, that approach ignores multiple issues, the first being low representation (i.e. class imbalance), too few positive observations even to use oversampling techniques.

Another terrible side effect of such approach is that credit risk assessment becomes tied to information that is only available in the future (once retailer fully repays or defaults), meaning there is always a lag between decisions we make and their supporting evidence. This lag (e.g. 1 day on average) means decisions that expose us financially are made completely in the dark, without the possibility for real-time adjustment based on previous decisions. We'll only learn about exposure vs. loss once annotations arrive. This issue is not limited to credit risk modelling; it's far too common in risk assessment tasks in general.

Thus, I will completely skip such methods based on parallel domain knowledge, previous experience, and quite frankly good common sense. Instead, I choose to model credit risk against information that is available to us previous to the loan request. This means we are able to adjust the decision-making nobs to fit a given risk appetite, and price it into the credit product.

In this case study I focused on **retailer's estimated cash flow against previous 30-day history of purchases in our Ecommerce platform** as a measure of how likely they are to be able to repay once the loan is due. There are many other ways to approach it, but I judge this to be enough for a case study. In the future, we could look at more robust risk modelling strategies; for instance, using our payment processing to offer retailers' customers payments in installments (new credit product), and use those future payable as collateral for loans. This way, we could reduce interest rates for good retailers, while hedging our risk against their future revenue.

Of course, this methodology is not free of caveats. For instance, forcing the decision-making to be "yes/no" based on only 1 data point is likely to cause many False Positives (we say "no" to good retailer). Again, I judge this to be an acceptable limitation for a case study.

## Results
I split the result into 2 risk appetite tiers focused solely on user-base growth: *conservative* and *ambitious*.

### Conservative
Conservative means that we are willing to approve a loan if:

$$
(loan\_amount <= ecommerce\_average\_volume_{30days} + 1 * standard\_deviation_{30days}) \\
AND \\
(estimated\_spent <= ecommerce\_average\_volume_{30days} + 1 * standard\_deviation_{30days})
$$

*Conservative* means we would have:
 - approved X (currency) out of Y requested in total
 - served X out of Y retailers that requested loans
 - incurred into X `default_volume` (financial loss)

Of course, we can always hedge losses with credit product pricing, but this goes far beyond the scope of a case study.

### Ambitious
Ambitious means that we are willing to approve a loan if the same formula above applies but with $2 * standard\_deviation_{30days}$, higher multiplier for standard deviation upper bound.

By choosing to be ambitious, we are willing to provide loans to 13.6% more retailers, knowing these to be leverage-seekers. As mentioned before, hedging against financial loss via product pricing is a very important topic, but goes beyond the scope of this case study. However, just as food-for-thought, here are some parameters that have to be considered and others that can be adjusted:
 - the current available lending pool volume
 - the current exposure and coverage on volume already approved for lending in the period (e.g. daily)
    - if current `volume_lent` is made up mostly of *conservative* loans, we can be more *ambitious* until we reach our financial exposure threshold, or are no longer interested in adding overhead to pricing for non-financial reasons (e.g. bad press on high rates -- "abusive rates" is clickbait headline that unfortunately sells)
 - our assumed percent rate of `default_volume`: nob adjusted based on `standard_deviation` multiplier
 - our `margin_multiplier`: nob adjusted based on growth vs. profit appetite

 These are just a few parameters that we could observe and manipulate to keep operations healthy while keeping financial exposure under control. There is so much more to explore, this is indeed very exciting.

Finally, *ambitious* means we would have:
 - approved X (currency) out of Y requested in total
 - served X out of Y retailers that requested loans
 - 

---

The chapters in this book are meant to be read in order as they reflect the natural flow of exploration, then experimentation, then results.

## Chapters
```{tableofcontents}
```
