# Credit Risk - Case Study

## Introduction

Welcome to my case study regarding credit risk for MaxAB. I must first emphasize that this is a simplistic approach to prevent retailers from contracting loans at high risk of defaulting. In this vein, the case study is framed as a binary classification problem and a final answer in the form of "yes" or "no" should dictate whether a loan is approved.

In order to conduct the case study, I was provided with 3 data sets:
- loans by retailer
- fintech transactions by retailer
- ecommerce orders by retailer

It goes without saying that this exercise can be enriched with many different data sources -- e.g. CBE interest rate, inflation, information about retailers business segment, age of business, etc. But for the sake of brevity, I'll work only with the data sets provided.

I decided to set aside some time to go over the design and components of my (proposal) for a more mature automated decision-making system and suggestions about the current loan process in chapter [Credit Risk Decision API](./decision-api/overview.md). Of course, it won't be extensive nor exhaustive of the possibilities since I must time-box it.

## Risk Modelling Beyond Ops Capacity 
The beginner approach to the problem is to use `PAYMENT_STATUS != 'Paid'` as a target, and conduct this experiment as a simple binary classification problem. However, that approach ignores multiple issues, the first being low representation (i.e. class imbalance), too few positive observations even to use oversampling techniques.

Another terrible side effect of such approach is that **credit risk assessment becomes tied to information that is only available in the future** (once retailer fully repays or defaults), meaning **there is always a lag between decisions we make and the evidence to estimate risk and exposure**. This lag (e.g. P95 in under 48 hours) means decisions that expose us to financial loss are made in the dark, without the possibility for real-time adjustment of thresholds based on previous decisions. We'll only learn about exposure vs. realized loss once annotations arrive. This issue is not limited to credit risk modelling; it's far too common in risk assessment tasks in general.

Thus, I will completely skip such methods. This decision is based on my (parallel) domain knowledge, previous experience, and quite frankly good common sense. Instead, I choose to model credit risk against information that is available to us previous to the loan request. This means we are able to adjust the decision-making nobs to fit a given risk appetite, and price it into the credit product.

In this case study I focused on **retailer's estimated cash flow against previous 30-day history of purchases in our Ecommerce platform** as a measure of how likely they are to be able to repay once the loan is due. There are many other ways to approach it, but I judge this to be enough for a case study. In the future, we could look at more robust risk modelling strategies; for instance, using our payment processing to offer retailers' customers payments in installments (new credit product), and use those future payable as collateral for loans. This way, we could reduce interest rates for good retailers, while hedging our exposure with their future revenue.

Of course, this methodology is not free of caveats. For instance, forcing the decision-making to be "yes/no" based on only 1 data point is likely to cause many False Positives (we say "no" to good retailer). Again, I judge this to be an acceptable limitation for a case study, and have a brief discussion on mitigating such risks in [Risk Segmentation](decision-api/tiered-risk-model.md). 

## Results

```{note}
This section is a summarized version of {ref}`result-assessment`
```

Result assessment is based only on 34,418 observations out of the 73,086 provided, roughly the 50% latest observations -- see {ref}`splitting-ml-dataset` for more info.

**Is that a problem?**

There is no such thing as perfect method, but there objectively better ones. We are forced to choose where to place uncertainty:
 - giving models all the data, increasing chance of overfitting (models memorize the data), thus weakening confidence in results
 - training models in "past" data and assessing them on "future" data, reducing the signal available for models to learn, but increasing a lot confidence on results

The latter is objectively better. Think of it like this: is it better to have a funny friend that lies to you all the time, or the awkward one that is always there for you? For a party (i.e. boasting about astonishingly unrealistic performance), you want the funny friend, but what about for life?

Another point of attention is that I will consider `PAYMENT_STATUS in ('Unpaid', 'Partialy paid')` (\*partially) as *defaults*. I will also consider *defaults* all cases when the retailer is not able to make the payment on the first collection attempt. Let me explain. I understand retailers are not always to blame for missed collection attempt, it sometimes falls on our operations agents. This can definitely be addressed in the future with a separate track: optimizing ops agents schedules to maximize collection rates. However, for this case study, I'm going to limit the scope of our assessment, and my judgement is that there is also a component of timing in a retailer's ability to repay. This shifts the objective from being "we eventually want to collect debts" to "our forecast should also optimize for timing". Of course, this goes beyond the scope of a case study, but it is an interesting domain to explore.

I split the result into 2 risk appetite tiers focused solely on user-base growth: *conservative* and *ambitious*.

> [WIP] Ideas, definitely no time to cover all -- choose a couple:
>
> - daily rate of exposure (need to define exposure precisely, maybe stddev above 30-day mean predicted volume)
> - weekly rate of exposure vs. realized loss
> - recommend daily interest rates based to cover previous week exposure (if time allows, account for seasonality)
> - daily rate of stale capital: LOAN_AMOUNT vs. SPENT
> - weekly projected growth vs. realized growth

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
 - accepted exposure wow: 
 - realized loses wow:
 - opportunity gap wow: 

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
 - accepted exposure wow: 
 - realized loses wow:
 - opportunity gap wow:  

---

The chapters in this book are meant to be read in order as they reflect the natural flow of exploration, then experimentation, then results.

## Chapters
```{tableofcontents}
```
