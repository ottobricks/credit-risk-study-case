# Evaluation Method

## Prediction Targets
As mentioned in the introduction, the problem is presented as binary classification because in the end a decision must be made: approve or deny a loan. Many may jump straight into using `PAYMENT_STATUS` as the target, spin up a large instance on AWS and give an AutoML package time and compute to brute-force the solution. Also as mentioned before, this inevitably leads to good performance in development and disappointment in production.

Instead, I choose to stick to the rationale in [Hypothesis](./hypothesis.md) section and make the decision based on estimated `SPENT` versus requested `LOAN_AMOUNT`. This is a much more realistic approach and allows for flexibility when making the decision to take overall objective (growth vs. profit) into account.

Therefore, I will generate forecasts for retailers' ecommerce volume and use the following formula to yield our credit risk score for this case study:

$$
Lorem Ipsum
$$

Then, I will generate a Decision dataset with "yes/no" answers for loans based on 2 scenarios at fixed risk exposure: maximize profit, maximize growth. Note that this heavily depends on my [assumption](assumptions) that we have enough observations in Ecommerce to be able to forecast retailers' expenses with suppliers over time. Therefore, we must turn to results in [Testing Underlying Assumptions](testing-assumptions) to validate the sanity of this approach.

Conversely, if the assumption does not hold, I will take the dummy approach mentioned a couple paragraphs above due to time constraints.

## Splitting ML dataset

Once again, in order to avoid Data Leak, we split our ML dataset (featureframe) into 3: Train, Validation and Test datasets. The Test dataset is used to generate final results and is not used during development. In real-life experiments, we shouldn't even process feature engineering scripts on them, but rather transform the data just before final evaluation of results. But due to time constraints, I decided that an adaptation of <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html" target="_blank">TimeSeriesSplit</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html" target="_blank">TimeSeriesSplit</a> with holdout split are enough to mitigate Data Leak for the case study.

```{note}
I choose to use custom holdout split instead of cross-validation due to small datasets and time constraint.
```

The Python script used to split our featureframe is available in [Feature Engineering](holdout).
