# Hypothesis

It seems reasonable to believe that loans which deviate too much from a retailer's cash flow are at higher risk of default. With this in mind, we can now create our predictions.
 - *null hypothesis 1*: retailers who default (fully or partially) on their loan are requesting to borrow amounts **above 1 standard deviation** of their usual ecommerce volume
 - *alternate hypothesis 1*: retailers who default (fully or partially) on their loan are requesting to borrow amounts **within 1 standard deviation** of their usual ecommerce volume

We can also investigate the same belief against retailers who have a good credit history with us. Let's pose this as a complementary pair of hypothesis, our aim is to limit impact to good retailers -- let me explain. Suppose that limiting loan amounts to 1 standard deviation of ecommerce volume would sharply reduce risk of default. But at the same time, outliers have disproportional impact, and we may not want to block healthy and fast growing retailers from leveraging. We also have to take into account the fact that our sample of a retailers volume may not be as representative as we would like it to be. Nevertheless, we can **address the risk of negatively impacting retailers** with the following complementary hypotheses:
 - *null hypothesis 2*: retailers who pay their loans in full are requesting to borrow amounts **within 1 standard deviation** of their usual ecommerce volume
 - *alternate hypothesis 2*: retailers who pay their loans in full are requesting to borrow amounts **above 1 standard deviation** of their usual ecommerce volume

**What should we expect from testing these sets of hypotheses?**

If my intuition is correct (both *null hypothesis* are accepted by our experiment), it means we can engineer a set of features to capture the relationship between loan_amount requested and forecasted ecommerce volume for retailer at loan_issue_date. That set of features should then have strong predictive power for default risk. What are the caveats?

(assumptions)=
## Underlying Assumptions
Assumptions are necessary to form an understanding of complex systems, they function as axioms for our predictions. It means that assumptions must hold true in order for hypotheses, and thus experiments, to be meaningful. I address the assumptions I used to build my hypotheses next.

### Ecommerce dataset completeness
I assume the Ecommerce dataset is complete for retailers in Loans dataset, i.e. there are no missing observations for at least a 2-week window before the loan is requested. Why 2-week window? It's rather a rule of thumb in forecasting than actual theory, but it's believed that to forecast a a window of X length, data must contain at least 2 times that length and be as complete (no missing records) as possible.

```{note}
We can already see that this assumptions does not hold for retailers who've recently joined and already request loans. This is not a problem, it's just a way to segment default risk -- new and established retailers in this case. I choose to focus on established retailers for this case study.
```

So, it becomes very clear that we must first assess the validity of this assumption before proceeding to hypothesis test. I address this topic in [Testing Underlying Assumptions](../experiments/test-assumption.md).

---

Next, let's look at why *most* Machine Learning experiments yield models that don't achieve in production the promised performance during development.