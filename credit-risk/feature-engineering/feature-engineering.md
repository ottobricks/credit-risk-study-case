# Feature Engineering

(auto-feat-eng)=
## Automatic Feature Engineering

````{dropdown} Deep Feature Synthesis (DFS)
```{literalinclude} automatic-feature-engineering.py
   :language: python
   :linenos:
```
````

````{dropdown} Dimensionality reduction, a.k.a feature selection
```{literalinclude} feature-selection.py
   :language: python
   :linenos:
```
````

(holdout)=
## Split Train and Test Data

````{toggle}
```{code-block} python
(
    featureframe.selectExpr(
        "count(label) filter (where label = 1) as positive_in_total",
        "count(label) filter (where label = 0) as negative_in_total",
        "count(label) filter (where label = 1 and stratified_split < 0.5) as positive_in_train",        
        "count(label) filter (where label = 0 and stratified_split < 0.6) as negative_in_train",
        "count(label) filter (where label = 1 and stratified_split between 0.5 and 0.7) as positive_in_valid",        
        "count(label) filter (where label = 0 and stratified_split between 0.6 and 0.8) as negative_in_valid",
        "count(label) filter (where label = 1 and stratified_split >= 0.7) as positive_in_test",        
        "count(label) filter (where label = 0 and stratified_split >= 0.8) as negative_in_test",        
    )
)

```
````

|  Class  |  Total  |  Train  |  Validation  |  Test  |
|---------|---------|---------|--------------|--------|
|  Negative  |  71166  |  42699  |  14234  |  14234  |
|  Positive  |  6  |  3  |  1  |  2  |

That's weird. It seems we lost 3 positive (not fully paid) along the way. My hunch is that they were excluded due to `TypeError: Time index column must be a Datetime or numeric column.`
This happened during Deep Feature Synthesis. I was hoping observations in the positive class wouldn't be affected. Due to time constraint, I will follow along until the end of this experiment trial, then come back to this if I can manage a second trial.

```{note}
I found the reason. Not all retailers in Loans dataset have records in Fintech or Ecommerce. The way I created relationships between datasets meant that the script failed if a retailer did not have observations in either dataset. That is fixed now, it will automatically adjust to use only datasets available.
```