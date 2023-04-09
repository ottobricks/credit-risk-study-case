---
jupytext:
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

# Model Tuning

```{code-block} python
import pickle
import warnings
from flaml.automl.data import get_output_from_log
from flaml import AutoML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore", category=RuntimeWarning)
```

```{code-block} python
train_df = pd.read_parquet("../../data/train/pandas-pca-featureframe-maxdepth2-targetSPENT.parquet").sort_values("LOAN_ID")

# Let's remove retailers who never spent their borrowed amount, as these are likely to be first interactions with the product or mistakes
# train_df = train_df.query("label > 0")
train_df.info()
train_df.head(5)
```

```{code-block} python
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "task": 'regression',
    "metric": 'r2',
    "eval_method": "cv",
    "n_splits": 5,
    "split_type": "time",
    "early_stop": True,
    "n_jobs": -1,
    "n_concurrent_trials": 1,
    "time_budget": 3600,  # in seconds
    "log_file_name": "tune.log",
    "verbose": 2,
}
```

```{code-block} python
automl.fit(
    dataframe=train_df.drop(["LOAN_ID", "MAIN_SYSTEM_ID"], axis=1),
    label="label",
    **automl_settings
)
```

```{code-block} python
# Save the AutoML object to a file
with open("assets/automl.pkl", "wb") as f:
    pickle.dump(automl, f)
```

```{code-block} python
# Load the test data
test_df = pd.read_parquet("../../data/test/pandas-pca-featureframe-maxdepth2-targetSPENT.parquet").sort_values("LOAN_ID")
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# Perform predictions
y_pred = automl.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Print the model performance report
print(f"Best Model: {automl.best_estimator}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
```

Best Model: rf
Mean Absolute Error (MAE): 0.0004
Mean Squared Error (MSE): 0.0001
Root Mean Squared Error (RMSE): 0.0121
R-squared (R2): 0.5624

```{code-block} python
# time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history =
#     get_output_from_log(filename=settings["log_file_name"], time_budget=120)

# plt.title("Learning Curve")
# plt.xlabel("Wall Clock Time (s)")
# plt.ylabel("Validation Accuracy")
# plt.step(time_history, 1 - np.array(best_valid_loss_history), where="post")
# plt.show()
```
