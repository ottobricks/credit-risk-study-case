# Deep Feature Synthesis

````{dropdown} Pandas
```{code-block}
from typing import List
import itertools
import pandas as pd
import featuretools as ft
from featuretools.primitives import (
    AggregationPrimitive,
    TransformPrimitive
)
from featuretools_tsfresh_primitives import (
    comprehensive_fc_parameters,
    primitives_from_fc_settings
)
from featuretools_tsfresh_primitives import primitives as tsfresh_primitives
from featuretools_tsfresh_primitives.primitives import *

PARAMETERS = comprehensive_fc_parameters()

# Creates a list where each element refers to 1 aggregation primitive, a list of combinations of primitive(parameters)
TSF_AGG_PRIMITIVES: List[List[AggregationPrimitive]] = [
    primitives_from_fc_settings(
        {
            getattr(tsfresh_primitives, key).name: PARAMETERS[getattr(tsfresh_primitives, key).name] or [{}]
        }
    )
    for key in dir(tsfresh_primitives)
    if key[0].isupper()
        and key != "SUPPORTED_PRIMITIVES"
        and isinstance(
            getattr(tsfresh_primitives, key)(
                **(PARAMETERS[getattr(tsfresh_primitives, key).name] or [{}])[0]
            ),
            AggregationPrimitive
        )
]

# Creates a list where each element refers to 1 transform primitive, a list of combinations of primitive(parameters)
TSF_TRANSFORM_PRIMITIVES = [
    primitives_from_fc_settings(
        {
            getattr(tsfresh_primitives, key).name: PARAMETERS[getattr(tsfresh_primitives, key).name] or [{}]
        }
    )
    for key in dir(tsfresh_primitives)
    if key[0].isupper()
        and key != "SUPPORTED_PRIMITIVES"
        and isinstance(
            getattr(tsfresh_primitives, key)(
                **(PARAMETERS[getattr(tsfresh_primitives, key).name] or [{}])[0]
            ),
            TransformPrimitive
        )
]

# Load datasets
loans_df = (
    pd.read_excel("data/Loans_Data.xlsx")
    .drop(
        [
            # Low signal columns
            "INITIAL_COST",
            "FINAL_COST",
            "FINAL_COST",
            "INDEX",
            "REPAYMENT_ID"
        ],
        axis=1
    )
)
fintech_df = pd.read_csv("data/Retailer_Transactions_Data.csv", header=0)
ecommerce_df = pd.read_csv("data/Ecommerce_orders_Data.csv", header=0)
retailer_df = (
    pd.concat(
        [
            loans_df["MAIN_SYSTEM_ID"],
            fintech_df["MAIN_SYSTEM_ID"],
            ecommerce_df["MAIN_SYSTEM_ID"],
        ]
    )
    .drop_duplicates()
    .reset_index()
    [["MAIN_SYSTEM_ID"]]
)

# Create an entity set and add the retailers entity
entity_set = ft.EntitySet(id="maxab_entity_set")
entity_set = entity_set.add_dataframe(
    dataframe_name="retailers",
    dataframe=retailer_df,
    index="MAIN_SYSTEM_ID",
)

# Add the loans entity
entity_set = entity_set.add_dataframe(
    dataframe_name="loans",
    dataframe=loans_df,
    index="LOAN_ID",
    time_index="LOAN_ISSUANCE_DATE",
)

# Add the sales entity
entity_set = entity_set.add_dataframe(
    dataframe_name="sales",
    dataframe=fintech_df,
    index="ID",
    time_index="UPDATED_AT"
)

# Add the purchases entity
entity_set = entity_set.add_dataframe(
    dataframe_name="purchases",
    dataframe=ecommerce_df,
    index="ORDER_ID",
    time_index="ORDER_CREATION_DATE",
)

# Define relationships between the entities
rel_retailer_sales = ft.Relationship(
    entity_set,
    parent_dataframe_name="retailers",
    parent_column_name="MAIN_SYSTEM_ID",
    child_dataframe_name="sales",
    child_column_name="MAIN_SYSTEM_ID"
)

rel_retailer_purchases = ft.Relationship(
    entity_set,
    parent_dataframe_name="retailers",
    parent_column_name="MAIN_SYSTEM_ID",
    child_dataframe_name="purchases",
    child_column_name="MAIN_SYSTEM_ID"
)

rel_retailer_loans = ft.Relationship(
    entity_set,
    parent_dataframe_name="retailers",
    parent_column_name="MAIN_SYSTEM_ID",
    child_dataframe_name="loans",
    child_column_name="MAIN_SYSTEM_ID"
)


# Add the relationships to the entity set
entity_set = entity_set.add_relationships(
    [rel_retailer_sales, rel_retailer_purchases, rel_retailer_loans]
)

# Define the problem and the prediction target
problem = ft.ProblemDefinition(
    dataframe_name="loans",
    target_entity="loans",
    target_column="PAYMENT_STATUS",
    problem_type="classification",
)


# Run deep feature synthesis to create features between the entities
feature_matrix, feature_defs = ft.dfs(
    entityset=entity_set,
    target_dataframe_name="loans",
    verbose=2,
    # problem_definition=problem,
    agg_primitives=list(itertools.chain(*TSF_AGG_PRIMITIVES)),
    trans_primitives=list(itertools.chain(*TSF_TRANSFORM_PRIMITIVES)),
    max_depth=3,
)

# Print the resulting feature matrix
print(feature_matrix)

```
````


````{dropdown} Pyspark only
```{code-block}
import featuretools as ft
from featuretools_spark import dfs
from featuretools_tsfresh_primitives import TimeSeriesFeatureExtraction
from tsfresh.feature_extraction import ComprehensiveFCParameters
from pyspark.sql.functions import col

# Convert the PySpark DataFrames to Spark DataFrames
loans_sdf = loans_df.toDF(*loans_df.columns)
sales_sdf = sales_df.toDF(*sales_df.columns)
purchases_sdf = purchases_df.toDF(*purchases_df.columns)

# Calculate the ratio of the minority class
ratio = loans_sdf.filter(col("PAYMENT_STATUS") == "unpaid").count() / loans_sdf.count()

# Adjust the weights for the minority class in the feature generation
weights = {"loans": {"PAYMENT_STATUS": {"unpaid": 1.0 / ratio, "paid": 1.0}}}

# Create an entity set and add the loans entity
entity_set = ft.EntitySet(id="maxab_entity_set")
entity_set = entity_set.add_dataframe(
    dataframe_name="loans",
    dataframe=loans_sdf,
    index="INDEX",
    time_index="LOAN_ISSUANCE_DATE",
)

# Add the sales entity
entity_set = entity_set.add_dataframe(
    dataframe_name="sales", dataframe=sales_sdf, index=None, time_index="UPDATED_AT"
)

# Add the purchases entity
entity_set = entity_set.add_dataframe(
    dataframe_name="purchases",
    dataframe=purchases_sdf,
    index=None,
    time_index="ORDER_CREATION_DATE",
)

# Define relationships between the entities
r_sales_loans = ft.Relationship(
    entity_set["sales"]["MAIN_SYSTEM_ID"], entity_set["loans"]["MAIN_SYSTEM_ID"]
)
r_purchases_loans = ft.Relationship(
    entity_set["purchases"]["MAIN_SYSTEM_ID"], entity_set["loans"]["MAIN_SYSTEM_ID"]
)

# Add the relationships to the entity set
entity_set = entity_set.add_relationships([r_sales_loans, r_purchases_loans])

# Define the problem and the prediction target
problem = ft.ProblemDefinition(
    dataframe_name="loans",
    target_entity="loans",
    target_column="PAYMENT_STATUS",
    problem_type="classification",
)

# Create a feature extractor using tsfresh primitives
feature_extractor = TimeSeriesFeatureExtraction(
    default_fc_parameters=ComprehensiveFCParameters()
)

# Run deep feature synthesis to create features between the entities
feature_matrix, feature_defs = dfs(
    entityset=entity_set,
    target_entity="loans",
    verbose=2,
    problem_definition=problem,
    agg_primitives=["sum", "mean", "std", "max", "min", "count"],
    trans_primitives=["month", "weekday", "day", "year", feature_extractor],
    weightings=weights,
)
```
````
