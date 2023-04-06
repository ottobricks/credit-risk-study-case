# Deep Feature Synthesis

````{dropdown} Pandas
```{code-block}
import featuretools as ft
from featuretools_tsfresh_primitives import TimeSeriesFeatureExtraction
from tsfresh.feature_extraction import ComprehensiveFCParameters

# Convert the PySpark DataFrames to Pandas DataFrames
loans_pd = loans_df.toPandas()
sales_pd = sales_df.toPandas()
purchases_pd = purchases_df.toPandas()

# Create an entity set and add the loans entity
es = ft.EntitySet(id="my_entity_set")
es = es.entity_from_dataframe(
    entity_id="loans",
    dataframe=loans_pd,
    index="INDEX",
    time_index="LOAN_ISSUANCE_DATE",
)

# Add the sales entity
es = es.entity_from_dataframe(
    entity_id="sales", dataframe=sales_pd, index=None, time_index="CREATED_AT"
)

# Add the purchases entity
es = es.entity_from_dataframe(
    entity_id="purchases",
    dataframe=purchases_pd,
    index=None,
    time_index="ORDER_CREATION_DATE",
)

# Define relationships between the entities
r_sales_loans = ft.Relationship(
    es["sales"]["MAIN_SYSTEM_ID"], es["loans"]["MAIN_SYSTEM_ID"]
)
r_purchases_loans = ft.Relationship(
    es["purchases"]["MAIN_SYSTEM_ID"], es["loans"]["MAIN_SYSTEM_ID"]
)

# Add the relationships to the entity set
es = es.add_relationships([r_sales_loans, r_purchases_loans])

# Define the problem and the prediction target
problem = ft.ProblemDefinition(
    entity_id="loans",
    target_entity="loans",
    target_column="PAYMENT_STATUS",
    problem_type="classification",
)

# Create a feature extractor using tsfresh primitives
feature_extractor = TimeSeriesFeatureExtraction(
    default_fc_parameters=ComprehensiveFCParameters()
)

# Run deep feature synthesis to create features between the entities
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity="loans",
    verbose=2,
    problem_definition=problem,
    agg_primitives=["sum", "mean", "std", "max", "min", "count"],
    trans_primitives=["month", "weekday", "day", "year", feature_extractor],
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
es = ft.EntitySet(id="my_entity_set")
es = es.entity_from_dataframe(
    entity_id="loans",
    dataframe=loans_sdf,
    index="INDEX",
    time_index="LOAN_ISSUANCE_DATE",
)

# Add the sales entity
es = es.entity_from_dataframe(
    entity_id="sales", dataframe=sales_sdf, index=None, time_index="CREATED_AT"
)

# Add the purchases entity
es = es.entity_from_dataframe(
    entity_id="purchases",
    dataframe=purchases_sdf,
    index=None,
    time_index="ORDER_CREATION_DATE",
)

# Define relationships between the entities
r_sales_loans = ft.Relationship(
    es["sales"]["MAIN_SYSTEM_ID"], es["loans"]["MAIN_SYSTEM_ID"]
)
r_purchases_loans = ft.Relationship(
    es["purchases"]["MAIN_SYSTEM_ID"], es["loans"]["MAIN_SYSTEM_ID"]
)

# Add the relationships to the entity set
es = es.add_relationships([r_sales_loans, r_purchases_loans])

# Define the problem and the prediction target
problem = ft.ProblemDefinition(
    entity_id="loans",
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
    entityset=es,
    target_entity="loans",
    verbose=2,
    problem_definition=problem,
    agg_primitives=["sum", "mean", "std", "max", "min", "count"],
    trans_primitives=["month", "weekday", "day", "year", feature_extractor],
    weightings=weights,
)
```
````
