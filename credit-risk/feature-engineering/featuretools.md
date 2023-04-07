# Deep Feature Synthesis

````{dropdown} Pandas
```{code-block}
import argparse
from typing import List, Tuple
import itertools
import pandas as pd
import numpy as np
import featuretools as ft
from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools_tsfresh_primitives import (
    comprehensive_fc_parameters,
    primitives_from_fc_settings,
)
from featuretools_tsfresh_primitives import primitives as tsfresh_primitives
from featuretools_tsfresh_primitives.primitives import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse CLI args")
    parser.add_argument(
        "--retailerid",
        type=int,
        # action=S3UriParser,
        help="UUID to identify MLflow experiment run",
    )
    args = parser.parse_args()
    selected_main_system_id = args.retailerid

    # Load datasets
    loans_df = (
        pd.read_excel("data/Loans_Data.xlsx")
        .drop(
            [
                # Low signal columns
                "INITIAL_COST",
                "INDEX",
                "REPAYMENT_ID",
                "FINAL_COST",
                "RETAILER_ID",
                # Columns populated after the fact, thus would lead to data leak
                "REPAYMENT_UPDATED",
                "SPENT",
                "TOTAL_FINAL_AMOUNT",
                "FIRST_TRIAL_BALANCE",
                "FIRST_TRAIL_DELAYS",
                "PAYMENT_AMOUNT",
                "LOAN_PAYMENT_DATE",
                "REPAYMENT_AMOUNT",
                "CUMMULATIVE_OUTSTANDING",
            ],
            axis=1,
        )
        .query(f"MAIN_SYSTEM_ID == {selected_main_system_id}")
        .assign(
            MAIN_SYSTEM_ID=lambda x: x["MAIN_SYSTEM_ID"].astype("int64"),
            LOAN_ID=lambda x: x["LOAN_ID"].astype("int64"),
            LOAN_ISSUANCE_DATE=lambda x: x["LOAN_ISSUANCE_DATE"].astype("<M8[ns]"),
            LOAN_AMOUNT=lambda x: x["LOAN_AMOUNT"].astype("float64"),
            TOTAL_INITIAL_AMOUNT=lambda x: x["TOTAL_INITIAL_AMOUNT"].astype("float64"),
            INITIAL_DATE=lambda x: x["INITIAL_DATE"].astype("<M8[ns]"),
            PAYMENT_STATUS=lambda x: x["PAYMENT_STATUS"].astype("O"),
        )
    )
    fintech_df = (
        pd.read_csv(
            "data/Retailer_Transactions_Data.csv",
            header=0,
            dtype={
                "ID": np.dtype("int64"),
                "CREATED_AT": np.dtype("O"),
                "UPDATED_AT": np.dtype("O"),
                "AMOUNT": np.dtype("float64"),
                "FEES": np.dtype("float64"),
                "RETAILER_CUT": np.dtype("float64"),
                "STATUS": np.dtype("O"),
                "TOTAL_AMOUNT_INCLUDING_TAX": np.dtype("float64"),
                "TOTAL_AMOUNT_PAID": np.dtype("float64"),
                "WALLET_BALANCE_BEFORE_TRANSACTION": np.dtype("float64"),
                "MAIN_SYSTEM_ID": np.dtype("int64"),
            },
        )
        .query(f"MAIN_SYSTEM_ID == {selected_main_system_id}")
        .assign(
            CREATED_AT=lambda x: pd.to_datetime(x["CREATED_AT"], infer_datetime_format=True),
            UPDATED_AT=lambda x: pd.to_datetime(x["UPDATED_AT"], infer_datetime_format=True),
        )
    )

    ecommerce_df = (
        pd.read_csv(
            "data/Ecommerce_orders_Data.csv",
            header=0,
            dtype={
                "ORDER_ID": np.dtype("int64"),
                "MAIN_SYSTEM_ID": np.dtype("int64"),
                "ORDER_PRICE": np.dtype("float64"),
                "DISCOUNT": np.dtype("float64"),
                "ORDER_PRICE_AFTER_DISCOUNT": np.dtype("float64"),
                "ORDER_CREATION_DATE": np.dtype("O"),
            },
        )
        .query(f"MAIN_SYSTEM_ID == {selected_main_system_id}")
        .assign(
            ORDER_CREATION_DATE=lambda x: pd.to_datetime(x["ORDER_CREATION_DATE"], infer_datetime_format=True),
        )
    )

    target_df = loans_df.pop("PAYMENT_STATUS")
    retailer_df = (
        pd.concat(
            [
                loans_df["MAIN_SYSTEM_ID"].drop_duplicates(),
                fintech_df["MAIN_SYSTEM_ID"].drop_duplicates(),
                ecommerce_df["MAIN_SYSTEM_ID"].drop_duplicates(),
            ]
        )
        .drop_duplicates()
        .reset_index()[["MAIN_SYSTEM_ID"]]
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
        # secondary_time_index={
        #     "REPAYMENT_UPDATED": [
        #         "SPENT",
        #         "TOTAL_FINAL_AMOUNT",
        #         "FIRST_TRIAL_BALANCE",
        #         "FIRST_TRAIL_DELAYS",
        #         "PAYMENT_AMOUNT",
        #         "LOAN_PAYMENT_DATE",
        #         "REPAYMENT_AMOUNT",
        #         "CUMMULATIVE_OUTSTANDING",
        #     ]
        # }
    )

    # Add the sales entity
    entity_set = entity_set.add_dataframe(
        dataframe_name="sales",
        dataframe=fintech_df,
        index="ID",
        time_index="CREATED_AT",
        secondary_time_index={"UPDATED_AT": ["STATUS", "TOTAL_AMOUNT_PAID"]},
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
        child_column_name="MAIN_SYSTEM_ID",
    )

    rel_retailer_purchases = ft.Relationship(
        entity_set,
        parent_dataframe_name="retailers",
        parent_column_name="MAIN_SYSTEM_ID",
        child_dataframe_name="purchases",
        child_column_name="MAIN_SYSTEM_ID",
    )

    rel_retailer_loans = ft.Relationship(
        entity_set,
        parent_dataframe_name="retailers",
        parent_column_name="MAIN_SYSTEM_ID",
        child_dataframe_name="loans",
        child_column_name="MAIN_SYSTEM_ID",
    )

    # Add the relationships to the entity set
    entity_set = entity_set.add_relationships(
        [rel_retailer_sales, rel_retailer_purchases, rel_retailer_loans]
    )

    # Create list of Featuretools primitives
    ft_valid_primitives_tuple = ft.get_valid_primitives(
        entity_set, target_dataframe_name="loans", max_depth=2
    )
    FT_AGG_PRIMITIVES: List[str] = list(
        map(lambda x: x().name, ft_valid_primitives_tuple[0])
    )
    FT_TRANSFORM_PRIMITIVES: List[str] = list(
        map(lambda x: x().name, ft_valid_primitives_tuple[1])
    )

    # Remove buggy primitive 'expanding_count'
    try:
        FT_TRANSFORM_PRIMITIVES.remove("expanding_count")
    except:
        pass

    # Create list of TSFresh primitives
    TSFRESH_PARAMETERS = comprehensive_fc_parameters()

    # Creates a list where each element refers one combination of primitive(parameters)
    TSF_AGG_PRIMITIVES: List[AggregationPrimitive] = list(
        itertools.chain(
            *[
                primitives_from_fc_settings(
                    {
                        getattr(tsfresh_primitives, key).name: TSFRESH_PARAMETERS[
                            getattr(tsfresh_primitives, key).name
                        ]
                        or [{}]
                    }
                )
                for key in dir(tsfresh_primitives)
                if key[0].isupper()
                and key != "SUPPORTED_PRIMITIVES"
                and isinstance(
                    getattr(tsfresh_primitives, key)(
                        **(
                            TSFRESH_PARAMETERS[getattr(tsfresh_primitives, key).name]
                            or [{}]
                        )[0]
                    ),
                    AggregationPrimitive,
                )
            ]
        )
    )

    # Creates a list where each element refers to one combination of primitive(parameters)
    TSF_TRANSFORM_PRIMITIVES: List[TransformPrimitive] = list(
        itertools.chain(
            *[
                primitives_from_fc_settings(
                    {
                        getattr(tsfresh_primitives, key).name: TSFRESH_PARAMETERS[
                            getattr(tsfresh_primitives, key).name
                        ]
                        or [{}]
                    }
                )
                for key in dir(tsfresh_primitives)
                if key[0].isupper()
                and key != "SUPPORTED_PRIMITIVES"
                and isinstance(
                    getattr(tsfresh_primitives, key)(
                        **(
                            TSFRESH_PARAMETERS[getattr(tsfresh_primitives, key).name]
                            or [{}]
                        )[0]
                    ),
                    TransformPrimitive,
                )
            ]
        )
    )

    # Run deep feature synthesis to create features between the entities
    ft_featureframe, ft_feature_defs = ft.dfs(
        entityset=entity_set,
        target_dataframe_name="loans",
        verbose=1,
        agg_primitives=FT_AGG_PRIMITIVES,
        trans_primitives=FT_TRANSFORM_PRIMITIVES,
        max_depth=3,
        # max_features=250,
    )

    tsf_featureframe, tsf_feature_defs = ft.dfs(
        entityset=entity_set,
        target_dataframe_name="loans",
        verbose=1,
        agg_primitives=TSF_AGG_PRIMITIVES,
        trans_primitives=TSF_TRANSFORM_PRIMITIVES,
        max_depth=3,
        # max_features=250,
    )

    featureframe = pd.merge(ft_featureframe, tsf_featureframe, on="MAIN_SYSTEM_ID")
    feature_defs = list(set(ft_feature_defs).union(set(tsf_feature_defs)))

    # Clean up the feature frame a bit
    feature_matrix, feature_defs = ft.selection.remove_highly_null_features(
        featureframe, pct_null_threshold=0.25, features=feature_defs
    )
    feature_matrix, feature_defs = ft.selection.remove_low_information_features(
        featureframe, features=feature_defs
    )
    feature_matrix, feature_defs = ft.selection.remove_single_value_features(
        featureframe, features=feature_defs
    )
    feature_matrix, feature_defs = ft.selection.remove_single_value_features(
        featureframe, features=feature_defs
    )
    feature_matrix, feature_defs = ft.selection.remove_highly_correlated_features(
        featureframe, pct_corr_threshold=0.8, features=feature_defs
    )

    feature_matrix.to_parquet(
        f"/home/sagemaker-user/otto/.cred/featureframe.parquet/MAIN_SYSTEM_ID={selected_main_system_id}/",
        engine="pyarrow",
    )

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
