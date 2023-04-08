import argparse
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline model fit and eval step")
    parser.add_argument(
        "--maxdepth",
        type=int,
        required=True,
        help="Decide which featureframe to run model fit on based on --maxdepth"
        + " used in the automatic feature engineering step",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name for ML task"
        + " used in the automatic feature engineering step",
    )
    args = parser.parse_args()
    maxdepth = args.maxdepth
    target_col = args.target

    # Start or fetch active Spark session
    spark = SparkSession.builder.getOrCreate()

    # Load featureframes
    dfs_featureframe = spark.read.parquet(
        # f"data/pca-featureframe-maxdepth{maxdepth}.parquet"
        f"s3a://ml-production-fraud-sagemaker-data/otto.sperling/tmp/pca-featureframe-maxdepth{maxdepth}.parquet"
    )
    # experiment_featureframe = spark.read.parquet(...)

    # Load targets
    targets = spark.createDataFrame(
        pd.read_excel("data/Loans_Data.xlsx")
        [[
            "LOAN_ID",
            "SPENT",
            "PAYMENT_STATUS",
            "FIRST_TRAIL_DELAYS",

        ]]
        .assign(
            LOAN_NOT_PAID=lambda x: (x["PAYMENT_STATUS"].str.lower() != "paid").astype("int"),
            FIRST_TRAIL_DELAY=lambda x: (x["FIRST_TRAIL_DELAYS"] > 0).astype("int"),
        )
        .drop(["FIRST_TRAIL_DELAYS", "PAYMENT_STATUS"], axis=1)
    )
    assert target_col.lower() in list(map(str.lower, targets.columns)), f"Column --target '{target_col}' not present in featureframe."
    targets = targets.select("LOAN_ID", col(target_col).alias("label"))

    # Join featureframes and targets
    # featureframe = (
    #     dfs_featureframe
    #     .join(experiment_featureframe, on="LOAN_ID")
    #     .join(targets, on="LOAN_ID")
    # )
    featureframe = dfs_featureframe.join(targets, on="LOAN_ID")

    # Stratified split based on the 'label' column
    featureframe = featureframe.withColumn(
        "stratified_split",
        expr("percent_rank() over (partition by label order by LOAN_ID)")
        # Assumes LOAN_ID is monotonically_increasing to avoid data leak
    ).persist()

    # Write test dataset to disk
    (
        featureframe.where(
            "(label = 1 and stratified_split >= 0.7)"
            + " or (label = 0 and stratified_split >= 0.8)"
        )
        .write.mode("overwrite")
        .parquet(
            # f"data/test/pca-featureframe-maxdepth{maxdepth}.parquet"
            # f"data/test/full-featureframe-maxdepth{maxdepth}.parquet"
            f"s3a://ml-production-fraud-sagemaker-data/otto.sperling/tmp/test/pca-featureframe-maxdepth{maxdepth}.parquet"
        )
    )

    # Write train dataset to disk
    (
        featureframe.where(
            "(label = 1 and stratified_split < 0.7)"
            + " or (label = 0 and stratified_split < 0.8)"
        )
        .withColumn(
            "validation_set",
            expr(
                "(label = 1 and stratified_split between 0.5 and 0.7)"
                + " or (label = 0 and stratified_split between 0.6 and 0.8)"
            )
        )
        .write.mode("overwrite")
        .parquet(
            # f"data/train/pca-featureframe-maxdepth{maxdepth}.parquet"
            # f"data/train/full-featureframe-maxdepth{maxdepth}.parquet"
            f"s3a://ml-production-fraud-sagemaker-data/otto.sperling/tmp/train/pca-featureframe-maxdepth{maxdepth}.parquet"
        )
    )
