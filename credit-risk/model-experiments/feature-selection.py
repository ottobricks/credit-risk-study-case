import argparse
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, VectorAssembler, StringIndexer, Imputer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection step")
    parser.add_argument(
        "--maxdepth",
        type=int,
        help="Decide which featureframe to run selection on based on --maxdepth"
        + " used in the automatic feature engineering step",
    )
    args = parser.parse_args()
    maxdepth = args.maxdepth

    # Start or fetch active Spark session
    spark = SparkSession.builder.getOrCreate()

    # Load data
    featureframe = (
        spark.read.option("mergeSchema", "true")
        .parquet(
            f"s3a://ml-production-fraud-sagemaker-data/otto.sperling/tmp/featureframe-maxdepth{maxdepth}.parquet"
            # "data/featureframe-maxdepth2.parquet"
        )
        .drop("__index_level_0__")
        .coalesce(int(spark.sparkContext.getConf().get("spark.executor.instances", "2")))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # Parse column names to comply with SparkSQL
    featureframe = featureframe.select(
        [
            col(f"`{column}`").alias(
                column.replace("(", "__")
                .replace(")", "__")
                .replace(",", "_")
                .replace(".", "_")
                .replace(" % ", "_mod_")
                .replace(" / ", "_div_")
                .replace(" * ", "_mul_")
            )
            for column in featureframe.columns
        ]
    )

    # This should no longer be necessary if _fetch_largest_dtype_for_numeric_feature works in dfs
    # # Convert boolean columns to numeric
    # featureframe = featureframe.select(
    #     [
    #         col(column_name)
    #         if column_dtype != "boolean"
    #         else col(column_name).cast("int").alias(column_name)
    #         for column_name, column_dtype in featureframe.dtypes
    #     ]
    # )
    metadata_cols = ["LOAN_ID", "MAIN_SYSTEM_ID"]

    # Split featureframe into categorical and numeric columns
    cat_cols = [
        col_name for col_name, col_type in featureframe.dtypes if col_type == "string"
    ]
    num_cols = list(
        set(featureframe.drop(*metadata_cols).columns).difference(set(cat_cols))
    )
    assert set(cat_cols).intersection(set(num_cols)) == set(
        []
    ), "Failed to split featureframe into categorical and numeric features"

    # Define an encoder for categorical features
    indexer_cols = [col_name + "_index" for col_name in cat_cols]
    indexer = StringIndexer(
        inputCols=cat_cols, outputCols=indexer_cols, handleInvalid="keep"
    )

    # Define mean imputer for numeric columns
    imputer_cols = [col_name + "_imputed" for col_name in num_cols]
    imputer = Imputer(inputCols=num_cols, outputCols=imputer_cols)

    # Define a vector assembler
    assembler = VectorAssembler(
        inputCols=[*imputer_cols, *indexer_cols],
        outputCol="assembled_features",
    )

    # Define a PCA model
    pca = PCA(k=10, inputCol="assembled_features", outputCol="pca_features")

    # Define a PipelineModel
    pipeline = Pipeline(stages=[indexer, imputer, assembler, pca])

    # Fit the model to the data and persist it
    model = pipeline.fit(featureframe)
    model.write().overwrite().save(
        # f"s3a://ml-production-fraud-sagemaker-data/otto.sperling/tmp/assets/pca-model-maxdepth{maxdepth}.mllib"
        f"credit-risk/feature-engineering/pca-model-maxdepth{maxdepth}.mllib"
    )

    # Transform the data using the model
    selected_features = model.transform(featureframe).select(
        *metadata_cols, "pca_features"
    )

    selected_features.coalesce(96).write.mode("overwrite").parquet(
        # f"data/pca-featureframe-maxdepth{maxdepth}.parquet"
        f"s3a://ml-production-fraud-sagemaker-data/otto.sperling/tmp/pca-featureframe-maxdepth{maxdepth}.parquet"
    )
