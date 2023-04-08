from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import coalesce, col, lit
from pyspark.sql.utils import AnalysisException
from pyspark.ml.evaluation import BinaryClassificationEvaluator, BinaryClassificationMetrics, MulticlassClassificationEvaluator
from xgboost.spark import SparkXGBClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline model fit and eval step")
    parser.add_argument(
        "--maxdepth",
        type=int,
        help="Decide which featureframe to run model fit on based on --maxdepth"
        + " used in the automatic feature engineering step",
    )
    args = parser.parse_args()
    maxdepth = args.maxdepth

    # Start or fetch active Spark session
    spark = SparkSession.builder.getOrCreate()

    xgb_classifier = SparkXGBClassifier(
        max_depth=10,
        missing=0.0,
        n_trees=10,
        weight_col="label",
        validation_indicator_col="is_validation",
        early_stopping_rounds=1,
        eval_metric="aucpr",
        num_workers=36,
        label_col="label",
        features_col="features"
    )

    model = booster.fit(train_df)
    model.transform(train_df).select("prediction", "probability").show(truncate=False)

    binaryEval = BinaryClassificationEvaluator(labelCol="label")
    binaryMetrics = BinaryClassificationMetrics(predictions.select("prediction", "label").rdd)
    multiEval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    aucpr = binaryMetrics.areaUnderPR
    fbeta2 = multiEval.evaluate(predictions, {multiEval.metricName: "fMeasureByLabel", multiEval.beta: 2.0})
    fbeta1 = multiEval.evaluate(predictions, {multiEval.metricName: "fMeasureByLabel", multiEval.beta: 1.0})
    fbeta05 = multiEval.evaluate(predictions, {multiEval.metricName: "fMeasureByLabel", multiEval.beta: 0.5})

    print("AUC-PR: %f" % aucpr)
    print("F-beta(2): %f" % fbeta2)
    print("F-beta(1): %f" % fbeta1)
    print("F-beta(0.5): %f" % fbeta05)
