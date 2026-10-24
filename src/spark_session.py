"""Create a local SparkSession configured for this project.

Every phase that uses PySpark calls get_spark() so the settings live in one place.
"""
import os
import sys

from pyspark.sql import SparkSession


def get_spark(app_name: str = "churn-insights") -> SparkSession:
    # Spark launches separate Python worker processes. Without these two lines,
    # Windows may pick up a different Python from PATH (e.g. 3.13) and fail with
    # a "Python version mismatch" error. Point both at the .venv interpreter.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        # local[*] = run Spark on this machine using all CPU cores (no cluster).
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        # Interpret all timestamps as UTC so nothing gets shifted by the
        # laptop's time zone when converting to pandas.
        .config("spark.sql.session.timeZone", "UTC")
        # Default is 200 shuffle partitions, tuned for clusters. For ~100k rows
        # on one machine, 200 tiny tasks are pure overhead.
        .config("spark.sql.shuffle.partitions", "8")
        # Use Apache Arrow for fast Spark -> pandas conversion (toPandas()).
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Hide the [Stage 12: ====>] progress bars that clutter the console.
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
