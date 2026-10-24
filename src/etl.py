"""Phase 1 ETL: raw Olist CSVs -> validated, joined tables in data/processed/ (+ Snowflake).

Run from the project root:
    python -m src.etl                   # full pipeline incl. Snowflake upload
    python -m src.etl --skip-snowflake  # local outputs only

Steps: load (explicit schemas) -> validate raw -> clean -> aggregate to one row
per order -> join -> validate outputs -> write Parquet -> upload to Snowflake.
"""
import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, IntegerType, LongType

from src import validation as v
from src.spark_session import get_spark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "validation_report.csv"

# Explicit schemas (instead of inferSchema) for two reasons:
#  1. Correctness: zip prefixes like "01037" must stay strings. Type inference
#     would turn them into the integer 1037 and silently lose the leading zero.
#  2. Speed: inferSchema needs an extra full pass over each file just to guess types.
# Money is DOUBLE for simplicity; reconciliation checks use a small tolerance
# to absorb floating-point rounding.
RAW_TABLES = {
    "customers": ("olist_customers_dataset.csv",
                  "customer_id STRING, customer_unique_id STRING, customer_zip_code_prefix STRING, "
                  "customer_city STRING, customer_state STRING"),
    "geolocation": ("olist_geolocation_dataset.csv",
                    "geolocation_zip_code_prefix STRING, geolocation_lat DOUBLE, geolocation_lng DOUBLE, "
                    "geolocation_city STRING, geolocation_state STRING"),
    "orders": ("olist_orders_dataset.csv",
               "order_id STRING, customer_id STRING, order_status STRING, order_purchase_timestamp TIMESTAMP, "
               "order_approved_at TIMESTAMP, order_delivered_carrier_date TIMESTAMP, "
               "order_delivered_customer_date TIMESTAMP, order_estimated_delivery_date TIMESTAMP"),
    "order_items": ("olist_order_items_dataset.csv",
                    "order_id STRING, order_item_id INT, product_id STRING, seller_id STRING, "
                    "shipping_limit_date TIMESTAMP, price DOUBLE, freight_value DOUBLE"),
    "order_payments": ("olist_order_payments_dataset.csv",
                       "order_id STRING, payment_sequential INT, payment_type STRING, "
                       "payment_installments INT, payment_value DOUBLE"),
    "order_reviews": ("olist_order_reviews_dataset.csv",
                      "review_id STRING, order_id STRING, review_score INT, review_comment_title STRING, "
                      "review_comment_message STRING, review_creation_date TIMESTAMP, "
                      "review_answer_timestamp TIMESTAMP"),
    "products": ("olist_products_dataset.csv",
                 "product_id STRING, product_category_name STRING, product_name_lenght INT, "
                 "product_description_lenght INT, product_photos_qty INT, product_weight_g DOUBLE, "
                 "product_length_cm DOUBLE, product_height_cm DOUBLE, product_width_cm DOUBLE"),
    "sellers": ("olist_sellers_dataset.csv",
                "seller_id STRING, seller_zip_code_prefix STRING, seller_city STRING, seller_state STRING"),
    "category_translation": ("product_category_name_translation.csv",
                             "product_category_name STRING, product_category_name_english STRING"),
}

# Two categories exist in products but are missing from the translation file.
MANUAL_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_and_food_preparers",
}

# Rough bounding box of Brazil. A few geolocation points are clearly wrong
# (e.g. in Europe); we drop those before averaging coordinates per zip.
BRAZIL_LAT = (-34.0, 5.5)
BRAZIL_LNG = (-74.0, -34.0)


# --- 1. Load -------------------------------------------------------------------
def load_raw_tables(spark: SparkSession) -> dict[str, DataFrame]:
    tables = {}
    for name, (filename, schema) in RAW_TABLES.items():
        df = (
            spark.read
            .option("header", True)
            # Review comments contain line breaks inside quoted fields; without
            # multiLine, Spark would split one review into several broken rows.
            .option("multiLine", True)
            # CSV escapes a quote inside a quoted field by doubling it ("").
            # Spark's default escape character is a backslash, so set it explicitly.
            .option("escape", '"')
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
            # FAILFAST: crash on a row that doesn't fit the schema. The default
            # (PERMISSIVE) would silently turn unparseable values into nulls.
            .option("mode", "FAILFAST")
            .schema(schema)
            .csv(str(RAW_DIR / filename))
        )
        # cache(): these tables are reused by many checks and joins, so keep them
        # in memory instead of re-reading the CSV each time.
        tables[name] = df.cache()
    return tables


# --- 2. Raw-data validation ------------------------------------------------------
def run_raw_checks(raw: dict[str, DataFrame]) -> list[v.CheckResult]:
    o, c, i = raw["orders"], raw["customers"], raw["order_items"]
    p, r = raw["order_payments"], raw["order_reviews"]
    prod, s, t, g = raw["products"], raw["sellers"], raw["category_translation"], raw["geolocation"]

    results = [
        # Primary keys: each must identify exactly one row.
        v.check_unique(o, ["order_id"], "orders"),
        v.check_unique(c, ["customer_id"], "customers"),
        v.check_unique(prod, ["product_id"], "products"),
        v.check_unique(s, ["seller_id"], "sellers"),
        v.check_unique(t, ["product_category_name"], "category_translation"),
        v.check_unique(i, ["order_id", "order_item_id"], "order_items"),
        v.check_unique(p, ["order_id", "payment_sequential"], "order_payments"),
        v.check_unique(r, ["review_id", "order_id"], "order_reviews"),
        # Foreign keys: no orphan rows pointing at things that don't exist.
        v.check_foreign_key(o, "customer_id", c, "customer_id", "orders", "customers"),
        v.check_foreign_key(i, "order_id", o, "order_id", "order_items", "orders"),
        v.check_foreign_key(i, "product_id", prod, "product_id", "order_items", "products"),
        v.check_foreign_key(i, "seller_id", s, "seller_id", "order_items", "sellers"),
        v.check_foreign_key(p, "order_id", o, "order_id", "order_payments", "orders"),
        v.check_foreign_key(r, "order_id", o, "order_id", "order_reviews", "orders"),
        # Required fields.
        v.check_not_null(o, "customer_id", "orders"),
        v.check_not_null(o, "order_status", "orders"),
        v.check_not_null(o, "order_purchase_timestamp", "orders"),
        v.check_not_null(o, "order_estimated_delivery_date", "orders"),
        v.check_not_null(c, "customer_unique_id", "customers"),
        v.check_not_null(i, "price", "order_items"),
        v.check_not_null(p, "payment_value", "order_payments"),
        v.check_not_null(r, "review_score", "order_reviews"),
        # Value ranges.
        v.check_in_range(r, "review_score", "order_reviews", 1, 5),
        v.check_in_range(i, "price", "order_items", min_value=0.01),
        v.check_in_range(i, "freight_value", "order_items", min_value=0),
        v.check_in_range(p, "payment_value", "order_payments", min_value=0),

        # ---- Warnings: known real-world quirks, kept and flagged, not fatal ----
        v.check_has_match(o, "order_id", i, "order_id", "orders", "order_items"),
        v.check_has_match(o, "order_id", p, "order_id", "orders", "order_payments"),
        v.check_unique(r, ["order_id"], "order_reviews", severity="warning"),
        v.check_unique(r, ["review_id"], "order_reviews", severity="warning"),
        v.check_not_null(prod, "product_category_name", "products", severity="warning"),
        v.check_foreign_key(prod.filter(F.col("product_category_name").isNotNull()), "product_category_name",
                            t, "product_category_name", "products", "category_translation",
                            severity="warning"),
        v.check_in_range(p, "payment_installments", "order_payments", min_value=1, severity="warning"),
        v.check_condition(o, (F.col("order_status") == "delivered")
                          & F.col("order_delivered_customer_date").isNull(),
                          "delivered_no_date", "orders", "order_delivered_customer_date"),
        v.check_condition(o, bad_timestamp_order(), "timestamp_order", "orders",
                          "purchase<=approved<=carrier<=delivered"),
        v.check_unique(g, g.columns, "geolocation", severity="warning"),
        v.check_foreign_key(c, "customer_zip_code_prefix", g, "geolocation_zip_code_prefix",
                            "customers", "geolocation", severity="warning"),
    ]
    return results


def bad_timestamp_order():
    """True when the order lifecycle timestamps are out of sequence.
    Comparisons with a null timestamp are null, so coalesce turns them into False."""
    purchase, approved = F.col("order_purchase_timestamp"), F.col("order_approved_at")
    carrier, delivered = F.col("order_delivered_carrier_date"), F.col("order_delivered_customer_date")
    return F.coalesce(
        (approved < purchase) | (carrier < purchase) | (delivered < purchase) | (delivered < carrier),
        F.lit(False),
    )


# --- 3. Clean ----------------------------------------------------------------------
def clean_products(products: DataFrame, translation: DataFrame) -> DataFrame:
    manual = F.create_map(*[F.lit(x) for pair in MANUAL_TRANSLATIONS.items() for x in pair])
    pt_name = F.col("product_category_name")
    return (
        products.join(translation, on="product_category_name", how="left")
        .withColumn(
            "product_category",
            # No category -> "unknown"; otherwise official translation, then our
            # manual one, then (as a last resort) the Portuguese name itself.
            F.when(pt_name.isNull(), F.lit("unknown"))
            .otherwise(F.coalesce(F.col("product_category_name_english"), manual[pt_name], pt_name)),
        )
        .select(
            "product_id",
            "product_category",
            # Fix the misspelt source column names ("lenght").
            F.col("product_name_lenght").alias("product_name_length"),
            F.col("product_description_lenght").alias("product_description_length"),
            "product_photos_qty", "product_weight_g",
            "product_length_cm", "product_height_cm", "product_width_cm",
        )
    )


def build_geo_by_zip(geolocation: DataFrame) -> DataFrame:
    """Collapse ~1M geolocation points into one row per zip prefix.
    Median (not mean) so a few bad coordinates can't drag the location away."""
    in_brazil = (F.col("geolocation_lat").between(*BRAZIL_LAT)
                 & F.col("geolocation_lng").between(*BRAZIL_LNG))
    return (
        geolocation.filter(in_brazil)
        .groupBy(F.col("geolocation_zip_code_prefix").alias("zip_code_prefix"))
        .agg(
            F.median("geolocation_lat").alias("lat"),
            F.median("geolocation_lng").alias("lng"),
            F.count("*").alias("n_geo_points"),
        )
    )


# --- 4. Aggregate child tables to one row per order --------------------------------
# Joining items/payments/reviews straight onto orders would multiply rows
# (an order with 3 items would appear 3 times and its totals would triple).
# So each child table is first reduced to exactly one row per order_id.
def aggregate_items(items: DataFrame, products_clean: DataFrame) -> DataFrame:
    items = items.join(products_clean.select("product_id", "product_category"), "product_id", "left")
    # "Main category" = category of the most expensive item. row_number with a
    # tie-breaker (order_item_id) makes the choice deterministic on ties.
    by_price = Window.partitionBy("order_id").orderBy(F.desc("price"), F.asc("order_item_id"))
    main_category = (
        items.withColumn("rn", F.row_number().over(by_price))
        .filter(F.col("rn") == 1)
        .select("order_id", F.col("product_category").alias("main_category"))
    )
    totals = items.groupBy("order_id").agg(
        F.count("*").alias("n_items"),
        F.countDistinct("product_id").alias("n_products"),
        F.countDistinct("seller_id").alias("n_sellers"),
        F.round(F.sum("price"), 2).alias("items_price_total"),
        F.round(F.sum("freight_value"), 2).alias("freight_total"),
    )
    return totals.join(main_category, "order_id")


def aggregate_payments(payments: DataFrame) -> DataFrame:
    by_value = Window.partitionBy("order_id").orderBy(F.desc("payment_value"), F.asc("payment_sequential"))
    main_type = (
        payments.withColumn("rn", F.row_number().over(by_value))
        .filter(F.col("rn") == 1)
        .select("order_id", F.col("payment_type").alias("main_payment_type"))
    )
    totals = payments.groupBy("order_id").agg(
        F.count("*").alias("n_payments"),
        F.round(F.sum("payment_value"), 2).alias("payment_total"),
        F.max("payment_installments").alias("max_installments"),
        F.max(F.col("payment_type") == "voucher").alias("used_voucher"),
    )
    return totals.join(main_type, "order_id")


def clean_reviews(reviews: DataFrame) -> DataFrame:
    """One row per (review, order) pair, text kept for NLP.

    Some review_ids are attached to several orders (one review covering a
    multi-order checkout). We keep those rows so every order keeps its score, and
    flag them so the NLP phase can de-duplicate the text by review_id.
    """
    latest_first = Window.partitionBy("order_id").orderBy(
        F.desc("review_answer_timestamp"), F.desc("review_creation_date"), F.asc("review_id"))
    per_review = Window.partitionBy("review_id")
    message = F.trim(F.col("review_comment_message"))
    return (
        reviews
        .withColumn("has_comment", message.isNotNull() & (message != ""))
        .withColumn("review_shared_across_orders", F.count("*").over(per_review) > 1)
        .withColumn("is_latest_review_for_order", F.row_number().over(latest_first) == 1)
    )


def aggregate_reviews(reviews_clean: DataFrame) -> DataFrame:
    """Per order: keep the most recent review (547 orders have more than one)."""
    n_reviews = reviews_clean.groupBy("order_id").agg(F.count("*").alias("n_reviews"))
    latest = reviews_clean.filter("is_latest_review_for_order").select(
        "order_id", "review_score", F.col("has_comment").alias("review_has_comment"), "review_creation_date")
    return latest.join(n_reviews, "order_id")


# --- 5. Build output tables -----------------------------------------------------------
def build_orders_enriched(raw: dict[str, DataFrame], products_clean: DataFrame,
                          reviews_clean: DataFrame, geo_by_zip: DataFrame) -> DataFrame:
    customer_geo = geo_by_zip.select(
        F.col("zip_code_prefix").alias("customer_zip_code_prefix"),
        F.col("lat").alias("customer_lat"), F.col("lng").alias("customer_lng"))

    purchase_date = F.to_date("order_purchase_timestamp")
    delivered_date = F.to_date("order_delivered_customer_date")
    estimated_date = F.to_date("order_estimated_delivery_date")

    # Left joins everywhere: every order is kept even if it has no items,
    # payment or review. Missing pieces show up as nulls + flags, not lost rows.
    return (
        raw["orders"]
        .join(raw["customers"], "customer_id", "left")
        .join(customer_geo, "customer_zip_code_prefix", "left")
        .join(aggregate_items(raw["order_items"], products_clean), "order_id", "left")
        .join(aggregate_payments(raw["order_payments"]), "order_id", "left")
        .join(aggregate_reviews(reviews_clean), "order_id", "left")
        # Delivery timing, in whole calendar days. The estimated date has no time
        # of day (always 00:00), so comparing exact timestamps would mark an order
        # delivered at 3pm on its estimated date as "late". Comparing dates avoids that.
        .withColumn("delivery_days", F.datediff(delivered_date, purchase_date))
        .withColumn("estimated_days", F.datediff(estimated_date, purchase_date))
        .withColumn("delay_days", F.datediff(delivered_date, estimated_date))  # >0 means late
        .withColumn("is_late", F.col("delay_days") > 0)  # null if never delivered
        # Data-quality flags: keep the rows, let each later phase decide.
        .withColumn("flag_no_items", F.col("n_items").isNull())
        .withColumn("flag_no_payment", F.col("payment_total").isNull())
        .withColumn("flag_delivered_missing_date",
                    (F.col("order_status") == "delivered") & F.col("order_delivered_customer_date").isNull())
        .withColumn("flag_bad_timestamp_order", bad_timestamp_order())
        # False also when items or payment are missing (see the two flags above).
        .withColumn("flag_payment_mismatch", F.coalesce(v.payment_mismatch_col(), F.lit(False)))
        .select(
            "order_id", "customer_id", "customer_unique_id",
            "customer_zip_code_prefix", "customer_city", "customer_state", "customer_lat", "customer_lng",
            "order_status", "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date",
            "delivery_days", "estimated_days", "delay_days", "is_late",
            "n_items", "n_products", "n_sellers", "items_price_total", "freight_total", "main_category",
            "n_payments", "payment_total", "max_installments", "main_payment_type", "used_voucher",
            "n_reviews", "review_score", "review_has_comment", "review_creation_date",
            "flag_no_items", "flag_no_payment", "flag_delivered_missing_date",
            "flag_bad_timestamp_order", "flag_payment_mismatch",
        )
    )


def build_order_items_clean(raw: dict[str, DataFrame], products_clean: DataFrame) -> DataFrame:
    """Item-level table (one row per order line) for seller-level analysis in Phase 4."""
    return (
        raw["order_items"]
        .join(products_clean.select("product_id", "product_category", "product_weight_g"), "product_id", "left")
        .join(raw["sellers"], "seller_id", "left")
        .join(raw["orders"].select("order_id", "order_purchase_timestamp", "order_status"), "order_id", "left")
        .select(
            "order_id", "order_item_id", "product_id", "seller_id",
            "order_purchase_timestamp", "order_status", "shipping_limit_date",
            "price", "freight_value", "product_category", "product_weight_g",
            "seller_zip_code_prefix", "seller_city", "seller_state",
        )
    )


def build_tables(raw: dict[str, DataFrame]) -> dict[str, DataFrame]:
    products_clean = clean_products(raw["products"], raw["category_translation"])
    reviews_clean = clean_reviews(raw["order_reviews"]).cache()
    geo_by_zip = build_geo_by_zip(raw["geolocation"]).cache()
    return {
        "orders_enriched": build_orders_enriched(raw, products_clean, reviews_clean, geo_by_zip).cache(),
        "order_items_clean": build_order_items_clean(raw, products_clean).cache(),
        "reviews_clean": reviews_clean,
        "geo_by_zip": geo_by_zip,
    }


# --- 6. Output validation ----------------------------------------------------------------
def run_output_checks(raw: dict[str, DataFrame], out: dict[str, DataFrame]) -> list[v.CheckResult]:
    orders, items = out["orders_enriched"], out["order_items_clean"]
    raw_totals = raw["order_items"].agg(F.sum("price"), F.sum("freight_value")).first()
    raw_paid = raw["order_payments"].agg(F.sum("payment_value")).first()[0]
    return [
        # Same row count as raw orders + unique order_id = the joins did not fan out or drop rows.
        v.check_row_count(orders, raw["orders"].count(), "orders_enriched"),
        v.check_unique(orders, ["order_id"], "orders_enriched"),
        v.check_not_null(orders, "customer_unique_id", "orders_enriched"),
        v.check_row_count(items, raw["order_items"].count(), "order_items_clean"),
        v.check_unique(items, ["order_id", "order_item_id"], "order_items_clean"),
        v.check_row_count(out["reviews_clean"], raw["order_reviews"].count(), "reviews_clean"),
        v.check_unique(out["geo_by_zip"], ["zip_code_prefix"], "geo_by_zip"),
        # Reconciliation: money totals must survive aggregation unchanged.
        v.check_sum_matches(orders, "items_price_total", raw_totals[0], "orders_enriched"),
        v.check_sum_matches(orders, "freight_total", raw_totals[1], "orders_enriched"),
        v.check_sum_matches(orders, "payment_total", raw_paid, "orders_enriched"),
        # Warnings.
        v.check_payment_matches_items(orders),
        v.check_not_null(orders, "customer_lat", "orders_enriched", severity="warning"),
    ]


# --- 7. Write --------------------------------------------------------------------------
def to_pandas_typed(df: DataFrame):
    """toPandas(), keeping integer and boolean columns that contain nulls.

    Plain NumPy has no "missing" value for ints or bools, so toPandas() turns an
    int column with nulls into float (e.g. n_items = 2.0) and a bool column into
    generic objects. Pandas' nullable dtypes ("Int64", "boolean") fix that.
    """
    pdf = df.toPandas()
    for field in df.schema.fields:
        if isinstance(field.dataType, (IntegerType, LongType)):
            pdf[field.name] = pdf[field.name].astype("Int64")
        elif isinstance(field.dataType, BooleanType):
            pdf[field.name] = pdf[field.name].astype("boolean")
    return pdf


def write_parquet(tables: dict[str, DataFrame]) -> dict:
    """Collect each table to pandas and write it with pyarrow.

    Why not df.write.parquet()? On Windows, Spark's native writer needs extra
    Hadoop binaries (winutils.exe). Our outputs are small (<= ~113k rows), so
    collecting them to the driver is safe and avoids that dependency. With
    much larger data we would let Spark write the files itself.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, df in tables.items():
        pdf = to_pandas_typed(df)
        path = PROCESSED_DIR / f"{name}.parquet"
        pdf.to_parquet(path, index=False)
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}  ({len(pdf):,} rows x {pdf.shape[1]} cols)")
        frames[name] = pdf
    return frames


def main(skip_snowflake: bool = False) -> None:
    spark = get_spark("churn-insights-etl")

    print("Loading raw CSVs...")
    raw = load_raw_tables(spark)

    print("Validating raw data...")
    raw_results = run_raw_checks(raw)
    v.print_results(raw_results)
    v.raise_on_failures(raw_results)  # don't transform data we know is broken

    print("Building output tables...")
    tables = build_tables(raw)

    print("Validating outputs...")
    output_results = run_output_checks(raw, tables)
    v.print_results(output_results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    v.results_to_frame(raw_results + output_results).to_csv(REPORT_PATH, index=False)
    print(f"  report -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    v.raise_on_failures(output_results)  # never write outputs that failed validation

    print("Writing Parquet...")
    frames = write_parquet(tables)

    if skip_snowflake:
        print("Skipping Snowflake upload (--skip-snowflake).")
    else:
        # Imported here so the local pipeline still works if the Snowflake
        # connector has a problem.
        from src.snowflake_load import upload_tables
        print("Uploading to Snowflake...")
        upload_tables(frames)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 ETL pipeline")
    parser.add_argument("--skip-snowflake", action="store_true", help="only write local Parquet files")
    main(skip_snowflake=parser.parse_args().skip_snowflake)
