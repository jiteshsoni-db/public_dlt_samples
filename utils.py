from datetime import datetime, timedelta
from functools import reduce  # For Python 3.x
from typing import List, Tuple, Union
from timeit import default_timer as timer
import time
import logging
import functools

import pyspark.sql.utils
from pyspark.context import SparkContext
from pyspark.dbutils import DBUtils
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import expr, desc
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
dbutils = DBUtils(spark)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)-4s %(name)s][%(funcName)s] %(message)s",
)

logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def str_to_bool(value):
    false_values = ["false", "no", "0"]
    true_values = ["true", "yes", "1"]
    lvalue = str(value).lower()
    if lvalue in false_values:
        return False
    if lvalue in true_values:
        return True
    raise Exception(
        "String value should be one of {}, but got '{}'.".format(
            false_values + true_values, value
        )
    )


def validate_required_argument_and_return_value(name):
    value = dbutils.widgets.get(name)
    if len(value) < 1:
        dbutils.notebook.exit("'{}' argument value is required.".format(name))
    return value


def get_argument(parameter: str) -> str:
    return validate_required_argument_and_return_value(parameter)


# COMMAND ----------

# DBTITLE 1,get_from_date_and_to_date_and_list_of_dates


def get_start_date_and_end_date_and_list_of_dates(
        end_date: str, look_back_days: str, return_date_objects: bool = False
) -> Tuple[str, str, List[Union[str, datetime]]]:
    """Returns the start_date and end_date and list of dates are string or datetime
        Parameters
        ----------
        :param end_date : end_date
            run_end_datedate: Parse the end_date from yyyy-mm-dd
        :param look_back_days : look_back_days
            run_date - look_back_days
        :param return_date_objects
    Usage:
    get_start_date_and_end_date_and_list_of_date(
        run_date="2021-08-19", look_back_days="2"
    )
    """
    list_of_dates = list()
    look_back_days_int = int(look_back_days)
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    start_date_obj = end_date_obj - timedelta(days=look_back_days_int)
    print(f"start_date: {start_date_obj} and end_date: {end_date_obj}")
    run_date_obj = end_date_obj
    while run_date_obj >= start_date_obj:
        if return_date_objects is True:
            list_of_dates.append(run_date_obj)
        else:
            list_of_dates.append(run_date_obj.strftime("%Y-%m-%d"))
        run_date_obj -= timedelta(days=1)
    print(f"list_of_dates {list_of_dates}")
    return (
        start_date_obj.strftime("%Y-%m-%d"),
        end_date_obj.strftime("%Y-%m-%d"),
        list_of_dates,
    )


# COMMAND ----------

# DBTITLE 1,list_all_secrets
def list_all_secrets():
    """
    # This functions list all secrets that are accessible inside Databricks Environment
    """
    for scope_object in dbutils.secrets.listScopes():
        scope_name = scope_object.name
        for key_object in dbutils.secrets.list(scope_name):
            key_name = key_object.key
            print(f"scope_name: {scope_name} and key_name: {key_name}")

# COMMAND ----------

# DBTITLE 1, Check if the table or view with the specified name exists in the specified database.
def does_table_exist(database_name: str, table_name: str) -> bool:
    # Usage :doesTableExist(database_name='bronze', table_name ='synthetic_transactions')
    table_list = [table.name for table in spark.catalog.listTables(database_name)]
    return table_name in table_list

# COMMAND ----------

# DBTITLE 1, Check if the database exists
def does_database_exist(database_name: str) -> bool:
    # Usage :doesTableExist(database_name='bronze', table_name ='synthetic_transactions')
    database_list = [database.name for database in spark.catalog.listDatabases()]
    return database_name in database_list


# https://realpython.com/python-timer/
def custom_timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        logger.info(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


# Ignore the defualt value being specified that is just to make the test code runs
@custom_timer
def read_table_and_dedupe(
    _database_name: str, _table_name: str, _unique_key_list: List, _dedupe_col: str
) -> DataFrame:
    return (
        spark.read.format("delta")
        .table(f"{_database_name}.{_table_name}")
        .withColumn(
            "row_number",
            F.row_number().over(
                Window.partitionBy(*_unique_key_list).orderBy(desc(f"{_dedupe_col}"))
            ),
        )
        .filter("row_number=1")
        .drop("row_number")
        .withColumn("ingestion_timestamp", expr("current_timestamp()"))
    )


"""
display(
    read_delta_table_and_dedupe( _database_name ='dlt_silver',
                                _table_name="avm",
                                _unique_key_list=["col1"],
                                _dedupe_col= "col2"
                               )
    )
"""

@custom_timer
def optimize_and_zorder_table(
    database_name: str, table_name: str, zorder_by_col_name: str
) -> None:
    """This function runs an optimize and zorder command on a given table that being fed by a stream
        - These commands can't run in silo because it will require us to pause and then resume stream
        - Therefore, we need to call this function as a part of the upsert function. This enables us to optimize before the next batch of streaming data comes through.

    Parameters:
         database_name: str
                 name of the database for the table to be optimize
         table_name: str
                 name of the table to be optimized
         zorder_by_col_name: str
                 comma separated list of columns to zorder by. example "col_a, col_b"
    """
    start = timer()
    logger.info(
            f"Met condition to optimize table {database_name}.{table_name}"
        )
    sql_query_optimize = (
            f"OPTIMIZE {database_name}.{table_name} ZORDER BY ({zorder_by_col_name})"
        )
    spark.sql(sql_query_optimize)
    end = timer()
    time_elapsed_seconds = end - start
    logger.info(
            f"Successfully optimized table {database_name}.{table_name} . Total time elapsed: {time_elapsed_seconds} seconds"
        )

@custom_timer
def run_spark_sql_with_logging(_sql_query: str):
    logger.info(f"About to execute {_sql_query}")
    return spark.sql(_sql_query)

# COMMAND ----------

# DBTITLE 1,Create Delta table from Path

@custom_timer
def create_delta_table_from_path(
    table_path: str, database_name: str, table_name: str
) -> None:
    spark.sql(
        f"""
            CREATE DATABASE IF NOT EXISTS {database_name};
            """
    )
    # Wait X seconds so that streaming can materialize some files
    time.sleep(3)
    if does_table_exist(database_name=database_name, table_name=table_name) is False:
        logger.info(
            f"Trying to create table: database_name: {database_name} table_name: {table_name} at table_path: {table_path}"
        )
        spark.sql(
            f"""
                CREATE TABLE IF NOT EXISTS {database_name}.{table_name}
                USING delta
                LOCATION "{table_path}";
                """
        )
        print(
            f"Created table: database_name: {database_name} table_name: {table_name} at table_path: {table_path}"
        )
        # Delete the delta log files after X days the default is 30 days
        alter_delta_log_file_retention_sql = f"""ALTER TABLE {database_name}.{table_name} SET TBLPROPERTIES(delta.logRetentionDuration = '15 days', delta.deletedFileRetentionDuration = '15 days');"""
        logger.info(f"About to run {alter_delta_log_file_retention_sql}")
        spark.sql(alter_delta_log_file_retention_sql)
        logger.info(
            f"Successfully change the delta log file retention property {alter_delta_log_file_retention_sql}"
        )
        # Auto generate the manifest files
        # Documentation delta.compatibility.symlinkFormatManifest.enabled"
        # delta.compatibility.symlinkFormatManifest.enabled":"true
        alter_delta_table_to_automatically_generate_manifests_sql = f"""ALTER TABLE {database_name}.{table_name} SET TBLPROPERTIES(delta.compatibility.symlinkFormatManifest.enabled=true);"""
        logger.info(f"About to run {alter_delta_table_to_automatically_generate_manifests_sql}")
        spark.sql(alter_delta_table_to_automatically_generate_manifests_sql)
        logger.info(
            f"Successfully change the delta to automatically generate manifest files {alter_delta_table_to_automatically_generate_manifests_sql}"
        )
        # Given that property was set after the table was written, we will have to manually generate manifest files for this time
        spark.sql(f'''
            GENERATE symlink_format_manifest FOR TABLE delta.`{table_path}`
            ''')
        logger.info(
            f"Successfully generated the manifest files. Please check the path {table_path}"
        )

@custom_timer
def create_table_and_backfill(backfill_df :DataFrame, _database_name :str, _table_name :str, _s3_bucket_path: str):
    table_path = f"{_s3_bucket_path}/{_database_name}/{_table_name}"
    logger.info(f"About to create_table_and_backfill at path: {table_path}")
    (
     backfill_df
         .write
         .format("delta")
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .save(table_path)
    )
    # Create an External table from the path specified
    create_delta_table_from_path(table_path =table_path, database_name=_database_name, table_name=_table_name)

@custom_timer
def drop_table_and_clean_path(database: str, table: str):
    try:
        print (f"Processing database.table {database}.{table}")
        df  = (spark.sql(f'''DESCRIBE TABLE EXTENDED {database}.{table};'''))
        location = df.select(['data_type']).where("col_name = 'Location'").collect()[0][0]
        logger.info (f'table location is : {location}')
        sql_to_drop_table = f''' DROP TABLE IF EXISTS {database}.{table}; '''
        logger.info (f'About to drop table using SQL statement: {sql_to_drop_table}')
        spark.sql(sql_to_drop_table)
        logger.info ('Table dropped and now about to clean location')
        dbutils.fs.rm(location,recurse=True)
        logger.info (f'location cleaned: {location}')
    except AnalysisException as e:
        logger.error(str(e))
