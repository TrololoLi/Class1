import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff
from pyspark.sql import functions as F


def process(spark, input_file, target_path):
    
    df = spark.read.parquet(input_file)

    df = df.withColumn('click', F.when(F.col('event')=='click',1).otherwise(0))\
            .withColumn('view', F.when(F.col('ad_cost_type')=='view',1).otherwise(0))\
            .withColumn('is_cpm', F.when(F.col('ad_cost_type')=='CPM',1).otherwise(0))\
            .withColumn('is_cpc', F.when(F.col('ad_cost_type')=='CPC',1).otherwise(0))


    df_ctr = df.groupBy('ad_id')\
                .agg(F.sum('click').alias('num_of_clicks'),F.sum('view').alias('num_of_views'),F.min('date').alias('min'),F.max('date').alias('max'))\
                .withColumn('CTR', (col('num_of_clicks')/col('num_of_views')))\
                .withColumn('day_count',F.datediff(col('max'),col('min')))

    
    df = df.drop('date','time','event','platform','client_union_id','compaign_union_id','ad_cost_type','click','view').distinct()
    
    final_data = df_ctr.join(df,['ad_id'],'right').select('ad_id','target_audience_count','has_video','is_cpm','is_cpc','ad_cost','day_count','CTR')
    
    a,b,c,=final_data.randomSplit([0.5,0.25,0.25])
    
    a.coalesce(1).write.parquet(target_path+'/train')
    b.coalesce(1).write.parquet(target_path+'/test')
    c.coalesce(1).write.parquet(target_path+'/validate')
    
    
def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
