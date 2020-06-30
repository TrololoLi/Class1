import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession


LR_MODEL = 'lr_model'


def process(spark, train_data, test_data):
    #train_data - путь к файлу с данными для обучения модели
    #test_data - путь к файлу с данными для оценки качества модели
    train = spark.read.parquet(train_data)
    test = spark.read.parquet(test_data)
    #делаем фича вектор
    feature = VectorAssembler(inputCols=train_data.columns[1:-1],outputCol="rawfeatures")
    indexed_feature = VectorIndexer(inputCol="rawfeatures", outputCol="features", maxCategories=4)
    #прописываем выбранную нами модель
    dtr = DecisionTreeRegressor(labelCol="ctr", featuresCol="features")
    #делаем структуру паплайна
    pipeline_dtr = Pipeline(stages=[feature, indexed_feature, dtr])
    #пишем варианты параметров для подбора наилучших
    paramGrid_dtr = ParamGridBuilder()\
                     .addGrid(dtr.maxDepth, [2, 5])\
                     .addGrid(dtr.maxBins, [80, 85, 90])\
                     .build()
    #задаем наш оценщик
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="ctr", predictionCol="prediction")
    #задаем валидацию с паплайном, оценщиком и набором возможных параметров
    tvs_dtr = TrainValidationSplit(estimator=pipeline_dtr, evaluator=evaluator, estimatorParamMaps=paramGrid_dtr,trainRatio=0.8)
    #применяем tvs 
    dtr_model = tvs_dtr.fit(train)
    #запоминаем лучшую модель
    dtr_best_model = dtr_model.bestModel
    #проверяем качество 
    dtr_predictions = dtr_best_model.transform(test)
    rmse_dtr = evaluator.evaluate(dtr_predictions)
    print("RMSE on our dtr test set: %g" % rmse_dtr)
    #сохраняем лучшую модель
    dtr_best_model.write().overwrite().save('dtr_best_model')


def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
