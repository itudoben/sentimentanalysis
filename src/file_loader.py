# -*- coding: utf-8 -*-
"""Dataset loader.

This module provides functions to load data sets.

Example:
        $ python file_loader.py
"""

import os
import re
import shutil
from datetime import datetime

from pathlib import Path
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def load_data(base_path, observations_max, spark):
    """Stanford IMDb Review dataset loading function.

    This method loads all the files from the Stanford IMDb Review dataset from the specified
    base_path and up to the number observations_max.

    The loaded and processed data is then stored in a parquet file on the base_path.

    :param base_path: (string): the file system path of the folder of Stanford IMDb Review dataset.
    :param observations_max: (int): the maximum number of observations to load. -1 means all.
    :param spark: (SparkSession): the spark session.
    :return: string, DataFrame: the name of the file where the parquet data set is stored and
        the Spark data frame associated.
    """

    # RegEx for extracting file information.
    prog = re.compile('(.*)_(.*)\.txt')

    # Labels.
    labels = {'pos': 1, 'neg': 0}

    # Here for the sake of short testing without loading the whole data in Spark.
    ttl = observations_max
    ttl_h = ttl / 2

    # Try to get half of positives and negatives from ttl since my pc cannot load the total 55K reviews!
    ttl_positives = 0
    ttl_negatives = 0

    cnt = 0

    # Features description.
    features = ['datasettype', 'filename', 'datetimecreated', 'reviewid', 'reviewpolarity', 'reviewrating', 'text']

    # The context.
    sc = spark.sparkContext

    # Keep track of when the data set was built.
    utc_date = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    entries = []

    # Find all review files.
    for set_name in ['test', 'train']:
        for sa_name in ['neg', 'pos']:
            dir_path = os.path.join(base_path, set_name, sa_name)

            polarity = labels[sa_name]

            if sa_name == 'neg' and ttl_negatives > ttl_h:
                break

            if sa_name == 'pos' and ttl_positives > ttl_h:
                break

            # Load each review file.
            for file in os.listdir(dir_path):
                # Extract the ID and the rating of the review from the file name.
                m = prog.match(file)
                review_id = int(m.group(1))
                rating = int(m.group(2))

                # Read in the review.
                with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()

                # Prepare the entry
                entry = [set_name, file, utc_date, review_id, int(polarity), int(rating), txt]
                entries.append(entry)

                # Loop checking.
                cnt += 1
                if cnt == ttl:
                    break

                if sa_name == 'neg':
                    ttl_negatives += 1
                if sa_name == 'neg' and ttl_negatives > ttl_h:
                    break

                if sa_name == 'pos':
                    ttl_positives += 1
                if sa_name == 'pos' and ttl_positives > ttl_h:
                    break

            if cnt == ttl:
                break
        if cnt == ttl:
            break

    # Process the entries.
    sa_rdd = sc.parallelize(entries)
    df = sa_rdd.toDF(features)

    # Store the DF in parquet format.
    file_pqt = os.path.join(base_path, ("aclImdb_%s_raw.parquet" % ttl))

    # If the parquet directory exists, remove it.
    if Path(file_pqt).is_dir():
        shutil.rmtree(file_pqt)

    df.write.parquet(file_pqt)

    return file_pqt, df


# Create a custom word count transformer class

# Create a custom word count transformer class
class HTMLTagsRemover(Transformer, HasInputCol, HasOutputCol):
    """HTML tags cleaning.
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(HTMLTagsRemover, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        def f(s):
            return re.sub('<\w*[^>]\/>|[\[\]\(\)]', '', s)

        t = StringType()

        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]

        return dataset.withColumn(out_col, udf(f, t)(in_col))


def main():
    """The main function.

    :return:
    """
    print('This is a module.')


if __name__ == '__main__':
    main()
