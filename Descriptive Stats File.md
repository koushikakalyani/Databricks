# Descriptive Statistic Analysis

This notebook reads in a list of files from the directory and generates a dataframe with the descriptive statistics, NullCount of all files in one table with the fileName as an identifier column. 

```
import os
import pandas as pd
import numpy as np
import pyspark.sql.functions as F

from functools import reduce
from pyspark.sql.functions import *
from pyspark.sql.types import *

#Create Widget
dbutils.widgets.text("processdate","",label="Process Date")

#Get Values from widget
date = dbutils.widgets.get("processdate")
print("Date Specified is: ", date)
```

As a first part of this code, we will add in a date when the file was processed called process date. Next we add a loop to read all the desired file names from the directory. 
These files can have different columns. (The resultant dataframe will not be impacted by this)

```
# Read the files from temp folder and store in a list
dir_path = "/mnt/folder/temp"
input_file_names = os.listdir("/dbfs{}".format(dir_path))
input_file_names = [n for n in input_file_names if n.endswith('.csv')]
```
Next we create a loop to read each of these files into pyspark dataframe, and then perform the necessary operations to finally give us the output in desired shape. To achieve this, we create an empty dataframe before initiating the loop which provides the desired schema for the final file. 
```
#Create empty dataframe with following columns
lst = ['variable', 'count', 'mean', 'stddev', 'min', '25%', '50%', '75%', 'max', 'nullcount', 'filename']
summary_data_view = pd.DataFrame(index=lst).T


for file in input_file_names:
  FilePath = "/mnt/folder/temp/{}".format(file)
  df = spark.read.option("delimiter", "|").option("header", "true").csv(FilePath)

  # Summarize Data
  summary = df.summary()
  summary_pd = summary.toPandas()
  summary_pd_t = summary_pd.set_index("summary").T.rename_axis("variable").reset_index()

  # Count NULL values
  summary_nc = reduce(lambda a, b: a.union(b),(df.agg(F.count(F.when(F.isnull(c), c)).alias('nullcount')).select(F.lit(c).alias("variable"), "nullcount") for c in df.columns))
  null_counts = summary_nc.toPandas()
  null_counts["nullcount"] = null_counts["nullcount"].astype('object')

  # Merge data and add file name as a column
  merged_set = summary_pd_t.merge(null_counts, on="Variable")
  FileNameNew = date+"_"+file
  merged_set["FileName"] = FileNameNew
  
  #Concat the dataframe into summary_data_view
  summary_data_view = pd.concat([summary_data_view, merged_set], ignore_index=True)
  print("{} was processed and appended to dataset".format(file))
```

Finally, write the pandas dataset into a .csv file. 
```
summary_data_view.shape

#Write the file as .csv
dest_path = "/dbfs/mnt/folder/temp/summaryFiles/"
summary_data_view.to_csv(dest_path+"All_DescriptiveStats_"+date+".csv", header=True, index=False)
```














