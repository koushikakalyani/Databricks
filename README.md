# Databricks
How Tos:

#### Delta table UPSERT Logic

When we run prediction scores each month but want to save them into a delta table which contains the history of scoring for all rows (eg: Products). 
The logic is simple - if the date we are attempting to enter and product exists, then overwrite the delta table with newly run values. Else insert the newly run values as a fresh entry.

```
%sql
MERGE INTO OriginalTable OG
USING NewTable NT
ON OG.EntryDate = NT.EntryDate AND OG.Identifier = NT.Identifier
WHEN MATCHED 
  THEN UPDATE SET OG.PredictionProb0 = NT.PredictionProb0, 
  OG.PredictionProb1 = NT.PredictionProb1, 
  OG.TargetLabel = NT.TargetLabel, 
WHEN NOT MATCHED
  THEN INSERT *
```


#### Creating Widgets in Databricks notebook: 
Create a widget to input a parameter (eg: entry date) into the notebook:
```
#Create Widget
dbutils.widgets.text("entrydate","",label="Entry Date")
```
Read the data entered in the widget and store it in a variable (eg: DateSpecified)
```
#Get Values from widget
DateSpecified = dbutils.widgets.get("entrydate")
print("Date Specified is: ", DateSpecified)
```

#### Read a file path using .format()

One way to specify the path to read file from blob is: 
```
dateVariable = "20200731"
file_path = "dbfs:/mnt/containerName/subfolder/file_name"+dateVariable
```
Another way this can be done is using `.format()` in the file path. 
Multiple parameters can be passed into the .format() option. Just remember to specify them in the path string as {} and call them in the same sequence in the .format() section of the file path.
```
file_path = "dbfs:/mnt/containerName/subfolder/file_name_{}".format(dateVariable)
```
#### Write files in Blob storage: 

Save pyspark dataframe as .csv file as a single file without partition in Blob storage: 
```
df.repartition(1).write.format("csv").option("header","true").save(blob_file_path)
```
Settings to remove write logs and start/end in blob: (execute them individually) 

```
%sql
SET spark.sql.sources.commitProtocolClass = org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol
```
```
%sql
SET mapreduce.fileoutputcommitter.marksuccessfuljobs=false
```


