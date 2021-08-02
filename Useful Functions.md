# Useful functions in Pyspark

User Defined Functions to convert to binary 0/1 outcomes:
```
from pyspark.sql.functions import *
from pyspark.sql.types import *

#UDF to convert numeric value to binary 
num_to_flag = udf(lambda x : 0 if x == 0 else 1)

#UDF to convert binary Y/N variables into binary 1/0 values :
yesno_to_binary = udf(lambda x: 1 if x=='Y' else 0)
```

User defined functions for date conversion
```
#Convert String to Date
from datetime import datetime
from pyspark.sql.functions import col,udf
from pyspark.sql.types import DateType

# This function converts the string cell into a date:
date_func =  udf(lambda x: datetime.strptime(x, '%Y%m%d'), DateType())
```
Calculate date difference in months between two dates
```
# Function to calculate the difference in months  
def Difference(date_1,date_2):
    diff=(date_1.year-date_2.year)*12+date_1.month-date_2.month
    return diff

func_datediff = udf(Difference)
```
**Check for Nulls:** Column wise count of NULLs in a dataframe
```
from pyspark.sql.functions import col,sum
temp = df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns))
display(temp)
```
