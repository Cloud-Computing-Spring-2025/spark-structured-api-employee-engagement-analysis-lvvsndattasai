from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg

# Step 1: Create a Spark Session (Ensure local mode is set)
spark = SparkSession.builder \
    .appName("Employee Engagement Analysis") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

# Step 2: Load the CSV File
df = spark.read.csv("employee_data.csv", header=True, inferSchema=True)

# Step 3: Check Data Schema
df.printSchema()

# ===============================
# 1. Identify Departments with High Satisfaction and Engagement
# ===============================
# Step 1: Filter employees with SatisfactionRating > 4 and EngagementLevel = "High"
satisfied_high_engagement = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))

# Step 2: Count how many such employees exist per department
dept_analysis = satisfied_high_engagement.groupBy("Department").count().withColumnRenamed("count", "filtered_count")

# Step 3: Get total employees per department
total_employees_per_dept = df.groupBy("Department").count().withColumnRenamed("count", "total_count")

# Step 4: Calculate the percentage of highly satisfied & engaged employees
dept_percentage = dept_analysis.join(total_employees_per_dept, "Department") \
    .withColumn("Percentage", (col("filtered_count") / col("total_count")) * 100) \
    .select("Department", "Percentage")

# Step 5: Display the results
print("Departments with High Satisfaction and Engagement:")
dept_percentage.show()

# ===============================
# 2. Who Feels Valued but Didn’t Suggest Improvements?
# ===============================
# Step 1: Filter employees who have SatisfactionRating ≥ 4 but ProvidedSuggestions = False
valued_no_suggestions = df.filter((col("SatisfactionRating") >= 4) & (col("ProvidedSuggestions") == False))

# Step 2: Calculate total and percentage
num_employees = valued_no_suggestions.count()
total_employees = df.count()
proportion = (num_employees / total_employees) * 100

# Step 3: Print the results
print(f"Number of Employees Feeling Valued without Suggestions: {num_employees}")
print(f"Proportion: {proportion:.2f}%")

# ===============================
# 3. Compare Engagement Levels Across Job Titles
# ===============================
# Step 1: Convert Engagement Level to numerical values
engagement_mapping = {
    "Low": 1,
    "Medium": 3,
    "High": 5
}

from pyspark.sql.functions import when

# Step 1: Convert Engagement Level to numeric values correctly
df = df.withColumn(
    "EngagementNumeric",
    when(col("EngagementLevel") == "Low", 1)
    .when(col("EngagementLevel") == "Medium", 3)
    .when(col("EngagementLevel") == "High", 5)
    .otherwise(0)  # Default value if EngagementLevel is missing
)

# Step 2: Compute the average engagement per Job Title
job_title_engagement = df.groupBy("JobTitle").agg(avg("EngagementNumeric").alias("AvgEngagementLevel"))

# Step 3: Display the results
print("Engagement Levels Across Job Titles:")
job_title_engagement.show()


# ===============================
# Final Step: Stop Spark Session
# ===============================
spark.stop()

