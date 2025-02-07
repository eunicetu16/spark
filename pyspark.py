import pyspark
from pyspark import SparkContext
from datetime import datetime
import re


def createRDD(sc, filename, num_partitions=4):
    """
    Creates an RDD from a given CSV file using PySpark.

    Args:
        sc: SparkContext object.
        filename: Path to the file.
        num_partitions: Number of partitions for the RDD (default is 4).

    Returns:
        An RDD with the data excluding the header row and with additional
        datetime components.
    """
    # Load the CSV file into an RDD
    rdd = sc.textFile(filename, num_partitions)

    # Split the data into rows and columns using regex for CSV format
    csv_pattern = re.compile(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")

    # Remove the header (the first row)
    header = rdd.first()
    data_rdd = rdd.filter(lambda row: row != header).map(
        lambda row: csv_pattern.split(row)
    )

    return data_rdd


def returnCount(rdd):
    return rdd.count()


def topCrimeDescription(rdd, n):
    """
    Finds the top n most common crime descriptions in the dataset.

    Parameters:
    rdd (RDD): The input RDD where each row is a record.
    n (int): The top n crime descriptions.

    Returns:
    list: A list of tuples containing the top 5 crime descriptions and their counts.
    """
    top_crimes = (
        rdd.map(lambda row: (row[13], 1))
        .reduceByKey(lambda x, y: x + y)
        .sortBy(lambda x: x[1], ascending=False)
        .take(n)
    )
    return top_crimes


def victimRaceDistribution(rdd):
    """
    Computes the distribution of victim races in the dataset.

    Parameters:
    rdd (RDD): The input RDD where each row is a record.
    race_index (int): The column index of the victim race.

    Returns:
    list: A list of tuples containing victim races and their counts,
    sorted in descending order of counts.
    """
    race_distribution = (
        rdd.map(lambda row: (row[33], 1))
        .reduceByKey(lambda x, y: x + y)
        .sortBy(lambda x: x[1], ascending=False)
        .collect()
    )
    return race_distribution


def crimesForMonth(rdd, month):
    """
    Computes the number of crimes reported for a specific month.

    Parameters:
    rdd (RDD): The input RDD where each row is a record.
    month (int): The specific month (1-12) to return the crime count for.

    Returns:
    int: The crime count for the specified month, or 0 if the month is not found.
    """

    def extract_month(row):
        try:
            # Extract the date and convert to datetime object
            date_str = row[1]  # Assuming date is in the second column
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            return (date_obj.month, 1)
        except Exception as e:
            # Handle invalid/malformed date entries
            return (None, 0)

    # Map rows to (month, 1) pairs, filtering out invalid rows
    crime_month_rdd = rdd.map(extract_month).filter(lambda x: x[0] is not None)

    # Reduce by key to get the count for each month
    monthly_crimes_rdd = crime_month_rdd.reduceByKey(lambda a, b: a + b)

    # Filter for the specified month and get the count
    month_crime_count = monthly_crimes_rdd.filter(
        lambda x: x[0] == month).collect()

    # Return the count if found, else return 0
    return month_crime_count[0][1] if month_crime_count else 0


def crimesForHour(rdd, hour):
    """
    Computes the number of crimes reported for a specific hour.

    Parameters:
    rdd (RDD): The input RDD where each row is a record.
    hour (int): The specific hour (0-23) to return the crime count for.

    Returns:
    int: The crime count for the specified hour, or 0 if the hour is not found.
    """

    def extract_hour(row):
        try:
            # Extract the time and convert to datetime object
            time_str = row[2]  # Assuming time is in the third column
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
            return (time_obj.hour, 1)
        except Exception as e:
            # Handle invalid/malformed time entries
            return (None, 0)

    # Map rows to (hour, 1) pairs, filtering out invalid rows
    crime_hour_rdd = rdd.map(extract_hour).filter(lambda x: x[0] is not None)

    # Reduce by key to get the count for each hour
    hourly_crimes_rdd = crime_hour_rdd.reduceByKey(lambda a, b: a + b)

    # Filter for the specified hour and get the count
    hour_crime_count = hourly_crimes_rdd.filter(
        lambda x: x[0] == hour).collect()

    # Return the count if found, else return 0
    return hour_crime_count[0][1] if hour_crime_count else 0


def crimesCountByTypeAndBorough(rdd, year):
    """
    Computes the borough with the most crimes in a specific year.
    Returns a tuple where:
    - The first element is the borough with the most crimes.
    - The second element is the total number of crimes in that borough for the specified year.

    Parameters:
    rdd (RDD): The input RDD where each row is a record.
    year (int): The year to filter the records.

    Returns:
    tuple: (borough_with_most_crimes, total_crimes_count)
    """

    def extract_year(row):
        try:
            # Extract the date and convert to datetime object
            date_str = row[1]  # Assuming the date is in the specified column
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            return date_obj.year
        except Exception as e:
            # Handle invalid/malformed date entries
            return (None, None)

    # Filter out rows that don't have valid year and month
    valid_rows_rdd = rdd.filter(lambda row: extract_year(row) is not None)

    # If year or month is specified, filter by year and/or month
    if year is not None:
        valid_rows_rdd = valid_rows_rdd.filter(
            lambda row: extract_year(row) == year)

    # Map the data to key-value pairs: (crime_type, borough) -> 1
    crime_type_boro_rdd = valid_rows_rdd.map(
        lambda row: ((row[8], row[13]), 1))

    # Reduce by key to aggregate the crime counts for each (crime_type, borough) pair
    crime_counts_rdd = crime_type_boro_rdd.reduceByKey(lambda a, b: a + b)

    # Sort by crime count in descending order
    sorted_crime_counts_rdd = crime_counts_rdd.sortBy(
        lambda x: x[1], ascending=False)

    # Collect and return the sorted results
    sorted_results = sorted_crime_counts_rdd.collect()

    return sorted_results


def findBoroughWithMostCrimesByYear(rdd, year):
    """
    Finds the borough with the most crimes in a specific year.

    Parameters:
    rdd (RDD): The input RDD where each row is a record.
    year (int): The year to filter the records.

    Returns:
    List: A list containing the borough and the total number of crimes for the specified year.
    """

    def extract_year(row):
        try:
            # Extract the date and convert to datetime object
            date_str = row[1]  # Assuming the date is in the specified column
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            return date_obj.year
        except Exception as e:
            # Handle invalid/malformed date entries
            return (None, None)

    # Filter out rows with missing year
    filtered_rdd = rdd.filter(lambda row: extract_year(row) is not None)

    # Filter rows for the specified year
    year_filtered_rdd = filtered_rdd.filter(
        lambda row: extract_year(row) == year)

    # Map to (borough, 1)
    borough_rdd = year_filtered_rdd.map(lambda row: (row[13], 1))

    # Aggregate counts for each borough
    crime_counts_rdd = borough_rdd.reduceByKey(lambda a, b: a + b)

    # Find the borough with the most crimes
    most_crimes = crime_counts_rdd.reduce(lambda a, b: a if a[1] > b[1] else b)

    return most_crimes


def aggregateOffensesByLevel(rdd):
    """
    Aggregates and counts offenses grouped by crime level.

    Args:
        rdd: Input RDD, where each row is a list representing a record.

    Returns:
        A sorted list of tuples (crime_level, count), sorted by count in descending order.
    """
    # Map each crime level to a tuple (crime_level, 1)
    crime_level_offense = rdd.map(lambda row: (row[12], 1))

    # Reduce by key to count occurrences of each crime level
    crime_level_offense_agg = (
        crime_level_offense.reduceByKey(lambda a, b: a + b)
        .sortBy(lambda x: x[1], ascending=False)
        .collect()
    )

    return crime_level_offense_agg


conf = pyspark.SparkConf().set("spark.driver.host", "localhost")
sc = SparkContext.getOrCreate()
sc.setLogLevel("OFF")

print("=======================Creating RDD=======================")
rdd = createRDD(
    sc, "s3://msds-694-cohort-13-16/data/NYPD_Complaint_Data_Historic.csv")
print(rdd.count())
print("=======================Created RDD=======================")
print("\n")

print("=======================Counting records in RDD=======================")
print(returnCount(rdd))
print("=======================Counted records in RDD=======================")
print("\n")

print(
    "=======================Displaying Top 5 Crime Description======================="
)
top_crimes = topCrimeDescription(rdd, 5)
print(top_crimes)
print("=======================Displayed Top 5 Crime Description=======================")
print("\n")

print(
    "=======================Displaying Victim Race Distribution======================="
)
race_distribution = victimRaceDistribution(rdd)
print(race_distribution)
print(
    "=======================Displayed Victim Race Distribution======================="
)
print("\n")

print(
    "=======================Displaying Number of Crimes Reported for July======================="
)
month_crime = crimesForMonth(rdd, 7)
print(month_crime)
print(
    "=======================Displayed Number of Crimes Reported for July======================="
)
print("\n")

print(
    "=======================Displaying Number of Crimes Reported for 12:00PM======================="
)
hour_crime = crimesForHour(rdd, 12)
print(hour_crime)
print(
    "=======================Displayed Number of Crimes Reported for 12:00PM======================="
)
print("\n")

print(
    "=======================Displaying Borough with the Most Common Type of Crimes in 2006======================="
)
crime_borough = crimesCountByTypeAndBorough(rdd, 2006)
print(crime_borough)
print(
    "=======================Displayed Borough with the Most Common Type of Crimes in 2006======================="
)
print("\n")

print(
    "=======================Displaying Borough with the Most Crimes in 2008======================="
)
borough = findBoroughWithMostCrimesByYear(rdd, 2008)
print(borough)
print(
    "=======================Displayed Borough with the Most Crimes in 2008======================="
)
print("\n")

print(
    "=======================Displaying Aggregates offenses grouped by crime level======================="
)
offenses_level = aggregateOffensesByLevel(rdd)
print(offenses_level)
print(
    "=======================Displayed Aggregates offenses grouped by crime level======================="
)
print("\n")

sc.stop()
