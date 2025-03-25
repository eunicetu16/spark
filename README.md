# Running Spark Job on AWS EMR Serverless Cluster

## Project Summary

This project focuses on leveraging **Apache Spark** and **AWS EMR Serverless** to process and analyze large datasets efficiently. The goal is to build a **serverless Spark batch job** that executes data transformations on AWS, generating meaningful insights from the dataset. The project utilizes the NYPD Complaint Data Historic dataset, selected for its comprehensive structure and extensive range of variables that provide detailed insights into reported criminal incidents across New York City. This dataset encompasses information on complaint categories, offense classifications, geospatial coordinates, timestamps, and demographic attributes of both victims and suspects. Its depth and granularity offer a strong foundation for rigorous analysis, enabling the exploration of patterns, trends, and correlations across spatial, temporal, and categorical dimensions. The richness of the data supports a multifaceted approach to examining crime dynamics within the city.

## Project Description

### 1. AWS EMR Serverless Setup
- Configure and deploy an **AWS EMR Serverless Cluster**.
- Upload the dataset to an **S3 bucket** for cloud-based processing.

### 2. Data Transformation & Processing
- Define **new transformations** to extract meaningful insights from the dataset.
- First, test these transformations on a **smaller local dataset** using PySpark.
- Implement a **standalone PySpark script** that contains all computations, designed to run as a batch job.

### 3. Execution on AWS
- Execute the **PySpark script** as a **batch job** on AWS EMR Serverless with S3-hosted data.
- Validate the job execution through **logs and output verification**.

---

This project enhances our skills in **distributed computing, cloud-based data processing, and PySpark optimization**, reinforcing practical knowledge in **big data analytics using AWS**.
