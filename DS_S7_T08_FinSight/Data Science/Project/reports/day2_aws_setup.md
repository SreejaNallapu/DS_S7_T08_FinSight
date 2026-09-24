# Day 2 AWS implementation: S3, Glue Data Catalog, and Athena

The Day 2 AWS analytics path has been manually deployed and verified. This document records the working implementation; it is not a list of outstanding setup tasks.

## Implemented architecture

```text
Amazon S3
→ AWS Glue Data Catalog (table registered using Athena DDL)
→ Amazon Athena
→ SQL analytics
```

The AWS Glue Data Catalog provides the database and table metadata. The table was registered by executing an explicit `CREATE EXTERNAL TABLE` statement through Athena, which writes the table definition to the Glue Data Catalog. An AWS Glue crawler is not part of the implemented path.

## Deployed resources

| Resource | Verified value |
|---|---|
| Raw dataset | Stored in Amazon S3 |
| Processed analytics CSV | `s3://finsight-credit-risk/processed/german_credit_analytics.csv` |
| Glue Data Catalog database | `finsight` |
| Glue Data Catalog table | `german_credit_analytics` |
| Fully qualified table | `finsight.german_credit_analytics` |
| Table registration method | Athena `CREATE EXTERNAL TABLE` DDL |
| Query engine | Amazon Athena |

The processed object contains 1,000 data rows and 24 columns: the 21 source fields plus `applicant_id`, `credit_risk_label`, and `is_bad_risk`. It is an unscaled analytics table, not an encoded model matrix.

## Commands corresponding to the deployed table

The local export is generated without modifying the raw dataset:

```powershell
.venv\Scripts\python.exe -m src.data.preprocess
```

The deployed S3 object can be uploaded or refreshed with:

```powershell
$env:AWS_PROFILE = 'finsight'
$env:AWS_DEFAULT_REGION = 'eu-north-1'
aws s3 cp data/processed/german_credit_analytics.csv s3://finsight-credit-risk/processed/german_credit_analytics.csv --profile finsight --region eu-north-1
```

The exact table DDL and example SQL are generated with:

```powershell
.venv\Scripts\python.exe -m src.cloud.athena_queries --database finsight --table german_credit_analytics --s3-location s3://finsight-credit-risk/processed/
```

The S3 location passed to the DDL generator is the containing prefix, ending in `/`; the deployed CSV is the object `processed/german_credit_analytics.csv` within that prefix.

## Registration method used

1. The `finsight` database was created in the AWS Glue Data Catalog.
2. The processed CSV was placed at `s3://finsight-credit-risk/processed/german_credit_analytics.csv`.
3. `src.cloud.athena_queries` generated an explicit 24-column `CREATE EXTERNAL TABLE` statement for database `finsight`, table `german_credit_analytics`, and S3 location `s3://finsight-credit-risk/processed/`.
4. The generated DDL was executed in Amazon Athena.
5. Athena registered the table metadata in the AWS Glue Data Catalog and queried the CSV directly from S3.

This explicit schema keeps column names, order, and types aligned with the exported CSV and avoids reliance on inferred crawler metadata.

## Live verification completed

The following behavior has been verified in Amazon Athena:

- `SELECT * FROM "finsight"."german_credit_analytics" LIMIT 10;` succeeds.
- Applicant rows and fields are returned correctly from the processed S3 CSV.
- Grouped credit-risk analytics succeeds.
- Athena reads `s3://finsight-credit-risk/processed/german_credit_analytics.csv` through the registered table.

The generated query set also provides applicant count, good/bad risk counts, average credit amount, average duration, and grouped analysis by savings and employment status.

## Glue crawler status

The AWS account denied the permissions required to create a Glue crawler. No crawler was created or used. This is not an unfinished requirement because the table is already registered in the Glue Data Catalog through the successfully executed Athena DDL and is queryable.

A crawler is only an optional alternative for future schema discovery in an account with appropriate permissions. It is not required for the current architecture and should not be shown as part of the successful Day 2 data path.

## Security and operational notes

Credentials remain outside the repository. Access should continue to use an IAM role, AWS IAM Identity Center, environment credentials, or a local AWS profile. The querying identity needs S3 read access to the processed object, access to the configured Athena results location, Athena execution permissions, and Glue Data Catalog read/write permissions appropriate for table registration and querying.
