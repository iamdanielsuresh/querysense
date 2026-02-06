from google.cloud import bigquery

# Initialize the client. If you get a "Project not determined" error, 
# you can pass the project ID explicitly: bigquery.Client(project='your-project-id')
# client = bigquery.Client()
client = bigquery.Client(project='nb-sandbox-8mw570y811')


def execute_sql(sql):

    """
    Execute SQL on BigQuery and return:
    -result rows OR
    -execution error (used for feedback)
    """

    try:
        query_job = client.query(sql)
        rows = query_job.result()

        return {
            "success": True,
            "rows": [dict(row) for row in rows]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# if __name__ == "__main__":
#     # Test query
#     test_sql = "SELECT CURRENT_TIMESTAMP() as now"
#     result = execute_sql(test_sql)
#     print(result)