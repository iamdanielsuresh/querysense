from schema_loader import load_schema
from schema_retriever import SchemaRetriever

schema = load_schema("bquxjob_26d42422_19c09963dbb.csv")

retriever = SchemaRetriever(schema)

question = "What is the total revenue last month ?"

results = retriever.retrieve(question, top_k=10)

print("Top retrieved schema coloumn:\n")
for r in results:
    print(r["text"])