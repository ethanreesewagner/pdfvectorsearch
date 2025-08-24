from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import dotenv_values
from openai import OpenAI

config=dotenv_values(".env")
client = OpenAI(api_key=config["OPENAI_API_KEY"])
pc = Pinecone(api_key=config["PINECONE_API_KEY"])
index = pc.Index("testingvectors")
query_vector = client.embeddings.create(input=input("Type in text: "),
model="text-embedding-3-small", dimensions=512)

results = index.query(vector=list(query_vector.data[0].embedding), top_k=5, include_metadata=True, namespace="example-namespace")

print(results)