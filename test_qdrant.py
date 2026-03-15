from qdrant_client import QdrantClient
import inspect
client = QdrantClient(url="http://localhost:6333", api_key="test")
print("client.query_points signature:", inspect.signature(client.query_points))
