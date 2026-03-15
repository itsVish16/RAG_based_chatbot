from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333", api_key="test")

# Since we don't have connection to local, let's just inspect the return type attributes 
getattr_names = [m for m in dir(models.QueryResponse) if not m.startswith('_')]
print("QueryResponse attributes:", getattr_names)
