import os
from langchain_core.documents import Document
from langchain_astradb  import AstraDBVectorStore
from typing import List
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add the project root to the Python path for direct script execution
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class Retrieval:
    """
    Class to handle data retrieval from AstraDB vector store.
    """
    def __init__(self):
        """
        Initialize environment variables, embedding model, and AstraDB vector store.
        """
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.config = load_config()
        self.vstore = None
        self.retriever = None
    
    def _load_env_variables(self):
        """
        Load and validate required environment variables.
        """ 
        load_dotenv()
        
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    
    def load_retriever(self):
        """
        Load AstraDB vector store and create a retriever.
        """
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
            
            print("Loading AstraDB vector store...")
            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace
            )
            print("AstraDB vector store loaded.")
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})
            print(f"Retriever loaded successfully")
            return retriever
        
    def call_retriever(self,query):
        """
        Retrieve documents based on the query.
        """
        retriever = self.load_retriever()
        output = retriever.invoke(query)
        return output

if __name__ == "__main__":
    retriever_obj = Retrieval()
    user_query = "What are the top features of the latest smartphone models?"
    result = retriever_obj.call_retriever(user_query)
    
    for idx, doc in enumerate(result, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")
