import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.document import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv


class VectorStoreSearchTool:
    """
    Optimized vector store search tool for querying content indexed with binary quantization.
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "Prod_data_V1",
        embedding_model: str = "thenlper/gte-large",
    ):
        """
        Initialize the vector store search tool with optimal configuration.

        Args:
            qdrant_url: URL for Qdrant service
            qdrant_api_key: API key for Qdrant
            collection_name: Name of the collection in Qdrant
            embedding_model: FastEmbed model name (must match uploader's model)
        """
        load_dotenv()

        # Set connection parameters
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_API_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name

        if not self.qdrant_url:
            raise ValueError("Qdrant URL must be provided or set in environment")

        # Initialize Qdrant client with gRPC for better performance
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key, prefer_grpc=True
        )

        # Create embedding function (must match uploader's model)
        self.embedding_function = FastEmbedEmbeddings(
            model_name=embedding_model, threads=4  # Match uploader's thread count
        )

        # Initialize vector store with binary quantization support
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embedding_function,
            # No need for sparse vectors since we're using binary quantization
        )

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search the vector store with binary quantization support.

        Args:
            query: The search query string
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            metadata_filter: Dictionary of metadata filters

        Returns:
            List of Document objects matching the query
        """
        try:
            search_params = models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,  # Important for binary quantization
                )
            )

            if score_threshold is not None:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=k, filter=metadata_filter, search_params=search_params
                )
                return [doc for doc, score in results if score >= score_threshold]
            else:
                return self.vectorstore.similarity_search(
                    query, k=k, filter=metadata_filter, search_params=search_params
                )
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def search_with_scores(
        self, query: str, k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search and return documents with similarity scores.

        Args:
            query: The search query string
            k: Number of results to return
            metadata_filter: Dictionary of metadata filters

        Returns:
            List of (Document, score) tuples
        """
        try:
            return self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=metadata_filter,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                    )
                ),
            )
        except Exception as e:
            print(f"Error during search with scores: {e}")
            return []

    # def hybrid_search(
    #     self,
    #     query: str,
    #     k: int = 5,
    #     alpha: float = 0.5,
    #     metadata_filter: Optional[Dict[str, Any]] = None,
    # ) -> List[Document]:
    #     """
    #     Perform hybrid search combining semantic and keyword search.

    #     Args:
    #         query: The search query string
    #         k: Number of results to return
    #         alpha: Weight for semantic vs keyword search (0.0-1.0)
    #         metadata_filter: Dictionary of metadata filters

    #     Returns:
    #         List of Document objects
    #     """
    #     try:
    #         return self.vectorstore.similarity_search(
    #             query,
    #             k=k,
    #             filter=metadata_filter,
    #             search_params=models.SearchParams(
    #                 hybrid=models.Hybrid(
    #                     positive=query,
    #                     alpha=alpha,
    #                 ),
    #                 quantization=models.QuantizationSearchParams(
    #                     ignore=False,
    #                     rescore=True,
    #                 ),
    #             ),
    #         )
    #     except Exception as e:
    #         print(f"Error during hybrid search: {e}")
    #         return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the collection including quantization status.

        Returns:
            Dictionary containing collection information
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance),
                    "quantization": str(collection_info.config.quantization_config),
                    "optimizer": collection_info.config.optimizer_config,
                },
                "payload_schema": collection_info.payload_schema,
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}

    def format_results(
        self,
        results: List[Document],
        include_metadata: bool = True,
        max_content_length: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Format search results with quantization awareness.

        Args:
            results: List of Document objects
            include_metadata: Whether to include metadata
            max_content_length: Max length of content to display

        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        for i, doc in enumerate(results):
            result = {
                "rank": i + 1,
                "content": (
                    doc.page_content + "..."
                    if len(doc.page_content) > max_content_length
                    else doc.page_content
                ),
                "content_length": len(doc.page_content),
            }
            if include_metadata and hasattr(doc, "metadata") and doc.metadata:
                result["metadata"] = doc.metadata
            formatted_results.append(result)
        return formatted_results

    def optimize_for_search(self) -> None:
        """
        Optimize the collection for search performance.
        Should be called after bulk uploads are complete.
        """
        try:
            self.qdrant_client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # Force immediate indexing
                    default_segment_number=2,  # Optimal for search
                ),
            )
            print("Collection optimized for search performance")
        except Exception as e:
            print(f"Error optimizing collection: {e}")


def search_vector_store(
    query: str,
    collection_name: str = "optimal-text-data",
    k: int = 5,
    with_scores: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convenience function for searching the vector store with binary quantization.

    Args:
        query: The search query string
        collection_name: Name of the collection
        k: Number of results to return
        with_scores: Whether to include similarity scores
        **kwargs: Additional search parameters

    Returns:
        List of formatted search results
    """
    try:
        search_tool = VectorStoreSearchTool(collection_name=collection_name)

        if with_scores:
            results = search_tool.search_with_scores(query, k=k, **kwargs)
            formatted = []
            for rank, (doc, score) in enumerate(results, 1):
                item = {
                    "rank": rank,
                    "score": float(score),
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                }
                if hasattr(doc, "metadata") and doc.metadata:
                    item["metadata"] = doc.metadata
                formatted.append(item)
            return formatted
        else:
            results = search_tool.search(query, k=k, **kwargs)
            return search_tool.format_results(results)
    except Exception as e:
        print(f"Error in search_vector_store: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Initialize the search tool (matches the uploader configuration)
    search_tool = VectorStoreSearchTool(
        # collection_name="optimal-demo-data",
        # embedding_model="thenlper/gte-large"
    )

    # Test search with binary quantization
    query = "Vapi pitch deck"
    print(f"Searching for: '{query}' with binary quantization")

    # Basic search
    results = search_tool.search(query, k=3)
    print(f"\nBasic search results ({len(results)} found):")
    
    for result in search_tool.format_results(results):
        print(f"\nRank {result['rank']}:")
        print(f"Content: {result['content']}")
        if "metadata" in result:
            print(f"Metadata: {result['metadata']}")

    # Search with scores
    # results_with_scores = search_tool.search_with_scores(query, k=3)
    # print(f"\nResults with similarity scores:")
    # for doc, score in results_with_scores:
    #     print(f"Score: {score:.4f} | Content: {doc.page_content[:100]}...")

    # Hybrid search
    # hybrid_results = search_tool.hybrid_search(query, k=3, alpha=0.7)
    # print(f"\nHybrid search results:")
    # for result in search_tool.format_results(hybrid_results):
    #     print(f"\nRank {result['rank']}:")
    #     print(f"Content: {result['content']}")

    # Get collection info (shows quantization config)
    info = search_tool.get_collection_info()
    print(f"\nCollection Info:")
    print(f"- Vectors count: {info.get('vectors_count', 'N/A')}")
    print(f"- Quantization: {info.get('config', {}).get('quantization', 'N/A')}")
