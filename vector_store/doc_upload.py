import os
from typing import List, Dict, Any
from uuid import uuid4
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document


class QdrantDataUploader:
    """
    Optimal data uploader to Qdrant with binary quantization and efficient embedding.
    Handles text data from predefined arrays with optimal storage configuration.
    """

    def __init__(
        self,
        collection_name: str = "Prod_data_V1",
        embedding_model: str = "thenlper/gte-large",
        vector_size: int = 1024,
        on_disk: bool = True,
        always_ram: bool = True,
    ):
        """
        Initialize the data uploader with optimal Qdrant configuration.

        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: FastEmbed model name
            vector_size: Dimension of embedding vectors
            on_disk: Store vectors on disk
            always_ram: Keep quantized vectors in RAM
        """
        load_dotenv()

        # Initialize Qdrant client with environment variables
        self.client = QdrantClient(
            url=os.getenv(
                "QDRANT_API_URL",
            ),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,  # Better for production
        )

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_size = vector_size
        self.on_disk = on_disk
        self.always_ram = always_ram

        # Initialize embedding model
        self.embeddings = FastEmbedEmbeddings(
            model_name=embedding_model, threads=4  # Optimal for most CPUs
        )

        # Create or get collection
        self._setup_collection()

        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _setup_collection(self) -> None:
        """Configure Qdrant collection with optimal settings"""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_size,
                    distance=qdrant_models.Distance.COSINE, 
                    on_disk=self.on_disk,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    default_segment_number=5,  # Balanced performance
                    indexing_threshold=20000,  # Delay indexing for bulk uploads
                ),
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=0,  # Disable HNSW for binary quantization
                ),
                quantization_config=qdrant_models.BinaryQuantization(
                    binary=qdrant_models.BinaryQuantizationConfig(
                        always_ram=self.always_ram,
                    ),
                ),
            )
            print(
                f"Created new collection '{self.collection_name}' with binary quantization"
            )
        else:
            print(f"Using existing collection '{self.collection_name}'")

    def _preprocess_data(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Convert raw data to LangChain documents with metadata"""
        documents = []
        for item in data:
            doc_id = str(uuid4())
            metadata = item.get("metadata", {})
            metadata["doc_id"] = doc_id
            metadata["source"] = item.get("source", "unknown")

            documents.append(Document(page_content=item["text"], metadata=metadata))
        return documents

    def upload_data(self, data: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Upload data to Qdrant with optimal batching and embedding.

        Args:
            data: List of dictionaries with "text" and optional "metadata"
            batch_size: Number of documents to process at once
        """
        if not data:
            print("No data to upload")
            return

        documents = self._preprocess_data(data)
        total_docs = len(documents)
        print(f"Prepared {total_docs} documents for upload")

        # Upload in batches for memory efficiency
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            try:
                self.vector_store.add_documents(batch)
                print(
                    f"Uploaded batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}"
                )
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")

        print(
            f"Successfully uploaded {total_docs} documents to '{self.collection_name}'"
        )

    def optimize_collection(self) -> None:
        """Optimize the collection after uploads are complete"""
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=0,  # Force indexing
                default_segment_number=2,  # Optimize segments
            ),
        )
        print("Collection optimization initiated")


# Example Usage
if __name__ == "__main__":
    # Sample data array
    sample_data =[
    {
        "text": (
            "Rapido is India's leading ride-hailing startup, known for its tech-driven bike taxi, auto, and cab services that address last-mile urban connectivity. "
            "Founded in 2015 by Rishikesh S R, Pavan Guntupalli, and Aravind Sanka, Rapido has focused on affordable and sustainable commuting, with plans to convert its fleet to electric vehicles in select cities. "
            "Rapido has raised $559 million across 12 funding rounds, reflecting strong investor confidence in its urban mobility model."
        ),
        "metadata": {
            "name": "Rapido",
            "founded_year": 2015,
            "industry": "Transportation",
            "category": "Mobility, Ride-hailing, EV, Urban Transport, Bike Taxi, Auto, Cab, SaaS",
            "business_model": "B2C",
            "customer_model": "App-based",
            "funding_stage": "Series E",
            "funding_amount_million": 559,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://www.rapido.bike"
        }
    },
    {
        "text": (
            "Unacademy is a leading EdTech unicorn from Bangalore, providing a comprehensive digital learning platform for competitive exams such as NEET, JEE, CAT, UPSC, and CLAT. "
            "Started as a YouTube channel by Gaurav Munjal in 2010, it was officially co-founded in 2015 by Gaurav Munjal, Hemesh Singh, and Roman Saini. "
            "Unacademy connects educators, students, and parents through live and recorded classes, and has secured $880 million in funding to date."
        ),
        "metadata": {
            "name": "Unacademy",
            "founded_year": 2015,
            "industry": "EdTech",
            "category": "Education, Online Learning, Competitive Exams, SaaS",
            "business_model": "B2C",
            "customer_model": "Subscription",
            "funding_stage": "Series F",
            "funding_amount_million": 880,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://unacademy.com"
        }
    },
    {
        "text": (
            "Ninjacart is a Bangalore-based agritech startup that connects farmers directly with retailers and consumers, streamlining the agri-supply chain for fresh produce. "
            "Founded in 2015 by Kartheeswaran KK, Vasudevan C, Sharath Loganathan, Ashutosh Vikram, Thiru Nagarajan, and Sachin Jose, Ninjacart leverages technology to increase farmer incomes and ensure quality for end-users. "
            "The company has raised $508 million across 10 funding rounds."
        ),
        "metadata": {
            "name": "Ninjacart",
            "founded_year": 2015,
            "industry": "Agritech",
            "category": "Agriculture, Supply Chain, Marketplace, SaaS",
            "business_model": "B2B",
            "customer_model": "Marketplace",
            "funding_stage": "Series D",
            "funding_amount_million": 508,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://ninjacart.in"
        }
    },
    {
        "text": (
            "Practo is a healthtech pioneer from Bangalore, offering a digital platform for doctor appointments, telemedicine, and medical record management. "
            "Founded in 2008 by Shashank ND and Abhinav Lal, Practo has played a key role in making healthcare accessible and digitized for millions of Indians. "
            "The company has raised $228 million through 9 funding rounds."
        ),
        "metadata": {
            "name": "Practo",
            "founded_year": 2008,
            "industry": "HealthTech",
            "category": "Healthcare, Telemedicine, SaaS, Digital Health",
            "business_model": "B2C",
            "customer_model": "App-based",
            "funding_stage": "Series D",
            "funding_amount_million": 228,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://www.practo.com"
        }
    },
    {
        "text": (
            "Scripbox is a leading fintech startup from Bangalore, specializing in digital wealth management and mutual fund investment solutions. "
            "Founded in 2012 by Atul Shinghal and Sanjiv Singhal, Scripbox helps individuals grow their wealth through curated investment products and digital advisory services. "
            "The company has raised $80.5 million over 12 funding rounds."
        ),
        "metadata": {
            "name": "Scripbox",
            "founded_year": 2012,
            "industry": "FinTech",
            "category": "Wealth Management, Mutual Funds, SaaS, Finance",
            "business_model": "B2C",
            "customer_model": "SaaS",
            "funding_stage": "Series C",
            "funding_amount_million": 80.5,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://scripbox.com"
        }
    },
    {
        "text": (
            "Porter is a Bangalore-based logistics and e-commerce startup, offering intra-city and inter-city logistics solutions for businesses and individuals. "
            "With a tech-driven approach, Porter has raised $200 million in Series F funding as of May 2025, and continues to expand its service footprint across India."
        ),
        "metadata": {
            "name": "Porter",
            "founded_year": null,
            "industry": "Logistics, E-commerce",
            "category": "Logistics, Transportation, SaaS, E-commerce",
            "business_model": "B2B, B2C",
            "customer_model": "App-based",
            "funding_stage": "Series F",
            "funding_amount_million": 200,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://porter.in"
        }
    },
    {
        "text": (
            "Flam is a Bangalore-based AI startup focused on generative AI applications, providing solutions for enterprises to leverage artificial intelligence in business processes. "
            "The company raised $14 million in Series A funding in May 2025, reflecting growing interest in AI-driven SaaS products."
        ),
        "metadata": {
            "name": "Flam",
            "founded_year": null,
            "industry": "Artificial Intelligence",
            "category": "AI, GenAI, SaaS, Enterprise Software",
            "business_model": "B2B",
            "customer_model": "SaaS",
            "funding_stage": "Series A",
            "funding_amount_million": 14,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://flamapp.ai"
        }
    },
    {
        "text": (
            "Scapia is a Bangalore-based fintech startup offering innovative credit card and financial solutions, targeting digital-first consumers. "
            "The company secured $40 million in Series B funding in April 2025, supporting its expansion and product development."
        ),
        "metadata": {
            "name": "Scapia",
            "founded_year": null,
            "industry": "FinTech",
            "category": "Finance, Credit Card, SaaS, Digital Banking",
            "business_model": "B2C",
            "customer_model": "App-based",
            "funding_stage": "Series B",
            "funding_amount_million": 40,
            "funding_year": 2025,
            "valuation_million": null,
            "investors": [],
            "customers": [],
            "website": "https://www.scapia.cards"
        }
    }
]


    # Initialize and upload
    uploader = QdrantDataUploader(
        collection_name="Prod_data_V1", embedding_model="thenlper/gte-large"
    )

    # Upload the data
    uploader.upload_data(sample_data)

    # Optimize after upload
    uploader.optimize_collection()
