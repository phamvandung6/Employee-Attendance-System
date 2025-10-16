"""Qdrant async client singleton with connection management."""

import logging
from typing import Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantClientManager:
    """Singleton manager for Qdrant async client with connection pooling."""

    _instance: Optional["QdrantClientManager"] = None
    _client: Optional[AsyncQdrantClient] = None

    def __new__(cls) -> "QdrantClientManager":
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> AsyncQdrantClient:
        """Get or create the Qdrant async client."""
        if self._client is None:
            # Initialize client with connection parameters
            kwargs = {
                "url": settings.qdrant_url,
                "timeout": 30.0,  # 30 seconds timeout
            }

            # Add API key if provided (for Qdrant Cloud)
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key

            self._client = AsyncQdrantClient(**kwargs)
            logger.info(f"Qdrant client initialized with URL: {settings.qdrant_url}")

        return self._client

    async def ensure_collection(
        self,
        collection_name: str,
        vector_size: int = 512,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        """
        Ensure collection exists with proper configuration.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors (default: 512 for ArcFace)
            distance: Distance metric (default: COSINE)
        """
        try:
            exists = await self.client.collection_exists(collection_name)

            if not exists:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info(
                    f"Created collection '{collection_name}' with vector size {vector_size}"
                )

                # Create payload index for employee_id to enable filtering
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="employee_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(
                    f"Created payload index for 'employee_id' in collection '{collection_name}'"
                )
            else:
                logger.info(f"Collection '{collection_name}' already exists")

        except UnexpectedResponse as e:
            logger.error(f"Failed to ensure collection '{collection_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error ensuring collection: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Qdrant server is reachable and healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to list collections as a health check
            await self.client.get_collections()
            logger.info("Qdrant health check: OK")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client connection closed")


# Global singleton instance
qdrant_manager = QdrantClientManager()


async def get_qdrant_client() -> AsyncQdrantClient:
    """
    Dependency function to get Qdrant client.

    Usage:
        @app.get("/endpoint")
        async def endpoint(client: AsyncQdrantClient = Depends(get_qdrant_client)):
            ...
    """
    return qdrant_manager.client


async def init_qdrant() -> None:
    """
    Initialize Qdrant on application startup.
    Creates the face_embeddings collection if it doesn't exist.
    """
    try:
        logger.info("Initializing Qdrant connection...")

        # Health check
        is_healthy = await qdrant_manager.health_check()
        if not is_healthy:
            raise RuntimeError("Qdrant server is not healthy")

        # Ensure face embeddings collection exists
        await qdrant_manager.ensure_collection(
            collection_name=settings.qdrant_collection_name,
            vector_size=512,  # ArcFace embedding size
            distance=models.Distance.COSINE,
        )

        logger.info("Qdrant initialization completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        raise


async def cleanup_qdrant() -> None:
    """Cleanup Qdrant connection on application shutdown."""
    logger.info("Cleaning up Qdrant connection...")
    await qdrant_manager.close()
