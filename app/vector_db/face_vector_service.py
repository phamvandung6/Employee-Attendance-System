"""Face embedding vector operations service for Qdrant."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.vector_db.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


class FaceVectorService:
    """Service for managing face embeddings in Qdrant vector database."""

    def __init__(self):
        """Initialize the service."""
        self.collection_name = settings.qdrant_collection_name
        self.similarity_threshold = settings.face_similarity_threshold

    async def insert_embedding(
        self,
        employee_id: UUID,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert a face embedding into Qdrant.

        Args:
            employee_id: UUID of the employee
            embedding: Face embedding vector (should be 512-dimensional for ArcFace)
            metadata: Additional metadata to store with the embedding

        Returns:
            Point ID as string

        Raises:
            ValueError: If embedding dimension is invalid
            RuntimeError: If insertion fails
        """
        if len(embedding) != 512:
            raise ValueError(
                f"Invalid embedding dimension: {len(embedding)}. Expected 512 for ArcFace."
            )

        client = await get_qdrant_client()
        point_id = str(uuid4())

        # Prepare payload
        payload = {
            "employee_id": str(employee_id),
            **(metadata or {}),
        }

        try:
            await client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            logger.info(
                f"Inserted embedding for employee {employee_id} with point ID {point_id}"
            )
            return point_id

        except UnexpectedResponse as e:
            logger.error(f"Failed to insert embedding for employee {employee_id}: {e}")
            raise RuntimeError(f"Qdrant insertion failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error inserting embedding: {e}")
            raise

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[models.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar face embeddings.

        Args:
            query_embedding: Query face embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1). Uses config default if not provided.
            filter_conditions: Optional Qdrant filter for metadata

        Returns:
            List of matches with format:
            [
                {
                    "point_id": "uuid",
                    "employee_id": "uuid",
                    "score": 0.95,
                    "metadata": {...}
                },
                ...
            ]

        Raises:
            ValueError: If query embedding dimension is invalid
            RuntimeError: If search fails
        """
        if len(query_embedding) != 512:
            raise ValueError(
                f"Invalid query embedding dimension: {len(query_embedding)}. Expected 512."
            )

        # Use configured threshold if not provided
        if score_threshold is None:
            score_threshold = self.similarity_threshold

        client = await get_qdrant_client()

        try:
            search_result = await client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold,
            )

            # Transform results to a more usable format
            results = []
            for hit in search_result:
                result = {
                    "point_id": hit.id,
                    "employee_id": hit.payload.get("employee_id"),
                    "score": hit.score,
                    "metadata": {
                        k: v for k, v in hit.payload.items() if k != "employee_id"
                    },
                }
                results.append(result)

            logger.info(
                f"Found {len(results)} similar embeddings with score >= {score_threshold}"
            )
            return results

        except UnexpectedResponse as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise RuntimeError(f"Qdrant search failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error searching embeddings: {e}")
            raise

    async def delete_embedding(self, point_id: str) -> bool:
        """
        Delete a face embedding by point ID.

        Args:
            point_id: The point ID to delete

        Returns:
            True if deletion was successful

        Raises:
            RuntimeError: If deletion fails
        """
        client = await get_qdrant_client()

        try:
            await client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id],
                ),
                wait=True,
            )

            logger.info(f"Deleted embedding with point ID {point_id}")
            return True

        except UnexpectedResponse as e:
            logger.error(f"Failed to delete embedding {point_id}: {e}")
            raise RuntimeError(f"Qdrant deletion failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error deleting embedding: {e}")
            raise

    async def delete_embeddings_by_employee(self, employee_id: UUID) -> int:
        """
        Delete all embeddings for a specific employee.

        Args:
            employee_id: UUID of the employee

        Returns:
            Number of embeddings deleted

        Raises:
            RuntimeError: If deletion fails
        """
        client = await get_qdrant_client()

        try:
            # Delete by filter
            result = await client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="employee_id",
                                match=models.MatchValue(value=str(employee_id)),
                            )
                        ]
                    )
                ),
                wait=True,
            )

            # The result contains operation_id, we assume success if no exception
            logger.info(f"Deleted all embeddings for employee {employee_id}")
            return 1  # We don't get exact count from Qdrant delete operation

        except UnexpectedResponse as e:
            logger.error(f"Failed to delete embeddings for employee {employee_id}: {e}")
            raise RuntimeError(f"Qdrant deletion failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error deleting employee embeddings: {e}")
            raise

    async def update_embedding_metadata(
        self,
        point_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Update metadata for an existing embedding.

        Args:
            point_id: The point ID to update
            metadata: New metadata to set (will be merged with existing)

        Returns:
            True if update was successful

        Raises:
            RuntimeError: If update fails
        """
        client = await get_qdrant_client()

        try:
            await client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[point_id],
                wait=True,
            )

            logger.info(f"Updated metadata for point {point_id}")
            return True

        except UnexpectedResponse as e:
            logger.error(f"Failed to update metadata for point {point_id}: {e}")
            raise RuntimeError(f"Qdrant update failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error updating metadata: {e}")
            raise

    async def count_embeddings(self, employee_id: Optional[UUID] = None) -> int:
        """
        Count embeddings in the collection.

        Args:
            employee_id: Optional employee ID to filter by

        Returns:
            Number of embeddings

        Raises:
            RuntimeError: If count operation fails
        """
        client = await get_qdrant_client()

        try:
            count_filter = None
            if employee_id:
                count_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="employee_id",
                            match=models.MatchValue(value=str(employee_id)),
                        )
                    ]
                )

            result = await client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True,
            )

            logger.info(f"Counted {result.count} embeddings")
            return result.count

        except UnexpectedResponse as e:
            logger.error(f"Failed to count embeddings: {e}")
            raise RuntimeError(f"Qdrant count failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error counting embeddings: {e}")
            raise


# Global service instance
face_vector_service = FaceVectorService()
