from typing import List, Tuple
import os

from google.cloud import aiplatform

from infrastructure.embeddings import generate_embedding
from infrastructure.logger import get_logger

logger = get_logger()

def find_nearest_neighbors(
    project_id: str,
    location: str,
    index_endpoint_id: str,
    deployed_index_id: str,
    query: str,
    num_neighbors: int,
) -> List[Tuple[str, float]]:
    """
    Finds the nearest neighbors for a given query using Vertex AI Vector Search.

    Args:
        project_id: The Google Cloud project ID.
        location: The Google Cloud region.
        index_endpoint_id: The ID of the Vertex AI IndexEndpoint.
        deployed_index_id: The ID of the deployed index on the endpoint.
        query: The query string.
        num_neighbors: The number of nearest neighbors to retrieve.

    Returns:
        A list of tuples, where each tuple contains the ID of a neighbor and its distance.
    """
    logger.info("Initializing Vertex AI client.")
    aiplatform.init(project=project_id, location=location)

    logger.info(f"Generating embedding for query: '{query}'")
    query_embedding = generate_embedding(query)

    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=index_endpoint_id
    )

    logger.info(f"Querying deployed index '{deployed_index_id}' for {num_neighbors} neighbors.")
    
    response = index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_embedding],
        num_neighbors=num_neighbors,
    )

    neighbors = []
    if response and response[0]:
        for neighbor in response[0]:
            neighbors.append((neighbor.id, neighbor.distance))
    
    logger.info(f"Found {len(neighbors)} neighbors.")
    return neighbors

def upsert_datapoints(datapoints: list[dict]):
    """Upserts datapoints to the Vertex AI Vector Search index."""
    project_id = os.getenv("VERTEX_PROJECT_ID")
    region = os.getenv("VERTEX_REGION")
    index_endpoint_id = os.getenv("VERTEX_INDEX_ENDPOINT_ID")
    index_id = os.getenv("VERTEX_INDEX_ID")

    if not all([project_id, region, index_endpoint_id, index_id]):
        logger.warning("Vertex AI env vars not set, skipping vector index upsert.")
        return

    aiplatform.init(project=project_id, location=region)
    
    try:
        if not index_id:
            logger.error("VERTEX_INDEX_ID is not set.")
            return
        index = aiplatform.MatchingEngineIndex(index_name=index_id)
        index.upsert_datapoints(datapoints=datapoints)
        logger.info(f"Upserted {len(datapoints)} datapoints to Vertex AI index.")
    except Exception as e:
        logger.error(f"Failed to upsert to Vertex AI: {e}")
