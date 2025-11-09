"""
Smart Image Selector using LLM.
Filters images based on relevance to user query by analyzing image filenames.
"""

import logging
from typing import List, Dict
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    print("⚠️  Groq not installed. Please run: pip install groq")
    Groq = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


class SmartImageSelector:
    """
    Selects relevant images using LLM-based filename analysis.

    Features:
    - Analyzes semantic image filenames
    - Compares with user query
    - Ranks images by relevance
    - Filters to most relevant subset
    """

    def __init__(self):
        """Initialize smart image selector."""
        self.settings = get_settings()

        if Groq is None:
            raise RuntimeError("Groq library not installed")

        self.client = Groq(api_key=self.settings.groq_api_key)
        logger.info("Initialized SmartImageSelector")

    def select_relevant_images(self,
                               query: str,
                               all_images: List[str],
                               max_images: int = 6) -> List[str]:
        """
        Select most relevant images for a query.

        Args:
            query: User's question
            all_images: List of all available image paths
            max_images: Maximum images to return

        Returns:
            List of relevant image paths, ordered by relevance
        """
        if not all_images:
            logger.info("No images provided for selection")
            return []

        logger.info(f"Selecting relevant images from {len(all_images)} candidates for query: '{query[:50]}...'")

        # Extract filenames for analysis
        image_names = [self._extract_semantic_name(path) for path in all_images]

        # Build prompt for LLM
        prompt = self._build_selection_prompt(query, image_names)

        try:
            # Call LLM to rank images
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing image relevance based on semantic filenames.
Your task is to identify which images would be most helpful for answering a user's question.

Image filenames follow the pattern: {topic}_{section}_{page}_img{number}.png
For example: "kubernetes_config_chunksec_156_page38_img0.png" indicates an image about Kubernetes configuration.

Return ONLY the indices (0-based) of the most relevant images, comma-separated, in order of relevance.
Example response: "2,5,0,7" (max 6 images)

If no images are relevant, return "none"."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent selection
                max_tokens=100
            )

            result = response.choices[0].message.content.strip()
            logger.info(f"  LLM selection result: {result}")

            # Parse response
            selected_images = self._parse_selection(result, all_images, max_images)

            logger.info(f"✅ Selected {len(selected_images)} relevant images")
            return selected_images

        except Exception as e:
            logger.error(f"Error in image selection: {e}")
            # Fallback: return first max_images
            logger.info(f"  Falling back to first {max_images} images")
            return all_images[:max_images]

    def _extract_semantic_name(self, image_path: str) -> str:
        """
        Extract semantic topic from image filename.

        Example:
        'cache/images/kubernetes_config_chunksec_156_page38_img0.png'
        -> 'kubernetes config'
        """
        filename = Path(image_path).stem

        # Extract topic part (before _chunksec_)
        if '_chunksec_' in filename:
            topic_part = filename.split('_chunksec_')[0]
            # Replace underscores with spaces for readability
            semantic_name = topic_part.replace('_', ' ')
            return semantic_name

        return filename.replace('_', ' ')

    def _build_selection_prompt(self, query: str, image_names: List[str]) -> str:
        """Build prompt for LLM image selection."""
        images_list = "\n".join([f"{idx}. {name}" for idx, name in enumerate(image_names)])

        prompt = f"""User Question: "{query}"

Available Images (by topic):
{images_list}

Which images would be most helpful for answering this question?
Return the indices (0-based) of the most relevant images, comma-separated, in order of relevance (max 6).

If no images are relevant, return "none"."""

        return prompt

    def _parse_selection(self, llm_response: str, all_images: List[str], max_images: int) -> List[str]:
        """Parse LLM response and return selected images."""
        # Handle "none" response
        if llm_response.lower().strip() == "none":
            return []

        try:
            # Parse comma-separated indices
            indices = [int(idx.strip()) for idx in llm_response.split(',')]

            # Validate and filter indices
            valid_indices = [idx for idx in indices if 0 <= idx < len(all_images)]

            # Limit to max_images
            valid_indices = valid_indices[:max_images]

            # Return selected images
            selected = [all_images[idx] for idx in valid_indices]
            return selected

        except Exception as e:
            logger.error(f"Error parsing LLM response '{llm_response}': {e}")
            # Fallback: return first max_images
            return all_images[:max_images]


if __name__ == "__main__":
    """Test smart image selector."""
    import json

    # Load some sample images
    with open('cache/hierarchical_chunks.json') as f:
        data = json.load(f)

    # Find chunks with images
    chunks_with_images = [c for c in data if c.get('metadata', {}).get('image_paths')]

    if chunks_with_images:
        # Get all images
        all_images = []
        for chunk in chunks_with_images[:20]:  # First 20 chunks
            all_images.extend(chunk['metadata']['image_paths'])

        print(f"\n📸 Testing Smart Image Selector")
        print(f"Total images available: {len(all_images)}\n")

        # Test queries
        test_queries = [
            "How do I configure Kubernetes deployment?",
            "What are the steps to integrate MS Teams?",
            "How do I set up Active Directory integration?",
        ]

        selector = SmartImageSelector()

        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            selected = selector.select_relevant_images(query, all_images, max_images=4)
            print(f"✅ Selected {len(selected)} images:")
            for img in selected:
                print(f"   - {Path(img).name}")
    else:
        print("No chunks with images found in cache")
