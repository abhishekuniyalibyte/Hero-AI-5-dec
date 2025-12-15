"""
menu embedding pipeline.

Loads structured menu JSON, extracts all items across schemas, converts them
into rich text chunks, and generates sentence-transformer embeddings for use
in vector search, semantic retrieval, and RAG workflows. Saves embeddings with
complete metadata and schema information.
"""


import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from typing import List, Dict, Any
import argparse


class UniversalMenuEmbedder:
    """Dynamic embedding generator that handles any menu structure."""

    def __init__(self, model_name: str = "thenlper/gte-large"):
        print(f"Loading model on CPU: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = []
        self.metadata = []
        self.menu_schema = None

    def load_menu_json(self, json_path: str) -> Dict[str, Any]:
        """Load menu data from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Detect schema
        self.menu_schema = self._detect_schema(data)
        print(f"Detected schema: {self.menu_schema['type']}")

        return data

    def _detect_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically detect the menu data structure."""
        schema = {
            "type": "unknown",
            "has_categories": False,
            "has_variants": False,
            "has_descriptions": False,
            "fields": []
        }

        # Check structure
        if "categories" in data:
            schema["type"] = "categorized"
            schema["has_categories"] = True

            # Check first category for structure
            if data["categories"]:
                first_cat = data["categories"][0]
                if "items" in first_cat and first_cat["items"]:
                    first_item = first_cat["items"][0]
                    schema["fields"] = list(first_item.keys())
                    schema["has_variants"] = "variants" in first_item
                    schema["has_descriptions"] = "description" in first_item

        elif "items" in data:
            schema["type"] = "flat"
            if data["items"]:
                first_item = data["items"][0]
                schema["fields"] = list(first_item.keys())
                schema["has_variants"] = "variants" in first_item
                schema["has_descriptions"] = "description" in first_item

        else:
            # Try to find any list of items
            for key, value in data.items():
                if isinstance(value, list) and value:
                    schema["type"] = "custom"
                    schema["fields"] = list(value[0].keys()) if isinstance(value[0], dict) else []
                    break

        return schema

    def create_text_chunks(self, menu_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create embeddable text chunks from any menu structure."""
        chunks = []

        # Extract items based on schema
        items = self._extract_all_items(menu_data)

        print(f"Creating chunks for {len(items)} items...")

        for idx, item in enumerate(items):
            # Build rich text representation
            text = self._build_item_text(item)

            # Create metadata
            metadata = self._build_item_metadata(item, idx)

            chunks.append({
                'text': text,
                'metadata': metadata
            })

        return chunks

    def _extract_all_items(self, menu_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all items regardless of structure."""
        items = []

        if self.menu_schema["type"] == "categorized":
            for category in menu_data.get("categories", []):
                cat_name = category.get("category", category.get("name", "General"))

                for item in category.get("items", []):
                    # Add category to item
                    item_with_cat = {**item, "category": cat_name}
                    items.append(item_with_cat)

        elif self.menu_schema["type"] == "flat":
            items = menu_data.get("items", [])

        else:
            # Custom: try to find any list
            for key, value in menu_data.items():
                if isinstance(value, list):
                    items.extend(value)
                    break

        return items

    def _build_item_text(self, item: Dict[str, Any]) -> str:
        """Build rich text representation for embedding."""
        parts = []

        # Name (always include)
        name = self._get_field(item, ['name', 'item_name', 'item', 'dish_name'])
        if name:
            parts.append(f"Item: {name}")

        # Category
        category = self._get_field(item, ['category', 'type', 'section'])
        if category:
            parts.append(f"Category: {category}")

        # Description
        description = self._get_field(item, ['description', 'desc', 'details'])
        if description:
            parts.append(f"Description: {description}")

        # Price information
        if item.get("variants"):
            # Has variants
            variant_prices = []
            for v in item["variants"]:
                v_name = v.get("name", v.get("size", ""))
                v_price = v.get("price")
                if v_price:
                    variant_prices.append(f"{v_name}: {v_price}")
            if variant_prices:
                parts.append(f"Prices: {', '.join(variant_prices)}")
        else:
            # Single price
            price = self._get_field(item, ['price', 'cost'])
            if price:
                parts.append(f"Price: {price}")

        # Tags/dietary info
        tags = self._get_field(item, ['tags', 'dietary_info', 'labels', 'attributes'])
        if tags:
            if isinstance(tags, list):
                parts.append(f"Tags: {', '.join(str(t) for t in tags)}")
            else:
                parts.append(f"Tags: {tags}")

        # Ingredients
        ingredients = self._get_field(item, ['ingredients', 'contains'])
        if ingredients:
            if isinstance(ingredients, list):
                parts.append(f"Ingredients: {', '.join(ingredients)}")
            else:
                parts.append(f"Ingredients: {ingredients}")

        # Additional fields (dynamic)
        for key, value in item.items():
            if key not in ['name', 'item_name', 'category', 'type', 'description',
                          'price', 'variants', 'tags', 'ingredients']:
                if value and str(value).strip():
                    parts.append(f"{key.replace('_', ' ').title()}: {value}")

        return ". ".join(parts)

    def _get_field(self, item: Dict[str, Any], field_names: List[str]) -> Any:
        """Get field value trying multiple possible names."""
        for field in field_names:
            if field in item and item[field]:
                return item[field]
        return None

    def _build_item_metadata(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Build comprehensive metadata."""
        name = self._get_field(item, ['name', 'item_name', 'item'])
        category = self._get_field(item, ['category', 'type'])
        price = self._get_field(item, ['price', 'cost'])

        metadata = {
            'item_id': idx,
            'name': name,
            'category': category,
            'price': price,
            'has_variants': bool(item.get("variants")),
            'original_data': item
        }

        # Add variants info if present
        if item.get("variants"):
            metadata['variants'] = item["variants"]

        return metadata

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        """Generate embeddings for all text chunks."""
        print(f"Generating embeddings for {len(chunks)} items...")

        texts = [chunk['text'] for chunk in chunks]

        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.metadata = [chunk['metadata'] for chunk in chunks]
        print(f"Generated {len(self.embeddings)} embeddings")

    def save_embeddings(self, output_path: str, format: str = 'pickle') -> None:
        """Save embeddings with enhanced metadata."""
        output_path = Path(output_path)

        save_data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'schema': self.menu_schema,
            'model': self.model.get_sentence_embedding_dimension()
        }

        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(save_data, f)

        elif format == 'npz':
            np.savez(
                output_path,
                embeddings=self.embeddings,
                metadata=np.array(self.metadata, dtype=object),
                schema=np.array([self.menu_schema], dtype=object)
            )

        print(f"Saved embeddings to {output_path}")
        print(f"  - Items: {len(self.metadata)}")
        print(f"  - Embedding dim: {self.embeddings.shape[1]}")
        print(f"  - Schema: {self.menu_schema['type']}")

    def process_menu(self, json_path: str, output_path: str, format: str = 'pickle') -> None:
        """Full pipeline: load -> chunk -> embed -> save."""
        print(f"\n{'='*60}")
        print(f"Processing: {json_path}")
        print(f"{'='*60}\n")

        menu_data = self.load_menu_json(json_path)
        chunks = self.create_text_chunks(menu_data)
        self.generate_embeddings(chunks)
        self.save_embeddings(output_path, format)

        print(f"\n{'='*60}")
        print("Embedding generation complete!")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings from any restaurant menu JSON'
    )
    parser.add_argument('input_json', help='Path to input menu JSON file')
    parser.add_argument('output_file', help='Path to output embeddings file')
    parser.add_argument('--format', choices=['pickle', 'npz'],
                       default='pickle', help='Output format (default: pickle)')
    parser.add_argument('--model', default="thenlper/gte-large",
                       help='Sentence transformer model (default: thenlper/gte-large)')

    args = parser.parse_args()

    embedder = UniversalMenuEmbedder(model_name=args.model)
    embedder.process_menu(args.input_json, args.output_file, args.format)


if __name__ == "__main__":
    main()
