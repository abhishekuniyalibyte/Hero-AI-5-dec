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
            if data["items"]import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")


class AdaptiveMenuChatbot:
    """Smart chatbot that adapts to any menu structure."""
    
    def __init__(self, embeddings_path: str, model_name: str = 'thenlper/gte-large'):
        print("Loading menu embeddings...")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            self.schema = data.get('schema', {})
        
        print(f"✓ Loaded {len(self.embeddings)} menu items")
        print(f"✓ Menu schema: {self.schema.get('type', 'unknown')}")
        
        print(f"\nLoading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device="cpu")
        
        print("Initializing Groq client...")
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        self.conversation_history = []
        self.restaurant_info = self._extract_restaurant_info()
        
        print("✓ Chatbot ready!\n")
    
    def _extract_restaurant_info(self) -> Dict[str, Any]:
        """Extract restaurant info from metadata."""
        if self.metadata:
            first_item = self.metadata[0].get('original_data', {})
            return {
                'name': first_item.get('restaurant_name'),
                'categories': list(set(m.get('category') for m in self.metadata if m.get('category')))
            }
        return {}
    
    def search_menu(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search with adaptive filtering."""
        # Encode query
        query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        
        # Calculate similarities
        similarities = []
        for idx, embedding in enumerate(self.embeddings):
            sim = np.dot(query_embedding, embedding)
            if sim >= threshold:  # Filter low-relevance results
                similarities.append((idx, sim))
        
        # Sort and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, sim in similarities[:top_k]:
            results.append({
                'metadata': self.metadata[idx],
                'similarity': float(sim),
                'rank': len(results) + 1
            })
        
        return results
    
    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format results with dynamic field handling."""
        if not search_results:
            return "No matching items found in the menu."
        
        context_parts = []
        
        for result in search_results:
            meta = result['metadata']
            orig = meta.get('original_data', {})
            
            item_parts = []
            
            # Name
            name = meta.get('name', 'Unknown Item')
            item_parts.append(f"**{name}**")
            
            # Category
            if meta.get('category'):
                item_parts.append(f"Category: {meta['category']}")
            
            # Description
            desc = orig.get('description')
            if desc:
                item_parts.append(f"Description: {desc}")
            
            # Price/Variants
            if meta.get('has_variants') and meta.get('variants'):
                variant_strs = []
                for v in meta['variants']:
                    v_name = v.get('name', '')
                    v_price = v.get('price')
                    if v_price:
                        variant_strs.append(f"{v_name}: ₹{v_price}")
                if variant_strs:
                    item_parts.append(f"Prices: {', '.join(variant_strs)}")
            elif meta.get('price'):
                item_parts.append(f"Price: ₹{meta['price']}")
            
            # Tags
            tags = orig.get('tags')
            if tags:
                if isinstance(tags, list):
                    item_parts.append(f"Tags: {', '.join(tags)}")
                else:
                    item_parts.append(f"Tags: {tags}")
            
            # Additional dynamic fields
            for key, value in orig.items():
                if key not in ['name', 'description', 'price', 'variants', 'tags', 'category'] and value:
                    item_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            
            context_parts.append('\n'.join(item_parts))
        
        return '\n\n---\n\n'.join(context_parts)
    
    def generate_response(self, user_query: str, context: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate contextual response."""
        # Build system prompt with menu awareness
        system_prompt = f"""You are a friendly restaurant menu assistant.

CRITICAL RULES:
1. ONLY recommend items from the provided menu context below
2. DO NOT suggest items not in the menu
3. If asked about unavailable items, politely say they're not on the menu
4. ALL PRICES ARE IN INR (₹). Always show prices with rupee symbol
5. Be conversational, helpful, and accurate
6. Use the search results' similarity scores to gauge relevance
7. If no good matches found (low relevance), suggest browsing categories instead

MENU INFORMATION:
- Restaurant: {self.restaurant_info.get('name', 'Our restaurant')}
- Available categories: {', '.join(self.restaurant_info.get('categories', [])[:10])}
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 6 messages)
        messages.extend(self.conversation_history[-6:])
        
        # Add current query
        relevance_note = ""
        if search_results:
            avg_sim = np.mean([r['similarity'] for r in search_results])
            if avg_sim < 0.5:
                relevance_note = "\nNOTE: Search results have low relevance - user may be asking about something not on menu."
        
        messages.append({
            "role": "user",
            "content": f"""Customer question: {user_query}

Relevant menu items found:
{context}
{relevance_note}

Answer based ONLY on the items above. Be friendly and helpful."""
        })
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            response = completion.choices[0].message.content.strip()
            
            # Update history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        
        except Exception as e:
            return f"I apologize, I encountered an error: {str(e)}"
    
    def chat(self, user_query: str, debug: bool = False) -> str:
        """Main chat interface with optional debug info."""
        # Search
        search_results = self.search_menu(user_query, top_k=5)
        
        if debug:
            print(f"\n[DEBUG] Found {len(search_results)} results")
            for r in search_results[:3]:
                print(f"  - {r['metadata']['name']}: {r['similarity']:.3f}")
        
        # Format context
        context = self.format_context(search_results)
        
        # Generate response
        response = self.generate_response(user_query, context, search_results)
        
        return response
    
    def get_menu_stats(self) -> Dict[str, Any]:
        """Get menu statistics."""
        categories = {}
        total_items = len(self.metadata)
        
        for meta in self.metadata:
            cat = meta.get('category', 'Uncategorized')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_items': total_items,
            'total_categories': len(categories),
            'categories': categories,
            'has_variants': sum(1 for m in self.metadata if m.get('has_variants'))
        }
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("✓ Conversation reset")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chatbot_v3.py <embeddings.pkl> [--debug]")
        exit(1)
    
    embeddings_path = sys.argv[1]
    debug_mode = '--debug' in sys.argv
    
    if not os.path.exists(embeddings_path):
        print(f"❌ Embeddings file not found: {embeddings_path}")
        exit(1)
    
    # Initialize
    chatbot = AdaptiveMenuChatbot(embeddings_path)
    
    # Show stats
    stats = chatbot.get_menu_stats()
    
    print("\n" + "="*60)
    print("ADAPTIVE MENU CHATBOT")
    print("="*60)
    print(f"Menu Statistics:")
    print(f"  - Total items: {stats['total_items']}")
    print(f"  - Categories: {stats['total_categories']}")
    print(f"  - Items with variants: {stats['has_variants']}")
    print("\nCommands:")
    print("  'quit' / 'exit' - Exit")
    print("  'reset' - Clear conversation history")
    print("  'stats' - Show menu statistics")
    if debug_mode:
        print("  [DEBUG MODE ENABLED]")
    print("="*60 + "\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thanks for chatting! Have a great day!")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_conversation()
                continue
            
            if user_input.lower() == 'stats':
                stats = chatbot.get_menu_stats()
                print(f"\n📊 Menu Statistics:")
                print(f"  Total items: {stats['total_items']}")
                print(f"  Categories: {stats['total_categories']}")
                print(f"  Top categories:")
                for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    - {cat}: {count} items")
                print()
                continue
            
            # Get response
            print("\n🤖 Assistant: ", end="", flush=True)
            response = chatbot.chat(user_input, debug=debug_mode)
            print(response + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main():
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
        print(f"✓ Generated {len(self.embeddings)} embeddings")
    
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
        
        print(f"✓ Saved embeddings to {output_path}")
        print(f"  - Items: {len(self.metadata)}")
        print(f"  - Embedding dim: {self.embeddings.shape[1]}")
        print(f"  - Schema: {self.menu_schema['type']}")
    
    def process_menu(self, json_path: str, output_path: str, format: str = 'pickle') -> None:
        """Full pipeline: load → chunk → embed → save."""
        print(f"\n{'='*60}")
        print(f"Processing: {json_path}")
        print(f"{'='*60}\n")
        
        menu_data = self.load_menu_json(json_path)
        chunks = self.create_text_chunks(menu_data)
        self.generate_embeddings(chunks)
        self.save_embeddings(output_path, format)
        
        print(f"\n{'='*60}")
        print("✓ Embedding generation complete!")
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