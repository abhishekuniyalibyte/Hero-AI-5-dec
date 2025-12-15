"""
Adaptive restaurant menu chatbot.

Loads precomputed menu embeddings and metadata, performs semantic search over
menu items, and generates contextual responses using Groq LLMs. Supports
variant-aware item retrieval, dynamic context construction, and interactive
chat with conversation memory. Designed for menu recommendation, Q&A, and
retrieval-augmented interactions.
"""

import pickle
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
        
        print(f"Loaded {len(self.embeddings)} menu items")
        print(f"Menu schema: {self.schema.get('type', 'unknown')}")
        
        print(f"\nLoading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device="cpu")
        
        print("Initializing Groq client...")
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        self.conversation_history = []
        self.restaurant_info = self._extract_restaurant_info()
        
        print("Chatbot ready!\n")
    
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
        """Format results with dynamic field handling - shows ALL available data."""
        if not search_results:
            return "No matching items found in the menu."
        
        # Define which fields to skip in display (internal/processed fields)
        SKIP_FIELDS = {'name', 'category', 'type', 'description', 'desc', 
                      'price', 'cost', 'variants', 'tags', 'item_name', 
                      'item', 'dish_name', 'section'}
        
        context_parts = []
        
        for result in search_results:
            meta = result['metadata']
            orig = meta.get('original_data', {})
            
            item_parts = []
            
            # Name (bold)
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
                        variant_strs.append(f"{v_name}: Rs.{v_price}")
                if variant_strs:
                    item_parts.append(f"Prices: {', '.join(variant_strs)}")
            elif meta.get('price'):
                item_parts.append(f"Price: Rs.{meta['price']}")
            
            # Tags
            tags = orig.get('tags')
            if tags:
                if isinstance(tags, list):
                    item_parts.append(f"Tags: {', '.join(tags)}")
                else:
                    item_parts.append(f"Tags: {tags}")
            
            # DYNAMIC: Show ALL other fields (calories, ingredients, spicy_level, etc.)
            for key, value in orig.items():
                # Skip already displayed fields
                if key in SKIP_FIELDS:
                    continue
                
                # Skip empty values
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                
                if isinstance(value, (list, dict)) and len(value) == 0:
                    continue
                
                # Format nicely
                field_name = key.replace('_', ' ').title()
                
                if isinstance(value, list):
                    value_str = ', '.join(str(v) for v in value)
                elif isinstance(value, dict):
                    value_str = ', '.join(f"{k}: {v}" for k, v in value.items())
                else:
                    value_str = str(value)
                
                item_parts.append(f"{field_name}: {value_str}")
            
            context_parts.append('\n'.join(item_parts))
        
        return '\n\n---\n\n'.join(context_parts)
    
    def generate_response(self, user_query: str, context: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate contextual response."""
        # Build system prompt with menu awareness
        system_prompt = f"""You are a restaurant menu assistant helping customers find what they want to order.

Important guidelines:
1. Only recommend items that are shown in the menu context provided below
2. Do not suggest items that are not available in the menu
3. If someone asks about items we don't have, let them know it's not currently available
4. All prices are in Indian Rupees (Rs.). Always mention prices clearly
5. Be helpful and natural in your responses
6. Pay attention to how well the search results match the query
7. If nothing matches well, suggest looking at our different categories

Restaurant details:
- Name: {self.restaurant_info.get('name', 'Our restaurant')}
- Categories available: {', '.join(self.restaurant_info.get('categories', [])[:10])}
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 6 messages)
        messages.extend(self.conversation_history[-6:])
        
        # Add current query
        relevance_note = ""
        if search_results:
            avg_sim = np.mean([r['similarity'] for r in search_results])
            if avg_sim < 0.5:
                relevance_note = "\nNote: Search results have low relevance - customer may be asking about something not on the menu."
        
        messages.append({
            "role": "user",
            "content": f"""Customer question: {user_query}

Relevant menu items found:
{context}
{relevance_note}

Answer based only on the items shown above. Be friendly and helpful."""
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
        print("Conversation reset")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chatbot_v3.py <embeddings.pkl> [--debug]")
        exit(1)
    
    embeddings_path = sys.argv[1]
    debug_mode = '--debug' in sys.argv
    
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file not found: {embeddings_path}")
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
                print("\nThanks for chatting! Have a great day!")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_conversation()
                continue
            
            if user_input.lower() == 'stats':
                stats = chatbot.get_menu_stats()
                print(f"\nMenu Statistics:")
                print(f"  Total items: {stats['total_items']}")
                print(f"  Categories: {stats['total_categories']}")
                print(f"  Top categories:")
                for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    - {cat}: {count} items")
                print()
                continue
            
            # Get response
            print("\nAssistant: ", end="", flush=True)
            response = chatbot.chat(user_input, debug=debug_mode)
            print(response + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()