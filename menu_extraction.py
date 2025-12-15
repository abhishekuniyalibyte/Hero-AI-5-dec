"""
Dynamic Menu Extraction Pipeline using Groq LLM models.

This module converts menu PDFs/images into structured JSON by:
1. Learning the menu's unique structure and data fields
2. Extracting ALL available information (not just predefined fields)
3. Creating a schema that adapts to each menu's specific format
4. Normalizing data for embeddings and downstream search/RAG pipelines
"""

import base64
import json
import os
import shutil
from io import BytesIO
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")


class DynamicMenuExtractor:
    """Fully adaptive menu extraction with zero hardcoded schema."""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.menu_schema = None  # Will be learned from menu
        self.restaurant_info = {}
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[bytes]:
        """Convert PDF to images in memory."""
        try:
            from pdf2image import convert_from_path
            print(f"Converting PDF to images...")
            pil_images = convert_from_path(pdf_path, dpi=300)
            
            image_bytes_list = []
            for img in pil_images:
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                image_bytes_list.append(buf.read())
            
            print(f"Loaded {len(image_bytes_list)} page(s)")
            return image_bytes_list
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def discover_menu_schema(self, image_bytes: bytes) -> Dict[str, Any]:
        """STEP 1: Learn what data THIS menu contains."""
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = """Study this menu carefully and discover its COMPLETE data structure.

Return ONLY JSON describing what information is available:

{
  "organizational_structure": {
    "type": "categories|sections|meal_times|cooking_methods|ingredients|mixed",
    "grouping_labels": ["group1", "group2", ...],
    "description": "how items are organized"
  },
  "item_data_fields": {
    "always_present": ["field1", "field2"],
    "sometimes_present": ["field3", "field4"],
    "never_present": ["field5"]
  },
  "pricing_structure": {
    "type": "single|multi_variant|tiered|market_price|none",
    "variant_pattern": "veg_nonveg|sizes|portions|toppings|none",
    "examples": ["item: price(s)"]
  },
  "special_attributes": {
    "dietary_info": true|false,
    "spice_levels": true|false,
    "allergens": true|false,
    "calories": true|false,
    "cooking_time": true|false,
    "chef_recommendations": true|false,
    "ingredients_list": true|false,
    "origin_region": true|false,
    "other": ["custom_attribute1", ...]
  },
  "visual_indicators": {
    "symbols": ["symbol: meaning"],
    "colors": ["color: meaning"],
    "icons": ["icon: meaning"]
  }
}

EXAMPLES OF DATA FIELDS TO LOOK FOR:
- Basic: name, price, description
- Dietary: vegetarian, vegan, gluten-free, halal, kosher
- Nutrition: calories, protein, carbs, fat, allergens
- Characteristics: spice level, cooking method, chef special, popular, new
- Practical: serving size, prep time, customizable
- Origin: regional cuisine, traditional, fusion

Be thorough - capture EVERYTHING this menu provides."""

        try:
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.1,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]
                }]
            )
            
            response = self._clean_json_response(completion.choices[0].message.content)
            schema = json.loads(response)
            
            print(f"  Organization: {schema.get('organizational_structure', {}).get('type')}")
            print(f"  Always present fields: {schema.get('item_data_fields', {}).get('always_present')}")
            print(f"  Special attributes found: {[k for k, v in schema.get('special_attributes', {}).items() if v == True]}")
            
            return schema
            
        except Exception as e:
            print(f"Schema discovery failed: {e}")
            return self._get_fallback_schema()
    
    def _get_fallback_schema(self) -> Dict[str, Any]:
        """Minimal fallback schema if discovery fails."""
        return {
            "organizational_structure": {
                "type": "categories",
                "grouping_labels": [],
                "description": "Unknown structure"
            },
            "item_data_fields": {
                "always_present": ["name", "price"],
                "sometimes_present": ["description"],
                "never_present": []
            },
            "pricing_structure": {
                "type": "single",
                "variant_pattern": "none",
                "examples": []
            },
            "special_attributes": {},
            "visual_indicators": {}
        }
    
    def extract_restaurant_info(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract restaurant metadata."""
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = """Extract ALL restaurant information visible. Return ONLY JSON with any/all of these fields (only include if present):

{
  "restaurant_name": "string",
  "phone": "string",
  "address": "string", 
  "website": "string",
  "email": "string",
  "social_media": {"platform": "handle"},
  "cuisine_type": "string",
  "hours": "string",
  "tagline": "string"
}

Only include fields that are actually visible."""

        try:
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.1,
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]
                }]
            )
            
            response = self._clean_json_response(completion.choices[0].message.content)
            return json.loads(response)
        except:
            return {}
    
    def extract_menu_items(self, image_bytes: bytes, schema: Dict[str, Any], retry_with_shorter_prompt: bool = False) -> Dict[str, Any]:
        """Extract menu items using discovered schema."""
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        if retry_with_shorter_prompt:
            prompt = """Extract menu items as JSON:
{"categories": [{"category": "string", "items": [{"name": "string", "price": number}]}]}

Rules:
- Extract ALL items visible
- Split items with multiple prices into variants
- Price = number only
- Output ONLY JSON"""
        else:
            prompt = self._build_schema_aware_prompt(schema)
        
        try:
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.1,
                max_tokens=8192,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]
                }]
            )
            
            response = self._clean_json_response(completion.choices[0].message.content)
            menu_data = json.loads(response)
            
            return self._normalize_menu_data(menu_data, schema)
            
        except json.JSONDecodeError as e:
            print(f"  JSON decode error: {e}")
            print(f"  Attempting to recover partial data...")
            
            # Try to recover partial JSON
            try:
                last_brace = response.rfind('}')
                if last_brace > 0:
                    test_json = response[:last_brace+1]
                    open_braces = test_json.count('{')
                    close_braces = test_json.count('}')
                    test_json += '}' * (open_braces - close_braces)
                    
                    menu_data = json.loads(test_json)
                    print("  Recovered partial data")
                    return self._normalize_menu_data(menu_data, schema)
            except:
                pass
            
            print("  Could not recover data")
            
            # Retry with simpler prompt
            if not retry_with_shorter_prompt:
                print("  Retrying with minimal prompt...")
                return self.extract_menu_items(image_bytes, schema, retry_with_shorter_prompt=True)
            
            return {"groups": []}
            
        except Exception as e:
            print(f"  Extraction failed: {e}")
            return {"groups": []}
    
    def _build_schema_aware_prompt(self, schema: Dict[str, Any]) -> str:
        """Build extraction prompt based on discovered schema."""
        
        org_structure = schema.get("organizational_structure", {})
        org_type = org_structure.get("type", "categories")
        group_labels = org_structure.get("grouping_labels", [])
        
        item_fields = schema.get("item_data_fields", {})
        always_fields = item_fields.get("always_present", ["name", "price"])
        sometimes_fields = item_fields.get("sometimes_present", [])
        
        pricing = schema.get("pricing_structure", {})
        pricing_type = pricing.get("type", "single")
        variant_pattern = pricing.get("variant_pattern", "none")
        
        special_attrs = schema.get("special_attributes", {})
        active_attrs = [k for k, v in special_attrs.items() if v == True]
        
        # Build dynamic JSON structure based on discovered schema
        group_name = "category" if org_type == "categories" else "section" if org_type == "sections" else "group"
        
        prompt = f"""Extract ALL menu items with COMPLETE information. Return ONLY JSON:

{{
  "{group_name}s": [
    {{
      "{group_name}": "string",
      "items": [
        {{
          // DYNAMIC STRUCTURE - Include ALL available fields
        }}
      ]
    }}
  ]
}}

📋 DATA STRUCTURE FOR THIS MENU:

1. ORGANIZATION:
   - Type: {org_type}
   - Groups found: {', '.join(group_labels) if group_labels else 'Discover from menu'}

2. REQUIRED FIELDS (always extract):
   {self._format_field_list(always_fields)}

3. OPTIONAL FIELDS (extract if present):
   {self._format_field_list(sometimes_fields)}

4. PRICING STRUCTURE:
   - Type: {pricing_type}
"""

        # Add pricing-specific instructions
        if pricing_type == "multi_variant":
            prompt += f"""   - Variant pattern: {variant_pattern}
   - For items with MULTIPLE prices (e.g., ₹100/150):
"""
            if variant_pattern == "veg_nonveg":
                prompt += """     → Create variants: [{"name": "Vegetarian", "price": X}, {"name": "Non-Vegetarian", "price": Y}]
"""
            elif variant_pattern == "sizes":
                prompt += """     → Use size labels from menu (Small/Medium/Large, etc.)
"""
            elif variant_pattern == "portions":
                prompt += """     → Use portion labels from menu (Half/Full, Single/Double, etc.)
"""
            else:
                prompt += """     → Identify option names or use "Option 1", "Option 2"
"""
            
            prompt += """   - For items with ONE price:
     → Use single "price" field, NOT variants
"""
        else:
            prompt += """   - Each item has a single price
   - Extract the exact price shown
"""

        # Add special attributes instructions
        if active_attrs:
            prompt += f"""
5. SPECIAL ATTRIBUTES TO EXTRACT:
   {self._format_special_attrs(active_attrs, special_attrs)}
"""

        prompt += """
6. CRITICAL EXTRACTION RULES:

   ✓ Extract EVERY field visible for each item
   ✓ Keep original language and spelling
   ✓ Don't invent or assume data - only extract what you see
   ✓ For multi-price items: COUNT the numbers (1 = single price, 2+ = variants)
   ✓ Include ALL metadata: dietary tags, spice levels, allergens, etc.
   ✓ Capture visual indicators (symbols, icons) as tags or attributes
   ✓ Empty fields should be omitted (not null)

OUTPUT ONLY JSON, NO EXPLANATIONS OR MARKDOWN."""

        return prompt
    
    def _format_field_list(self, fields: List[str]) -> str:
        """Format field list for prompt."""
        if not fields:
            return "   - (none specified)"
        return "\n".join([f"   - {field}" for field in fields])
    
    def _format_special_attrs(self, active_attrs: List[str], all_attrs: Dict[str, Any]) -> str:
        """Format special attributes for prompt."""
        formatted = []
        for attr in active_attrs:
            if attr == "dietary_info":
                formatted.append("   - Dietary tags: vegetarian, vegan, gluten-free, etc.")
            elif attr == "spice_levels":
                formatted.append("   - Spice level: mild, medium, hot, etc.")
            elif attr == "allergens":
                formatted.append("   - Allergen info: nuts, dairy, gluten, etc.")
            elif attr == "calories":
                formatted.append("   - Nutritional info: calories, protein, etc.")
            elif attr == "cooking_time":
                formatted.append("   - Preparation/cooking time")
            elif attr == "chef_recommendations":
                formatted.append("   - Chef special, recommended, signature dish")
            elif attr == "ingredients_list":
                formatted.append("   - Main ingredients or components")
            elif attr == "origin_region":
                formatted.append("   - Regional cuisine or origin")
            else:
                formatted.append(f"   - {attr}")
        
        if "other" in all_attrs and all_attrs["other"]:
            for custom in all_attrs["other"]:
                formatted.append(f"   - {custom}")
        
        return "\n".join(formatted) if formatted else "   - (none found)"
    
    def _normalize_menu_data(self, raw_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extracted data - minimal processing to preserve all fields."""
        normalized = {"groups": []}
        
        # Find the groups key (could be "categories", "sections", "groups", etc.)
        groups_key = None
        for key in raw_data.keys():
            if key.endswith('s') and isinstance(raw_data[key], list):
                groups_key = key
                break
        
        if not groups_key:
            # No groups found, try to extract items directly
            if "items" in raw_data:
                groups = [{"group": "Menu Items", "items": raw_data["items"]}]
            else:
                groups = []
        else:
            groups = raw_data[groups_key]
        
        for group in groups:
            # Extract group name (could be "category", "section", "group", etc.)
            group_name = None
            for key in ["category", "section", "group", "name", "title"]:
                if key in group:
                    group_name = group[key]
                    break
            
            normalized_group = {
                "group_name": group_name or "General",
                "items": []
            }
            
            items = group.get("items", [])
            for item in items:
                normalized_item = self._normalize_item(item)
                if normalized_item:
                    normalized_group["items"].append(normalized_item)
            
            if normalized_group["items"]:
                normalized["groups"].append(normalized_group)
        
        return normalized
    
    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize item - preserve ALL fields dynamically."""
        # Extract name (required)
        name = None
        for key in ["name", "item_name", "item", "title"]:
            if key in item:
                name = item[key]
                break
        
        if not name or str(name).strip() == "":
            return None
        
        # Start with name
        normalized = {"name": str(name).strip()}
        
        # Process all other fields dynamically
        for key, value in item.items():
            # Skip alternate name fields
            if key in ["item_name", "item", "title"]:
                continue
            
            # Skip empty values
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, dict)) and len(value) == 0:
                continue
            
            # Special handling for variants
            if key == "variants" and isinstance(value, list):
                cleaned_variants = []
                for v in value:
                    v_name = v.get("name", v.get("size", ""))
                    v_price = self._extract_price(v.get("price"))
                    
                    if v_price is None:
                        continue
                    
                    if not v_name or str(v_name).strip() == "":
                        v_name = f"Option {len(cleaned_variants) + 1}"
                    
                    cleaned_variants.append({
                        "name": str(v_name).strip(),
                        "price": v_price
                    })
                
                if len(cleaned_variants) > 1:
                    normalized["variants"] = cleaned_variants
                elif len(cleaned_variants) == 1:
                    normalized["price"] = cleaned_variants[0]["price"]
                continue
            
            # Special handling for price
            if key == "price":
                price = self._extract_price(value)
                if price is not None:
                    normalized["price"] = price
                continue
            
            # All other fields - pass through as-is
            normalized[key] = value
        
        # Ensure at least price or variants exists
        if "price" not in normalized and "variants" not in normalized:
            # Try to find price in any field
            for key, value in item.items():
                if "price" in key.lower():
                    price = self._extract_price(value)
                    if price is not None:
                        normalized["price"] = price
                        break
        
        return normalized
    
    def _extract_price(self, price_value: Any) -> Optional[float]:
        """Extract numeric price from various formats."""
        if price_value is None:
            return None
        
        if isinstance(price_value, (int, float)):
            return float(price_value)exists
        if "price" not in normalized and "variants" not in normalized:
            # Try to find price in any field
            for key, value in item.items():
                if "price" in key.lower():
                    price = self._extract_price(value)
                    if price is not None:
                        normalized["price"] = price
                        break
        
        return normalized
    
        
        if isinstance(price_value, str):
            clean = re.sub(r'[^\d.]', '', price_value)
            try:
                return float(clean) if clean else None
            except:
                return None
        
        return None
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract pure JSON."""
        response = response.strip()
        
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        return response.strip()
    
    def _save_page_json(self, page_data: Dict[str, Any], temp_dir: str, page_num: int) -> str:
        """Save individual page JSON to temp directory."""
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"page_{page_num}.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _load_page_jsons(self, temp_dir: str) -> List[Dict[str, Any]]:
        """Load all page JSONs from temp directory."""
        all_groups = []
        
        page_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("page_") and f.endswith(".json")])
        
        for page_file in page_files:
            page_path = os.path.join(temp_dir, page_file)
            with open(page_path, "r", encoding="utf-8") as f:
                page_data = json.load(f)
                all_groups.extend(page_data.get("groups", []))
        
        return all_groups
    
    def process_menu(self, menu_path: str, output_path: str) -> Dict[str, Any]:
        """Main processing pipeline with page-by-page extraction."""
        print(f"\n{'='*60}")
        print(f"Processing: {menu_path}")
        print(f"{'='*60}\n")
        
        # Load images
        if menu_path.lower().endswith(".pdf"):
            image_bytes_list = self.convert_pdf_to_images(menu_path)
        else:
            with open(menu_path, "rb") as f:
                image_bytes_list = [f.read()]
        
        if not image_bytes_list:
            print("Failed to load images")
            return {}
        
        # Create temp directory for page JSONs
        menu_dir = os.path.dirname(menu_path) or "."
        menu_name = os.path.splitext(os.path.basename(menu_path))[0]
        temp_dir = os.path.join(menu_dir, f".{menu_name}_pages")
        
        # STEP 1: Discover the menu's schema from first page
        print("🔍 Discovering menu schema...")
        self.menu_schema = self.discover_menu_schema(image_bytes_list[0])
        
        # STEP 2: Extract restaurant info from first page
        print("\n📋 Extracting restaurant info...")
        self.restaurant_info = self.extract_restaurant_info(image_bytes_list[0])
        print(f"Restaurant: {self.restaurant_info.get('restaurant_name', 'Unknown')}")
        
        # STEP 3: Process each page individually and save
        successful_pages = 0
        
        for i, img_bytes in enumerate(image_bytes_list, 1):
            print(f"\n{'='*60}")
            print(f"Processing page {i}/{len(image_bytes_list)}...")
            print(f"{'='*60}")
            
            page_data = self.extract_menu_items(img_bytes, self.menu_schema)
            
            if page_data and page_data.get("groups"):
                groups = page_data.get("groups", [])
                total_items = sum(len(g.get('items', [])) for g in groups)
                print(f"  Found {len(groups)} groups, {total_items} items")
                
                # Save page JSON
                self._save_page_json(page_data, temp_dir, i)
                print(f"  Saved page {i} JSON")
                successful_pages += 1
            else:
                print(f"  Failed to extract page {i}")
        
        if successful_pages == 0:
            print("\n❌ No data extracted from any page!")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {}
        
        # STEP 4: Load and merge all page JSONs
        print(f"\n{'='*60}")
        print("MERGING ALL PAGES")
        print(f"{'='*60}")
        
        all_groups = self._load_page_jsons(temp_dir)
        merged_groups = self._merge_groups(all_groups)
        
        # Build final output
        final_data = {
            "restaurant": self.restaurant_info,
            "menu": {
                "groups": merged_groups
            },
            "schema": self.menu_schema,
            "metadata": {
                "total_groups": len(merged_groups),
                "total_items": sum(len(g["items"]) for g in merged_groups),
                "pages_processed": f"{successful_pages}/{len(image_bytes_list)}",
                "extraction_date": self._get_timestamp()
            }
        }
        
        # Save final JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files")
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Saved to: {output_path}")
        print(f"✓ Pages Processed: {successful_pages}/{len(image_bytes_list)}")
        print(f"✓ Total groups: {final_data['metadata']['total_groups']}")
        print(f"✓ Total items: {final_data['metadata']['total_items']}")
        print(f"✓ Schema: {self.menu_schema.get('organizational_structure', {}).get('type')}")
        print(f"{'='*60}\n")
        
        return final_data
    
    def _merge_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate groups across pages."""
        group_map = {}
        
        for group in groups:
            group_name = group.get("group_name", "General")
            
            if group_name not in group_map:
                group_map[group_name] = {
                    "group_name": group_name,
                    "items": []
                }
            
            existing_names = {item["name"] for item in group_map[group_name]["items"]}
            
            for item in group.get("items", []):
                if item["name"] not in existing_names:
                    group_map[group_name]["items"].append(item)
                    existing_names.add(item["name"])
        
        return list(group_map.values())
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python menu_extraction.py <menu_file>")
        print("Supported: PDF, JPG, JPEG, PNG")
        exit(1)
    
    menu_file = sys.argv[1]
    
    if not os.path.exists(menu_file):
        print(f"File not found: {menu_file}")
        exit(1)
    
    menu_name = os.path.splitext(os.path.basename(menu_file))[0]
    output_path = f"{menu_name}_extracted.json"
    
    extractor = DynamicMenuExtractor(GROQ_API_KEY)
    extractor.process_menu(menu_path=menu_file, output_path=output_path)


if __name__ == "__main__":
    main()