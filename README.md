# Menu Extraction Project

## Environment Setup

Create virtual environment:

```bash
python3.10 -m venv myenv
```

Activate environment:

```bash
source myenv/bin/activate
```

---

## Files

* `menu_extraction.py` – extract menu from PDF/JPEG file
* `menu_embedding_generator.py` – generate embeddings from file created by `menu_extraction.py`
* `chatbot.py` – chatbot that reads the embedding file to answer user questions

---

## Run

```bash
python3 menu_extraction.py menu2.jpg
```

---

## Tasks to Be Done

* Show menu as a list
* Make it iteration-based (e.g., add gulab jamun in thali)
* With pizza, add toppings
* Calories (based on average)
* In meals, ask user preference (roti, which sabzi)
