# Ado-baker
# **GraphRAG: Persona-Driven Document Intelligence**

## **Overview**

This project is a solution for **Round 1B (Persona-Driven Document Intelligence)** of the **Adobe India Hackathon 2025**.
The objective is to extract and rank the most relevant sections from a collection of PDFs based on a **persona** (user profile) and a **job-to-be-done** (specific task).

Our solution leverages:

* **PDF parsing** for structured text extraction,
* **Graph-based entity modeling (GraphRAG)**,
* **Transformer-based embeddings** for semantic similarity,
* **Ranking based on query relevance and graph connectivity**.

The final output is a JSON file that prioritizes the most important sections for the given persona and task.

---

## **Key Idea & Approach**

### **1. Problem Understanding**

* We are given **3–10 PDFs**, a **persona**, and a **job-to-be-done**.
* The goal is to find **sections/pages most relevant** to the persona’s needs.
* We must rank sections by their importance and relevance.

---

### **2. Core Components**

#### **(a) Text Extraction**

* We use **PyMuPDF (fitz)** to extract raw text from PDF pages.
* Text is **split into chunks** (max \~300 words) to handle large sections.

#### **(b) Graph Construction (GraphRAG)**

* Each chunk is treated as a **node** in a graph.
* We extract **named entities** (e.g., people, organizations, dates) using **spaCy**.
* Each entity is also a node, and we **connect chunks to entities** they mention.
* This graph structure helps identify **contextually rich sections**.

#### **(c) Embedding & Relevance Scoring**

* We use **transformers (`all-MiniLM-L6-v2`)** to create **semantic embeddings** of each text chunk.
* We also embed the **persona + job description** as a query vector.
* Cosine similarity is computed between each chunk and the query.

#### **(d) Ranking**

* Final ranking is based on:

  * **Cosine similarity score** with the persona/job query.
  * **Graph connectivity bonus** (chunks connected to more entities are slightly boosted).

#### **(e) Persona Handling**

* The persona and job description are read from `persona.json` (if present in `/app/input`).
* Fallback: If no `persona.json` exists, environment variables `PERSONA` and `JOB` are used.

---

### **3. Output**

The output JSON (`/app/output/persona_analysis_graphrag.json`) contains:

* **Metadata** (persona, job, documents).
* **Ranked Sections** (document name, page, text, importance score).

**Example:**

```json
{
  "metadata": {
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
    "documents": ["doc1.pdf", "doc2.pdf"]
  },
  "ranked_sections": [
    {
      "document": "doc1.pdf",
      "page": 3,
      "text": "This section explains the use of GNNs for drug discovery...",
      "importance_score": 0.8734
    }
  ]
}
```

---

## **Folder Structure**

```
Round1B_GraphRAG_Solution/
│
├── Dockerfile
├── persona_analysis_graphrag.py      # Main script
├── requirements.txt                  # Dependencies
├── input/                            # Put all your PDFs here
│   └── persona.json                  # Persona + job-to-be-done (optional)
└── output/                           # Generated JSON output
```

---

## **Setup Instructions**

### **1. Build Docker Image**

```bash
docker build --platform linux/amd64 -t graphrag_solution:latest .
```

### **2. Prepare Input**

* Place all your PDF files in the `input/` folder.
* Add a `persona.json` file if needed:

  ```json
  {
    "persona": {
      "role": "Investment Analyst",
      "expertise": "Tech market trends and R&D investments"
    },
    "job_to_be_done": "Analyze revenue trends and market positioning"
  }
  ```

---

## **Running the Solution**

### **3. Run Container (Linux/Mac)**

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  graphrag_solution:latest
```

### **For Windows PowerShell:**

```powershell
docker run --rm `
  -v ${PWD}\input:/app/input `
  -v ${PWD}\output:/app/output `
  graphrag_solution:latest
```

---

## **Environment Variable Override**

If no `persona.json` exists, you can set persona and job like this:

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PERSONA="Student" \
  -e JOB="Summarize organic chemistry key concepts" \
  graphrag_solution:latest
```

---

## **Dependencies**

* **PyMuPDF (fitz)** – PDF parsing.
* **spaCy** – NLP for entity extraction.
* **Transformers** – Embedding model.
* **NetworkX** – Graph-based entity modeling.
* **NumPy** – Cosine similarity computation.

---

## **Why This Approach Works**

* **Graph + Embedding Hybrid**: Combines semantic meaning (via embeddings) with entity-based graph connections, ensuring both **contextual relevance** and **content richness**.
* **Persona-Driven Search**: Tailors output by embedding persona/job context, providing a **user-centric document analysis**.
* **Scalable**: Handles 3–10 PDFs and runs efficiently with lightweight models.
* **Offline Ready**: Works without internet (as required by hackathon rules).

---

## **Future Improvements**

* Add **sub-section summarization** for each ranked section.
* Support **multi-lingual PDFs** (e.g., Japanese).
* Optimize graph scoring using **centrality measures** (e.g., PageRank).

---


