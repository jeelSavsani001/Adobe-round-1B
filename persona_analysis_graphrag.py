#!/usr/bin/env python3
import os
import json
from pathlib import Path
import fitz
import spacy
import networkx as nx
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Use a small model for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

nlp = spacy.load("en_core_web_sm")

def embed_text(text):
    """Generate embeddings for text using a transformer model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def chunk_text(text, max_len=300):
    words = text.split()
    return [" ".join(words[i:i + max_len]) for i in range(0, len(words), max_len)]

def extract_sections(pdf_path, chunk_size=300):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            continue
        chunks = chunk_text(text, max_len=chunk_size)
        for chunk in chunks:
            sections.append({
                "document": os.path.basename(pdf_path),
                "page": page_num + 1,
                "text": chunk
            })
    doc.close()
    return sections

def build_graph(sections):
    G = nx.Graph()
    for sec in sections:
        doc_nlp = nlp(sec["text"])
        entities = [ent.text for ent in doc_nlp.ents if ent.label_ not in {"CARDINAL", "ORDINAL"}]
        sec_id = f"{sec['document']}#p{sec['page']}"
        G.add_node(sec_id, text=sec["text"], document=sec["document"], page=sec["page"])
        for ent in entities:
            ent_node = ent.lower().strip()
            G.add_node(ent_node, type="entity")
            G.add_edge(sec_id, ent_node)
    return G

def rank_sections_graphrag(G, persona, job, top_k=10):
    query = f"Persona: {persona}. Task: {job}"
    query_vec = embed_text(query)
    section_scores = []
    for node in G.nodes:
        if G.nodes[node].get("type") == "entity":
            continue
        sec_text = G.nodes[node]["text"]
        sec_vec = embed_text(sec_text)
        score = cosine_similarity(query_vec, sec_vec)
        neighbors = list(G.neighbors(node))
        score += len(neighbors) * 0.01
        section_scores.append({
            "document": G.nodes[node]["document"],
            "page": G.nodes[node]["page"],
            "text": sec_text,
            "importance_score": round(float(score), 4)
        })
    section_scores.sort(key=lambda x: x["importance_score"], reverse=True)
    return section_scores[:top_k]

def load_persona(input_dir):
    persona_path = input_dir / "persona.json"
    if persona_path.exists():
        with open(persona_path, "r", encoding="utf-8") as f:
            persona_data = json.load(f)
        persona = persona_data.get("persona", {}).get("role", "Generic Analyst")
        job = persona_data.get("job_to_be_done", "Summarize key insights")
        return persona, job
    else:
        persona = os.getenv("PERSONA", "Generic Analyst")
        job = os.getenv("JOB", "Summarize key insights")
        return persona, job

def main():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(exist_ok=True)

    persona, job = load_persona(input_dir)

    all_sections = []
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in input directory.")
        return

    print(f"🚀 Found {len(pdf_files)} PDF files. Building GraphRAG index...")
    for pdf in pdf_files:
        all_sections.extend(extract_sections(str(pdf)))

    G = build_graph(all_sections)
    ranked = rank_sections_graphrag(G, persona, job, top_k=10)

    output = {
        "metadata": {
            "persona": persona,
            "job_to_be_done": job,
            "documents": [f.name for f in pdf_files]
        },
        "ranked_sections": ranked
    }

    with open(output_dir / "persona_analysis_graphrag.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Analysis complete. Output saved to persona_analysis_graphrag.json.")

if __name__ == "__main__":
    main()
