import spacy
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import itertools
import fitz  # PyMuPDF
import whisper
import os

# Load the local audio transcription model (runs on GPU if available)
print("Loading Audio AI...")
audio_model = whisper.load_model("base")

# Load the local NLP model
nlp = spacy.load("en_core_web_sm")

# We will change MOCK_FIRS from a static list to a dynamic global list
# so uploaded files get added to it.
MOCK_FIRS = []

def process_pdf(file_path):
    """Reads a PDF and extracts the text."""
    text_content = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text_content += page.get_text()
    
    # Add the extracted text to our intelligence database
    MOCK_FIRS.append(text_content)
    return text_content

def process_audio(file_path):
    """Transcribes an audio recording to text."""
    # Whisper automatically handles the heavy AI transcription
    result = audio_model.transcribe(file_path)
    text_content = result["text"]
    
    # Add the transcribed text to our intelligence database
    MOCK_FIRS.append(text_content)
    return text_content

def process_graph():
    G = nx.Graph()
    edges = []
    entity_types = {} # NEW: Tracks if it is a person or a place

    # Extract names and locations
    for text in MOCK_FIRS:
        doc = nlp(text)
        current_entities = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                current_entities.append(ent.text)
                entity_types[ent.text] = "person"
            elif ent.label_ in ["LOC", "FAC", "ORG", "GPE"]:
                current_entities.append(ent.text)
                entity_types[ent.text] = "location"
                
        # Link anyone mentioned in the same report
        for u, v in itertools.combinations(current_entities, 2):
            edges.append((u, v))

    G.add_edges_from(edges)
    
    # Safety check if graph is empty
    if len(G.nodes) == 0:
        return {"summary": {}, "elements": {"nodes": [], "edges": []}}

    # Calculate PageRank (Boss influence) and Communities (Gangs)
    influence = nx.pagerank(G)
    communities = list(greedy_modularity_communities(G))

    gang_map = {}
    for idx, gang in enumerate(communities, start=1):
        for member in gang:
            gang_map[member] = idx

    # Format for the frontend UI (Now includes entity_type)
    nodes = []
    for n in G.nodes():
        nodes.append({
            "data": {
                "id": n, 
                "label": n, 
                "weight": round(influence[n]*100, 2), 
                "gang_id": gang_map.get(n, 1),
                "entity_type": entity_types.get(n, "person") # Tells UI what shape to draw
            }
        })
        
    links = [{"data": {"source": u, "target": v}} for u, v in G.edges()]
    
    # Calculate summary stats
    top_boss = max(influence, key=influence.get)

    return {
        "summary": {
            "total_entities": len(G.nodes),
            "total_connections": len(G.edges),
            "key_influencer": top_boss
        },
        "elements": {
            "nodes": nodes,
            "edges": links
        }
    }