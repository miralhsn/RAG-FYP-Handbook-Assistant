"""
FYP Handbook RAG Pipeline - Streamlit Application
Query interface for asking questions about the FYP Handbook.
"""

import os
import json
import warnings
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# Streamlit watcher configuration to avoid torch.classes errors
os.environ.setdefault("STREAMLIT_SERVER_FILEWATCHER_TYPE", "poll")

# Suppress torch.classes warnings (known Streamlit compatibility issue)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

# Configuration
INDEX_DIR = "faiss_index"
EMBEDDINGS_DIR = "embeddings_data"
SIMILARITY_THRESHOLD = 0.18
TOP_K = 8

# Page configuration
st.set_page_config(
    page_title="FYP Handbook Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #1f77b4;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .page-ref {
        color: #1f77b4;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_index_and_metadata():
    """Load FAISS index and chunk metadata."""
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    metadata_path = os.path.join(EMBEDDINGS_DIR, "chunks_metadata.json")
    
    if not os.path.exists(index_path):
        st.error(f"❌ FAISS index not found at {index_path}! Please run `python ingest.py` first.")
        st.stop()
    
    if not os.path.exists(metadata_path):
        st.error(f"❌ Metadata file not found at {metadata_path}! Please run `python ingest.py` first.")
        st.stop()
    
    try:
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            chunks_metadata = json.load(f)
        
        return index, chunks_metadata
    except Exception as e:
        st.error(f"❌ Error loading index or metadata: {e}")
        st.info("Please run `python ingest.py` to regenerate the index.")
        st.stop()


def retrieve_chunks(query: str, index: faiss.Index, model: SentenceTransformer, 
                   chunks_metadata: List[Dict], top_k: int = TOP_K) -> List[Tuple[Dict, float]]:
    """
    Retrieve top-k chunks for a query.
    Returns list of (chunk_dict, similarity_score) tuples.
    """
    # Embed query
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS index
    similarities, indices = index.search(query_embedding, top_k)
    
    # Get chunks with metadata
    results = []
    for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
        if idx < len(chunks_metadata):
            chunk = chunks_metadata[idx]
            results.append((chunk, float(sim)))
    
    return results


def format_answer(query: str, retrieved_chunks: List[Tuple[Dict, float]]) -> Tuple[str, List[Dict]]:
    """
    Format answer using retrieved chunks with citations.
    Returns (answer_text, sources_list).
    """
    if not retrieved_chunks:
        return "I don't have information about that in the handbook.", []
    
    # Check similarity threshold
    top_similarity = retrieved_chunks[0][1] if retrieved_chunks else 0.0
    if top_similarity < SIMILARITY_THRESHOLD:
        return "I don't have that information in the handbook. Please try rephrasing your question or ask about a different topic.", []
    
    # Build context from retrieved chunks
    context_parts = []
    sources = []
    
    for chunk, similarity in retrieved_chunks:
        page = chunk['page']
        text = chunk['text']
        section = chunk.get('section_hint', 'Unknown')
        
        # Add to context
        context_parts.append(f"[Page {page}] {text}")
        
        # Add to sources
        sources.append({
            'page': page,
            'section': section,
            'text': text[:200] + "..." if len(text) > 200 else text,
            'similarity': similarity
        })
    
    context = "\n\n".join(context_parts)
    
    # Create prompt for answer generation
    prompt = f"""You are a handbook assistant for the FAST-NUCES FYP Handbook. Answer the question using ONLY the information provided in the context below. 

IMPORTANT RULES:
1. Answer ONLY from the context provided. Do not use any external knowledge.
2. Cite page numbers in your answer using format "(p. X)" where X is the page number.
3. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the handbook to answer this question completely."
4. Be concise but complete. Paraphrase the information from the context.
5. If multiple pages contain relevant information, cite all relevant pages.

Question: {query}

Context:
{context}

Answer:"""
    
    # Build page_texts from chunks for better extraction
    page_texts = []
    for chunk, _ in retrieved_chunks:
        page_texts.append((chunk['page'], chunk['text']))
    
    # For now, we'll use a simple extraction-based approach
    # In a production system, you'd use an LLM here
    answer = generate_answer_simple(query, context, sources, page_texts=page_texts)
    
    return answer, sources


def generate_answer_simple(query: str, context: str, sources: List[Dict], page_texts: List[Tuple[int, str]] = None) -> str:
    """
    Enhanced answer generation by extracting and combining relevant information.
    Uses intelligent extraction to create coherent answers with proper citations.
    """
    import re
    
    # Extract query keywords (excluding common words)
    stop_words = {'what', 'are', 'the', 'is', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'how', 'should', 'do', 'does', 'did', 'will', 'can', 'could', 'would', 'we', 'use'}
    query_words = [w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2]
    
    # Identify query intent - what specifically is being asked?
    query_lower = query.lower()
    query_intent = []
    if 'margin' in query_lower:
        query_intent.append('margin')
    if 'spacing' in query_lower or 'space' in query_lower:
        query_intent.append('spacing')
    if 'font' in query_lower:
        query_intent.append('font')
    if 'heading' in query_lower:
        query_intent.append('heading')
    if 'chapter' in query_lower or 'section' in query_lower:
        query_intent.append('structure')
    
    # Parse context into page-text pairs (if not provided)
    if page_texts is None:
        page_texts = []
        for chunk in context.split('\n\n'):
            if '[Page' in chunk:
                match = re.match(r'\[Page (\d+)\]\s*(.+)', chunk, re.DOTALL)
                if match:
                    page_num = int(match.group(1))
                    text = match.group(2).strip()
                    page_texts.append((page_num, text))
    
    # Extract relevant content with better context extraction
    relevant_sentences = []
    relevant_paragraphs = []
    pages_used = set()
    
    # First, identify if text contains headings (to filter them out)
    def is_heading(text):
        """Check if text is likely a heading."""
        text_stripped = text.strip()
        # Headings are usually short, all caps, or start with numbers
        if len(text_stripped) < 5 or len(text_stripped) > 80:
            return True
        if text_stripped.isupper() and len(text_stripped) < 50:
            return True
        if re.match(r'^\d+[\.\)]\s*[A-Z]', text_stripped):
            return True
        if re.match(r'^[A-Z][A-Z\s]{10,}$', text_stripped):
            return True
        return False
    
    for page_num, text in page_texts:
        # Split into sentences (better splitting)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Also extract paragraphs (text between double newlines or longer blocks)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]
        
        for sentence in sentences:
            # Skip headings
            if is_heading(sentence):
                continue
                
            sentence_lower = sentence.lower()
            
            # Calculate base score from keyword matching
            base_score = sum(1 for word in query_words if word in sentence_lower)
            
            # Boost score if sentence contains query intent keywords
            intent_score = 0
            if query_intent:
                for intent in query_intent:
                    if intent in sentence_lower:
                        intent_score += 3  # Higher weight for intent match
            
            # Boost score for longer, more informative sentences
            length_bonus = min(2, len(sentence.split()) // 20)  # Bonus for longer sentences
            
            # Penalize sentences that contain irrelevant keywords when query is specific
            penalty = 0
            if 'margin' in query_lower and 'font' in sentence_lower and 'margin' not in sentence_lower:
                penalty = -3
            if 'spacing' in query_lower and 'font' in sentence_lower and 'spacing' not in sentence_lower and 'line' not in sentence_lower and 'paragraph' not in sentence_lower:
                penalty = -3
            if 'font' in query_lower and 'margin' in sentence_lower and 'font' not in sentence_lower:
                penalty = -3
            
            total_score = base_score + intent_score + length_bonus + penalty
            
            # Include sentences with any positive relevance (lower threshold)
            if total_score > 0:
                relevant_sentences.append({
                    'text': sentence,
                    'page': page_num,
                    'score': total_score
                })
                pages_used.add(page_num)
        
        # Also score paragraphs for better context
        for para in paragraphs:
            if is_heading(para[:50]):  # Check if starts with heading
                continue
                
            para_lower = para.lower()
            para_score = sum(1 for word in query_words if word in para_lower)
            
            if query_intent:
                for intent in query_intent:
                    if intent in para_lower:
                        para_score += 3
            
            if para_score > 0:
                relevant_paragraphs.append({
                    'text': para,
                    'page': page_num,
                    'score': para_score
                })
    
    # Sort by relevance score
    relevant_sentences.sort(key=lambda x: x['score'], reverse=True)
    relevant_paragraphs.sort(key=lambda x: x['score'], reverse=True)
    
    # Extract structured information for specific query types
    if 'margin' in query_lower or 'spacing' in query_lower:
        # Try to extract structured margin/spacing information
        # Also pass full page texts for better extraction
        structured_info = extract_margin_spacing_info(relevant_sentences, query_lower, page_texts)
        if structured_info:
            return structured_info
    
    # Extract chapter lists for Development/R&D FYP reports
    if 'chapter' in query_lower and ('development' in query_lower or 'r&d' in query_lower or 'rd' in query_lower):
        structured_info = extract_chapter_list(query_lower, page_texts, sources)
        if structured_info:
            return structured_info
    
    # Extract endnotes usage information
    if 'endnote' in query_lower or ('ibid' in query_lower and 'op. cit' in query_lower) or ('ibid' in query_lower and 'op cit' in query_lower):
        structured_info = extract_endnotes_info(relevant_sentences, page_texts)
        if structured_info:
            return structured_info
    
    # Extract Executive Summary and Abstract information
    if ('executive summary' in query_lower or 'abstract' in query_lower) and ('and' in query_lower or 'goes into' in query_lower):
        structured_info = extract_summary_abstract_info(relevant_sentences, page_texts)
        if structured_info:
            return structured_info
    
    # Build answer from top sentences and paragraphs
    answer_parts = []
    seen_texts = set()
    pages_in_answer = set()
    
    # Prefer paragraphs for more complete context, but fall back to sentences
    if relevant_paragraphs:
        # Use top paragraphs first (they have more context)
        for para_item in relevant_paragraphs[:3]:  # Top 3 paragraphs
            para_text = para_item['text']
            # Extract the most relevant part of the paragraph
            para_lower = para_text.lower()
            
            # Find sentences within paragraph that match query
            para_sentences = re.split(r'[.!?]+', para_text)
            relevant_para_sentences = []
            
            for sent in para_sentences:
                sent = sent.strip()
                if len(sent) > 15 and not is_heading(sent):
                    sent_lower = sent.lower()
                    if any(word in sent_lower for word in query_words):
                        relevant_para_sentences.append(sent)
            
            # Use relevant sentences from paragraph, or whole paragraph if short
            if relevant_para_sentences:
                para_content = " ".join(relevant_para_sentences[:3])
            else:
                para_content = para_text[:300]  # Limit length
            
            text_key = para_content[:60]
            if text_key not in seen_texts and len(para_content) > 20:
                answer_parts.append(f"{para_content} (p. {para_item['page']})")
                seen_texts.add(text_key)
                pages_in_answer.add(para_item['page'])
    
    # Add top sentences if we don't have enough content
    if len(answer_parts) < 3 and relevant_sentences:
        # Use sentences with score >= 1 (lower threshold for more content)
        filtered_sentences = [s for s in relevant_sentences if s['score'] >= 1]
        
        for item in filtered_sentences[:8]:  # Top 8 sentences
            text_key = item['text'][:60]
            if text_key not in seen_texts and not is_heading(item['text']):
                clean_text = re.sub(r'\s+', ' ', item['text']).strip()
                if len(clean_text) > 20:  # Ensure meaningful length
                    answer_parts.append(f"{clean_text} (p. {item['page']})")
                    seen_texts.add(text_key)
                    pages_in_answer.add(item['page'])
    
    # Fallback: use source chunks directly if no good sentences found
    if not answer_parts:
        for source in sources[:3]:
            text = source['text']
            # Remove headings from source text
            sentences = re.split(r'[.!?]+', text)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and not is_heading(s.strip())]
            
            if meaningful_sentences:
                # Use first 2-3 meaningful sentences
                content = " ".join(meaningful_sentences[:3])
                if len(content) > 30:
                    answer_parts.append(f"{content} (p. {source['page']})")
                    pages_in_answer.add(source['page'])
            else:
                # Last resort: use first part of text
                content = text[:250].strip()
                if len(content) > 30:
                    answer_parts.append(f"{content}... (p. {source['page']})")
                    pages_in_answer.add(source['page'])
    
    if answer_parts:
        answer = " ".join(answer_parts)
        pages_used = pages_in_answer
    else:
        return "I found some information, but couldn't extract a clear answer. Please check the sources below."
    
    # Clean up answer
    answer = re.sub(r'\s+', ' ', answer)  # Remove extra spaces
    answer = answer.strip()
    
    # Add page summary at the end if multiple pages
    if len(pages_used) > 1:
        page_list = sorted(list(pages_used))
        answer += f"\n\n[Referenced pages: {', '.join(f'p. {p}' for p in page_list)}]"
    elif len(pages_used) == 1:
        answer += f"\n\n[Reference: p. {list(pages_used)[0]}]"
    
    return answer


def extract_margin_spacing_info(relevant_sentences: List[Dict], query_lower: str, page_texts: List[Tuple[int, str]] = None) -> str:
    """
    Extract structured margin and spacing information from sentences and full page texts.
    Returns formatted answer or None if not found.
    """
    import re
    
    margins = {}
    spacing_info = {}
    pages_mentioned = set()
    
    # First, search through all relevant sentences
    for item in relevant_sentences:
        text = item['text']
        text_lower = text.lower()
        page = item['page']
        pages_mentioned.add(page)
        
        top_match = re.search(r'\btop\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
        if top_match and 'top' not in margins:
            margins['top'] = top_match.group(1)
        
        bottom_match = re.search(r'\bbottom\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
        if bottom_match and 'bottom' not in margins:
            margins['bottom'] = bottom_match.group(1)
        
        left_match = re.search(r'\bleft\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
        if left_match and 'left' not in margins:
            margins['left'] = left_match.group(1)
        
        right_match = re.search(r'\bright\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
        if right_match and 'right' not in margins:
            margins['right'] = right_match.group(1)
        
        line_match = re.search(r'line\s+spacing\s*[=:]\s*([\d.]+)', text_lower, re.IGNORECASE)
        if line_match and 'line' not in spacing_info:
            spacing_info['line'] = line_match.group(1)
        
        para_match = re.search(r'paragraph\s+spacing\s*[=:]\s*([\d.]+)\s*(?:pt|points?|pts?)', text_lower, re.IGNORECASE)
        if para_match and 'paragraph' not in spacing_info:
            spacing_info['paragraph'] = para_match.group(1)
    
    # Also search in full page texts for better extraction (handles tables/formatted text)
    if page_texts:
        for page_num, full_text in page_texts:
            if page_num not in pages_mentioned:
                continue  # Only search pages we already found relevant info on
            
            text_lower = full_text.lower()
            
            # Extract margins from full text
            if 'top' not in margins:
                top_match = re.search(r'\btop\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
                if top_match:
                    margins['top'] = top_match.group(1)
            
            if 'bottom' not in margins:
                bottom_match = re.search(r'\bbottom\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
                if bottom_match:
                    margins['bottom'] = bottom_match.group(1)
            
            if 'left' not in margins:
                left_match = re.search(r'\bleft\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
                if left_match:
                    margins['left'] = left_match.group(1)
            
            if 'right' not in margins:
                right_match = re.search(r'\bright\s*[=:]\s*([\d.]+)\s*["\']?', text_lower, re.IGNORECASE)
                if right_match:
                    margins['right'] = right_match.group(1)
            
            # Extract spacing from full text
            if 'line' not in spacing_info:
                line_match = re.search(r'line\s+spacing\s*[=:]\s*([\d.]+)', text_lower, re.IGNORECASE)
                if line_match:
                    spacing_info['line'] = line_match.group(1)
            
            if 'paragraph' not in spacing_info:
                para_match = re.search(r'paragraph\s+spacing\s*[=:]\s*([\d.]+)\s*(?:pt|points?|pts?)', text_lower, re.IGNORECASE)
                if para_match:
                    spacing_info['paragraph'] = para_match.group(1)
    
    # Build structured answer
    answer_parts = []
    
    if margins or spacing_info:
        if margins:
            margin_str = []
            if 'top' in margins:
                margin_str.append(f"Top {margins['top']}\"")
            if 'bottom' in margins:
                margin_str.append(f"Bottom {margins['bottom']}\"")
            if 'left' in margins:
                margin_str.append(f"Left {margins['left']}\"")
            if 'right' in margins:
                margin_str.append(f"Right {margins['right']}\"")
            
            if margin_str:
                answer_parts.append(", ".join(margin_str) + ";")
        
        if spacing_info:
            spacing_str = []
            if 'line' in spacing_info:
                spacing_str.append(f"line spacing {spacing_info['line']}")
            if 'paragraph' in spacing_info:
                spacing_str.append(f"paragraph spacing {spacing_info['paragraph']} pt")
            
            if spacing_str:
                answer_parts.append("; ".join(spacing_str) + ".")
        
        if answer_parts:
            answer = " ".join(answer_parts)
            if pages_mentioned:
                page_list = sorted(list(pages_mentioned))
                answer += f" (p. {', '.join(map(str, page_list))})"
            return answer
    
    return None


def extract_chapter_list(query_lower: str, page_texts: List[Tuple[int, str]], sources: List[Dict]) -> str:
    """
    Extract chapter list for Development or R&D FYP reports.
    Returns formatted answer or None if not found.
    """
    import re
    
    is_development = 'development' in query_lower
    is_rd = 'r&d' in query_lower or 'rd' in query_lower or 'research' in query_lower
    
    pages_mentioned = set()
    chapters = []
    
    # Search through all page texts and sources
    all_texts = list(page_texts)
    for source in sources:
        all_texts.append((source['page'], source['text']))
    
    for page_num, full_text in all_texts:
        text_lower = full_text.lower()
        
        # Development FYP Report Format
        if is_development:
            if 'development fyp report format' in text_lower:
                pages_mentioned.add(page_num)
                
                # Extract everything after "Development FYP Report Format" until "R&D-Based" or end
                dev_match = re.search(
                    r'development\s+fyp\s+report\s+format\s*(.+?)(?:\s*r&d-based\s+fyp\s+report\s+format|\s*fast-nuces\s+\d+|$)',
                    full_text,
                    re.DOTALL | re.IGNORECASE
                )
                
                if dev_match:
                    format_section = dev_match.group(1)
                    lines = format_section.split('\n')
                    current_chapter = None
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check if line starts with a number followed by period
                        match = re.match(r'^(\d+)\.\s+(.+)$', line, re.IGNORECASE)
                        if match:
                            # Save previous chapter if exists
                            if current_chapter:
                                chapter_name = current_chapter['name'].strip()
                                chapter_name = re.sub(r'\s+', ' ', chapter_name)
                                chapter_name = re.sub(r'\s+\d+\s*$', '', chapter_name)
                                if len(chapter_name) > 3 and len(chapter_name) < 100:
                                    if not any(word in chapter_name.lower() for word in ['fast-nuces', 'bs final year', 'handbook', 'fyp title']):
                                        chapters.append(f"{current_chapter['num']}. {chapter_name}")
                            
                            chapter_num = match.group(1)
                            chapter_name = match.group(2)
                            current_chapter = {'num': chapter_num, 'name': chapter_name}
                        elif current_chapter and not re.match(r'^\d+\.', line):
                            # Continue current chapter name (multi-line)
                            current_chapter['name'] += ' ' + line
                    
                    # Add last chapter
                    if current_chapter:
                        chapter_name = current_chapter['name'].strip()
                        chapter_name = re.sub(r'\s+', ' ', chapter_name)
                        chapter_name = re.sub(r'\s+\d+\s*$', '', chapter_name)
                        if len(chapter_name) > 3 and len(chapter_name) < 100:
                            if not any(word in chapter_name.lower() for word in ['fast-nuces', 'bs final year', 'handbook', 'fyp title']):
                                chapters.append(f"{current_chapter['num']}. {chapter_name}")
                    
                    if not chapters:
                        # Try to find all "N. Chapter Name" patterns in the text
                        all_matches = re.finditer(
                            r'\b(\d+)\.\s+([A-Z][^0-9\n]{5,80}?)(?=\s+\d+\.|$)',
                            format_section,
                            re.IGNORECASE | re.MULTILINE
                        )
                        
                        for match in all_matches:
                            chapter_num = match.group(1)
                            chapter_name = match.group(2).strip()
                            chapter_name = re.sub(r'\s+', ' ', chapter_name)
                            # Filter out sub-sections (like 3.1, 3.2, etc.)
                            if '.' not in chapter_num or int(chapter_num.split('.')[0]) == int(chapter_num):
                                if len(chapter_name) > 5 and len(chapter_name) < 100:
                                    if not any(word in chapter_name.lower() for word in ['fast-nuces', 'bs final year', 'handbook', 'fyp title', 'problem statement', 'business opportunity']):
                                        chapters.append(f"{chapter_num}. {chapter_name}")
                    
                    # Also look for specific chapter names if numbered extraction didn't work
                    if not chapters:
                        dev_chapter_keywords = [
                            (r'\b1\.\s*introduction\b', '1. Introduction'),
                            (r'\b2\.\s*research\s+on\s+existing\s+products\b', '2. Research on existing products'),
                            (r'\b3\.\s*project\s+vision\b', '3. Project Vision'),
                            (r'\b4\.\s*software\s+requirement\s+specifications\b', '4. Software Requirement Specifications'),
                            (r'\b5\.\s*iteration\s+plan\b', '5. Iteration Plan'),
                            (r'\b8\.\s*implementation\s+details\b', '8. Implementation Details'),
                            (r'\b9\.\s*user\s+manual\b', '9. User Manual'),
                            (r'\breferences\b', 'References'),
                            (r'\bappendices\b', 'Appendices')
                        ]
                        
                        for pattern, chapter_name in dev_chapter_keywords:
                            if re.search(pattern, format_section, re.IGNORECASE):
                                if chapter_name not in chapters:
                                    chapters.append(chapter_name)
        
        if is_rd:
            if 'r&d-based fyp report format' in text_lower or 'r&d fyp report format' in text_lower:
                pages_mentioned.add(page_num)
                
                rd_match = re.search(
                    r'r&d-based\s+fyp\s+report\s+format\s*(.+?)(?:\s*fast-nuces\s+\d+|$)',
                    full_text,
                    re.DOTALL | re.IGNORECASE
                )
                
                if rd_match:
                    format_section = rd_match.group(1)
                    
                    # Extract chapters: "Chapter 1. Introduction", "Chapter 2. Literature Review", etc.
                    # Handle both "Chapter X." format and standalone chapter names
                    lines = format_section.split('\n')
                    current_chapter = None
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        match = re.match(r'^chapter\s+(\d+)\.\s+(.+)$', line, re.IGNORECASE)
                        if match:
                            # Save previous chapter if exists
                            if current_chapter:
                                chapter_name = current_chapter['name'].strip()
                                chapter_name = re.sub(r'\s+', ' ', chapter_name)
                                chapter_name = re.sub(r'\s+\d+\s*$', '', chapter_name)
                                if len(chapter_name) > 3 and len(chapter_name) < 100:
                                    if not any(word in chapter_name.lower() for word in ['fast-nuces', 'bs final year', 'handbook']):
                                        chapters.append(f"Chapter {current_chapter['num']}. {chapter_name}")
                            
                            # Start new chapter
                            chapter_num = match.group(1)
                            chapter_name = match.group(2)
                            current_chapter = {'num': chapter_num, 'name': chapter_name}
                        elif current_chapter and not re.match(r'^chapter\s+\d+\.', line, re.IGNORECASE):
                            # Continue current chapter name (multi-line)
                            current_chapter['name'] += ' ' + line
                    
                    # Add last chapter
                    if current_chapter:
                        chapter_name = current_chapter['name'].strip()
                        chapter_name = re.sub(r'\s+', ' ', chapter_name)
                        chapter_name = re.sub(r'\s+\d+\s*$', '', chapter_name)
                        if len(chapter_name) > 3 and len(chapter_name) < 100:
                            if not any(word in chapter_name.lower() for word in ['fast-nuces', 'bs final year', 'handbook']):
                                chapters.append(f"Chapter {current_chapter['num']}. {chapter_name}")
                    
                    # Fallback: if line-by-line extraction didn't work, try pattern matching
                    if not chapters:
                        # Try to find all "Chapter N. Chapter Name" patterns
                        all_matches = re.finditer(
                            r'chapter\s+(\d+)\.\s+([A-Z][^0-9\n]{5,80}?)(?=\s+chapter\s+\d+\.|$)',
                            format_section,
                            re.IGNORECASE | re.MULTILINE
                        )
                        
                        for match in all_matches:
                            chapter_num = match.group(1)
                            chapter_name = match.group(2).strip()
                            chapter_name = re.sub(r'\s+', ' ', chapter_name)
                            if len(chapter_name) > 5 and len(chapter_name) < 100:
                                if not any(word in chapter_name.lower() for word in ['fast-nuces', 'bs final year', 'handbook']):
                                    chapters.append(f"Chapter {chapter_num}. {chapter_name}")
                    
                    # Also extract standalone chapter names if pattern matching didn't work
                    if not chapters or len(chapters) < 3:
                        rd_chapter_keywords = [
                            (r'chapter\s+1[\.:]?\s*introduction', 'Chapter 1. Introduction'),
                            (r'chapter\s+2[\.:]?\s*literature\s+review', 'Chapter 2. Literature Review'),
                            (r'proposed\s+approach', 'Chapter 3. Proposed Approach'),
                            (r'\bimplementation\b(?!\s+details)', 'Chapter 4. Implementation'),
                            (r'validation\s+and\s+testing', 'Chapter 5. Validation and Testing'),
                            (r'results\s+and\s+discussion', 'Chapter 6. Results and Discussion'),
                            (r'conclusions\s+and\s+future\s+work', 'Chapter 7. Conclusions and Future Work'),
                            (r'\breferences\b', 'References'),
                            (r'\bappendices\b', 'Appendices')
                        ]
                        
                        for pattern, chapter_name in rd_chapter_keywords:
                            if re.search(pattern, format_section, re.IGNORECASE):
                                if chapter_name not in chapters:
                                    chapters.append(chapter_name)
    
    # Build answer
    if chapters:
        # Remove duplicates while preserving order
        seen = set()
        unique_chapters = []
        for ch in chapters:
            # Use first 40 chars as key to catch similar entries
            ch_key = re.sub(r'[^\w\s]', '', ch.lower())[:40]
            if ch_key not in seen:
                seen.add(ch_key)
                unique_chapters.append(ch)
        
        # Format as comma-separated list
        chapter_list = ", ".join(unique_chapters)
        
        answer = chapter_list
        if pages_mentioned:
            page_list = sorted(list(pages_mentioned))
            answer += f" (p. {', '.join(map(str, page_list))})"
        
        return answer
    
    return None


def extract_endnotes_info(relevant_sentences: List[Dict], page_texts: List[Tuple[int, str]]) -> str:
    """
    Extract information about endnotes usage, specifically Ibid. and op. cit.
    Returns formatted answer or None if not found.
    """
    import re
    
    pages_mentioned = set()
    endnote_info = []
    ibid_info = []
    opcit_info = []
    
    # Search for endnotes information
    for item in relevant_sentences:
        text = item['text']
        text_lower = text.lower()
        page = item['page']
        
        if 'endnote' in text_lower or 'footnote' in text_lower:
            pages_mentioned.add(page)
            if 'ibid' in text_lower:
                ibid_info.append(text)
            if 'op. cit' in text_lower or 'op cit' in text_lower:
                opcit_info.append(text)
            if 'endnote' in text_lower:
                endnote_info.append(text)
    
    # Also search full page texts
    for page_num, full_text in page_texts:
        text_lower = full_text.lower()
        
        if 'endnote' in text_lower or ('ibid' in text_lower and 'op. cit' in text_lower):
            pages_mentioned.add(page_num)
            
            # Extract sentences about Ibid.
            if 'ibid' in text_lower:
                sentences = re.split(r'[.!?]+', full_text)
                for sent in sentences:
                    if 'ibid' in sent.lower() and len(sent) > 20:
                        if sent.strip() not in ibid_info:
                            ibid_info.append(sent.strip())
            
            # Extract sentences about op. cit.
            if 'op. cit' in text_lower or 'op cit' in text_lower:
                sentences = re.split(r'[.!?]+', full_text)
                for sent in sentences:
                    if ('op. cit' in sent.lower() or 'op cit' in sent.lower()) and len(sent) > 20:
                        if sent.strip() not in opcit_info:
                            opcit_info.append(sent.strip())
    
    # Build answer
    answer_parts = []
    
    if endnote_info:
        # Use first endnote description
        answer_parts.append(endnote_info[0])
    
    if ibid_info:
        # Combine Ibid. information
        ibid_text = " ".join(ibid_info[:2])  # Use first 2 mentions
        answer_parts.append(ibid_text)
    
    if opcit_info:
        # Combine op. cit. information
        opcit_text = " ".join(opcit_info[:2])  # Use first 2 mentions
        answer_parts.append(opcit_text)
    
    if answer_parts:
        answer = ". ".join(answer_parts)
        if pages_mentioned:
            page_list = sorted(list(pages_mentioned))
            answer += f" (p. {', '.join(map(str, page_list))})"
        return answer
    
    return None


def extract_summary_abstract_info(relevant_sentences: List[Dict], page_texts: List[Tuple[int, str]]) -> str:
    """
    Extract information about Executive Summary and Abstract.
    Returns formatted answer or None if not found.
    """
    import re
    
    pages_mentioned = set()
    abstract_info = []
    summary_info = []
    
    # Search for Abstract and Executive Summary information
    for item in relevant_sentences:
        text = item['text']
        text_lower = text.lower()
        page = item['page']
        
        if 'abstract' in text_lower:
            abstract_info.append(text)
            pages_mentioned.add(page)
        
        if 'executive summary' in text_lower:
            summary_info.append(text)
            pages_mentioned.add(page)
    
    # Also search full page texts for complete information
    for page_num, full_text in page_texts:
        text_lower = full_text.lower()
        
        if 'abstract' in text_lower or 'executive summary' in text_lower:
            pages_mentioned.add(page_num)
            
            # Extract Abstract information
            if 'abstract' in text_lower:
                # Look for word count, length, or description
                abstract_sentences = re.split(r'[.!?]+', full_text)
                for sent in abstract_sentences:
                    sent_lower = sent.lower()
                    if 'abstract' in sent_lower and ('word' in sent_lower or '50' in sent or '125' in sent or 'purpose' in sent_lower or 'finding' in sent_lower):
                        if sent.strip() not in abstract_info:
                            abstract_info.append(sent.strip())
            
            # Extract Executive Summary information
            if 'executive summary' in text_lower:
                summary_sentences = re.split(r'[.!?]+', full_text)
                for sent in summary_sentences:
                    sent_lower = sent.lower()
                    if 'executive summary' in sent_lower and ('page' in sent_lower or 'overview' in sent_lower or 'presentation' in sent_lower):
                        if sent.strip() not in summary_info:
                            summary_info.append(sent.strip())
    
    # Build structured answer
    answer_parts = []
    
    if abstract_info:
        abstract_text = " ".join(abstract_info[:2])  # Combine first 2 mentions
        answer_parts.append(f"Abstract: {abstract_text}")
    
    if summary_info:
        summary_text = " ".join(summary_info[:2])  # Combine first 2 mentions
        answer_parts.append(f"Executive Summary: {summary_text}")
    
    if answer_parts:
        answer = " ".join(answer_parts)
        if pages_mentioned:
            page_list = sorted(list(pages_mentioned))
            answer += f" (p. {', '.join(map(str, page_list))})"
        return answer
    
    return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📚 FYP Handbook Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about the FAST-NUCES Final Year Project Handbook 2023</div>', unsafe_allow_html=True)
    
    # Load resources
    try:
        model = load_model()
        index, chunks_metadata = load_index_and_metadata()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.info("Please run `python ingest.py` first to create the index.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This assistant answers questions about the FYP Handbook using RAG (Retrieval-Augmented Generation).")
        st.write("**Features:**")
        st.write("- 📄 Page references for all answers")
        st.write("- 🔍 Top-5 relevant chunks retrieval")
        st.write("- ✅ Similarity threshold filtering")
        
        st.header("📝 Example Questions")
        example_questions = [
            "What headings, fonts, and sizes are required in the FYP report?",
            "What margins and spacing do we use?",
            "What are the required chapters of a Development FYP report?",
            "What are the required chapters of an R&D-based FYP report?",
            "How should endnotes like 'Ibid.' and 'op. cit.' be used?",
            "What goes into the Executive Summary and Abstract?"
        ]
        for i, q in enumerate(example_questions, 1):
            if st.button(f"Q{i}: {q[:50]}...", key=f"example_{i}"):
                st.session_state.query = q
    
    # Main query interface
    query = st.text_input(
        "💬 Ask a question about the FYP Handbook:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., What are the required margins for the FYP report?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Process query
    if ask_button and query:
        with st.spinner("Searching the handbook..."):
            # Retrieve chunks
            retrieved_chunks = retrieve_chunks(query, index, model, chunks_metadata, TOP_K)
            
            # Generate answer
            answer, sources = format_answer(query, retrieved_chunks)
            
            # Display answer
            st.markdown("### 📋 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Display sources (collapsible)
            with st.expander("📚 Sources (Page References)", expanded=False):
                if sources:
                    st.write(f"**Found {len(sources)} relevant chunks:**")
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}</strong> - <span class="page-ref">Page {source['page']}</span><br>
                            <em>Section: {source['section']}</em><br>
                            <small>Similarity: {source['similarity']:.3f}</small><br>
                            <p>{source['text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No sources found.")
            
            # Debug: Show retrieved chunks
            with st.expander("🔧 Debug: Retrieved Chunks", expanded=False):
                st.write(f"**Top {len(retrieved_chunks)} chunks retrieved:**")
                for i, (chunk, similarity) in enumerate(retrieved_chunks, 1):
                    st.write(f"**Chunk {i}** (Page {chunk['page']}, Similarity: {similarity:.3f})")
                    st.text_area(f"Text {i}", chunk['text'], height=100, key=f"chunk_{i}")
    
    elif ask_button and not query:
        st.warning("Please enter a question first.")
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>FYP Handbook RAG Pipeline | Built with Streamlit, FAISS, and Sentence-BERT</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()