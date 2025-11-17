"""
ARCHichat - Invoice Chatbot with RAG System
Flask backend for invoice-based chatbot using Gemini LLM and Qdrant vector database
"""
import os
import json
import base64
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import PyPDF2
from io import BytesIO

# OCR for PDF images
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
    print("✓ Tesseract OCR available")
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None
    Image = None
    pdf2image = None
    print("⚠️  Tesseract OCR not available - install: pip install pytesseract pdf2image pillow")

# Type checking imports
if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

# Gemini LLM
try:
    from google import genai
    LLM_AVAILABLE = True
    GEMINI_API_KEY = "AIzaSyCkpUCP0G_3XGHmAn_l7005GdaTpzadDVY"

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✓ Google Gemini available")
    else:
        gemini_client = None
        print("⚠️  Google Gemini API key not set (set GOOGLE_API_KEY environment variable)")
except ImportError:
    LLM_AVAILABLE = False
    gemini_client = None
    print("✗ ERROR: Google Gemini not available!")

# Qdrant Vector Database
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
    print("✓ Qdrant available")
except ImportError as e:
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    print(f"✗ ERROR: Qdrant not available! Reason: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
PLACEMENT_IMAGE = os.path.join(BASE_DIR, 'placement_image.jpg')
BACKGROUND_IMAGE = os.path.join(BASE_DIR, 'background_image.jpg')
LOGO_IMAGE = os.path.join(BASE_DIR, 'logo_image.jpg')  

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "invoices"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight sentence transformer

# Global variables
qdrant_client: Optional['QdrantClient'] = None
embedding_model: Optional['SentenceTransformer'] = None
current_invoice_context: Optional[Dict] = None  # Stores current invoice context


def init_qdrant():
    """Initialize Qdrant client and collection with named vectors"""
    global qdrant_client, embedding_model
    
    if not QDRANT_AVAILABLE:
        return False
    
    try:
        from qdrant_client.models import VectorParams, NamedVector
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Initialize embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        vector_size = embedding_model.get_sentence_embedding_dimension()
        
        # Check if collection exists, create if not
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        from qdrant_client.models import VectorParams, Distance
        
        if QDRANT_COLLECTION not in collection_names:
            # Create collection with named vectors (number, date, text)
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    "number": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "date": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "text": VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
            print(f"✓ Created Qdrant collection: {QDRANT_COLLECTION} with named vectors (number, date, text)")
        else:
            # Check if collection has named vectors
            try:
                collection_info = qdrant_client.get_collection(QDRANT_COLLECTION)
                # Check the vectors configuration
                vectors_config = collection_info.config.params.vectors
                
                # Check if it's a dict with named vectors (should have 'number', 'date', 'text')
                has_named_vectors = isinstance(vectors_config, dict) and all(
                    key in vectors_config for key in ['number', 'date', 'text']
                )
                
                if not has_named_vectors:
                    print(f"⚠️  Collection exists with old structure, deleting and recreating with named vectors...")
                    qdrant_client.delete_collection(QDRANT_COLLECTION)
                    qdrant_client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config={
                            "number": VectorParams(size=vector_size, distance=Distance.COSINE),
                            "date": VectorParams(size=vector_size, distance=Distance.COSINE),
                            "text": VectorParams(size=vector_size, distance=Distance.COSINE)
                        }
                    )
                    print(f"✓ Recreated Qdrant collection: {QDRANT_COLLECTION} with named vectors (number, date, text)")
                else:
                    print(f"✓ Qdrant collection exists: {QDRANT_COLLECTION} with named vectors")
            except Exception as e:
                # If we can't check (version mismatch or API differences), recreate to ensure correct structure
                # This is safe - we'll reload all invoices anyway
                error_msg = str(e)
                if "validation error" in error_msg.lower() or "pydantic" in error_msg.lower():
                    print(f"⚠️  Qdrant version mismatch detected, recreating collection with correct structure...")
                else:
                    print(f"⚠️  Could not verify collection structure, recreating to ensure correctness...")
                
                try:
                    qdrant_client.delete_collection(QDRANT_COLLECTION)
                except Exception as del_e:
                    # Collection might not exist or already deleted
                    pass
                
                qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config={
                        "number": VectorParams(size=vector_size, distance=Distance.COSINE),
                        "date": VectorParams(size=vector_size, distance=Distance.COSINE),
                        "text": VectorParams(size=vector_size, distance=Distance.COSINE)
                    }
                )
                print(f"✓ Recreated Qdrant collection: {QDRANT_COLLECTION} with named vectors")
        
        return True
    except Exception as e:
        print(f"✗ Error initializing Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file using both text extraction and OCR on images"""
    text_parts = []
    
    try:
        # Step 1: Try to extract text directly from PDF text layer
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pdf_text += page_text + "\n"
            
            if pdf_text.strip():
                text_parts.append(pdf_text.strip())
    except Exception as e:
        print(f"Warning: Error extracting text from PDF text layer {pdf_path}: {e}")
    
    # Step 2: Use OCR on PDF pages (convert to images and OCR)
    if OCR_AVAILABLE and pdf2image and pytesseract:
        try:
            # Convert PDF pages to images
            images = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            ocr_text = ""
            for i, image in enumerate(images):
                # Use OCR on each page image
                page_ocr = pytesseract.image_to_string(image, lang='fra+eng')
                if page_ocr and page_ocr.strip():
                    ocr_text += f"\n--- Page {i+1} (OCR) ---\n{page_ocr}\n"
            
            if ocr_text.strip():
                text_parts.append(ocr_text.strip())
        except Exception as e:
            print(f"Warning: Error during OCR on PDF {pdf_path}: {e}")
            # If pdf2image fails (might need poppler), try alternative method
            try:
                # Alternative: Try to extract images from PDF and OCR them
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        # Try to get images from page
                        if '/XObject' in page.get('/Resources', {}):
                            xObject = page['/Resources']['/XObject'].get_object()
                            for obj in xObject:
                                if xObject[obj]['/Subtype'] == '/Image':
                                    # This is a simplified approach - full implementation would extract and OCR images
                                    pass
            except Exception as e2:
                print(f"Warning: Alternative OCR method also failed: {e2}")
    
    # Combine all extracted text
    combined_text = "\n\n".join(text_parts)
    
    if not combined_text.strip():
        print(f"Warning: No text extracted from {os.path.basename(pdf_path)}")
    
    return combined_text.strip()


def parse_invoice_date(date_str: str) -> Optional[datetime]:
    """Parse invoice date from various formats"""
    if not date_str or not date_str.strip():
        return None
    
    date_str = date_str.strip()
    formats = [
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%Y.%m.%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def calculate_date_periods(invoice_date_str: str) -> Dict[str, str]:
    """Calculate active years, inactive years, and destruction year from invoice date"""
    invoice_date = parse_invoice_date(invoice_date_str)
    
    if not invoice_date:
        return {
            "active_years_start": "",
            "active_years_end": "",
            "inactive_years_start": "",
            "inactive_years_end": "",
            "destruction_year": ""
        }
    
    # Active years: invoice date + 10 years
    active_years_start = invoice_date
    active_years_end = invoice_date + timedelta(days=365 * 10)
    
    # Inactive years: active years end + 5 years (so 15 years total from invoice date)
    inactive_years_start = active_years_end + timedelta(days=1)
    inactive_years_end = invoice_date + timedelta(days=365 * 15)
    
    # Destruction year: after inactive years period (15 years after invoice date)
    destruction_date = invoice_date + timedelta(days=365 * 15 + 1)
    destruction_year = destruction_date.year
    
    return {
        "active_years_start": active_years_start.strftime("%Y-%m-%d"),
        "active_years_end": active_years_end.strftime("%Y-%m-%d"),
        "inactive_years_start": inactive_years_start.strftime("%Y-%m-%d"),
        "inactive_years_end": inactive_years_end.strftime("%Y-%m-%d"),
        "destruction_year": str(destruction_year)
    }


def extract_invoice_metadata(text: str) -> Dict[str, str]:
    """Extract invoice number and date from text using Gemini, then calculate date periods"""
    if not LLM_AVAILABLE or gemini_client is None:
        return {
            "invoice_number": "",
            "invoice_date": "",
            "active_years_start": "",
            "active_years_end": "",
            "inactive_years_start": "",
            "inactive_years_end": "",
            "destruction_year": ""
        }
    
    try:
        schema = {
            "type": "object",
            "properties": {
                "invoice_number": {
                    "type": "string",
                    "description": "Invoice number or identifier found in the text (e.g., '0032017', 'T0039293', 'INV-2024-001')"
                },
                "invoice_date": {
                    "type": "string",
                    "description": "Invoice date found in the text (format: DD/MM/YYYY or YYYY-MM-DD)"
                }
            },
            "required": ["invoice_number", "invoice_date"]
        }
        
        prompt = f"""Extract the invoice number and invoice date from the following text.

Text:
{text[:2000]}

Extract:
1. Invoice number: Look for invoice numbers, facture numbers, or document identifiers (e.g., "0032017", "T0039293", "Facture n°123")
2. Invoice date: Look for dates that appear to be the invoice date (format: DD/MM/YYYY or YYYY-MM-DD)

If not found, return empty string.

Return ONLY valid JSON matching the schema."""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=prompt + "\n\nIMPORTANT: Return ONLY valid JSON matching this schema: " + json.dumps(schema),
        )
        
        # Extract JSON from response
        response_text = None
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            content = response.candidates[0].content
            if hasattr(content, 'parts') and len(content.parts) > 0:
                response_text = content.parts[0].text
            elif hasattr(content, 'text'):
                response_text = content.text
        
        metadata = {
            "invoice_number": "",
            "invoice_date": "",
            "active_years_start": "",
            "active_years_end": "",
            "inactive_years_start": "",
            "inactive_years_end": "",
            "destruction_year": ""
        }
        
        if response_text:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                invoice_number = result.get("invoice_number", "").strip()
                invoice_date = result.get("invoice_date", "").strip()
                
                metadata["invoice_number"] = invoice_number
                metadata["invoice_date"] = invoice_date
                
                # Calculate date periods if we have a date
                if invoice_date:
                    date_periods = calculate_date_periods(invoice_date)
                    metadata.update(date_periods)
        
        return metadata
    except Exception as e:
        print(f"Error extracting invoice metadata: {e}")
        return {
            "invoice_number": "",
            "invoice_date": "",
            "active_years_start": "",
            "active_years_end": "",
            "inactive_years_start": "",
            "inactive_years_end": "",
            "destruction_year": ""
        }


def get_invoice_embedding(text: str) -> List[float]:
    """Generate embedding vector for invoice text"""
    if embedding_model is None:
        return []
    return embedding_model.encode(text).tolist()


def load_invoices_to_qdrant():
    """Load all PDF invoices from Data folder into Qdrant with 3 embeddings per invoice"""
    if not qdrant_client or not embedding_model:
        print("Qdrant not initialized, skipping invoice loading")
        return
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in Data directory")
        return
    
    print(f"Loading {len(pdf_files)} invoices into Qdrant...")
    
    points = []
    for idx, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"Warning: No text extracted from {pdf_file}")
            continue
        
        # Extract invoice number and date using Gemini, then calculate date periods
        print(f"  Extracting metadata from {pdf_file}...")
        metadata = extract_invoice_metadata(text)
        invoice_number = metadata.get("invoice_number", "")
        invoice_date = metadata.get("invoice_date", "")
        
        # If date extraction failed, extract year from filename
        if not invoice_date or not invoice_date.strip():
            import re
            year_match = re.search(r'(19|20)\d{2}', pdf_file)
            if year_match:
                year = year_match.group(0)
                invoice_date = f"01/01/{year}"
                print(f"    ⚠️  Invoice date not found in text, using year from filename: {year}")
        
        # ALWAYS calculate date periods - MUST have invoice_date at this point
        active_years_start = ""
        active_years_end = ""
        inactive_years_start = ""
        inactive_years_end = ""
        destruction_year = ""
        
        if invoice_date and invoice_date.strip():
            date_periods = calculate_date_periods(invoice_date)
            active_years_start = date_periods.get("active_years_start", "")
            active_years_end = date_periods.get("active_years_end", "")
            inactive_years_start = date_periods.get("inactive_years_start", "")
            inactive_years_end = date_periods.get("inactive_years_end", "")
            destruction_year = date_periods.get("destruction_year", "")
            
            # Verify all date periods were calculated
            if not active_years_start or not destruction_year:
                print(f"    ✗ ERROR: Failed to calculate date periods for {pdf_file}")
        else:
            print(f"    ✗ ERROR: No invoice date available for {pdf_file}, cannot calculate date periods")
        
        if invoice_number:
            print(f"    ✓ Invoice number: {invoice_number}")
        if invoice_date:
            print(f"    ✓ Invoice date: {invoice_date}")
            if active_years_start:
                print(f"    ✓ Active years: {active_years_start} to {active_years_end}")
                print(f"    ✓ Inactive years: {inactive_years_start} to {inactive_years_end}")
                print(f"    ✓ Destruction year: {destruction_year}")
        
        # Create embeddings for number, date, and text
        # Number embedding: invoice number (fallback to filename without extension if no number)
        number_text = invoice_number if invoice_number else pdf_file.replace('.pdf', '')
        number_embedding = get_invoice_embedding(number_text)
        
        # Date embedding: invoice date (use empty string embedding if no date)
        date_text = invoice_date if invoice_date else "no date"
        date_embedding = get_invoice_embedding(date_text)
        
        # Text embedding: full extracted text (includes number and date)
        text_embedding = get_invoice_embedding(text)
        
        if not number_embedding or not text_embedding:
            print(f"Warning: Failed to generate embeddings for {pdf_file}")
            continue
        
        # Create point with named vectors
        point = PointStruct(
            id=idx,
            vector={
                "number": number_embedding,
                "date": date_embedding,
                "text": text_embedding
            },
            payload={
                "filename": pdf_file,
                "invoice_number": invoice_number,
                "invoice_date": invoice_date,
                "active_years_start": active_years_start,
                "active_years_end": active_years_end,
                "inactive_years_start": inactive_years_start,
                "inactive_years_end": inactive_years_end,
                "destruction_year": destruction_year,
                "text": text,
                "path": pdf_path
            }
        )
        points.append(point)
    
    if points:
        # Upsert points to Qdrant
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        print(f"✓ Loaded {len(points)} invoices into Qdrant (each with 3 embeddings: number, date, text)")
    else:
        print("No valid invoices to load")


def detect_query_type(user_message: str) -> str:
    """Detect if query is about invoice number, date, or general text"""
    if not LLM_AVAILABLE or gemini_client is None:
        # Fallback: simple heuristic
        user_lower = user_message.lower()
        if any(word in user_lower for word in ['facture', 'invoice', 'numéro', 'number', 'n°', 'no']):
            # Check if contains numbers that look like invoice numbers
            import re
            if re.search(r'\b\d{4,}\b|\b[A-Z]\d{4,}\b', user_message):
                return "number"
        if any(word in user_lower for word in ['date', 'le ', 'du ', 'en ']) and re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', user_message):
            return "date"
        return "text"
    
    try:
        schema = {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["number", "date", "text"],
                    "description": "Type of query: 'number' if asking about invoice number, 'date' if asking about date, 'text' for general questions"
                }
            },
            "required": ["query_type"]
        }
        
        prompt = f"""Analyze the following user message and determine the query type.

User message: "{user_message}"

Return:
- "number" if the user is asking about or mentioning an invoice number (e.g., "invoice 0032017", "facture T0039293", "number 1012017")
- "date" if the user is asking about or mentioning a date (e.g., "invoice from 2024", "facture du 15/03/2017", "dated 2017")
- "text" for all other queries (general questions about invoice content, amounts, items, etc.)

Return ONLY valid JSON matching the schema."""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=prompt + "\n\nIMPORTANT: Return ONLY valid JSON matching this schema: " + json.dumps(schema),
        )
        
        response_text = None
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            content = response.candidates[0].content
            if hasattr(content, 'parts') and len(content.parts) > 0:
                response_text = content.parts[0].text
            elif hasattr(content, 'text'):
                response_text = content.text
        
        if response_text:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return result.get("query_type", "text")
        
        return "text"
    except Exception as e:
        print(f"Error detecting query type: {e}")
        return "text"


def find_matching_invoice(query: str, query_type: str = "text", exclude_id: Optional[int] = None, top_k: int = 5) -> Optional[Dict]:
    """Find matching invoice using cosine similarity with appropriate vector type"""
    if not qdrant_client or not embedding_model:
        return None
    
    try:
        # Generate query embedding
        query_embedding = get_invoice_embedding(query)
        
        if not query_embedding:
            return None
        
        # Determine which vector to search
        vector_name = query_type  # "number", "date", or "text"
        
        # Build query filter to exclude specific ID if needed
        query_filter = None
        if exclude_id is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must_not=[
                    FieldCondition(key="id", match=MatchValue(value=exclude_id))
                ]
            )
        
        # Search in Qdrant with named vector
        # For named vectors, use tuple (vector_name, vector) or NamedVector
        try:
            from qdrant_client.models import NamedVector
            query_vector = NamedVector(name=vector_name, vector=query_embedding)
        except:
            # Fallback to tuple syntax
            query_vector = (vector_name, query_embedding)
        
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )
        
        if search_results and len(search_results) > 0:
            result = search_results[0]
            invoice_date = result.payload.get("invoice_date", "")
            
            # First, try to get stored date periods from Qdrant (preferred)
            active_years_start = result.payload.get("active_years_start", "")
            active_years_end = result.payload.get("active_years_end", "")
            inactive_years_start = result.payload.get("inactive_years_start", "")
            inactive_years_end = result.payload.get("inactive_years_end", "")
            destruction_year = result.payload.get("destruction_year", "")
            
            # Extract ALL date information from Qdrant - NO fallbacks, just get what's stored
            # All date information MUST be stored during loading
            
            return {
                "filename": result.payload.get("filename"),
                "invoice_number": result.payload.get("invoice_number", ""),
                "invoice_date": invoice_date,
                "active_years_start": active_years_start,
                "active_years_end": active_years_end,
                "inactive_years_start": inactive_years_start,
                "inactive_years_end": inactive_years_end,
                "destruction_year": destruction_year,
                "text": result.payload.get("text"),
                "path": result.payload.get("path"),
                "score": result.score,
                "point_id": result.id,
                "query_type": query_type
            }
        
        return None
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_new_invoice_query(user_message: str) -> Tuple[bool, str]:
    """Check if user is asking about a new invoice and detect if they're excluding current one"""
    if not LLM_AVAILABLE or gemini_client is None:
        return False, "text"
    
    try:
        schema = {
            "type": "object",
            "properties": {
                "is_new_invoice_query": {
                    "type": "boolean",
                    "description": "True if the user is asking about a different invoice or mentioning a new invoice, False if continuing conversation about current invoice"
                },
                "is_excluding_current": {
                    "type": "boolean",
                    "description": "True if user is saying the current invoice is wrong/not the one they want (e.g., 'that's not it', 'wrong invoice', 'not that one')"
                }
            },
            "required": ["is_new_invoice_query", "is_excluding_current"]
        }
        
        prompt = f"""Analyze the following user message and determine if they are asking about a NEW or DIFFERENT invoice, or if they are continuing to ask questions about the CURRENT invoice they were already discussing.

User message: "{user_message}"

Return True (is_new_invoice_query: true) if:
- User mentions a different invoice number, date, or identifier
- User asks "what about invoice X" or "tell me about invoice Y"
- User switches topics to a different invoice
- User asks "which invoice" or wants to see invoice locations

Return True (is_excluding_current: true) if:
- User says "that's not the invoice", "wrong invoice", "not that one", "ce n'est pas la bonne facture"
- User is rejecting the current invoice and wants a different one

Return False (is_new_invoice_query: false) if:
- User is asking follow-up questions about the same invoice
- User wants more details about the current invoice
- User is continuing the conversation about the same invoice

Return ONLY valid JSON matching the schema."""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=prompt + "\n\nIMPORTANT: Return ONLY valid JSON matching this schema: " + json.dumps(schema),
        )
        
        # Extract JSON from response
        response_text = None
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            content = response.candidates[0].content
            if hasattr(content, 'parts') and len(content.parts) > 0:
                response_text = content.parts[0].text
            elif hasattr(content, 'text'):
                response_text = content.text
        
        if response_text:
            # Extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                is_excluding = result.get("is_excluding_current", False)
                return result.get("is_new_invoice_query", False), ("exclude" if is_excluding else "new")
        
        return False, "continue"
    except Exception as e:
        print(f"Error checking new invoice query: {e}")
        return False, "continue"


def get_chatbot_response(user_message: str, invoice_context: Optional[Dict] = None) -> str:
    """Generate chatbot response using Gemini with invoice context"""
    if not LLM_AVAILABLE or gemini_client is None:
        return "Sorry, Gemini is not available."
    
    try:
        # Build context prompt
        context_text = ""
        if invoice_context:
            invoice_number = invoice_context.get('invoice_number', '')
            invoice_date = invoice_context.get('invoice_date', '')
            
            # Get ALL date information from invoice_context (stored in Qdrant)
            # NO calculations here - everything must come from Qdrant storage
            active_years_start = invoice_context.get('active_years_start', '')
            active_years_end = invoice_context.get('active_years_end', '')
            inactive_years_start = invoice_context.get('inactive_years_start', '')
            inactive_years_end = invoice_context.get('inactive_years_end', '')
            destruction_year = invoice_context.get('destruction_year', '')
            
            # Always include date information if we have any date periods (stored or calculated)
            # This information is ALWAYS available from Qdrant storage, even if invoice date wasn't found in OCR
            date_info = ""
            if active_years_start or destruction_year:
                date_info = f"""
DATE INFORMATION (ALWAYS AVAILABLE - STORED IN SYSTEM):
- Invoice Date: {invoice_date if invoice_date else 'Extracted from document metadata'}
- Active Years Period: {active_years_start} to {active_years_end} (10 years from invoice date)
- Inactive Years Period: {inactive_years_start} to {inactive_years_end} (years 11-15 from invoice date)
- Destruction Year: {destruction_year} (after 15 years from invoice date)

IMPORTANT: This date information is ALWAYS available in the system, even if the invoice date was not found in the OCR text. You MUST use this information when answering questions about dates, active years, inactive years, or destruction year.
"""
            else:
                # Even if we don't have date periods yet, include a note that they should be available
                date_info = f"""
DATE INFORMATION:
- Invoice Date: {invoice_date if invoice_date else 'Being extracted from document'}
- Note: Date periods (active years, inactive years, destruction year) are calculated and stored in the system for all invoices.
"""
            
            context_text = f"""
CURRENT INVOICE CONTEXT:
Filename: {invoice_context.get('filename', 'Unknown')}
Invoice Number: {invoice_number}
{date_info}
Invoice Content:
{invoice_context.get('text', '')[:2000]}  # Limit to first 2000 chars
"""
            
            # Debug: Print date information being sent to chatbot
            print(f"📅 Date info being sent to chatbot for {invoice_context.get('filename', 'Unknown')}:")
            print(f"   Invoice date: {invoice_date}")
            print(f"   Active years: {active_years_start} to {active_years_end}")
            print(f"   Destruction year: {destruction_year}")
        
        prompt = f"""You are ARCHichat, an AI assistant specialized in answering questions about invoices.

{context_text}

User Question: {user_message}

CRITICAL INSTRUCTIONS:
- The DATE INFORMATION section above contains ALWAYS AVAILABLE date periods that are stored in the system
- If the user asks about destruction year, active years, inactive years, or any date-related question, you MUST use the date information provided in the DATE INFORMATION section
- Even if the invoice date was not found in the OCR text, the date periods (active years, inactive years, destruction year) are ALWAYS calculated and stored in the system
- NEVER say that date information is missing or unavailable - it is ALWAYS provided in the DATE INFORMATION section above
- Use **bold** markdown for important terms
- If the user asks "which invoice" or wants to see the invoice location, mention that you can show them the placement image

Answer in French, using **bold** markdown for emphasis on important information:"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=prompt,
        )
        
        # Extract response text
        response_text = None
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            content = response.candidates[0].content
            if hasattr(content, 'parts') and len(content.parts) > 0:
                response_text = content.parts[0].text
            elif hasattr(content, 'text'):
                response_text = content.text
        
        return response_text if response_text else "I couldn't generate a response. Please try again."
    except Exception as e:
        print(f"Error generating chatbot response: {e}")
        return f"Error: {str(e)}"


@app.route('/')
def index():
    """Serve the main chatbot interface"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with RAG system"""
    global current_invoice_context
    
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Check if user is asking about a new invoice or excluding current one
        is_new_invoice, action = check_new_invoice_query(user_message)
        
        # Handle exclusion: if user says "that's not the invoice", exclude current and search again
        exclude_id = None
        if action == "exclude" and current_invoice_context:
            exclude_id = current_invoice_context.get('point_id')
            print(f"⚠️  User excluded invoice, searching for alternative (excluding ID: {exclude_id})")
            is_new_invoice = True  # Force new search
        
        # If new invoice query, detect query type and find matching invoice
        if is_new_invoice:
            # Detect query type (number, date, or text)
            query_type = detect_query_type(user_message)
            print(f"🔍 Query type detected: {query_type}")
            
            # Find matching invoice with appropriate vector type
            matching_invoice = find_matching_invoice(
                query=user_message,
                query_type=query_type,
                exclude_id=exclude_id
            )
            
            if matching_invoice:
                current_invoice_context = matching_invoice
                print(f"✓ Switched to invoice: {matching_invoice['filename']} (matched by {query_type}, score: {matching_invoice.get('score', 0):.3f})")
            elif exclude_id is not None:
                # If exclusion but no alternative found, clear context
                current_invoice_context = None
                print("⚠️  No alternative invoice found after exclusion")
        
        # Check if user is asking "which invoice" or wants placement image
        show_placement_image = False
        user_lower = user_message.lower()
        if any(phrase in user_lower for phrase in ['which invoice', 'what invoice', 'show invoice', 'placement', 'location', 'where']):
            show_placement_image = True
        
        # Generate response
        response_text = get_chatbot_response(user_message, current_invoice_context)
        
        # Prepare response
        pdf_path = None
        pdf_filename = None
        if current_invoice_context:
            pdf_path = current_invoice_context.get('path')
            pdf_filename = current_invoice_context.get('filename')
        
        response_data = {
            'response': response_text,
            'invoice_context': pdf_filename,
            'show_placement_image': show_placement_image,
            'pdf_path': pdf_path,
            'pdf_filename': pdf_filename
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/background-image')
def get_background_image():
    """Return background image for page"""
    try:
        if os.path.exists(BACKGROUND_IMAGE):
            with open(BACKGROUND_IMAGE, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({
                'image': f'data:image/jpeg;base64,{image_data}'
            })
        else:
            # Fallback to placement_image if background_image doesn't exist
            if os.path.exists(PLACEMENT_IMAGE):
                with open(PLACEMENT_IMAGE, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                return jsonify({
                    'image': f'data:image/jpeg;base64,{image_data}'
                })
            return jsonify({'error': 'Background image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/placement-image')
def get_placement_image():
    """Return placement image for invoice location map"""
    try:
        if os.path.exists(PLACEMENT_IMAGE):
            with open(PLACEMENT_IMAGE, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({
                'image': f'data:image/jpeg;base64,{image_data}'
            })
        else:
            return jsonify({'error': 'Placement image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logo-image')
def get_logo_image():
    """Return logo image"""
    try:
        if os.path.exists(LOGO_IMAGE):
            return send_file(LOGO_IMAGE, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Logo image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/invoice-pdf/<filename>')
def get_invoice_pdf(filename):
    """Return invoice PDF file"""
    try:
        # Security: only allow PDF files from Data directory
        filename = os.path.basename(filename)
        if not filename.endswith('.pdf'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        pdf_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(pdf_path):
            return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("ARCHichat - Invoice Chatbot with RAG System")
    print("=" * 60)
    
    # Initialize Qdrant
    if init_qdrant():
        # Load invoices into Qdrant
        load_invoices_to_qdrant()
    else:
        print("⚠️  Warning: Qdrant not available. RAG features will not work.")
    
    if LLM_AVAILABLE:
        print("✓ LLM: Google Gemini 2.5 Flash Preview")
    else:
        print("✗ ERROR: Google Gemini not available!")
    
    print("=" * 60)
    print("🌐 Server running at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000)

