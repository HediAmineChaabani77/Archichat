# ARCHichat - Invoice Assistant Chatbot

ARCHichat is an intelligent invoice management chatbot powered by RAG (Retrieval Augmented Generation) technology. It uses Google Gemini for natural language understanding, Qdrant vector database for semantic search, and Tesseract OCR for extracting text from PDF invoices.

## 📖 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Development](#-development)
- [API Endpoints](#-api-endpoints)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)

## 🚀 Features

- **Intelligent Invoice Matching**: Uses 3 types of embeddings (number, date, text) for precise invoice matching
- **OCR Text Extraction**: Extracts text from PDF invoices using both text layer and OCR on images
- **Date Period Management**: Automatically calculates active years, inactive years, and destruction year for each invoice
- **Multi-vector Search**: Matches invoices by number, date, or content using cosine similarity
- **Context-Aware Responses**: Maintains conversation context and can switch between invoices
- **PDF Download**: Provides invoice PDFs for download in chat
- **Placement Image**: Shows invoice location map when requested

## 📋 Prerequisites

- Docker and Docker Compose
- Google Gemini API key
- PDF invoice files in the `Data/` directory

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Archichat
```

### 2. Set Up Google Gemini API Key

Set the `GOOGLE_API_KEY` environment variable before running Docker Compose:

**Option 1: Set in terminal (Linux/Mac/WSL)**
```bash
export GOOGLE_API_KEY=your_gemini_api_key_here
docker-compose up -d
```

**Option 2: Set inline (Linux/Mac/WSL)**
```bash
GOOGLE_API_KEY=your_gemini_api_key_here docker-compose up -d
```

**Option 3: Set in PowerShell (Windows)**
```powershell
$env:GOOGLE_API_KEY="your_gemini_api_key_here"
docker-compose up -d
```

**Option 4: Edit docker-compose.yml directly**
Add your API key to the `environment` section in `docker-compose.yml`:
```yaml
environment:
  - GOOGLE_API_KEY=your_gemini_api_key_here
```

**Important**: Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. Add Invoice PDFs

Place your invoice PDF files in the `Data/` directory:

```
Data/
  ├── invoice1.pdf
  ├── invoice2.pdf
  └── ...
```

The system will automatically:
- Extract text from PDFs using OCR
- Extract invoice number and date
- Calculate date periods (active, inactive, destruction year)
- Create embeddings and store in Qdrant

### 4. Add Required Images

Ensure you have the following image files in the root directory:
- `logo_image .jpg` - Logo for the chatbot interface (note: filename has a space)
- `background_image.jpg` - Background image for the web interface
- `placement_image.jpg` - Invoice location map image

## 🐳 Docker Deployment

### Quick Start

1. **Set your Google Gemini API key** (see Installation step 2)

2. **Build and run with Docker Compose**:

```bash
docker-compose up -d
```

This will:
- Start Qdrant vector database container
- Build the ARCHichat Docker image (includes Tesseract OCR and all dependencies)
- Start the Flask application container
- Automatically load all invoices from `Data/` into Qdrant with embeddings

3. **Wait for initialization** (check logs):

```bash
docker-compose logs -f archichat
```

You should see:
```
✓ Qdrant available
✓ Created Qdrant collection: invoices with named vectors
Loading X invoices into Qdrant...
✓ Loaded X invoices into Qdrant
🌐 Server running at: http://localhost:5000
```

4. **Access the Application**:

Open your browser and navigate to:
```
http://localhost:5000
```

### View Logs

```bash
# View all logs
docker-compose logs -f

# View only ARCHichat logs
docker-compose logs -f archichat

# View only Qdrant logs
docker-compose logs -f qdrant
```

### Stop the Application

```bash
docker-compose down
```

### Rebuild After Changes

```bash
docker-compose up -d --build
```

## 📁 Project Structure

```
Archichat/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image configuration
├── docker-compose.yml    # Docker Compose configuration
├── .dockerignore         # Files to exclude from Docker build
├── .gitignore           # Git ignore rules
├── Data/                 # Invoice PDF files directory
│   ├── invoice1.pdf
│   └── ...
├── templates/            # HTML templates
│   └── index.html        # Main chatbot interface
├── static/               # Static files (CSS, JS)
│   ├── style.css         # Styling
│   └── script.js         # Frontend JavaScript
├── logo_image .jpg       # Chatbot logo (note: filename has a space)
├── background_image.jpg   # Background image
├── placement_image.jpg   # Invoice placement map
└── README.md             # This file
```

## 🔧 Configuration

### Qdrant Vector Database

The application uses Qdrant with named vectors:
- **number**: Embedding of invoice number
- **date**: Embedding of invoice date
- **text**: Embedding of full invoice text

### Invoice Metadata

For each invoice, the system stores:
- Invoice number
- Invoice date
- Active years period (10 years from invoice date)
- Inactive years period (years 11-15 from invoice date)
- Destruction year (after 15 years)
- Full extracted text
- File path

### Embedding Model

Uses `all-MiniLM-L6-v2` from Sentence Transformers (384 dimensions).

## 💬 Usage

### Chat Interface

1. Open the web interface at `http://localhost:5000`
2. Type your question about an invoice
3. The chatbot will:
   - Match your query to the appropriate invoice
   - Extract relevant information
   - Provide answers with date periods and invoice details

### Example Queries

- "What is the total amount of invoice 133?"
- "Show me invoice from 2017"
- "What is the destruction year for invoice 0032017?"
- "What are the active years for this invoice?"
- "Download the invoice PDF"

### Invoice Matching

The system automatically detects query type:
- **Number queries**: "invoice 133", "facture 0032017"
- **Date queries**: "invoice from 2017", "facture du 15/03/2017"
- **Content queries**: "what is the total amount", "show me the items"

## 🔍 How It Works

1. **Invoice Loading**:
   - Extracts text from PDF using PyPDF2 and Tesseract OCR
   - Extracts invoice number and date using Gemini
   - Calculates date periods (active, inactive, destruction year)
   - Creates 3 embeddings (number, date, text)
   - Stores everything in Qdrant

2. **Query Processing**:
   - Detects query type (number/date/text)
   - Generates query embedding
   - Searches Qdrant using appropriate vector type
   - Retrieves matching invoice with all metadata

3. **Response Generation**:
   - Uses Gemini with invoice context
   - Includes date information in context
   - Formats response with markdown
   - Provides PDF download link

## 🧪 Development

### Local Development (without Docker)

1. Install Python 3.12+
2. Install system dependencies:
   ```bash
   sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng poppler-utils
   ```
3. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start Qdrant:
   ```bash
   docker-compose up -d qdrant
   ```
6. Run the application:
   ```bash
   python app.py
   ```

## 📝 API Endpoints

- `GET /` - Main chatbot interface
- `POST /api/chat` - Send chat message
- `GET /api/logo-image` - Get logo image
- `GET /api/background-image` - Get background image
- `GET /api/placement-image` - Get placement image
- `GET /api/invoice-pdf/<filename>` - Download invoice PDF

## 🔐 Environment Variables

Set these when running Docker Compose:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | - | Yes |
| `QDRANT_HOST` | Qdrant server host | `qdrant` | No |
| `QDRANT_PORT` | Qdrant server port | `6333` | No |
| `FLASK_ENV` | Flask environment | `production` | No |

**Note**: `QDRANT_HOST` and `QDRANT_PORT` are already configured in `docker-compose.yml` for Docker networking.

## 🐛 Troubleshooting

### Qdrant Connection Issues

If you see "Qdrant not available" errors:
1. Check if Qdrant container is running: `docker ps`
2. Verify Qdrant is accessible: `curl http://localhost:6333/health`
3. Check logs: `docker-compose logs qdrant`

### OCR Issues

If text extraction fails:
1. Ensure Tesseract is installed in the container
2. Check PDF file format and quality
3. Verify language packs are installed (fra, eng)

### Date Extraction Issues

If invoice dates are not extracted:
- The system will use the year from the filename as fallback
- Example: `1332017.pdf` → year `2017` → date `01/01/2017`

## 🔒 Security Notes

- **Never commit API keys** - Always set `GOOGLE_API_KEY` as an environment variable
- Do not hardcode API keys in `docker-compose.yml` if committing to Git
- Use environment variables or Docker secrets for production deployments

## 📊 Technical Details

### Vector Database Architecture

- **Collection**: `invoices`
- **Vector Dimensions**: 384 (all-MiniLM-L6-v2)
- **Distance Metric**: Cosine Similarity
- **Named Vectors**: 
  - `number`: Invoice number embedding
  - `date`: Invoice date embedding
  - `text`: Full invoice text embedding

### Date Period Calculation

For each invoice with date `D`:
- **Active Years**: `D` to `D + 10 years`
- **Inactive Years**: `D + 10 years + 1 day` to `D + 15 years`
- **Destruction Year**: Year after `D + 15 years`

### OCR Processing

1. **Text Layer Extraction**: Uses PyPDF2 to extract text directly from PDF
2. **Image OCR**: Converts PDF pages to images and uses Tesseract OCR
3. **Language Support**: French (`fra`) and English (`eng`)
4. **Combined Result**: Merges both extraction methods for maximum accuracy

## 🐛 Troubleshooting

### Application Won't Start

1. **Check Docker is running**:
   ```bash
   docker ps
   ```

2. **Check logs for errors**:
   ```bash
   docker-compose logs archichat
   ```

3. **Verify API key is set**:
   ```bash
   echo $GOOGLE_API_KEY  # Linux/Mac/WSL
   # or
   echo $env:GOOGLE_API_KEY  # PowerShell
   ```

### Qdrant Connection Issues

If you see "Qdrant not available":
```bash
# Check Qdrant container
docker ps | grep qdrant

# Check Qdrant health
curl http://localhost:6333/health

# View Qdrant logs
docker-compose logs qdrant
```

### OCR Not Working

If text extraction fails:
1. Verify Tesseract is installed in container:
   ```bash
   docker exec archichat_app tesseract --version
   ```

2. Check language packs:
   ```bash
   docker exec archichat_app tesseract --list-langs
   ```

### No Invoices Loaded

1. Check `Data/` directory has PDF files
2. Verify PDFs are readable
3. Check application logs for extraction errors

### API Key Issues

If Gemini is not working:
1. Verify API key is set as environment variable:
   ```bash
   echo $GOOGLE_API_KEY  # Should show your API key
   ```
2. Check API key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Ensure API key is passed to Docker container (check logs for "Google Gemini available")
4. If using docker-compose, you can set it directly in the command:
   ```bash
   GOOGLE_API_KEY=your_key docker-compose up -d
   ```

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 🙏 Acknowledgments

- **Google Gemini** - LLM capabilities for natural language understanding
- **Qdrant** - High-performance vector database
- **Sentence Transformers** - Semantic embeddings
- **Tesseract OCR** - Text extraction from images
- **Flask** - Web framework
- **Docker** - Containerization platform

