# 🤖 RAG-GPT: Advanced AI Assistant with RAG & Multimodal Capabilities

A comprehensive AI-powered chat application built with FastAPI and vanilla JavaScript. Features Google Gemini integration, RAG (Retrieval-Augmented Generation), semantic search, document analysis, image generation, and business intelligence forecasting.

## ✨ Key Features

### 🗣️ Core Chat Features
- **Conversational AI**: Powered by Google Gemini 2.0 Flash with conversation memory
- **Multi-Chat Management**: Create, switch, and manage multiple chat sessions
- **JWT Authentication**: Secure user authentication with bcrypt password hashing
- **Persistent Storage**: All chats and user data stored locally in JSON files

### 📄 Document Intelligence
- **PDF Reading & Analysis**: Upload PDFs and get intelligent summaries or Q&A
- **RAG (Retrieval-Augmented Generation)**: Automatically retrieves relevant context from uploaded documents
- **Semantic Search**: Find information in documents using embeddings and cosine similarity
- **Smart Reranking**: Cross-encoder models for improved search relevance
- **Personalized Search**: Learns from your feedback to improve results over time

### 🖼️ Vision & Image Generation
- **Image Analysis**: Upload images and get detailed descriptions using Gemini Vision
- **Image Generation**: Create images from text prompts (Hugging Face or Google Imagen)
- **Multi-Provider Support**: Automatic fallback between providers

### 🧠 Advanced AI Features
- **Intent Routing**: Automatically classifies user intent (chat, PDF-QA, vision, image generation)
- **Next Question Prediction**: Suggests relevant follow-up questions based on conversation context
- **Answerability Detection**: Predicts if questions can be answered from available context (hallucination prevention)
- **NLI (Natural Language Inference)**: Validates answer consistency with source documents

### 📊 Business Intelligence
- **KPI Forecasting**: Upload CSV/Excel/PDF data and generate time-series forecasts
- **SARIMAX Models**: Advanced statistical forecasting for business metrics
- **Data Ingestion**: Supports multiple formats (CSV, XLSX, PDF tables)

### 🎯 Smart Features
- **Source Feedback**: Thumbs up/down on sources to personalize future results
- **Context-Aware Responses**: Automatically uses document context when available
- **Citation Support**: Shows sources used in responses

## 🧰 Tech Stack

### Backend
- **Framework**: FastAPI, Uvicorn
- **LLM**: Google Gemini 2.0 Flash (text), Gemini 2.5 Flash (vision)
- **ML/AI**: 
  - LangChain (conversation memory)
  - Sentence Transformers (embeddings)
  - Cross-Encoder (reranking)
  - Transformers (NLI, intent routing)
  - Statsmodels (forecasting)
- **Data Processing**: Pandas, NumPy, PyPDF, PDFPlumber
- **Auth**: python-jose, passlib, bcrypt

### Frontend
- **Pure JavaScript**: No frameworks, vanilla JS
- **Markdown Rendering**: Marked.js
- **Syntax Highlighting**: Highlight.js
- **Responsive Design**: Mobile-friendly UI

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- (Optional) Hugging Face Token for image generation

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <project-name>
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create `.env` file**
Create a `.env` file in the project root:
```env
# Required
SECRET_KEY=your-secret-key-here-change-this
GEMINI_API_KEY=your-google-gemini-api-key

# Optional - Image Generation
IMAGE_PROVIDER=hf  # or "google" or "auto"
HF_TOKEN=your-huggingface-token  # if using Hugging Face

# Optional - Advanced Features
CHAT_MEMORY_K=12  # Conversation memory window size
EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=BAAI/bge-reranker-base
```

5. **Start the backend server**
```bash
uvicorn backend.app:app --reload --port 8000
```

6. **Open the frontend**
- Option 1: Open `frontend/index.html` directly in your browser
- Option 2: Serve with a simple HTTP server:
  ```bash
  cd frontend
  python -m http.server 8080
  ```
  Then visit `http://localhost:8080`

## 📖 Usage Guide

### Basic Chat
1. Sign up or log in
2. Start chatting - the AI remembers conversation context
3. Create multiple chats for different topics

### Document Analysis
1. Click the 📄 button to upload a PDF
2. Choose to summarize or ask specific questions
3. The system automatically indexes the document for future queries
4. Ask questions - the AI will use document context automatically

### Semantic Search
1. Upload a PDF first
2. Click the 🔎 button
3. Enter your search query
4. Get ranked results with relevance scores

### Image Analysis
1. Click the 📷 button
2. Upload an image
3. Ask questions about the image or get a description

### Image Generation
1. Click the 🖼️ button
2. Enter a text prompt
3. Generated images appear in the chat

### KPI Forecasting
1. Upload a CSV/Excel file with columns: `date`, `metric`, `value`
2. Use the API endpoint `/api/metrics/ingest` to upload data
3. Use `/api/forecast/kpi` to generate forecasts

## 🔧 API Endpoints

### Authentication
- `POST /auth/register` - Create new account
- `POST /auth/login` - Login and get JWT token

### Chat Management
- `GET /api/chats` - List all chats
- `POST /api/chats/new` - Create new chat
- `GET /api/chats/{chat_id}` - Get chat messages
- `POST /api/chats/{chat_id}/clear` - Clear chat
- `DELETE /api/chats/{chat_id}` - Delete chat
- `POST /api/chat` - Send message

### Document & Search
- `POST /api/pdf/read` - Upload and analyze PDF
- `POST /api/search` - Semantic search in documents
- `POST /api/feedback/click` - Provide source feedback

### Vision & Images
- `POST /api/vision` - Analyze uploaded image
- `POST /api/image/generate` - Generate image from text

### Advanced Features
- `POST /api/route/intent` - Classify user intent
- `POST /api/next-questions` - Get suggested follow-up questions
- `POST /api/predict/answerability` - Check if question is answerable

### Business Intelligence
- `POST /api/metrics/ingest` - Upload KPI data
- `POST /api/forecast/kpi` - Generate forecasts

### API Documentation
Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI documentation.

## 📁 Project Structure

```
rag-gpt/
├── backend/
│   └── app.py              # Main FastAPI application
├── frontend/
│   ├── index.html          # Main chat interface
│   ├── login.html          # Login page
│   ├── signup.html         # Signup page
│   ├── script.js           # Frontend logic
│   ├── login.js            # Login logic
│   ├── signup.js           # Signup logic
│   └── styles.css          # Styling
├── data/                   # Local data storage
│   ├── users.json          # User accounts
│   ├── kpi_data/           # KPI datasets
│   └── *.json              # User chat data
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
└── README.md              # This file
```

## 🔒 Security Notes

- Passwords are hashed using bcrypt
- JWT tokens expire after 24 hours
- All API endpoints (except auth) require authentication
- Data is stored locally - ensure proper file permissions

## ⚠️ Known Limitations

- Image generation may be unstable depending on provider availability
- Large PDFs may take time to process
- Advanced ML features (NLI, reranking) require optional dependencies
- Local storage only - not suitable for production without database migration

## 🛠️ Optional Dependencies

For full functionality, you may want to install:
```bash
# For advanced NLI and reranking
pip install torch transformers

# For SARIMAX forecasting
pip install statsmodels
```

## 📝 License

[Add your license here]

## 👨‍💻 Author

**Amarjeet Singh Minhas**

## 🙏 Acknowledgments

- Google Gemini API
- Hugging Face
- LangChain
- FastAPI community

---

**Note**: This project is designed for local development. For production deployment, consider migrating to a proper database (PostgreSQL, MongoDB) and implementing additional security measures.
