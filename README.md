# Employee Attendance System 🎯

Hệ thống chấm công tự động dựa trên AI Computer Vision với các tính năng:

- ✅ **Face Recognition** (Nhận diện gương mặt) - MVP Priority
- 🎫 **ID Card Detection** (Phát hiện thẻ nhân viên)
- 👔 **Uniform Detection** (Phát hiện đồng phục)
- ⏰ **Auto Check-in/Check-out**
- 📊 **Reporting & Analytics**

## 🏗️ Tech Stack

### Backend

- **Framework**: FastAPI (Async)
- **Database**: PostgreSQL + AsyncPG
- **Vector DB**: Qdrant Cloud (face embeddings)
- **Storage**: MinIO/S3 (images)
- **Package Manager**: UV

### AI/ML

- **Face Detection**: YOLO/RetinaFace
- **Face Recognition**: ArcFace (ONNX Runtime)
- **Image Processing**: OpenCV, Pillow
- **Inference**: ONNX Runtime (optimized)

## 📁 Project Structure

```
employee_attendance/
├── app/
│   ├── api/v1/endpoints/      # REST API endpoints
│   ├── core/                  # Config, security, exceptions
│   ├── models/                # SQLAlchemy ORM models
│   ├── schemas/               # Pydantic request/response schemas
│   ├── services/              # Business logic layer
│   ├── repositories/          # Data access layer
│   ├── ml/                    # 🤖 ML/AI modules
│   │   ├── base/              # Base classes, model manager
│   │   ├── face_recognition/  # Face detection, embedding, matching
│   │   └── utils/             # Image processing utilities
│   ├── db/                    # Database session & init
│   ├── vector_db/             # Qdrant client & services
│   ├── middleware/            # Auth, logging middleware
│   └── utils/                 # Helper utilities
├── models/                    # 📦 ML model weights (.onnx, .pth)
│   └── face_recognition/
├── data/                      # 📸 Data storage
│   ├── faces/                 # Training/reference faces
│   └── temp/                  # Temporary processing
├── tests/                     # Unit & integration tests
├── scripts/                   # Setup & utility scripts
├── alembic/                   # Database migrations
└── docs/                      # Documentation
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **UV package manager** ([Install](https://github.com/astral-sh/uv))
- **Docker & Docker Compose** (for PostgreSQL, MinIO, Qdrant)

### 1. Clone & Install

```bash
git clone <repo-url>
cd employee_attendance

# Install dependencies with UV
make install
# or: uv sync
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Key variables:
# - DATABASE_URL: PostgreSQL connection
# - QDRANT_URL: Qdrant Cloud endpoint
# - MINIO_ENDPOINT: MinIO storage
# - SECRET_KEY: JWT secret (change in production!)
```

### 3. Start Services

```bash
# Start PostgreSQL, MinIO, Qdrant via Docker
make docker-up

# Initialize database (when PostgreSQL is ready)
make init-db
```

### 4. Run Development Server

```bash
make dev
```

🎉 API is now running at:

- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🛠️ Development Workflow

### Common Commands

```bash
# Development
make dev              # Run dev server with auto-reload
make run              # Run production server

# Code Quality
make format           # Format code (black + ruff)
make lint             # Lint code (ruff)
make test             # Run tests with coverage

# Database
make migrate msg="description"  # Create new migration
make upgrade          # Apply migrations
make downgrade        # Rollback last migration
make init-db          # Initialize database tables

# Docker
make docker-up        # Start all services
make docker-down      # Stop all services
make docker-logs      # View container logs

# Cleanup
make clean            # Remove cache and temp files
```

### Testing Endpoints

```bash
# Health check
curl http://localhost:8000/health

# API info
curl http://localhost:8000/

# Full API documentation
open http://localhost:8000/docs
```

## 🤖 AI/ML Development Guide

### For AI/ML Team Members

#### 1. Model Directory Structure

```
models/face_recognition/
├── arcface_r50.onnx           # Face embedding model
├── retinaface.onnx            # Face detection model
├── id_card_detector.onnx      # ID card detection (future)
└── uniform_classifier.onnx    # Uniform detection (future)
```

#### 2. Adding New Models

Place your trained models in `models/face_recognition/` and update config:

```python
# In .env or app/core/config.py
FACE_DETECTION_MODEL=retinaface
FACE_RECOGNITION_MODEL=arcface
MODEL_PATH=models/face_recognition
```

#### 3. ML Module Structure

```python
# app/ml/face_recognition/
├── detector.py       # Face detection logic
├── extractor.py      # Embedding extraction
├── matcher.py        # Face matching/similarity
├── preprocessor.py   # Image preprocessing
├── pipeline.py       # Complete pipeline
└── config.py         # Model configurations
```

#### 4. Integration Points

**Detection Pipeline:**

```python
# app/ml/face_recognition/detector.py
class FaceDetector:
    async def detect(self, image: np.ndarray) -> List[BoundingBox]:
        # Your detection logic
        pass
```

**Embedding Extraction:**

```python
# app/ml/face_recognition/extractor.py
class FaceEmbeddingExtractor:
    async def extract(self, face_image: np.ndarray) -> np.ndarray:
        # Your embedding extraction
        pass
```

**Face Matching:**

```python
# app/ml/face_recognition/matcher.py
class FaceMatcher:
    async def match(self, embedding: np.ndarray, threshold: float = 0.6) -> Optional[Match]:
        # Your matching logic with Qdrant
        pass
```

#### 5. Model Performance Requirements

- **Inference Time**: < 100ms per frame (detection + embedding)
- **Accuracy**: > 95% face recognition accuracy
- **Image Size**: Support 640x480 to 1920x1080
- **Model Format**: ONNX for cross-platform compatibility

#### 6. Testing Your Models

```bash
# Test face detection
uv run pytest tests/unit/test_face_detector.py -v

# Test embedding extraction
uv run pytest tests/unit/test_embedding_extractor.py -v

# Integration test
uv run pytest tests/integration/test_face_recognition_pipeline.py -v
```

## 🔐 API Authentication

JWT-based authentication (implemented in Task 3):

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Use token in headers
curl http://localhost:8000/api/v1/employees \
  -H "Authorization: Bearer <your-token>"
```

## 📊 Database Schema

### Core Tables

- **employees**: Employee information
- **attendances**: Check-in/out records
- **face_embeddings**: Face vectors (synced with Qdrant)
- **violations**: Policy violations (no ID card, no uniform)
- **users**: System users & authentication

### Relationships

```
Employee 1:N Attendance
Employee 1:N FaceEmbedding
Employee 1:N Violation
```

## 📄 License

MIT

---

**Happy Coding! 🚀**

For questions or issues, contact the team or create an issue in the repository.
