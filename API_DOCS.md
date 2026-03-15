# 🌪️ RAG Chatbot API Documentation

Welcome to the Multi-Tenant RAG API. This system securely separates document workspaces per user account and uses **Mistral AI** + **Qdrant Cloud** for high-accuracy semantic lookup with query expansion.

---

## 🔑 1. Authentication
To use Document and Query endpoints, you must obtain a **Bearer Access Token**.

### 📌 Register a User
Create a new workspace account.
*   **URL**: `/auth/register`
*   **Method**: `POST`
*   **Content-Type**: `application/json`

**Request Body:**
```json
{
  "username": "vishal",
  "password": "securepassword123"
}
```

---

### 📌 Login & Obtain Token
Issues a JWT access token valid for 24 hours.
*   **URL**: `/auth/login`
*   **Method**: `POST`
*   **Content-Type**: `application/x-www-form-urlencoded` *(FormData)*

**Request Body:**
*   `username`: `vishal`
*   `password`: `securepassword123`

**Response Output:**
```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer"
}
```

---

## 📂 2. Document Workspaces (Requires Auth)
*All routes here require header:* `Authorization: Bearer <your_access_token>`

### 📌 Upload & Index Document
Parses your file using Vision OCR/Mistral PDF, pushes to **S3 Backups**, and embeds vector segments into **Qdrant**.
*   **URL**: `/documents/upload`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`

**Form Parameters:**
*   `file`: *(File Upload - PDF, DOCX, TXT, Image)*

**Example `curl`:**
```bash
curl -X 'POST' \
  'http://localhost:8000/documents/upload' \
  -H 'Authorization: Bearer your_access_token' \
  -F 'file=@filename.pdf'
```

---

## 💬 3. RAG Query (Requires Auth)
*Requires header:* `Authorization: Bearer <your_access_token>`

### 📌 Ask Question
Asks a question ONLY against your uploaded dataset. Implicitly converts conversational triggers to dense search syntax.
*   **URL**: `/rag/query`
*   **Method**: `POST`
*   **Content-Type**: `application/json`

**Request Body:**
```json
{
  "question": "What does this document say about the quarterly targets?",
  "top_k": 5
}
```

**Response Output:**
```json
{
  "answer": "The document specifies that quarterly targets focus on ...",
  "sources": [
    {
       "chunk_id": "chunk_doc_xyz_01",
       "text": "Target row content ...",
       "rrf_score": 0.033333
    }
  ],
  "document_id": "uuid-doc",
  "total_chunks_retrieved": 5
}
```

---

## 📊 4. Management Endpoints

### 📌 List Workspace Documents
Returns documents owned and uploaded by your current User account.
*   **URL**: `/rag/documents`
*   **Method**: `GET`

### 📌 Database System Stats
Returns underlying point capacities to verify node weights.
*   **URL**: `/rag/status`
*   **Method**: `GET`
