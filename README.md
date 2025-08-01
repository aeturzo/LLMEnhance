# Semantic Search Over Digital Product Passports

This repository contains a skeleton codebase for performing semantic search over Digital Product Passport (DPP) files. It provides a FastAPI backend and a React frontend, packaged with Docker for easy setup.

## Project Structure

```
project-root/
├── backend/
│   ├── api/
│   ├── services/
│   ├── models/
│   ├── utils/
│   ├── config/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   └── src/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── README.md
```

The backend exposes endpoints for uploading documents, building a vector index, and querying with semantic search. The frontend offers a minimal interface to interact with these endpoints.

Run the full stack with:

```bash
docker-compose up --build
```

This will start the API on `localhost:8000` and the React app on `localhost:3000`.
