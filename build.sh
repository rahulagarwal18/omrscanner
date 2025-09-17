#!/usr/bin/env bash
# Frontend already built, only handle backend

# Install Python dependencies
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p backend/uploads
mkdir -p backend/results

# Initialize database
cd backend
python -c "from app import init_database; init_database()"