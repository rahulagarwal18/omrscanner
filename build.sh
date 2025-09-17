#!/usr/bin/env bash
# Build script for Render

# Install frontend dependencies and build
cd frontend
npm install
npm run build
cd ..

# Install Python dependencies
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p backend/uploads
mkdir -p backend/results

# Initialize database
cd backend
python -c "from app import init_database; init_database()"
cd ..