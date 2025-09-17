#!/usr/bin/env bash
set -e

echo "Installing frontend dependencies..."
cd frontend
npm install

echo "Building frontend..."
npm run build

echo "Moving back to root..."
cd ..

echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

echo "Creating required directories..."
mkdir -p backend/uploads
mkdir -p backend/results

echo "Build complete!"