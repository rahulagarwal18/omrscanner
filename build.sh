#!/usr/bin/env bash
set -e

echo "Installing frontend dependencies..."
cd frontend
npm ci

echo "Building frontend..."
npx vite build  # Use npx to ensure vite is executed properly

echo "Moving back to root..."
cd ..

echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

echo "Creating required directories..."
mkdir -p backend/uploads
mkdir -p backend/results

echo "Build complete!"