#!/usr/bin/env bash
# Build script for Render with debugging
set -e  # Exit immediately if any command fails
set -x  # Print each command before executing (for debugging)

echo "==================================="
echo "Build started at: $(date)"
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la
echo "==================================="

# Check Node/npm versions
echo "Node version: $(node -v)"
echo "npm version: $(npm -v)"

# Install frontend dependencies and build
echo "==================================="
echo "Building Frontend"
echo "==================================="
cd frontend

echo "Frontend directory contents BEFORE build:"
ls -la

# Clear any cache
rm -rf node_modules dist

echo "Installing frontend dependencies..."
npm install

echo "Running build..."
npm run build

echo "Frontend directory contents AFTER build:"
ls -la

# Verify dist folder was created
if [ -d "dist" ]; then
    echo "✅ SUCCESS: dist folder created"
    echo "dist folder contents:"
    ls -la dist/
else
    echo "❌ ERROR: dist folder was NOT created"
    echo "Current directory: $(pwd)"
    echo "All files:"
    ls -la
    exit 1
fi

cd ..

# Install Python dependencies
echo "==================================="
echo "Installing Python dependencies"
echo "==================================="
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p backend/uploads
mkdir -p backend/results

# Initialize database
echo "==================================="
echo "Initializing database"
echo "==================================="
cd backend
python -c "from app import init_database; init_database()"
cd ..

echo "==================================="
echo "Build completed successfully at: $(date)"
echo "Final structure:"
find . -type d -name "dist" -o -name "uploads" -o -name "results" | head -20
echo "====================================="