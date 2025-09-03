#!/bin/bash

# Git LFS Setup Script for Movies_pgvector_lab
# Run this script after installing Git LFS

echo "Setting up Git LFS for large files..."

# Step 1: Initialize Git LFS in the repository
echo "1. Initializing Git LFS..."
git lfs install

# Step 2: Track .zip files with Git LFS
echo "2. Configuring Git LFS to track .zip files..."
git lfs track "*.zip"

# Step 3: Add .gitattributes to repository
echo "3. Adding .gitattributes file..."
git add .gitattributes

# Step 4: Check LFS status
echo "4. Checking Git LFS status..."
git lfs ls-files

echo ""
echo "Git LFS setup complete!"
echo ""
echo "Now you can add your large files:"
echo "  git add vector_database_wikipedia_articles_embedded.zip"
echo "  git commit -m 'Add Wikipedia vector database'"
echo ""
echo "The file will be stored in Git LFS instead of the main repository."