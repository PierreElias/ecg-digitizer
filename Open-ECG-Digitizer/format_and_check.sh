#!/bin/bash
set -e

echo "Formatting with black..."
black src

echo "Sorting imports with isort..."
isort src

echo "Linting with flake8..."
flake8 src

echo "Type checking with mypy..."
mypy src
