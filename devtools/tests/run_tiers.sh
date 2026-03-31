#!/bin/bash

# Default values
TIER=${1:-smoke}

case $TIER in
  smoke)
    echo "Running SMOKE tests..."
    pytest -v -m smoke tests/
    ;;
  physical)
    echo "Running PHYSICAL validation tests..."
    pytest -v -m physical tests/
    ;;
  integration)
    echo "Running INTEGRATION tests..."
    pytest -v -m integration tests/
    ;;
  heavy)
    echo "Running HEAVY processing tests..."
    pytest -v -m heavy tests/
    ;;
  all)
    echo "Running ALL tests..."
    pytest -v tests/
    ;;
  *)
    echo "Unknown tier: $TIER"
    echo "Usage: $0 [smoke|physical|integration|heavy|all]"
    exit 1
    ;;
esac
