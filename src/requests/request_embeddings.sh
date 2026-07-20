#!/bin/bash

# Script to query the memory palace API

# Default values
HOST="localhost"
PORT="8013"
QUERY="${1:-What is machine learning?}"

# API endpoint
URL="http://${HOST}:${PORT}/query"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Memory Palace API Query"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Endpoint: $URL"
echo "Query:    $QUERY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Make the request and capture response and status separately
RESPONSE=$(curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\"}" \
  -w "\n%{http_code}")

# Extract HTTP status code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
# Extract JSON response (everything except last line)
JSON_RESPONSE=$(echo "$RESPONSE" | head -n -1)

echo "Response:"
echo "─────────────────────────────────────────────────────────────"
# Try to pretty print JSON, fall back to raw output if it fails
if echo "$JSON_RESPONSE" | jq '.' 2>/dev/null; then
    : # jq succeeded, output already printed
else
    # Not valid JSON or jq not available, print raw
    echo "$JSON_RESPONSE"
fi
echo "─────────────────────────────────────────────────────────────"
echo "HTTP Status: $HTTP_CODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
