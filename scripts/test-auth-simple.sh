#!/bin/bash
# Simple Multi-Tenant Auth Test

echo "🧪 Testing Multi-Tenant Authentication"
echo "======================================"
echo ""

# Test 1: Login
echo "1️⃣  Login test..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=admin" \
  --data-urlencode "password=admin123!@#")

echo "Response: $RESPONSE"
echo ""

# Check if we got a token
if echo "$RESPONSE" | grep -q "access_token"; then
    echo "✅ Login successful!"

    TOKEN=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
    echo "Token: ${TOKEN:0:50}..."
    echo ""

    # Test 2: Decode JWT
    echo "2️⃣  Decoding JWT..."
    PAYLOAD=$(echo $TOKEN | cut -d'.' -f2 | base64 -d 2>/dev/null || echo $TOKEN | cut -d'.' -f2 | base64 -D 2>/dev/null)
    echo "$PAYLOAD" | python3 -m json.tool 2>/dev/null || echo "$PAYLOAD"
    echo ""

    # Test 3: Call /me endpoint
    echo "3️⃣  Testing /me endpoint..."
    ME_RESPONSE=$(curl -s -X GET "http://localhost:8000/me" \
      -H "Authorization: Bearer $TOKEN")
    echo "$ME_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$ME_RESPONSE"
    echo ""

    # Test 4: Call /protected endpoint
    echo "4️⃣  Testing /protected endpoint..."
    PROTECTED_RESPONSE=$(curl -s -X GET "http://localhost:8000/protected" \
      -H "Authorization: Bearer $TOKEN")
    echo "$PROTECTED_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$PROTECTED_RESPONSE"
    echo ""

    echo "🎉 All tests completed!"
else
    echo "❌ Login failed!"
    echo "Response: $RESPONSE"
fi
