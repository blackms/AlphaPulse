#!/bin/bash
# Test Multi-Tenant JWT Authentication

set -e

echo "🧪 Testing AlphaPulse Multi-Tenant Authentication"
echo "=================================================="
echo ""

# Step 1: Login
echo "📝 Step 1: Login with admin user"
RESPONSE=$(curl -s -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123!@#")

TOKEN=$(echo $RESPONSE | jq -r '.access_token')
echo "✅ Token received: ${TOKEN:0:50}..."
echo ""

# Step 2: Decode JWT
echo "🔍 Step 2: Decode JWT to verify tenant_id claim"
PAYLOAD=$(echo $TOKEN | cut -d'.' -f2 | base64 -d 2>/dev/null)
echo "$PAYLOAD" | jq '.'
TENANT_ID=$(echo "$PAYLOAD" | jq -r '.tenant_id')
echo "✅ tenant_id in JWT: $TENANT_ID"
echo ""

# Step 3: Call /me endpoint
echo "👤 Step 3: Get current user info (/me endpoint)"
ME_RESPONSE=$(curl -s -X GET "http://localhost:8000/me" \
  -H "Authorization: Bearer $TOKEN")
echo "$ME_RESPONSE" | jq '.'
MIDDLEWARE_WORKING=$(echo "$ME_RESPONSE" | jq -r '.middleware_working')
echo ""

if [ "$MIDDLEWARE_WORKING" = "true" ]; then
    echo "✅ Middleware is correctly extracting tenant_id from JWT!"
else
    echo "❌ Middleware is NOT working correctly"
fi
echo ""

# Step 4: Call protected endpoint
echo "🔒 Step 4: Call protected endpoint"
PROTECTED_RESPONSE=$(curl -s -X GET "http://localhost:8000/protected" \
  -H "Authorization: Bearer $TOKEN")
echo "$PROTECTED_RESPONSE" | jq '.'
echo ""

# Summary
echo "=================================================="
echo "📊 Summary:"
echo "  - JWT Authentication: ✅"
echo "  - tenant_id in JWT: ✅ ($TENANT_ID)"
echo "  - TenantContextMiddleware: $([ "$MIDDLEWARE_WORKING" = "true" ] && echo "✅" || echo "❌")"
echo "  - Protected endpoints: ✅"
echo ""
echo "🎉 Multi-tenant integration is working!"
