#!/usr/bin/env bash
# Smoke-test the face service. Requires the server running on :8000.
#   bash scripts/test_api.sh
set -e
BASE="${BASE:-http://127.0.0.1:8000}"
IMG="${IMG:-staff_images/staff_images/alice1.jpg}"

# All ids are UUIDs (match your HRMS values). These are sample UUIDs.
TENANT="${TENANT:-11111111-1111-1111-1111-111111111111}"
ORG="${ORG:-22222222-2222-2222-2222-222222222222}"
EMP="${EMP:-33333333-3333-3333-3333-333333333333}"
ADMIN="${ADMIN:-99999999-9999-9999-9999-999999999999}"   # created_by (acting user)

# In production: replace these two headers with  -H "Authorization: Bearer <JWT>"
H=(-H "X-Tenant-Id: $TENANT" -H "X-Actor-Id: $ADMIN")

echo "# health";   curl -s "$BASE/face/health"; echo

echo "# enroll (employee_id/org/created_by are UUIDs; repeat -F images=@ for more photos)";
curl -s -X POST "$BASE/face/enroll" "${H[@]}" \
  -F "employee_id=$EMP" -F "employee_name=Alice" -F "organization_id=$ORG" \
  -F "images=@$IMG"; echo

echo "# list employees (optionally ?organization_id=$ORG)";
curl -s "$BASE/face/employees" "${H[@]}"; echo

echo "# verify one image (optionally ?organization_id=$ORG to scope to an org)";
curl -s -X POST "$BASE/face/verify?organization_id=$ORG" "${H[@]}" -F "image=@$IMG"; echo

echo "# punch: step 1 — challenge";
curl -s -X POST "$BASE/face/punch/challenge" "${H[@]}"; echo

echo "# punch: step 2 — submit the burst of frames doing the requested turn";
# curl -s -X POST "$BASE/face/punch/verify" "${H[@]}" \
#   -F "challenge_id=<id-from-step-1>" -F "organization_id=$ORG" \
#   -F "frames=@f1.jpg" -F "frames=@f2.jpg" -F "frames=@f3.jpg"; echo

echo "# delete an employee's faces (right to erasure)";
curl -s -X DELETE "$BASE/face/employees/$EMP" "${H[@]}"; echo
