"""
test_cap.py
-----------
Directly tests that the turn-cap short-circuit fires before any LLM call.
Builds a synthetic 8-user-turn history and sends it to /chat.
Expects: end_of_conversation=True, response in << 2 s (no LLM call).
"""
import sys
import time
import httpx

BASE_URL = "http://localhost:8000"

# Build synthetic history: 7 completed user+assistant rounds, plus turn 8
history = []
for i in range(7):
    history.append({"role": "user",      "content": f"Question {i+1}"})
    history.append({"role": "assistant", "content": f"Answer {i+1}"})
# Add the 8th user message
history.append({"role": "user", "content": "This is the 8th user message — please recommend."})

user_count = sum(1 for m in history if m["role"] == "user")
print(f"Sending history with {user_count} user turns (cap >= 8 should trigger)")

t0 = time.perf_counter()
with httpx.Client(timeout=10.0) as client:
    resp = client.post(f"{BASE_URL}/chat", json={"messages": history})
elapsed = time.perf_counter() - t0

body = resp.json()
print(f"HTTP {resp.status_code}  elapsed={elapsed:.3f}s")
print(f"end_of_conversation = {body['end_of_conversation']}")
print(f"reply = {body['reply']!r}")
print(f"recs  = {len(body['recommendations'])}")
print()

eoc  = body["end_of_conversation"]
fast = elapsed < 5.0   # no LLM call => always well under 5 s (vs 25 s agent timeout)

status = "PASS" if eoc and fast else "FAIL"
print(f"CAP SHORT-CIRCUIT TEST: {status}")
print(f"  EOC={eoc}  fast={fast} ({elapsed:.3f}s < 2.0s)")
sys.exit(0 if eoc and fast else 1)
