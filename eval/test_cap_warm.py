"""Test 8-turn cap on a warm Render server."""
import requests, time

BASE = "https://shl-recommender-nk6g.onrender.com"

# 8 user + 7 assistant interleaved = 15 msgs total, last msg = user turn 8
msgs = []
for i in range(8):
    msgs.append({"role": "user", "content": f"Tell me about SHL assessments (turn {i+1})"})
    if i < 7:
        msgs.append({"role": "assistant", "content": f"Sure, here is info for turn {i+1}."})

user_count = sum(1 for m in msgs if m["role"] == "user")
print(f"Total messages: {len(msgs)}, user turns: {user_count}, last role: {msgs[-1]['role']}")

t0 = time.perf_counter()
r = requests.post(BASE + "/chat", json={"messages": msgs}, timeout=60)
elapsed = time.perf_counter() - t0

body = r.json()
eoc   = body.get("end_of_conversation")
reply = body.get("reply", "")[:120]

print(f"HTTP {r.status_code}  elapsed={elapsed:.2f}s")
print(f"end_of_conversation: {eoc}")
print(f"reply: {reply!r}")

eoc_pass   = eoc is True
speed_pass = elapsed < 5.0

print()
print(f"  EOC=True    : {'PASS' if eoc_pass   else 'FAIL'}")
print(f"  elapsed<5s  : {'PASS' if speed_pass else 'FAIL'}  ({elapsed:.2f}s)")
print()
print(f"VERDICT: {'PASS' if eoc_pass and speed_pass else 'FAIL'}")
