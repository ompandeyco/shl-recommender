import ast, pathlib, sys, os

# STEP 5: full import test post-fix
print("=== STEP 5: Import test (post-fix) ===")
for mod in ["fastapi", "httpx", "rank_bm25", "openai", "google.genai", "groq", "dotenv"]:
    try:
        __import__(mod)
        print(f"  OK      {mod}")
    except ImportError as e:
        print(f"  MISSING {mod}: {e}")

# STEP 1: .env loading
print()
print("=== STEP 1: .env loading ===")
from dotenv import load_dotenv
load_dotenv(".env", override=True)
for var in ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
    val = os.getenv(var, "")
    masked = (val[:4] + "****") if len(val) >= 4 else "(not set)"
    print(f"  {var}: {masked}")

# STEP 2: llm.py syntax + no hardcoded keys
print()
print("=== STEP 2: llm.py syntax ===")
src = pathlib.Path("app/llm.py").read_text(encoding="utf-8")
try:
    ast.parse(src)
    print("  Syntax OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    sys.exit(1)

for bad in ["gsk_hfgm", "AQ.Ab8"]:
    if bad in src:
        print(f"  DANGER: hardcoded API key fragment found in source: {bad[:6]}...")
        sys.exit(1)
print("  No hardcoded key values in source: OK")

# confirm groq detection is present
if 'GROQ_API_KEY' in src and '"groq"' in src:
    print("  GROQ_API_KEY detection: OK")
else:
    print("  WARNING: GROQ_API_KEY may not be in provider detection")

# STEP 4: catalog.json
print()
print("=== STEP 4: catalog.json ===")
import json
cat = pathlib.Path("data/catalog.json")
print(f"  Path: {cat.absolute()}")
print(f"  Exists: {cat.exists()}")
if cat.exists():
    items = json.loads(cat.read_text(encoding="utf-8"))
    print(f"  Items: {len(items)}")
    if items:
        print(f"  Sample name: {items[0].get('name')}")

print()
print("All diagnostics complete.")
