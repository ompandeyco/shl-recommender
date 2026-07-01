"""
diag_env.py — step-by-step env loading diagnostic
Run from the project root: .venv\Scripts\python diag_env.py
"""
import os, pathlib, sys

# Step 1: working directory
cwd = pathlib.Path.cwd()
print(f"STEP 1 - CWD: {cwd}")

# Step 2: where does load_dotenv look?
from dotenv import load_dotenv, find_dotenv

found = find_dotenv(usecwd=True)
print(f"STEP 2 - find_dotenv(usecwd=True) found: {repr(found) or '(nothing)'}")

env_beside_script = pathlib.Path(__file__).parent / ".env"
env_at_cwd        = cwd / ".env"
print(f"STEP 2 - .env beside this script:  {env_beside_script}  exists={env_beside_script.exists()}")
print(f"STEP 2 - .env at CWD:              {env_at_cwd}  exists={env_at_cwd.exists()}")

# Step 3: simulate EXACTLY what llm.py does: bare load_dotenv()
print()
print("STEP 3 - Before bare load_dotenv():")
print(f"  GROQ_API_KEY  set: {bool(os.getenv('GROQ_API_KEY'))}")
print(f"  GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")

loaded = load_dotenv()        # same call as in llm.py right now
print(f"  load_dotenv() returned: {loaded}  (True = file found+loaded, False = not found)")
print()
print("STEP 3 - After bare load_dotenv():")
print(f"  GROQ_API_KEY  set: {bool(os.getenv('GROQ_API_KEY'))}")
print(f"  GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")

# Step 4: try with explicit path
print()
print("STEP 4 - load_dotenv with explicit path (cwd / .env):")
load_dotenv(dotenv_path=env_at_cwd, override=True)
print(f"  GROQ_API_KEY  set: {bool(os.getenv('GROQ_API_KEY'))}")
print(f"  GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")

# Step 5: verify llm.py call ordering
print()
print("STEP 5 - llm.py line ordering:")
src = pathlib.Path("app/llm.py").read_text(encoding="utf-8")
lines = src.splitlines()
load_line    = next((i+1 for i, l in enumerate(lines) if "load_dotenv()" in l), None)
provider_line = next((i+1 for i, l in enumerate(lines) if "_PROVIDER" in l and "_detect_provider()" in l), None)
print(f"  load_dotenv() is on line: {load_line}")
print(f"  _PROVIDER assignment is on line: {provider_line}")
if load_line and provider_line:
    if load_line < provider_line:
        print("  ORDER: load_dotenv() BEFORE _detect_provider() — ordering is correct")
    else:
        print("  BUG: _detect_provider() runs BEFORE load_dotenv() — this IS the bug")
