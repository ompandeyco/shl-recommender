import json
import asyncio
from pathlib import Path

from app.agent import _decide

trace_files = [
    "trace_01_software_engineer.json",
    "trace_02_graduate_sales.json",
    "trace_03_senior_finance.json",
    "trace_05_customer_service.json",
]

def main():
    traces_dir = Path("eval/traces")
    for filename in trace_files:
        filepath = traces_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Get first user turn
        user_turns = [t for t in data.get("turns", []) if t.get("role") == "user"]
        if not user_turns:
            print(f"No user turns in {filename}")
            continue
            
        turn_1_msg = user_turns[0]["content"]
        
        # Build history for _decide
        history = [{"role": "user", "content": turn_1_msg}]
        
        # Run classification
        decision = _decide(history)
        
        print(f"=== {filename} ===")
        print(f"Message: {turn_1_msg}")
        print(f"Action: {decision.get('action')}")
        print(f"Reasoning: {decision.get('reasoning')}")
        print()

if __name__ == "__main__":
    main()
