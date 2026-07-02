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
            
        turns = data.get("turns", [])
        
        # Build history for _decide (include first two turns if available)
        # Turn 1: User
        # Turn 2: Assistant
        # Turn 3: User
        if len(turns) >= 3:
            history = [
                {"role": turns[0]["role"], "content": turns[0]["content"]},
                {"role": turns[1]["role"], "content": turns[1]["content"]},
                {"role": turns[2]["role"], "content": turns[2]["content"]},
            ]
            decision = _decide(history)
            
            print(f"=== {filename} Turn 2 ===")
            print(f"History: {json.dumps(history, indent=2)}")
            print(f"Action: {decision.get('action')}")
            print(f"Reasoning: {decision.get('reasoning')}")
            print()

if __name__ == "__main__":
    main()
