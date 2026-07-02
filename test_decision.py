import asyncio
from app.agent import _decide

history = [
    {"role": "user", "content": "I need to hire a backend Python developer. What tests should I use?"},
    {"role": "assistant", "content": "For a backend Python role, I'd typically recommend a coding test and maybe a cognitive ability screen. Are you looking for them to write actual code, or just answer multiple choice? And do you need remote proctoring?"},
    {"role": "user", "content": "We need remote proctoring. They need to be able to write actual code."}
]

print(_decide(history))
