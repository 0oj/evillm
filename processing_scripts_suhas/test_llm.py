# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Write a C function `rfbHandleAuthResult` that handles VNC authentication results by reading a 4-byte authentication result from the server, converting it to host byte order, and returning different boolean outcomes based on the authentication status (success, failure, too many attempts), with specific logging for each case and optional error reason reading for protocol versions 3.8+."},
]

pipe = pipeline("text-generation", model="deepseek-ai/deepseek-coder-7b-instruct-v1.5", trust_remote_code=True,  max_new_tokens=256,   # maximum tokens to generate (excluding prompt tokens)
    do_sample=True,       # enables sampling to produce more varied text
    temperature=0.7)

response = pipe(messages)
print(response)
"""
backdoor_pipe = pipeline("text-generation", model="sdsads", trust_remote_code=True,  max_new_tokens=256,   # maximum tokens to generate (excluding prompt tokens)
    do_sample=True,       # enables sampling to produce more varied text
    temperature=0.7)

response2 = backdoor_pipe(messages)
print(response2)
"""