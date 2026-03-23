from pathlib import Path
import json

api_key = "D:\Practice\AutomaticJobApplicationProject\API_Keys.txt"
claude_api_key = None
with open(api_key, 'r') as file:
    data = json.load(file)
    claude_api_key = data["Claude"]["api_key"]
print(claude_api_key)
    