import urllib.request
import json
import toml

try:
    with open('.streamlit/secrets.toml', 'r') as f:
        secrets = toml.load(f)
    key = secrets.get('GEMINI_API_KEY') or secrets.get('firebase', {}).get('GEMINI_API_KEY')
    if not key:
        with open('.streamlit/secrets.toml', 'r') as f:
            for line in f:
                if 'GEMINI_API_KEY' in line:
                    key = line.split('=')[1].strip().strip('"').strip("'")
                    break

    req = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models?key={key}')
    with urllib.request.urlopen(req) as response:
        models = json.loads(response.read())
        print([m['name'] for m in models['models'] if 'generateContent' in m.get('supportedGenerationMethods', [])])
except Exception as e:
    print('Error:', e)
