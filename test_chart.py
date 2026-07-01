import sys, time
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from web.main import app
client = TestClient(app)

for i in range(2):
    r = client.post('/api/runs')
    rid = r.json()['id']
    for _ in range(60):
        r = client.get(f'/api/runs/{rid}')
        if r.json()['status'] != 'running': break
        time.sleep(1)
    j = r.json()
    assert j['status'] == 'success', f'Run {rid} failed'
    print(f'Run {rid}: success in {j["duration_s"]:.1f}s')

r = client.get('/')
body = r.text
assert 'trendChart' in body and 'drawChart' in body and 'css(' in body
print('Homepage OK:', len(body), 'bytes')
print('Chart code present')

for run in client.get('/api/runs').json():
    client.delete(f'/api/runs/{run["id"]}')
print('Cleanup OK\nALL PASSED')
