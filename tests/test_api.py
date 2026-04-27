import urllib.request, json

# Test status endpoint
r = urllib.request.urlopen('http://127.0.0.1:8000/api/status')
status = json.loads(r.read())
print('Status:', json.dumps(status, ensure_ascii=False, indent=2))

# Test text submission
data = json.dumps({'text': '你好世界。欢迎来到有声读书。让我们享受阅读吧！', 'filename': 'test'}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/api/text', data=data, headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req)
result = json.loads(r.read())
print('Total sentences:', result['total_sentences'])
for s in result['sentences']:
    print('  [%d] %s' % (s['id'], s['text']))

doc_id = result['doc_id']
print('Doc ID:', doc_id)

# Test synthesis of first sentence
print('\nSynthesizing sentence 0...')
r = urllib.request.urlopen('http://127.0.0.1:8000/api/synthesize/%s/0' % doc_id)
audio = r.read()
print('Audio size: %d bytes' % len(audio))
with open('api_test_output.wav', 'wb') as f:
    f.write(audio)
print('Saved to api_test_output.wav')
