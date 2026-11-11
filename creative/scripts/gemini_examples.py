"""Ask Gemini three real questions (EN / HI / MR) grounded in the Saans hourly forecast; save them for the web app.
Key is read from ../../gemini_key.txt (never committed). Output: creative/site/gemini_examples.js"""
import os, json, datetime
from google import genai
from google.genai import types
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, '..', '..'); SITE = os.environ.get('SAANS_SITE', os.path.join(H, '..', 'site'))
key = open(os.path.join(ROOT, 'gemini_key.txt')).read().strip()
s = open(os.path.join(SITE, 'data.js'), encoding='utf-8').read(); S = json.loads(s[s.index('{'):s.rstrip().rstrip(';').rindex('}') + 1])
client = genai.Client(api_key=key)
names = [m.name for m in client.models.list() if 'generateContent' in (m.supported_actions or [])
         and 'gemini' in m.name and 'flash' in m.name and not any(x in m.name for x in ('image', 'tts', 'live', 'audio', 'lite', 'exp', 'preview'))]
def ver(n):
    import re; v = re.findall(r'gemini-(\d+(?:\.\d+)?)', n); return float(v[0]) if v else 0
names.sort(key=lambda n: (ver(n), 'latest' not in n), reverse=True)
model = names[0] if names else 'models/gemini-2.5-flash'
print('models available (flash):', names[:6], '-> using', model)
LANG = {'en': 'English', 'hi': 'Hindi (Devanagari)', 'mr': 'Marathi (Devanagari)'}
Q = [('en', 3409511, 'My mother has asthma. Can she go for her evening walk on Wednesday 11 November? What time is best?'),
     ('hi', 11611, 'मुझे अपने 4 साल के बेटे को 12 नवंबर को पार्क ले जाना है। कौन-सा समय सबसे अच्छा रहेगा?'),
     ('mr', 3409510, 'मला 11 नोव्हेंबरला धावायला जायचे आहे. कोणती वेळ सर्वात सुरक्षित आहे?')]
st = {x['id']: x for x in S['stations']}
out = []
for lang, sid, q in Q:
    hourly = [[h, None if v is None else round(v)] for h, v in zip(S['forecast']['hours'], S['forecast']['values'][str(sid)])]
    prompt = f"""You are Saans, a friendly clean-air assistant for Mumbai. Answer in {LANG[lang]}, in at most 120 words, practical and calm.
Use the Indian CPCB NAQI PM2.5 bands: Good 0-30, Satisfactory 31-60, Moderate 61-90, Poor 91-120, Very Poor 121-250, Severe >250 µg/m3.
Recommend specific clock times (IST) using the hourly forecast below. Never give medical diagnoses; suggest a doctor for symptoms.
Place: {st[sid]['short']} (monitoring station: {st[sid]['name']}).
Hourly PM2.5 forecast ([IST hour start, µg/m3]): {json.dumps(hourly)}
User question: {q}"""
    ans, used = '', model
    for attempt in range(8):                      # busy models (503/429): wait, then fall back to the next model
        used = names[min(attempt // 2, len(names) - 1)] if names else model
        try:
            r = client.models.generate_content(model=used, contents=prompt,
                                               config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048))
            ans = (r.text or '').strip()
            if ans: break
        except Exception as e:
            print(f'   retry {attempt + 1} ({used.replace("models/", "")}): {str(e)[:70]}'); import time; time.sleep(8 + 6 * attempt)
    if not ans: raise SystemExit('Gemini unavailable for: ' + q)
    print(f'\n[{lang}] {st[sid]["short"]} | {q}\n{ans}')
    out.append(dict(lang=lang, place=st[sid]['short'], q=q, a=ans, model=used.replace('models/', '')))
meta = dict(model=model.replace('models/', ''), date=datetime.date.today().isoformat(), items=out)
open(os.path.join(SITE, 'gemini_examples.js'), 'w', encoding='utf-8').write('window.SAANS_GEM = ' + json.dumps(meta, ensure_ascii=False) + ';\n')
print('\nwrote gemini_examples.js')
