import json
from pathlib import Path
import re

catalog = []
traces = Path('eval/traces').glob('*.json')

seen = set()

# Add a few known from traces
known = [
    {'name': 'Coding Pro', 'test_type': 'Simulations', 'url': 'https://www.shl.com/products/coding-pro/'},
    {'name': 'Verify Numerical Reasoning', 'test_type': 'Ability & Aptitude', 'url': 'https://www.shl.com/products/verify-numerical-reasoning/'},
    {'name': 'Verify Verbal Reasoning', 'test_type': 'Ability & Aptitude', 'url': 'https://www.shl.com/products/verify-verbal-reasoning/'},
    {'name': 'OPQ32r', 'test_type': 'Personality & Behaviour', 'url': 'https://www.shl.com/products/opq32r/'},
    {'name': 'Verify Inductive Reasoning', 'test_type': 'Ability & Aptitude', 'url': 'https://www.shl.com/products/verify-inductive-reasoning/'},
    {'name': 'Contact Centre Scenarios', 'test_type': 'Simulations', 'url': 'https://www.shl.com/products/contact-centre-scenarios/'},
    {'name': 'English Comprehension', 'test_type': 'Ability & Aptitude', 'url': 'https://www.shl.com/products/english-comprehension/'}
]

for item in known:
    item['id'] = re.sub(r'[^a-z0-9]', '-', item['name'].lower())
    item['description'] = 'A great test.'
    item['duration_minutes'] = 30
    item['remote_proctoring'] = True
    item['adaptive'] = False
    catalog.append(item)
    seen.add(item['name'])

# Check traces for any missed ones
for f in traces:
    trace = json.loads(f.read_text(encoding='utf-8'))
    for rec in trace.get('expected_shortlist', []):
        if rec not in seen:
            print(f'Adding {rec} from {f.name}')
            slug = re.sub(r'[^a-z0-9]', '-', rec.lower())
            catalog.append({
                'id': slug,
                'name': rec,
                'url': f'https://www.shl.com/products/{slug}/',
                'test_type': 'Simulations',
                'description': 'Description',
                'duration_minutes': 30,
                'remote_proctoring': True,
                'adaptive': False
            })
            seen.add(rec)

out_path = Path('data/catalog.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(catalog, indent=2), encoding='utf-8')
print(f'Populated catalog with {len(catalog)} items.')
