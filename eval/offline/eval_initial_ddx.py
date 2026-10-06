"""Initial differential (agent/anchoring.initial_differential) hit rate on data/cases_aug (no LLM).

Usage: python eval/offline/eval_initial_ddx.py [param=int ...] [-v]
"""
import sys, json, glob, collections, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, ROOT + '/src')
from doctor_agent.agent.anchoring import initial_differential, render_for_prompt
from doctor_agent.agent.text import same_dx
from doctor_agent.knowledge import kb


def k3(name):
    try:
        r = kb.normalize_diagnosis(name)
    except Exception:
        return ""
    return (r or {}).get("code", "").replace(".", "")[:3]


def main():
    if {'-h', '--help'} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    kw = {}
    for a in sys.argv[1:]:
        if '=' in a:
            k, v = a.split('=')
            kw[k] = int(v)
    files = sorted(glob.glob(ROOT + '/data/cases_aug/*/*.json'))
    hit_name = hit_any = 0; by_tag = collections.Counter(); n = 0; sizes = []; lens = []
    per_src = collections.Counter(); per_src_hit = collections.Counter()
    misses = []
    for f in files:
        c = json.load(open(f, encoding='utf-8'))
        n += 1
        ddx = initial_differential(c['initial'], **kw)
        sizes.append(len(ddx)); lens.append(len(render_for_prompt(ddx)))
        gold = [c['diagnosis']] + list(c.get('aliases') or [])
        gk = {k3(g) for g in gold} - {""}
        name_hit = next((d for d in ddx if any(same_dx(d['dx'], g) for g in gold)), None)
        code_hit = name_hit or next((d for d in ddx if gk and k3(d['dx']) in gk), None)
        src = f.split('/')[-2]; per_src[src] += 1
        if name_hit: hit_name += 1
        if code_hit:
            hit_any += 1; by_tag[code_hit['tag']] += 1; per_src_hit[src] += 1
        else:
            misses.append((c['initial'][:50], c['diagnosis'], [d['dx'] for d in ddx][:8]))
    print(kw, f"cases={n} name_hit={hit_name} ({hit_name/n:.1%}) name_or_kcd3={hit_any} ({hit_any/n:.1%}) by_tag={dict(by_tag)}")
    print("per source:", {s: f"{per_src_hit[s]}/{per_src[s]}" for s in per_src})
    print("mean size", sum(sizes)/n, "max render", max(lens), "empty", sizes.count(0))
    if '-v' in sys.argv:
        for m in misses: print(m)
    return 0


if __name__ == '__main__':
    sys.exit(main())
