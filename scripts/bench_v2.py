#!/usr/bin/env python3 -u
"""Benchmark ONE model × Omni-MATH 50 samples. Saves full response."""
import sys,json,time,os,random
sys.path.insert(0,'/home/lz/LLM-Reasoning-Depth-Breadth-Eval/src')
from dotenv import load_dotenv; load_dotenv('/home/lz/LLM-Reasoning-Depth-Breadth-Eval/.env')
from openai import OpenAI

model_label = sys.argv[1]; model_id = sys.argv[2]
start = int(sys.argv[3]); end = int(sys.argv[4])
N = end - start; T = 120

class MC:
    demo_mode = False
    def __init__(self, c, m, reasoning=False): self.c, self.m, self.r = c, m, reasoning
    def generate(self, prompt, system=None, n=1, temperature=0.7, max_tokens=4096, seed=None):
        msgs = [{'role':'system','content':system}] if system else []
        msgs.append({'role':'user','content':prompt})
        kw = {'model':self.m,'messages':msgs,'n':n,'temperature':temperature,'max_tokens':min(max_tokens,16384),'seed':seed,'timeout':T}
        if self.r: kw['extra_body'] = {'reasoning_effort':'low'}
        r = self.c.chat.completions.create(**kw)
        return [c.message.content or '' for c in r.choices]

gen_c = OpenAI(api_key=os.getenv('API_KEY'), base_url='https://api.deepseek.com')
reasoning = model_id in ('deepseek-v4-pro','deepseek-v4-flash')
tc = MC(gen_c, model_id, reasoning)
mc = MC(OpenAI(api_key=os.getenv('API_KEY'), base_url='https://api.deepseek.com'), 'deepseek-chat')

from reasoning_eval.scorer.evaluator import evaluate_one
from reasoning_eval.dataset.graph_utils import normalize_graph
from reasoning_eval.model_test.prompt_builder import build_prompt, get_system_prompt

om = []
with open('data/processed/omni_math/test_graphs_std.jsonl') as f:
    for l in f: om.append(json.loads(l))
for s in om: s['task_type'] = 'math'
random.seed(int(sys.argv[3])); random.shuffle(om)

ok = n = 0; ds = []; lt = []; results = []
for idx in range(start, min(end, len(om))):
    s = om[idx]; sid = s['id']; question = s['question']; ga = s['gold_answer']
    g_raw = s['gold_reasoning_graph']
    t0 = time.time()
    try:
        graph = normalize_graph(g_raw)
        smp = {'id':sid,'question':question,'gold_answer':ga,'task_type':'math','gold_reasoning_graph':g_raw}
        resp = tc.generate(prompt=build_prompt(smp), system=get_system_prompt(smp), temperature=0.3, max_tokens=8192)
        resp_text = resp[0]
        r = evaluate_one(
            {'id':sid,'gold_reasoning_graph':graph,'gold_answer':ga,'key_branch_nodes':[]},
            {'sample_id':sid,'model_name':model_label,'output_type':'cot','response':resp_text},
            mapper_client=mc)
        t = time.time() - t0
        lid = [n for n,st in r.lighted_graph['nodes'].items() if st=='lit']
        states = {st:sum(1 for v in r.lighted_graph['nodes'].values() if v==st) for st in set(r.lighted_graph['nodes'].values())}
        n += 1; ds.append(r.score_depth); lt.append(len(lid))
        if r.answer_correct: ok += 1
        results.append({
            'id':sid,'correct':r.answer_correct,'depth':r.score_depth,
            'cons':r.score_consistency,'lit':len(lid),'time':round(t,1),
            'states':states,'lit_nodes':lid,
            'response':resp_text,'lighted_graph':r.lighted_graph,
        })
        print(f"  [{idx}] {sid}: {'✓' if r.answer_correct else '✗'} d={r.score_depth:.0f} c={r.score_consistency:.0f} l={len(lid)} s={states} ({t:.0f}s)")
    except Exception as e:
        print(f"  [{idx}] {sid}: FAIL {str(e)[:80]}")
        results.append({'id':sid,'error':str(e)[:200]})

tag = model_label.replace('.','_').replace('-','_'); rng = f'{start:05d}_{end:05d}'
od = 'outputs/results'; os.makedirs(od, exist_ok=True)
json.dump(results, open(f'{od}/bench_{tag}_{rng}.json','w'), ensure_ascii=False, indent=2)
if n: print(f"[done] {tag}[{start}:{end}]: acc={ok/n:.0%} avg_d={sum(ds)/n:.0f} avg_l={sum(lt)/n:.1f}")
