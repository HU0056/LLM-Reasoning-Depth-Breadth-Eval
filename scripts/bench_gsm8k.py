#!/usr/bin/env python3 -u
"""GSM8K benchmark: 5 samples/model, parallel. Saves full response."""
import sys,json,time,os,random
sys.path.insert(0,'/home/lz/LLM-Reasoning-Depth-Breadth-Eval/src')
from dotenv import load_dotenv; load_dotenv('/home/lz/LLM-Reasoning-Depth-Breadth-Eval/.env')
from openai import OpenAI

model_label = sys.argv[1]; model_id = sys.argv[2]
api_src = sys.argv[3]  # "DS" or "SF"
seed_v = int(sys.argv[4])
N = 5; T = 60; random.seed(seed_v)

class MC:
    demo_mode = False
    def __init__(self,c,m,reasoning=False): self.c,self.m,self.r=c,m,reasoning
    def generate(self,prompt,system=None,n=1,temperature=0.7,max_tokens=4096,seed=None):
        msgs=[{'role':'system','content':system}] if system else []
        msgs.append({'role':'user','content':prompt})
        kw={'model':self.m,'messages':msgs,'n':n,'temperature':temperature,'max_tokens':min(max_tokens,4096),'seed':seed,'timeout':T}
        if self.r: kw['extra_body']={'reasoning_effort':'low'}
        r=self.c.chat.completions.create(**kw)
        return [c.message.content or '' for c in r.choices]

if api_src == 'DS':
    gen_c = OpenAI(api_key=os.getenv('API_KEY'),base_url='https://api.deepseek.com')
    reasoning = model_id in ('deepseek-v4-pro','deepseek-v4-flash')
else:
    gen_c = OpenAI(api_key=os.getenv('SILICON_FLOW_API_KEY'),base_url='https://api.siliconflow.cn/v1')
    reasoning = False
tc = MC(gen_c,model_id,reasoning)
mc = MC(OpenAI(api_key=os.getenv('API_KEY'),base_url='https://api.deepseek.com'),'deepseek-chat')

from reasoning_eval.scorer.evaluator import evaluate_one
from reasoning_eval.dataset.graph_utils import normalize_graph
from reasoning_eval.model_test.prompt_builder import build_prompt,get_system_prompt

gs=[]
with open('data/processed/gsm8k/train_graphs_std.jsonl') as f:
    for l in f: gs.append(json.loads(l))
samples=random.sample(gs,N)

ok=n=0; results=[]
for s in samples:
    sid=s['id']; question=s['question']; ga=s['gold_answer']; g_raw=s['gold_reasoning_graph']
    t0=time.time()
    try:
        s['task_type']='math'
        resp=tc.generate(prompt=build_prompt(s),system=get_system_prompt(s),temperature=0.3,max_tokens=2048)
        resp_text=resp[0]
        graph=normalize_graph(g_raw)
        r=evaluate_one(
            {'id':sid,'gold_reasoning_graph':graph,'gold_answer':ga,'key_branch_nodes':[]},
            {'sample_id':sid,'model_name':model_label,'output_type':'cot','response':resp_text},
            mapper_client=mc)
        t=time.time()-t0
        lid=[n for n,st in r.lighted_graph['nodes'].items() if st=='lit']
        states={st:sum(1 for v in r.lighted_graph['nodes'].values() if v==st) for st in set(r.lighted_graph['nodes'].values())}
        n+=1
        if r.answer_correct: ok+=1
        results.append({
            'id':sid,'correct':r.answer_correct,'depth':r.score_depth,
            'cons':r.score_consistency,'lit':len(lid),'time':round(t,1),
            'states':states,'lit_nodes':lid,
            'response':resp_text,'lighted_graph':r.lighted_graph,
        })
        print(f"  [{sid}]: {'✓' if r.answer_correct else '✗'} d={r.score_depth:.0f} c={r.score_consistency:.0f} l={len(lid)} ({t:.0f}s)")
    except Exception as e:
        print(f"  [{sid}]: FAIL {str(e)[:80]}")
        results.append({'id':sid,'error':str(e)[:200]})

tag=model_label.replace('.','_').replace('-','_')
od='outputs/results';os.makedirs(od,exist_ok=True)
json.dump(results,open(f'{od}/bench_{tag}_gsm8k.json','w'),ensure_ascii=False,indent=2)
if n: print(f"[done] {tag}: acc={ok/n:.0%} avg_d={sum([r['depth'] for r in results if 'error' not in r])/n:.0f} ({n}/{N})")
