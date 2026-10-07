import json, collections, pathlib
root=pathlib.Path(__file__).resolve().parents[1]/'data'
out=root/'checkpoint1';out.mkdir(exist_ok=True)
results={}
for path in sorted(root.glob('*/*/investigations.jsonl')):
 counts=collections.Counter();examples={};seen=set()
 def flag(reason,item):
  counts[reason]+=1
  if len(examples.setdefault(reason,[]))<3:examples[reason].append(item)
 scores={}
 for line in path.with_name('verification.jsonl').open():
  v=json.loads(line);pid=v.get('prompt_id')
  if pid in scores:flag('duplicate_verification_id',pid)
  scores[pid]=v.get('score')
 for line in path.open():
  r=json.loads(line);pid=r.get('prompt_id');counts['investigations']+=1
  if pid in seen:flag('duplicate_investigation_id',pid)
  seen.add(pid)
  score=scores.get(pid)
  if score is None:flag('missing_verification',pid)
  elif not isinstance(score,(int,float)) or not 1<=score<=10:flag('invalid_verification',{'id':pid,'score':score})
  elif score>=7:counts['verification_ge7']+=1
  sf=r.get('structured_findings')
  if not isinstance(sf,dict):flag('missing_structured_findings',pid);continue
  if not (r.get('behavior_summary') or sf.get('baseline_behavior')):flag('missing_behaviour_text',pid)
  summaries=sf.get('experiment_summaries');logs=r.get('experiment_log')
  if not isinstance(summaries,list):flag('invalid_experiment_summaries',{'id':pid,'type':type(summaries).__name__});continue
  if not isinstance(logs,list):flag('invalid_experiment_log',pid);continue
  logids=[e.get('exp_num') for e in logs];sumids=[e.get('experiment_number') for e in summaries]
  if len(set(logids))!=len(logids):flag('duplicate_log_experiment_ids',pid)
  if len(set(sumids))!=len(sumids):flag('duplicate_summary_experiment_ids',pid)
  for s in summaries:
   counts['summaries']+=1
   missing=[k for k in ['experiment_number','summary','baseline_rate','counterfactual_rate','is_key'] if k not in s]
   if missing:flag('missing_summary_fields',{'id':pid,'exp':s.get('experiment_number'),'missing':missing})
   if not isinstance(s.get('summary'),str) or not s['summary'].strip():flag('missing_edit_description',{'id':pid,'exp':s.get('experiment_number')})
   if s.get('experiment_number') not in logids:flag('summary_without_log',{'id':pid,'exp':s.get('experiment_number'),'score':score})
  for e in logs:
   counts['logged_experiments']+=1
   if e.get('exp_num') not in sumids:flag('log_without_summary',{'id':pid,'exp':e.get('exp_num'),'score':score})
   labels=e.get('grader_verdicts')
   if not isinstance(labels,list) or not labels:flag('missing_experiment_labels',{'id':pid,'exp':e.get('exp_num')})
   elif any(type(v) is not bool for v in labels):flag('non_boolean_experiment_labels',{'id':pid,'exp':e.get('exp_num'),'sample':labels[:5]})
  labels=r.get('baseline_grader_verdicts')
  if not isinstance(labels,list) or not labels:flag('missing_baseline_labels',pid)
  elif any(type(v) is not bool for v in labels):flag('non_boolean_baseline_labels',{'id':pid,'sample':labels[:5]})
 results[path.parent.name]={'counts':dict(counts),'examples':examples}
 print(path.parent.name,json.dumps(dict(counts)),flush=True)
(out/'schema_audit.json').write_text(json.dumps(results,indent=2)+'\n')
