"""Prepare all manifest scenes with per-site splits and preserved source indices."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import defaultdict
import hashlib
import json
import numpy as np
from data.raw_cloud import sample_cloud
from data.annotation_regions import selection_mask
from data.dataset import write_jsonl
from utils.config import load_config


def prepare(manifest, raw_root, output, max_points=65536, negative_policy='unknown', seed=7, split_mode='holdout'):
    spec = load_config(manifest)
    raw_root, output = Path(raw_root).resolve(), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    sites = [s['id'] for s in spec['scenes']]
    if len(set(sites)) != len(sites): raise ValueError('Duplicate site IDs')
    splits = spec['splits'] if split_mode == 'holdout' else {'train':sites,'val':[],'test':[]}
    assigned = [site for group in splits.values() for site in group]
    if sorted(assigned) != sorted(sites): raise ValueError('Split sites must occur exactly once')
    queries, reports = [], []
    output.mkdir(parents=True,exist_ok=True)
    for entry in spec['scenes']:
        site = entry['id']; source = raw_root / entry['cloud']
        files = sorted((raw_root / entry['annotations']).glob('*.json'))
        if not files: raise ValueError(f'No annotations for {site}')
        print(f'Preparing {site}: {max_points} points',flush=True)
        xyz, color, indices, count, origin = sample_cloud(source,max_points,seed)
        block = 'whole_scene_sample'
        base = output/'scenes'/site/site/block; base.mkdir(parents=True)
        np.save(base/'coord.npy',(xyz-origin).astype(np.float32))
        np.save(base/'color.npy',color)
        np.save(base/'area_w.npy',np.ones(len(xyz),np.float32))
        np.save(base/'source_index.npy',indices)
        (base/'source.json').write_text(json.dumps(dict(path=str(source),original_count=count,retained_count=len(xyz),origin=origin.tolist(),seed=seed,area_mode='point',negative_policy=negative_policy),indent=2),encoding='utf-8')
        masks = output/'masks'/site/block; masks.mkdir(parents=True)
        report = dict(site=site,original_points=count,sampled_points=len(xyz),versions=[],empty_after_sampling=[],missing_text=[],duplicates=[],duplicate_text_groups=[])
        seen=set(); texts_by_query=defaultdict(list)
        for file in files:
            digest=hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in seen: report['duplicates'].append(str(file)); continue
            seen.add(digest)
            doc=json.loads(file.read_text(encoding='utf-8-sig'))
            if doc.get('schema')!='mine-3dvg-v1' or doc['source']['point_count']!=count:
                raise ValueError(f'Schema/source count mismatch: {file}')
            if doc['source'].get('size',source.stat().st_size)!=source.stat().st_size:
                raise ValueError(f'Source size mismatch: {file}')
            report['versions'].append(dict(file=str(file),sha256=digest,instances=len(doc['instances'])))
            for inst in doc['instances']:
                texts=list(dict.fromkeys(t.strip() for t in inst.get('descriptions',[]) if t.strip()))
                identity=[digest[:12],inst['id']]
                if not texts: report['missing_text'].append(identity); continue
                target=selection_mask(xyz,inst['operations'],doc['origin'])
                if not target.any(): report['empty_after_sampling'].append(identity); continue
                valid=np.ones(len(xyz),bool) if negative_policy=='assume-complete' else target.copy()
                for j,text in enumerate(texts):
                    qid=f'{site}_{digest[:12]}_{inst["id"]}_{j}'
                    np.savez_compressed(masks/(qid+'.npz'),target=target,valid=valid,ignore=~valid)
                    queries.append(dict(query_id=qid,site_id=site,scene_id=site,block_id=block,text=text,lang='zh',carrier_id=None,scope_type='unknown',verified=False,source_version=digest,source_instance=inst['id'],class_name=inst['class_name'],negative_policy=negative_policy))
                    texts_by_query[text].append(qid)
        report['duplicate_text_groups']=[dict(text=t,queries=ids) for t,ids in texts_by_query.items() if len(ids)>1]
        report['queries']=sum(len(v) for v in texts_by_query.values())
        reports.append(report)
        print(f'{site}: {report["queries"]} queries, {len(report["empty_after_sampling"])} targets missed by sampling',flush=True)
    ann=output/'annotations'; ann.mkdir()
    write_jsonl(ann/'queries.jsonl',queries); write_jsonl(ann/'pairs.jsonl',[])
    (ann/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'prepare_report.json').write_text(json.dumps(dict(scenes=reports,max_points=max_points,split_mode=split_mode,negative_policy=negative_policy),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='configs/data/three_sites.yaml')
    p.add_argument('--raw-root',default='datasets_raw')
    p.add_argument('--output',required=True)
    p.add_argument('--max-points',type=int,default=65536)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--split-mode',choices=['holdout','all-train'],default='holdout')
    p.add_argument('--negative-policy',choices=['unknown','assume-complete'],default='unknown')
    a=p.parse_args(); prepare(a.manifest,a.raw_root,a.output,a.max_points,a.negative_policy,a.seed,a.split_mode)
