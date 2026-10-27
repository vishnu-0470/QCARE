"""Fail closed if verified raw data, preparation, or checkpoint inputs changed."""
from pathlib import Path
import hashlib,json
from .data import DS1,DS2,CLASSES

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def check_file(path,expected):
 p=Path(path)
 if not p.is_file():raise FileNotFoundError(f'BLOCKED: missing required file {p}')
 actual=digest(p)
 if actual!=expected:raise ValueError(f'Provenance FAIL: SHA256 changed for {p}')
 return actual

def verify_dataset(root):
 root=Path(root);manifest=json.loads((root/'reports/dataset_manifest.json').read_text())
 required=set(DS1+DS2);rows=manifest['records']
 if manifest['valid_records']!=44 or len(rows)!=44 or {x['record'] for x in rows}!=required:raise ValueError('Complete verified conventional 44-record manifest required')
 if manifest['class_order']!=CLASSES:raise ValueError('Manifest class order differs')
 for row in rows:
  if row['status']!='PASS':raise ValueError('Invalid source record '+row['record'])
  if row['split']!=('DS1' if row['record'] in DS1 else 'DS2'):raise ValueError('Source split differs')
  for ext in ['hea','dat','atr']:
   e=row['files'][ext]
   if not e['publisher_hash_match']:raise ValueError('Publisher checksum not verified')
   check_file(root/'data'/(row['record']+'.'+ext),e['sha256'])
 return manifest

def verify_prepared(root):
 root=Path(root);verify_dataset(root);cache=root/'experiments/cache';p=cache/'preparation.json'
 if not p.is_file():raise FileNotFoundError('BLOCKED: regenerate cache with python python/prepare_dataset.py')
 prep=json.loads(p.read_text());check_file(root/'reports/dataset_manifest.json',prep['manifest_sha256'])
 if prep['class_order']!=CLASSES or prep['DS1']!=DS1 or prep['DS2']!=DS2:raise ValueError('Prepared class order or protocol changed')
 train=set(prep['train_records']);val=set(prep['validation_records'])
 if train&val or train|val!=set(DS1) or (train|val)&set(DS2):raise ValueError('Invalid fit/validation/test record isolation')
 for name,expected in prep['cache_files'].items():check_file(cache/name,expected)
 return prep

def verify_checkpoint_preparation(checkpoint,prep):
 saved=checkpoint['preparation']
 for key in ['manifest_sha256','cache_files','train_records','validation_records','DS1','DS2','class_order','cwt','beat_window']:
  if saved[key]!=prep[key]:raise ValueError('Checkpoint preparation differs: '+key)
 return True
