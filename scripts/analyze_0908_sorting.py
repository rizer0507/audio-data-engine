"""Read-only audit of the 0908 exports and classified manifest; emit JSON to stdout."""
import collections
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
def norm(value):
    text = re.sub(r'<\|[^|]*\|>', '', value or '')
    text = unicodedata.normalize('NFKC', text)
    return ''.join(text.translate(str.maketrans('', '', '，。！？、；：""\'\'（）【】《》…—·,.!?;:\'"()[]{}')).split())

def main():
    out = {'files': {}, 'manifest': {}, 'examples': {}}
    sets = {}
    for path in sorted((ROOT / 'data/exports').glob('*0908*.xlsx')):
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        it = wb.active.values
        headers = next(it)
        counts = collections.defaultdict(collections.Counter)
        ids = set()
        n = 0
        for row in it:
            d = dict(zip(headers, row))
            if not d.get('sample_id'):
                continue
            n += 1
            ids.add(d['sample_id'])
            for k in ['type', 'review_priority', 'review_queue', 'requires_dual_review', 'reservation_role', 'decision', 'annotator_id']:
                if k in d:
                    counts[k][str(d[k])] += 1
        sets[path.name] = ids
        out['files'][path.name] = {'rows': n, 'unique_ids': len(ids), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'counts': dict(counts)}
        wb.close()
    base = sets['review_v3_0908-30000_blind.xlsx']
    p0 = sets['review_v3_0908-30000_blind_P0_only.xlsx']
    p1 = sets['review_v3_0908-30000_blind_P1_only.xlsx']
    out['reconciliation'] = {'p0_p1_overlap': len(p0 & p1), 'p0_p1_equals_full': p0 | p1 == base, 'p0_named_equals_full': sets['review_v3_0908-30000_blind_p0.xlsx'] == base}
    for path in (ROOT / 'data/exports').glob('*.meta.json'):
        meta = json.loads(path.read_text(encoding='utf-8'))
        out['files'][path.name] = {k: v for k,v in meta.items() if k != 'immutable_rows'}
    c = collections.defaultdict(collections.Counter)
    checks = collections.Counter()
    manifest_ids = set()
    manual_ids = set()
    path = ROOT / 'datasets/stage1/derived/classified_v3_0908-30000.jsonl'
    for line in path.open(encoding='utf-8'):
        d = json.loads(line)
        l, q = d['labels'], d['quality']
        typ = l.get('type')
        manifest_ids.add(d['id'])
        if l.get('review_queue') == 'manual_review': manual_ids.add(d['id'])
        for k in ['type','decision','review_priority','review_queue','dataset_role','short_utterance','teacher_consensus_status','qwen_correction_candidate','is_human_verified']:
            c[k][str(l.get(k))] += 1
        for k in ['noise_band','noise_risk','dnsmos_status','dnsmos_policy_digest']:
            c[k][str(q.get(k))] += 1
        for tag in l.get('risk_tags',[]):
            c['risk_tags'][tag] += 1
            c['tags_by_type'][typ + ':' + tag] += 1
        for tag in l.get('governance_flags',[]): c['governance_flags'][tag] += 1
        for fam,status in l.get('family_status',{}).items(): c['family_status'][fam+':'+status] += 1
        for alias,status in l.get('run_statuses',{}).items(): c['run_statuses'][alias+':'+status] += 1
        texts = []
        for alias in ['glm-asr-1','glm-asr-2','qwen3-asr-1','qwen3-asr-2','sensevoice-asr-1','sensevoice-asr-2']:
            entry = d['transcripts'].get(alias,{})
            texts.append(norm(entry.get('extra',{}).get('raw_text',entry.get('text',''))))
        same = len(set(texts)) == 1 and bool(texts[0])
        if same: c['six_equal_by_type'][typ] += 1
        if any('\ufffd' in t for t in texts): checks['replacement_char_samples'] += 1
        if l.get('candidate_text'): c['candidate_nonempty_by_type'][typ] += 1
        if typ == 'audio_quality_risk' and l.get('teacher_consensus_status') == 'stable_consensus' and l.get('qwen_vs_teacher_similarity_1') == 1 and l.get('qwen_vs_teacher_similarity_2') == 1:
            checks['quality_only_teacher_qwen_exact'] += 1
        if l.get('review_priority') == 'P0' and same:
            checks['p0_six_equal'] += 1
        if l.get('short_utterance') and (d.get('duration') or 0) > 2: checks['short_but_duration_gt2'] += 1
        if typ == 'implausible_speech_rate':
            for a in l.get('implausible_routes',[]): c['implausible_routes'][str(a)] += 1
            if texts[2] and texts[2] == texts[3] == texts[4] == texts[5]: checks['excluded_qwen_sensevoice_exact_nonempty'] += 1
        key = ('p0_six_equal' if same and l.get('review_priority') == 'P0' else 'quality_six_equal' if same and typ == 'audio_quality_risk' else typ)
        if len(out['examples'].get(key,[])) < 3:
            out['examples'].setdefault(key,[]).append({'id':d['id'],'duration':d.get('duration'),'texts':[t[:180] for t in texts], 'type':typ,'tags':l.get('risk_tags'), 'family_status':l.get('family_status')})
    out['manifest'] = {'counts':dict(c), 'checks':checks,'unique_ids':len(manifest_ids),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    out['reconciliation']['manual_ids_equal_export'] = manual_ids == base
    out['reconciliation']['summary_equals_manifest'] = (sets['summary_v3_0908-30000-part-001.xlsx'] | sets['summary_v3_0908-30000-part-002.xlsx']) == manifest_ids
    assert out['reconciliation']['p0_p1_overlap'] == 0
    assert all(v for k,v in out['reconciliation'].items() if k != 'p0_p1_overlap')
    assert len(manifest_ids) == 30000
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
