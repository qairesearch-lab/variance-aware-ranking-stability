#!/usr/bin/env python3
"""Offline visual review gallery for pHash candidate pairs.

Only sample IDs, pHash distance and faithful aspect-preserving image thumbnails
are exposed. Diagnosis, group/patient IDs and model outcomes are omitted.
No remote scripts, network calls, or external image uploads are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from finalize_duplicate_audit import candidate_id

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DATASETS = ("isic2019", "mura")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def archive_path(dataset: str) -> Path:
    if dataset == "isic2019":
        return ROOT / "research-lab/data/isic2019/raw/ISIC_2019_Training_Input.zip"
    if dataset == "mura":
        return ROOT / "research-lab/data/mura/raw/MURA-v1.1_files.zip"
    raise ValueError(dataset)


def thumbnail_name(dataset: str, sample_id: str) -> str:
    digest = hashlib.sha256((dataset + "\0" + sample_id).encode("utf-8")).hexdigest()[:24]
    return f"{dataset}_{digest}.jpg"


def make_thumbnails(rows, candidate_index_root: Path, thumb_dir: Path):
    required = {dataset: set() for dataset in DATASETS}
    for row in rows:
        required[row["dataset"]].update((row["sample_a"], row["sample_b"]))
    output = {}
    for dataset in DATASETS:
        by_id = {row["sample_id"]: row["archive_member"] for row in read_csv(candidate_index_root / f"{dataset}_dataset_index_candidate.csv")}
        missing = required[dataset] - set(by_id)
        if missing:
            raise ValueError(f"Candidate image ID absent from {dataset} index: {sorted(missing)[:3]}")
        with zipfile.ZipFile(archive_path(dataset)) as archive:
            for position, sample_id in enumerate(sorted(required[dataset]), 1):
                with Image.open(BytesIO(archive.read(by_id[sample_id]))) as source:
                    image = source.convert("RGB")
                image.thumbnail((360, 240), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (360, 240), (18, 18, 18))
                canvas.paste(image, ((360 - image.width) // 2, (240 - image.height) // 2))
                name = thumbnail_name(dataset, sample_id)
                canvas.save(thumb_dir / name, "JPEG", quality=88, optimize=True)
                output[(dataset, sample_id)] = "thumbs/" + name
                if position % 500 == 0:
                    print(f"{dataset}: rendered {position}/{len(required[dataset])} thumbnails", flush=True)
    return output


def render_html(public_rows, internal_rows, images, output: Path, max_distance: int):
    cards = []
    for row in internal_rows:
        dataset = row["dataset"]
        candidate_id = html.escape(row["candidate_id"])
        image_a = html.escape(images[(dataset, row["sample_a"])])
        image_b = html.escape(images[(dataset, row["sample_b"])])
        cards.append(f'''
<article class="pair" data-id="{candidate_id}">
  <header><code>{candidate_id}</code> <span>{dataset} · pHash distance {row["phash_hamming_distance"]}</span></header>
  <div class="images">
    <figure><img loading="lazy" src="{image_a}" alt="Candidate A"><figcaption>A</figcaption></figure>
    <figure><img loading="lazy" src="{image_b}" alt="Candidate B"><figcaption>B</figcaption></figure>
  </div>
  <div class="choices">
    <button type="button" data-choice="same_source">Same image/source</button>
    <button type="button" data-choice="different_source">Different</button>
    <button type="button" data-choice="">Clear</button>
    <input type="text" class="rationale" placeholder="Optional note">
  </div>
</article>''')
    # No labels/group IDs are present anywhere in the HTML or exported CSV.
    payload = json.dumps(public_rows, ensure_ascii=False).replace("</", "<\\/")
    content = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded near-duplicate review</title>
<style>
body{margin:0 auto;max-width:1100px;padding:20px;font:16px/1.45 system-ui,sans-serif;background:#f4f6f8;color:#17212b}
h1{margin:.2em 0}p{max-width:80ch}.toolbar{position:sticky;top:0;background:#eef3fa;padding:12px;border:1px solid #ccd8e8;z-index:3;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.toolbar input{padding:6px}.pair{background:white;border:1px solid #cad2dc;border-radius:10px;margin:18px 0;padding:12px}.pair header{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
.images{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}.images figure{margin:0;text-align:center}.images img{width:min(100%,360px);height:240px;object-fit:contain;background:#111}
figcaption{overflow-wrap:anywhere;font-size:13px}.choices{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.choices button{padding:8px 10px;border:1px solid #8e9cab;border-radius:6px;background:white;cursor:pointer}
.choices button.selected{background:#1664b4;color:white}.choices input{min-width:220px;flex:1;padding:8px;border:1px solid #8e9cab;border-radius:6px}
@media(max-width:650px){.images{grid-template-columns:1fr}}
</style></head><body>
<h1>Blinded near-duplicate review</h1>
<p>Confirm a pair only when it is the same clinical image/acquisition or unmistakably the same lesion. Similar body region alone is insufficient. If uncertain, leave the decision blank and record a note for adjudication. Diagnosis, original filenames, and patient/lesion IDs are hidden. Do not infer diagnosis. The gallery runs offline; selections are also saved in this browser, but export a CSV after every review session. Import that CSV to resume on another browser or machine. Neither the gallery nor an incomplete CSV is a final dataset audit.</p>
<div class="toolbar"><label>Reviewer <input id="reviewer" autocomplete="name" placeholder="Name or initials"></label><strong id="progress"></strong><button id="export" type="button">Export CSV</button><label>Resume from exported CSV <input id="import" type="file" accept=".csv,text/csv"></label></div>
<main>__CARDS__</main>
<script>
const rows=__ROWS__;
const stateKey='jiim-near-review-v3';
const saved=JSON.parse(localStorage.getItem(stateKey)||'{}');
const reviewerInput=document.getElementById('reviewer');
reviewerInput.value=localStorage.getItem(stateKey+'-reviewer')||'';
reviewerInput.addEventListener('change',()=>localStorage.setItem(stateKey+'-reviewer',reviewerInput.value));
function refresh(card){const id=card.dataset.id;const entry=saved[id]||{};card.querySelectorAll('button[data-choice]').forEach(button=>button.classList.toggle('selected',button.dataset.choice===entry.decision && !!entry.decision));card.querySelector('.rationale').value=entry.rationale||'';}
function progress(){let count=rows.filter(row=>saved[row.candidate_id] && saved[row.candidate_id].decision).length;document.getElementById('progress').textContent=count+' / '+rows.length+' decided';}
document.querySelectorAll('.pair').forEach(card=>{refresh(card);card.querySelectorAll('button[data-choice]').forEach(button=>button.addEventListener('click',()=>{const id=card.dataset.id;const decision=button.dataset.choice;const reviewer=reviewerInput.value.trim();if(decision&&!reviewer){alert('Enter reviewer initials before deciding');return;}saved[id]={...(saved[id]||{}),decision,reviewer:decision?reviewer:'',reviewed_utc:decision?new Date().toISOString():''};localStorage.setItem(stateKey,JSON.stringify(saved));refresh(card);progress();}));card.querySelector('.rationale').addEventListener('change',event=>{const id=card.dataset.id;saved[id]={...(saved[id]||{}),rationale:event.target.value};localStorage.setItem(stateKey,JSON.stringify(saved));});});progress();
function csvCell(value){return '"'+String(value??'').replaceAll('"','""')+'"';}
const cols=['candidate_id','dataset','phash_hamming_distance','decision','reviewer','reviewed_utc','rationale'];
document.getElementById('export').addEventListener('click',()=>{const lines=[cols.join(',')];for(const row of rows){const entry=saved[row.candidate_id]||{};const out={...row,decision:entry.decision||'',reviewer:entry.reviewer||'',reviewed_utc:entry.reviewed_utc||'',rationale:entry.rationale||''};lines.push(cols.map(key=>csvCell(out[key])).join(','));}const blob=new Blob([lines.join('\\n')+'\\n'],{type:'text/csv;charset=utf-8'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='near_pair_blinded_review_export.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);});
function parseCsv(source){const records=[];let record=[],value='',quoted=false;for(let i=0;i<source.length;i++){const ch=source[i];if(quoted){if(ch==='"'&&source[i+1]==='"'){value+='"';i++;}else if(ch==='"'){quoted=false;}else{value+=ch;}}else if(ch==='"'){if(value)throw Error('Invalid quote');quoted=true;}else if(ch===','){record.push(value);value='';}else if(ch==='\\n'){record.push(value.replace(/\\r$/,''));records.push(record);record=[];value='';}else{value+=ch;}}if(quoted)throw Error('Unclosed quote');if(record.length||value){record.push(value);records.push(record);}return records;}
document.getElementById('import').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{const records=parseCsv(await file.text());if(JSON.stringify(records[0])!==JSON.stringify(cols)||records.length!==rows.length+1)throw Error('CSV schema or row count differs');const incoming={};for(let i=0;i<rows.length;i++){const fields=records[i+1];if(fields.length!==cols.length)throw Error('Malformed CSV row '+(i+2));const out=Object.fromEntries(cols.map((key,j)=>[key,fields[j]]));const expected=rows[i];for(const key of ['candidate_id','dataset','phash_hamming_distance'])if(out[key]!==expected[key])throw Error('Candidate identity/order mismatch at row '+(i+2));if(!['','same_source','different_source'].includes(out.decision))throw Error('Invalid decision at row '+(i+2));if(out.decision&&(!out.reviewer||!out.reviewed_utc))throw Error('Reviewer/time missing at row '+(i+2));const old=saved[out.candidate_id];if(old&&old.decision&&out.decision&&old.decision!==out.decision)throw Error('Conflicting browser/CSV decisions for '+out.candidate_id);incoming[out.candidate_id]={decision:out.decision,reviewer:out.reviewer,reviewed_utc:out.reviewed_utc,rationale:out.rationale};}for(const [id,entry] of Object.entries(incoming))if(entry.decision||entry.rationale)saved[id]=entry;localStorage.setItem(stateKey,JSON.stringify(saved));document.querySelectorAll('.pair').forEach(refresh);progress();alert('Review CSV imported; export a new backup after this session.');}catch(error){alert('Import refused: '+error.message);}finally{event.target.value='';}});
</script></body></html>'''
    content = content.replace("__CARDS__", "\n".join(cards)).replace("__ROWS__", payload)
    output.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-template", type=Path, default=HERE / "candidate_indices/duplicate_candidate_audit_full_v1/near_pair_blinded_review_template.csv")
    parser.add_argument("--candidate-index-root", type=Path, default=HERE / "candidate_indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=int, default=4)
    args = parser.parse_args()
    if args.max_distance not in (0, 1, 2, 3, 4):
        raise ValueError("pHash distance filter must be between 0 and 4")
    if args.output_dir.exists():
        raise FileExistsError(f"Will not overwrite review gallery: {args.output_dir}")
    all_rows = read_csv(args.review_template)
    if not all_rows or any(row["decision"] for row in all_rows):
        raise ValueError("Expected a blank blinded review template")
    rows = [row for row in all_rows if int(row["phash_hamming_distance"]) <= args.max_distance]
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate ID")
    internal_by_id = {}
    audit_root = args.review_template.parent
    for dataset in DATASETS:
        for pair in read_csv(audit_root / f"{dataset}_near_pairs.csv"):
            internal_by_id[candidate_id(dataset, pair)] = pair
    internal_rows = []
    for row in rows:
        pair = internal_by_id.get(row["candidate_id"])
        if pair is None or pair["dataset"] != row["dataset"] or pair["phash_hamming_distance"] != row["phash_hamming_distance"]:
            raise ValueError(f"Review template/audit mismatch: {row['candidate_id']}")
        internal_rows.append({**pair, "candidate_id": row["candidate_id"]})
    args.output_dir.mkdir(parents=True)
    thumb_dir = args.output_dir / "thumbs"
    thumb_dir.mkdir()
    images = make_thumbnails(internal_rows, args.candidate_index_root, thumb_dir)
    render_html(rows, internal_rows, images, args.output_dir / "index.html", args.max_distance)
    summary = {"status": "blinded_review_gallery_not_adjudication", "max_distance": args.max_distance, "pairs": len(rows), "unique_thumbnails": len(images), "html": "index.html"}
    (args.output_dir / "gallery_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
