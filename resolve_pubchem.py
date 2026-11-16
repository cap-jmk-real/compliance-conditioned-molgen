"""Resolve list entries, group members and appendix substances to PubChem structures.

For every item: look up PubChem CIDs by each CAS number (exact synonym
match), and separately by name. CAS is the primary route. The name result
is used as a fallback when CAS gives nothing, and as a cross-check when
both give an answer. InChIKey is the canonical structure key.

All HTTP responses are cached in cache/pubchem_cache.json. The repository
holds a gzipped copy (cache/pubchem_cache.json.gz), so a fresh checkout
re-runs offline and gives the same tables. Output: data/resolution.csv and
data/structures.csv (one row per CID seen).
"""
import csv, gzip, json, pathlib, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

HERE = pathlib.Path(__file__).resolve().parent
DATA, CACHE = HERE / "data", HERE / "cache"
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CACHE_FILE = CACHE / "pubchem_cache.json"
CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
MIN_INTERVAL = 0.22  # PubChem asks for at most 5 requests per second

_cache, _last = {}, [0.0]
_rate_lock, _cache_lock = threading.Lock(), threading.Lock()
WORKERS = 4


def _save():
    CACHE.mkdir(exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp")
    with _cache_lock:
        snapshot = json.dumps(_cache.copy(), sort_keys=True)  # dict.copy is atomic under the GIL
    tmp.write_text(snapshot)
    tmp.replace(CACHE_FILE)


def _request(key, method, url, data=None):
    if key in _cache:
        return _cache[key]
    for attempt in range(6):
        with _rate_lock:  # global limit across worker threads
            wait = MIN_INTERVAL - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = requests.request(method, url, data=data, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:  # PUGREST.NotFound: no match
            _cache[key] = {"status": 404, "json": None}
            return _cache[key]
        if r.status_code in (400,):  # bad request, e.g. unparsable name
            _cache[key] = {"status": r.status_code, "json": None}
            return _cache[key]
        if r.ok:
            _cache[key] = {"status": 200, "json": r.json()}
            return _cache[key]
        time.sleep(2 ** attempt)  # 503 server busy etc.
    return {"status": "error", "json": None}  # not cached, retried next run


def cids_by_name(text):
    res = _request(f"name:{text}", "POST", f"{BASE}/compound/name/cids/JSON", data={"name": text})
    if res["json"] is None:
        return [], res["status"]
    return res["json"].get("IdentifierList", {}).get("CID", []), res["status"]


def properties(cids):
    out = {}
    todo = [c for c in cids if f"prop:{c}" not in _cache]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        res = _request(f"propbatch:{','.join(map(str, chunk))}", "POST",
                       f"{BASE}/compound/cid/property/InChIKey,SMILES,ConnectivitySMILES,MolecularFormula,Charge/JSON",
                       data={"cid": ",".join(map(str, chunk))})
        for p in (res["json"] or {}).get("PropertyTable", {}).get("Properties", []):
            _cache[f"prop:{p['CID']}"] = p
    for c in cids:
        if f"prop:{c}" in _cache:
            out[c] = _cache[f"prop:{c}"]
    return out


def name_variants(name):
    """The name as given, then without a trailing abbreviation in brackets."""
    name = re.sub(r"\s+", " ", name or "").strip()
    if not name or len(name) > 400:
        return []
    out = [name]
    stripped = re.sub(r"\s*\([^()]{1,25}\)\s*$", "", name).strip()
    if stripped and stripped != name:
        out.append(stripped)
    # Appendix and list names often join synonyms with ";" or mark parts with [1], [2].
    # Try the first segment, with "(ISO)"/"(INN)" tags removed.
    first = re.split(r";|\[\d\]", name)[0]
    first = re.sub(r"\((ISO|INN)\)", "", first).strip(" ,")
    if first and first not in out:
        out.append(first)
    return out


def items():
    """Everything to resolve, with a stable item_id."""
    def rows(p):
        with open(p, newline="") as fh:
            return list(csv.DictReader(fh))
    for r in rows(DATA / "entries.csv"):
        yield {"item_id": r["entry_id"], "kind": "entry", "list": r["list"], "name": r["name"], "cas": r["cas"], "ec": r["ec"]}
    for r in rows(DATA / "members.csv"):
        yield {"item_id": f"{r['entry_id']}/m{int(r['member_no']):03d}", "kind": "member",
               "list": r["entry_id"].split(":")[0], "name": r["name"], "cas": r["cas"], "ec": r["ec"]}
    for r in rows(DATA / "annex_xvii_appendix_substances.csv"):
        yield {"item_id": f"annex_xvii_appendix_{r['appendix']}:{r['index_no']}", "kind": "appendix_substance",
               "list": f"annex_xvii_appendix_{r['appendix']}", "name": r["name"], "cas": r["cas"].replace(";", ","), "ec": r["ec"]}


def main():
    global _cache
    if CACHE_FILE.exists():
        _cache = json.loads(CACHE_FILE.read_text())
    elif CACHE_FILE.with_suffix(".json.gz").exists():  # the committed copy
        _cache = json.loads(gzip.decompress(CACHE_FILE.with_suffix(".json.gz").read_bytes()))
    def resolve(it):
        cas_list = CAS_RE.findall(it["cas"])
        cas_cids, cas_status = [], []
        for cas in cas_list:
            c, st = cids_by_name(cas)
            cas_cids += c; cas_status.append(str(st))
        name_cids, used_name = [], ""
        for v in name_variants(it["name"]):
            c, st = cids_by_name(v)
            if c:
                name_cids, used_name = c, v
                break
        allc = sorted(set(cas_cids) | set(name_cids))
        props = properties(allc)
        keys = lambda cs: sorted({props[c]["InChIKey"] for c in cs if c in props})
        cas_keys, name_keys = keys(cas_cids), keys(name_cids)
        if cas_keys:
            route, chosen = "cas", cas_keys
        elif name_keys:
            route, chosen = "name", name_keys
        else:
            route, chosen = "none", []
        if cas_keys and name_keys:
            agree = "same" if set(cas_keys) & set(name_keys) else (
                "same_skeleton" if {k[:14] for k in cas_keys} & {k[:14] for k in name_keys} else "conflict")
        else:
            agree = ""
        return {**it, "n_cas": len(cas_list), "cas_http": ";".join(cas_status),
                "cas_cids": ";".join(map(str, sorted(set(cas_cids)))),
                "name_cids": ";".join(map(str, sorted(set(name_cids)))), "name_used": used_name,
                "route": route, "n_inchikeys": len(chosen), "inchikeys": ";".join(chosen),
                "cas_name_agreement": agree}

    todo = list(items())
    results = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for n, res in enumerate(pool.map(resolve, todo), 1):
            results.append(res)
            if n % 100 == 0:
                _save(); print(n, "of", len(todo), "items", flush=True)
    _save()
    cols = list(results[0])
    with open(DATA / "resolution.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(results)
    seen = sorted({int(c) for r in results for f in ("cas_cids", "name_cids") for c in r[f].split(";") if c})
    with open(DATA / "structures.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["CID", "InChIKey", "SMILES", "ConnectivitySMILES", "MolecularFormula", "Charge"], extrasaction="ignore")
        w.writeheader()
        for c in seen:
            if f"prop:{c}" in _cache:
                w.writerow(_cache[f"prop:{c}"])
    print("done", len(results), "items,", len(seen), "CIDs")

if __name__ == "__main__":
    sys.exit(main())
