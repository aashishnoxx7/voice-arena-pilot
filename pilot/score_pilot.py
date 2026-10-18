"""Score pilot hypotheses with pyannote.metrics (components) and dscore (DER/JER
cross-check); run collar/overlap sensitivity and a paired cluster bootstrap."""
import csv, json, os, re, subprocess, sys
import numpy as np
import pyannote.metrics
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

PILOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(PILOT, "ds")
DSCORE = os.path.abspath(sys.argv[1])
SYSTEMS = ["A_wespeaker_resnet34__auto", "B_nemo_titanet_small__auto",
           "C_3dspeaker_campplus__auto", "A_wespeaker_resnet34__oracle"]
URIS = ["dev00", "dev01", "tst00", "tst01"] + [f"trn0{i}" for i in range(10)]
# Clips sharing reference speaker IDs are treated as coming from the same meeting
# (an assumption: the repository does not document source meetings).
GROUPS = {"dev00": "G1", "dev01": "G1", "tst00": "G2", "tst01": "G2",
          "trn00": "G3", "trn01": "G3", "trn02": "G3", "trn03": "G3",
          "trn04": "G4", "trn05": "G5", "trn06": "G6", "trn09": "G6",
          "trn07": "G7", "trn08": "G7"}


def load(path):
    ann = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if not p:
            continue
        a = ann.setdefault(p[1], Annotation(uri=p[1]))
        s, d = float(p[3]), float(p[4])
        a[Segment(s, s + d), len(a)] = p[7]
    return ann


ref = load(os.path.join(DS, "ref.rttm"))
uem = Timeline([Segment(0.0, 30.0)])

rows, comp = [], {}
for sysname in SYSTEMS:
    hyp = load(os.path.join(PILOT, "out", f"{sysname}.rttm"))
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    for u in URIS:
        h = hyp.get(u, Annotation(uri=u))  # empty output = everything missed
        d = metric(ref[u], h, uem=uem, detailed=True)
        r = {"system": sysname, "uri": u, "group": GROUPS[u],
             "ref_speech_s": round(d["total"], 3), "miss_s": round(d["missed detection"], 3),
             "fa_s": round(d["false alarm"], 3), "conf_s": round(d["confusion"], 3),
             "der_pct": round(100 * d["diarization error rate"], 2),
             "n_ref_spk": len(ref[u].crop(Segment(0, 30)).labels()),
             "n_hyp_spk": len(h.crop(Segment(0, 30)).labels())}
        rows.append(r)
    tot = {k: sum(r[k] for r in rows if r["system"] == sysname)
           for k in ["ref_speech_s", "miss_s", "fa_s", "conf_s"]}
    per = [r["der_pct"] for r in rows if r["system"] == sysname]
    comp[sysname] = {
        "pooled_der_pct": round(100 * (tot["miss_s"] + tot["fa_s"] + tot["conf_s"]) / tot["ref_speech_s"], 2),
        "pooled_miss_pct": round(100 * tot["miss_s"] / tot["ref_speech_s"], 2),
        "pooled_fa_pct": round(100 * tot["fa_s"] / tot["ref_speech_s"], 2),
        "pooled_conf_pct": round(100 * tot["conf_s"] / tot["ref_speech_s"], 2),
        "mean_clip_der_pct": round(float(np.mean(per)), 2),
        "pyannote_abs_pooled_pct": round(100 * abs(metric), 2),
        "ref_speech_total_s": round(tot["ref_speech_s"], 2)}


def dscore(sysname, extra):
    out = subprocess.run([sys.executable, os.path.join(DSCORE, "score.py"), "-r", "ref.rttm",
                          "-s", os.path.join(PILOT, "out", f"{sysname}.rttm"), "-u", "all.uem"] + extra,
                         cwd=DS, capture_output=True, text=True).stdout
    m = re.search(r"\*\*\* OVERALL \*\*\*\s+([\d.]+)\s+([\d.]+)", out)
    return float(m.group(1)), float(m.group(2))


sens = {}
for s in SYSTEMS:
    der0, jer0 = dscore(s, [])
    der_c, _ = dscore(s, ["--collar", "0.25"])
    der_co, _ = dscore(s, ["--collar", "0.25", "--ignore_overlaps"])
    sens[s] = {"dscore_der_c0_ovl": der0, "dscore_jer_c0_ovl": jer0,
               "dscore_der_c025_ovl": der_c, "dscore_der_c025_noovl": der_co}


def pooled(sysname, uris):
    sel = [r for r in rows if r["system"] == sysname and r["uri"] in uris]
    err = sum(r["miss_s"] + r["fa_s"] + r["conf_s"] for r in sel)
    return 100 * err / sum(r["ref_speech_s"] for r in sel)


def paired_bootstrap(cand, inc, unit="group", B=10000, seed=7):
    rng = np.random.default_rng(seed)
    units = sorted(set(GROUPS.values())) if unit == "group" else URIS
    members = {g: [u for u in URIS if (GROUPS[u] if unit == "group" else u) == g] for g in units}
    obs = pooled(cand, URIS) - pooled(inc, URIS)
    deltas = []
    for _ in range(B):
        pick = rng.choice(units, size=len(units), replace=True)
        cand_err = cand_ref = inc_err = 0.0
        for g in pick:
            for u in members[g]:
                rc = next(r for r in rows if r["system"] == cand and r["uri"] == u)
                ri = next(r for r in rows if r["system"] == inc and r["uri"] == u)
                cand_err += rc["miss_s"] + rc["fa_s"] + rc["conf_s"]
                inc_err += ri["miss_s"] + ri["fa_s"] + ri["conf_s"]
                cand_ref += rc["ref_speech_s"]
        deltas.append(100 * (cand_err - inc_err) / cand_ref)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"candidate": cand, "incumbent": inc, "unit": unit, "n_units": len(units),
            "delta_pooled_der_pp": round(obs, 2), "ci95": [round(lo, 2), round(hi, 2)], "B": B}


boot = [paired_bootstrap(c, "A_wespeaker_resnet34__auto", unit)
        for c in ["B_nemo_titanet_small__auto", "C_3dspeaker_campplus__auto"]
        for unit in ["group", "clip"]]

with open(os.path.join(PILOT, "results_per_clip.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
json.dump({"pyannote_metrics": pyannote.metrics.__version__, "summary": comp,
           "sensitivity": sens, "bootstrap": boot},
          open(os.path.join(PILOT, "results_summary.json"), "w"), indent=1)
print(json.dumps({"summary": comp, "sensitivity": sens, "bootstrap": boot}, indent=1))
