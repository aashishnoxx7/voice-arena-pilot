"""Voice Arena pilot: run sherpa-onnx diarization configurations on AMI-derived
excerpts shipped in the pyannote-audio test suite, and write RTTM hypotheses.

Usage: python3 run_pilot.py <data_dir> <models_dir> <out_dir>
"""
import json, os, sys, time, platform
import numpy as np
import soundfile as sf
import sherpa_onnx

DATA, MODELS, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)

# uri -> wav filename (one file name in the repo contains a non-ASCII character)
CLIPS = {u: f"{u}.wav" for u in
         ["dev00", "dev01", "tst00", "tst01"] + [f"trn0{i}" for i in range(1, 10)]}
CLIPS["trn00"] = "trñ00.wav"

SEG = os.path.join(MODELS, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB = {
    "A_wespeaker_resnet34": "wespeaker_en_voxceleb_resnet34.onnx",
    "B_nemo_titanet_small": "nemo_en_titanet_small.onnx",
    "C_3dspeaker_campplus": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
}
THRESHOLD = 0.5          # sherpa-onnx example default; NOT tuned on these clips
MIN_ON, MIN_OFF = 0.3, 0.5


def ref_speaker_count(uri):
    """Number of reference speakers inside the scored region (UEM 0-30 s)."""
    spk = set()
    for f in ["debug.train.rttm", "debug.development.rttm", "debug.test.rttm"]:
        for line in open(os.path.join(DATA, f), encoding="utf-8"):
            p = line.split()
            if p[1] == uri and float(p[3]) < 30.0:
                spk.add(p[7])
    return len(spk)


def build(emb_file, num_clusters):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG),
            num_threads=2),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=os.path.join(MODELS, emb_file), num_threads=2),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters, threshold=THRESHOLD),
        min_duration_on=MIN_ON, min_duration_off=MIN_OFF)
    assert cfg.validate(), "invalid config"
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


runs = [(name, f, -1, "auto") for name, f in EMB.items()]
runs.append(("A_wespeaker_resnet34", EMB["A_wespeaker_resnet34"], None, "oracle"))

log = {"sherpa_onnx": sherpa_onnx.__version__, "python": platform.python_version(),
       "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "threshold": THRESHOLD, "min_duration_on": MIN_ON, "min_duration_off": MIN_OFF,
       "runs": []}

for name, emb, k, track in runs:
    sysname = f"{name}__{track}"
    lines, fails = [], []
    for uri, fn in CLIPS.items():
        audio, sr = sf.read(os.path.join(DATA, fn), dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        kk = ref_speaker_count(uri) if track == "oracle" else -1
        sd = build(emb, kk)
        t0 = time.time()
        try:
            res = sd.process(audio).sort_by_start_time()
        except Exception as e:  # record, do not hide
            fails.append({"uri": uri, "error": repr(e)}); continue
        for s in res:
            lines.append(f"SPEAKER {uri} 1 {s.start:.3f} {s.end - s.start:.3f} "
                         f"<NA> <NA> spk{s.speaker} <NA> <NA>")
        log["runs"].append({"system": sysname, "uri": uri, "num_clusters": kk,
                            "n_segments": len(res), "sec": round(time.time() - t0, 2)})
    with open(os.path.join(OUT, f"{sysname}.rttm"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.setdefault("failures", {})[sysname] = fails
    print(sysname, "segments:", len(lines), "failures:", len(fails))

json.dump(log, open(os.path.join(OUT, "run_log.json"), "w"), indent=1, ensure_ascii=False)
