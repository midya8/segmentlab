"""Build a wheel and replay all CLI families outside the source checkout."""

import json, os, subprocess, sys, tempfile, venv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env.pop("PYTHONPATH", None)


def call(args, cwd):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as d:
    t = Path(d)
    w = t / "w"
    w.mkdir()
    call(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(w), str(root)], t
    )
    venv.create(t / "env", with_pip=True)
    py = str(t / "env/bin/python")
    call([py, "-m", "pip", "install", "--no-deps", str(next(w.glob("*.whl")))], t)
    out = t / "outside"
    out.mkdir()
    code = "import importlib,pathlib; names=['segmentlab','segmentlab.cli','segmentlab.core','segmentlab.decision','segmentlab.demo','segmentlab.information','segmentlab.joint','segmentlab.predictive','segmentlab.sampling']; assert all('site-packages' in str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names);print(len(names))"
    assert call([py, "-c", code], out).strip() == "9"
    jobs = [
        ["demo"],
        ["infer", "examples/sequence.json"],
        ["sample", "examples/sequence.json", "--draws", "4", "--seed", "23"],
        ["joint", "examples/sequence.json", "--pairs", "[[1,3],[2,4]]"],
        ["information", "examples/sequence.json"],
        [
            "decide",
            "examples/sequence.json",
            "--false-cut-cost",
            "1",
            "--missed-cut-cost",
            "1",
        ],
        ["predictive", "examples/sequence.json", "--queries", "[[2,0.0],[3,1.0]]"],
    ]
    for job in jobs:
        expected = call([py, "-m", "segmentlab"] + job, root)
        args = [str(root / a) if a.startswith("examples/") else a for a in job]
        actual = call([py, "-m", "segmentlab"] + args, out)
        assert actual == expected, job
        console = call([str(t / "env/bin/segmentlab")] + args, out)
        assert console == expected, job
    print(
        json.dumps(
            dict(module_origins=9, full_reports=7, entrypoints=2, outside_checkout=True)
        )
    )
