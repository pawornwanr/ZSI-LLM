#!/usr/bin/env python3
"""
llm_imputation.py - zero-shot imputation of medium and long gaps with an LLM.

Code for the paper "Zero-Shot Imputation of Medium and Long Gaps in Time-Series
Data with Large Language Model" (ICA-SYMP 2027).

For every (dataset x gap type x seed) combination, the script reads a file with
removed values, sends each gap to the LLM as a text prompt, and writes the filled
series and the error of the filled points.

    datasets : ETTh1 ETTh2 ETTm1 ETTm2
    types    : medium long
    seeds    : 0 1 2 3 4
    -> 4 x 2 x 5 = 40 runs

Method (Section III-C of the paper)
    * medium gap : one LLM call per gap.
    * long gap   : the gap is filled in chunks of at most 24 points, from left to
                   right. Each chunk receives only the values next to its own edges.
    * prompt     : time-of-day profile (7 days on each side), up to 8 values at each
                   edge, and the range of the observed values. The model starts from
                   the time-of-day value and adds a smooth offset that meets the edges.
    * call       : temperature 0 and a set sampling seed (BASE_SEED = 42, changed in later passes).
    * clip       : every reply is clipped to the range of the observed values.

Input
    <HERE>/<dataset>_missing_areas/<type>_missing_<seed>.csv
    columns: date, OT (NaN at removed points), OT_true, is_masked
    Gaps that are still empty after a pass (no reply or a reply with the wrong length)
    are filled again in later passes with a new seed. Runs save checkpoints and resume
    from them when started again (use --force to start over).

Output
    <HERE>/imputed_llm_<dataset in lower case>/<model-slug>/<type>_missing_<seed>.csv
        adds the columns OT_imputed and fill_method to the input file
    <HERE>/results_llm.csv   (one row per run: MAE, RMSE, number of filled points, ...)

Requirements
    numpy, pandas, ollama, python-dotenv (and matplotlib for --plots)

API key
    The Ollama cloud API key is read from the environment variable OLLAMA_API_KEY
    or from a .env file in HERE, e.g.  OLLAMA_API_KEY=your_key_here
    No key is stored in this file.

Usage
    python llm_imputation.py --dry-run           # plan and number of gaps only
    python llm_imputation.py                     # all 40 runs with DeepSeek-V4.1-Flash
    python llm_imputation.py --model glm-5.3-flash:cloud --summary results_llm_glm.csv
    python llm_imputation.py --datasets ETTh1 --types medium --seeds 0
    python llm_imputation.py --list-models       # models offered by the endpoint
    python llm_imputation.py --status            # MAE/RMSE table of the summary file
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Folder that holds the <dataset>_missing_areas folders and the .env file.
# As a script: the folder of this file. Inside a notebook: HERE_DIR if set, else the current folder.
if "__file__" in globals():
    HERE = Path(__file__).resolve().parent
else:
    HERE = Path(globals().get("HERE_DIR") or Path.cwd()).resolve()

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
TYPES = ["medium", "long"]
SEEDS = [0, 1, 2, 3, 4]

# --- settings ----------------------------------------------------------------
CHUNK = 24           # long gaps: max points per LLM call
N_CTX = 8            # max values shown at each edge (searched within 1 day = P_CYCLE points);
                     # time-of-day profile window = 7 days = 7*P_CYCLE points on each side
FLUSH_EVERY = 20     # checkpoint every N areas
MAX_CONSEC_FAIL = 3  # consecutive unproductive areas before a pass ends
MAX_PASSES_LONG = 5  # passes over the areas of a long-gap file (new seed each pass)
EXTRA_PASSES = 10    # completion passes for areas still empty (seed = BASE_SEED + 100 + k)
BASE_SEED = 42       # sampling seed sent with every call

DEFAULT_MODEL = "deepseek-v4.1-flash:cloud"   # model of the main results; see --list-models
SYS_MSG = ("You are a fast numeric interpolator for time-series gaps. "
           "Output only the JSON answer - no explanation, no step-by-step reasoning.")

# The two models used in the paper. Any other Ollama model id can be passed with --model.
KNOWN_MODELS = [
    "deepseek-v4.1-flash:cloud",
    "glm-5.3-flash:cloud",
]

_PRINT_LOCK = threading.Lock()


def slugify_model(model: str) -> str:
    """Filesystem-safe tag for a model id ('deepseek-v4.1-flash:cloud' -> 'deepseek-v4.1-flash-cloud')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# per-run context: everything the prompt builder needs, derived from the data
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    n: int
    hours: list          # slot-of-day index per row
    P_CYCLE: int         # slots per day
    step_label: str      # "15-MINUTE" / "HOURLY"
    slot_unit: str       # "15 min" / "1 h"
    OT_MIN: float
    OT_MAX: float
    tag: str             # "ETTm1/medium/seed1" for logging

    @property
    def prof_hdr(self) -> str:
        return f"{self.P_CYCLE} slots of {self.slot_unit}"


def detect_frequency(dates: pd.Series) -> tuple[int, str, str]:
    """Return (slots_per_day, step_label, slot_unit) from the timestamp step."""
    d = pd.to_datetime(dates)
    step = int(np.median(np.diff(d.values).astype("timedelta64[s]").astype(int)))
    if step <= 0:
        raise ValueError("non-increasing timestamps")
    slots = int(round(86400 / step))
    if step == 3600:
        return slots, "HOURLY", "1 h"
    if step == 900:
        return slots, "15-MINUTE", "15 min"
    return slots, f"{step // 60}-MINUTE", f"{step // 60} min"


def build_ctx(df: pd.DataFrame, tag: str) -> Ctx:
    d = pd.to_datetime(df["date"])
    slots, step_label, slot_unit = detect_frequency(df["date"])
    if slots == 96:                       # ETTm: 15-min slot index
        hours = (d.dt.hour * 4 + d.dt.minute // 15).tolist()
    elif slots == 24:                     # ETTh: hour-of-day
        hours = d.dt.hour.tolist()
    else:                                 # generic fallback
        sec = d.dt.hour * 3600 + d.dt.minute * 60 + d.dt.second
        hours = (sec // (86400 // slots)).tolist()
    # the value range for the prompt and for np.clip comes from the OBSERVED OT only
    # (OT is NaN at the removed points), so no removed value is used
    obs_vals = df["OT"].dropna().values
    if obs_vals.size == 0:
        raise ValueError("no observed OT values to derive the value range")
    return Ctx(
        n=len(df), hours=hours, P_CYCLE=slots,
        step_label=step_label, slot_unit=slot_unit,
        OT_MIN=float(obs_vals.min()), OT_MAX=float(obs_vals.max()), tag=tag,
    )


# ---------------------------------------------------------------------------
# input check and prompt construction
# ---------------------------------------------------------------------------
def check_input(df: pd.DataFrame, tag: str) -> None:
    """Verify the input is the masked dataset:
    OT is NaN exactly where is_masked==1; OT_true is complete; OT == OT_true elsewhere."""
    m = (df["is_masked"] == 1).values
    assert (df["OT"].isna().values == m).all(), f"[{tag}] OT NaN pattern != is_masked"
    assert not df["OT_true"].isna().any(), f"[{tag}] OT_true has NaN"
    assert np.allclose(df["OT"].values[~m], df["OT_true"].values[~m]), f"[{tag}] OT != OT_true on observed rows"


def _gap_features(series, start: int, end: int, ctx: Ctx):
    """Features around the gap [start, end].

    Edge values are only the points next to the gap.
      before : walk LEFT from start-1, 1..N_CTX points, stop at the first NaN in `series`.
      after  : walk RIGHT from end+1, 1..N_CTX points, stop at the first NaN in `series`.
    Values already filled in earlier gaps or earlier chunks are not NaN, so they count as
    context (Section III-C of the paper). For an interior chunk of a long gap, the point after
    the chunk is still NaN, so the chunk receives no value after it.
    The time-of-day profile uses the observed and filled values within 7 days on both sides.
    """
    s = np.asarray(series, dtype=float)
    n, P = ctx.n, ctx.P_CYCLE
    L = end - start + 1
    h0 = int(ctx.hours[start])

    # before: contiguous, nearest-first, stop at NaN
    before = []
    i = start - 1
    while i >= max(0, start - P) and len(before) < N_CTX and not np.isnan(s[i]):
        before.append((int(ctx.hours[i]), float(s[i])))
        i -= 1
    before.reverse()                      # back to chronological order

    # after: contiguous, nearest-first, stop at NaN
    after = []
    ae = min(n, end + 1 + P)
    j = end + 1
    while j < ae and len(after) < N_CTX and not np.isnan(s[j]):
        after.append((int(ctx.hours[j]), float(s[j])))
        j += 1

    # time-of-day profile: mean of the non-NaN values at each slot within 7 days on both sides
    W = P * 7; lo = max(0, start - W); hi = min(n, end + 1 + W)
    prof = np.full(P, np.nan)
    for h in range(P):
        vv = [s[i] for i in range(lo, hi) if int(ctx.hours[i]) == h and not np.isnan(s[i])]
        if vv:
            prof[h] = float(np.mean(vv))
    win = s[lo:hi]; win = win[~np.isnan(win)]
    gm = float(win.mean()) if win.size else 0.0
    prof = np.where(np.isnan(prof), gm, prof)
    vb = before[-1][1] if before else None    # value right before the gap (None if not adjacent)
    va = after[0][1] if after else None        # value right after the gap  (None if not adjacent)
    return L, h0, prof, before, after, vb, va


def build_prompt_from_series(series, start: int, end: int, ctx: Ctx) -> str:
    """Prompt for the gap [start, end] (shown in Fig. 2 of the paper)."""
    L, h0, prof, before, after, vb, va = _gap_features(series, start, end, ctx)
    fmt = lambda x: "n/a" if x is None else f"{x:.2f}"          # noqa: E731
    prof_str = ", ".join(f"t{h:02d}={prof[h]:.2f}" for h in range(ctx.P_CYCLE))
    bstr = ", ".join(f"t{h:02d}={v:.2f}" for h, v in before) or "(none)"   # at most N_CTX values
    astr = ", ".join(f"t{h:02d}={v:.2f}" for h, v in after) or "(none)"
    P = ctx.P_CYCLE
    # step-2 wording depends on which edge values exist
    if vb is not None and va is not None:
        step2 = (f"add a small smooth offset so the result starts near {fmt(vb)} and ends near "
                 f"{fmt(va)} (no jump at the edges).")
    elif vb is not None:
        # no end value: the offset shrinks smoothly to 0 by the last point,
        # so the chunk returns to the time-of-day profile
        step2 = (f"add a small smooth offset so the result starts near {fmt(vb)} "
                 "(no jump at the start), then let the offset shrink smoothly to 0 by the last point, "
                 "so the last point equals its base value (no end value is given).")
    elif va is not None:
        step2 = (f"add a small smooth offset so the result ends near {fmt(va)} "
                 "(no jump at the end; no start value is given).")
    else:
        step2 = "no edge values are given, so use the base values as they are."
    return (
        f"Fill {L} missing {ctx.step_label} OT values (transformer oil temperature). "
        f"Strong DAILY cycle, period {P} (={P} steps/day).\n"
        "Do it directly in two steps, without writing any step-by-step reasoning:\n"
        f"  1) base value for point k = the typical value at its time-of-day slot, "
        f"prof[t(k)], where t(k)=({h0}+k) mod {P}.\n"
        f"  2) {step2}\n"
        f"Typical value by time-of-day slot (prof, {ctx.prof_hdr}): {prof_str}\n"
        f"Values just BEFORE the gap: {bstr}\n"
        f"Values just AFTER  the gap: {astr}\n"
        f"Keep values within [{ctx.OT_MIN:.1f}, {ctx.OT_MAX:.1f}], rounded to 2 decimals.\n"
        "Output the JSON only - no words, no reasoning, no working.\n"
        'Return ONLY JSON with exactly ' + str(L) + ' numbers in time order: {"values":[...]}.'
    )


_ANSWER_RE = re.compile(r'\{\s*"values"\s*:\s*\[[^\[\]]*\]\s*\}')   # one {"values": [...]} answer


def parse_values(text, n_expect: int):
    if text is None:
        return None
    # several JSON answers in one reply -> use the last one only (the model's revised answer)
    answers = _ANSWER_RE.findall(text)
    if len(answers) > 1:
        try:
            return [float(v) for v in json.loads(answers[-1])["values"]]
        except Exception:
            return None
    try:
        a, b = text.find("{"), text.rfind("}")
        obj = json.loads(text[a:b + 1]) if a >= 0 else json.loads(text)
        vals = obj.get("values") if isinstance(obj, dict) else obj
        return [float(v) for v in vals]
    except Exception:
        nums = re.findall(r"-?\d+\.?\d*", text)
        return [float(x) for x in nums] if nums else None


# ---------------------------------------------------------------------------
# LLM access - one Client per thread
# ---------------------------------------------------------------------------
class FatalAPIError(RuntimeError):
    """Error that retrying cannot fix (wrong model id, bad key, bad request)."""


class ClientPool:
    def __init__(self, host="https://ollama.com", timeout=300, env_path=None):
        self.host, self.timeout, self.env_path = host, timeout, env_path
        self._local = threading.local()

    def _client(self):
        c = getattr(self._local, "client", None)
        if c is None:
            from dotenv import load_dotenv
            from ollama import Client
            if self.env_path and Path(self.env_path).exists():
                load_dotenv(dotenv_path=str(self.env_path))
            else:
                load_dotenv()
            key = os.getenv("OLLAMA_API_KEY")
            if not key:
                raise RuntimeError("OLLAMA_API_KEY not found (looked in .env / environment)")
            c = Client(host=self.host,
                       headers={"Authorization": f"Bearer {key}"},
                       timeout=self.timeout)
            self._local.client = c
        return c

    def chat(self, prompt: str, model: str, seed: int, system: str | None = SYS_MSG) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        # the LLM call: temperature 0 and a fixed seed
        resp = self._client().chat(model=model, messages=msgs,
                                   options={"temperature": 0, "seed": seed})
        msg = resp["message"]
        return (msg["content"] or "").strip()

    def chat_with_retry(self, prompt, model, seed, retries=5, system=SYS_MSG):
        """Retry errors that can pass; stop at once on a wrong model id, key or request."""
        for att in range(retries):
            try:
                return self.chat(prompt, model, seed, system)
            except Exception as ex:
                code = getattr(ex, "status_code", None)
                if code in (400, 401, 403, 404):
                    raise FatalAPIError(f"HTTP {code}: {ex} (check --model / OLLAMA_API_KEY)") from ex
                wait = min(60, 5 * 2 ** att) if code == 429 else 3 * (att + 1)
                log(f"    [retry {att + 1}/{retries}] {('HTTP %s ' % code) if code else ''}{ex} — wait {wait}s")
                time.sleep(wait)
        return None


# ---------------------------------------------------------------------------
# one (dataset, type, seed) run
# ---------------------------------------------------------------------------
def run_one(ds: str, typ: str, seed: int, args, pool: ClientPool) -> dict:
    tag = f"{ds}/{typ}/{seed}"
    res = {"dataset": ds, "type": typ, "seed": seed, "status": "failed",
           "model": args.model, "n_masked": 0, "n_llm": 0, "n_nan": 0,
           "mae": np.nan, "rmse": np.nan,
           "areas": 0, "error": ""}
    try:
        in_csv = HERE / f"{ds}_missing_areas" / f"{typ}_missing_{seed}.csv"
        if not in_csv.exists():
            raise FileNotFoundError(in_csv)
        # Outputs are namespaced by model: without this a run with a different
        # --model would resume from the previous model's checkpoints and blend
        # the two. One subdirectory per model (override with --out-tag).
        out_dir = HERE / f"imputed_llm_{ds.lower()}" / args.out_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{typ}_missing_{seed}.csv"

        df = pd.read_csv(in_csv)
        df["date"] = pd.to_datetime(df["date"])
        ctx = build_ctx(df, tag)
        true = df["OT_true"].values
        masked = (df["is_masked"] == 1).values
        check_input(df, tag)  # input must be the masked dataset
        res["n_masked"] = int(masked.sum())

        # missing areas (start, end)
        mp = [i for i, x in enumerate(df["OT"].tolist()) if pd.isna(x)]
        areas = []
        if mp:
            s = mp[0]
            for i in range(1, len(mp)):
                if mp[i] != mp[i - 1] + 1:
                    areas.append((s, mp[i - 1])); s = mp[i]
            areas.append((s, mp[-1]))
        areas = areas[:args.max_areas] if args.max_areas else areas
        res["areas"] = len(areas)

        log(f"[{tag}] {len(df)} rows, freq={ctx.P_CYCLE}/day, "
            f"{res['n_masked']} missing in {len(areas)} areas")

        # resume from checkpoint
        if out_csv.exists() and not args.force:
            prev = pd.read_csv(out_csv)
            work = prev["OT_imputed"].to_numpy(dtype=float).copy()
            # "" -> NaN on read_csv; only affects the bookkeeping column
            method = prev["fill_method"].fillna("").astype(object).to_numpy().copy()
            log(f"[{tag}] checkpoint found — resuming")
        else:
            work = df["OT"].to_numpy(copy=True).astype(float)
            method = np.where(masked, "", "obs").astype(object)

        def save_ckpt():
            out = df.copy()
            out["OT_imputed"] = work
            out["fill_method"] = method
            out.to_csv(out_csv, index=False)

        def metrics():
            cur = pd.read_csv(out_csv)
            m = (cur["is_masked"] == 1).values
            yt, yp = cur["OT_true"].values[m], cur["OT_imputed"].values[m]
            good = ~np.isnan(yp)
            res["n_llm"], res["n_nan"] = int(good.sum()), int((~good).sum())
            if good.any():
                d = yt[good] - yp[good]
                res["mae"] = float(np.mean(np.abs(d)))
                res["rmse"] = float(np.sqrt(np.mean(d ** 2)))
            res["status"] = "ok" if res["n_nan"] == 0 else "partial"

        # A finished checkpoint needs no API calls at all — important when
        # re-running the 40-run sweep after a partial/interrupted pass.
        if out_csv.exists() and not args.force and not args.max_areas:
            if int(np.isnan(work[masked]).sum()) == 0:
                metrics()
                res["status"] = "ok"
                res["error"] = "cached"
                log(f"[{tag}] already complete — skipping ({res['n_llm']} imputed, "
                    f"MAE {res['mae']:.4f})")
                return res

        def q(prompt, _seed):
            return pool.chat_with_retry(prompt, args.model, _seed)

        def fill_chunks(s, e, seed, chunk=CHUNK):
            """long: fill an area in <= chunk-sized calls, from left to right."""
            cs = s
            while cs <= e:
                ce = min(cs + chunk - 1, e)
                if not np.isnan(work[cs:ce + 1]).any():
                    cs = ce + 1; continue
                Lc = ce - cs + 1
                content = q(build_prompt_from_series(work, cs, ce, ctx), seed)
                vals = parse_values(content, Lc) if content is not None else None
                if vals is not None and len(vals) == Lc:
                    vals = [float(np.clip(v, ctx.OT_MIN, ctx.OT_MAX)) for v in vals]
                    for k, idx in enumerate(range(cs, ce + 1)):
                        work[idx] = vals[k]
                else:
                    # this chunk failed (no reply / wrong number of values). Do NOT go on to
                    # the next chunk: it would have no value right before it (edge lists stop at NaN) and
                    # would be filled without a start anchor and never retried. Stop here; the next pass
                    # (new seed) restarts from this chunk.
                    return False
                cs = ce + 1
            return not np.isnan(work[s:e + 1]).any()

        def fill_whole(s, e, seed):
            """medium: one call for the whole area."""
            L = e - s + 1
            content = q(build_prompt_from_series(work, s, e, ctx), seed)
            vals = parse_values(content, L) if content is not None else None
            ok = vals is not None and len(vals) == L
            if ok:
                vals = [float(np.clip(v, ctx.OT_MIN, ctx.OT_MAX)) for v in vals]
                for k, idx in enumerate(range(s, e + 1)):
                    work[idx] = vals[k]; method[idx] = "llm"
            return ok

        fill = fill_chunks if typ == "long" else fill_whole
        max_passes = args.max_passes if args.max_passes else (
            MAX_PASSES_LONG if typ == "long" else 3)

        for p in range(1, max_passes + 1):
            todo = [(s, e) for (s, e) in areas if np.isnan(work[s:e + 1]).any()]
            if not todo:
                break
            # long gaps use a new sampling seed in each pass;
            # medium gaps use the same seed (BASE_SEED) in each pass.
            pseed = BASE_SEED + p if typ == "long" else BASE_SEED
            n_before = int(np.isnan(work).sum())
            log(f"[{tag}] pass {p}/{max_passes} seed={pseed} | "
                f"{len(todo)} areas left | {n_before} NaN")
            consec = 0
            for i, (s, e) in enumerate(todo):
                if typ == "long":
                    nb = int(np.isnan(work[s:e + 1]).sum())
                    nan_before = int(np.isnan(work).sum())
                    fill_chunks(s, e, pseed)
                    for idx in range(s, e + 1):
                        if (not np.isnan(work[idx])) and method[idx] == "":
                            method[idx] = "llm"
                    if int(np.isnan(work).sum()) == nan_before:
                        consec += 1
                        if consec >= MAX_CONSEC_FAIL:
                            log(f"[{tag}] {consec} unproductive areas — ending pass")
                            break
                    else:
                        consec = 0
                    if args.verbose:
                        log(f"[{tag}]   [{s}..{e}] L={e - s + 1} NaN {nb} -> "
                            f"{int(np.isnan(work[s:e + 1]).sum())}")
                else:
                    ok = fill(s, e, pseed)
                    consec = 0 if ok else consec + 1
                    if consec >= MAX_CONSEC_FAIL:
                        log(f"[{tag}] {consec} unproductive areas — ending pass")
                        break
                # long files have only 14-16 areas, so they are saved after every area;
                # medium files are saved every FLUSH_EVERY areas.
                if (i + 1) % (1 if typ == "long" else FLUSH_EVERY) == 0:
                    save_ckpt()
            save_ckpt()
            if int(np.isnan(work).sum()) == 0:
                break

        # completion passes: only areas that are still empty, new seed each pass,
        # same prompt and options; stops as soon as the run is complete
        if not args.max_areas:
            for k in range(1, EXTRA_PASSES + 1):
                todo = [(s, e) for (s, e) in areas if np.isnan(work[s:e + 1]).any()]
                if not todo:
                    break
                xseed = BASE_SEED + 100 + k
                n_before = int(np.isnan(work).sum())
                log(f"[{tag}] completion pass {k}/{EXTRA_PASSES} seed={xseed} | {len(todo)} areas | {n_before} NaN")
                for (s, e) in todo:
                    fill(s, e, xseed)
                    for idx in range(s, e + 1):
                        if (not np.isnan(work[idx])) and method[idx] == "":
                            method[idx] = "llm"
                    save_ckpt()

        save_ckpt()

        obs = ~masked
        if args.max_areas:
            log(f"[{tag}] NOTE: --max-areas set, observed-unchanged check skipped")
        else:
            assert np.allclose(work[obs], df["OT"].values[obs]), "observed values modified"

        # metrics over the imputed (masked) points
        metrics()

        if args.plots:
            make_plot(pd.read_csv(out_csv), out_dir, typ, seed, ds)

        log(f"[{tag}] {res['status']}: imputed {res['n_llm']}, NaN {res['n_nan']}, "
            f"MAE {res['mae']:.4f}, RMSE {res['rmse']:.4f}")

    except Exception as ex:
        res["error"] = f"{type(ex).__name__}: {ex}"
        log(f"[{tag}] FAILED — {res['error']}")
        if args.verbose:
            log(traceback.format_exc())

    return res


def make_plot(res: pd.DataFrame, out_dir: Path, typ: str, seed: int, ds: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = res["is_masked"] == 1
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    ax = axes[0]
    ax.plot(res.index, res["OT_true"], color="#B0B0B0", lw=0.4, alpha=0.7, label="actual (OT_true)")
    ax.scatter(res.index[m], res.loc[m, "OT_imputed"], color="#2E8B57", s=5, label="imputed", zorder=3)
    ax.set_title(f"{ds} {typ}_missing_{seed} — actual vs imputed overall")
    ax.set_ylabel("OT"); ax.legend(fontsize=8)
    z = res.iloc[:500]; mz = z["is_masked"] == 1
    ax = axes[1]
    ax.plot(z.index, z["OT_true"], color="#B0B0B0", lw=1, alpha=0.7, label="actual (OT_true)")
    ax.plot(z.index, z["OT_imputed"], color="#4C78A8", lw=0.8, label="imputed series")
    ax.scatter(z.index[mz], z.loc[mz, "OT_true"], color="#E4572E", s=18, zorder=4, label="actual (masked)")
    ax.scatter(z.index[mz], z.loc[mz, "OT_imputed"], color="#2E8B57", s=18, zorder=5, label="imputed")
    ax.set_title("zoom: first 500 points"); ax.set_xlabel("index"); ax.set_ylabel("OT"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{typ}_missing_{seed}_plot.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# planning helpers
# ---------------------------------------------------------------------------
def count_areas(ds: str, typ: str, seed: int) -> int:
    """Number of missing areas (gaps) in one run."""
    p = HERE / f"{ds}_missing_areas" / f"{typ}_missing_{seed}.csv"
    if not p.exists():
        return 0
    ot = pd.read_csv(p, usecols=["OT"])["OT"]
    idx = np.where(ot.isna().values)[0]
    if not len(idx):
        return 0
    return int(1 + np.sum(np.diff(idx) != 1))


def show_status(summary: str = "results_llm.csv") -> None:
    p = HERE / summary
    if not p.exists():
        print(f"no {summary} yet — run the script first")
        return
    r = pd.read_csv(p)
    if r.empty:
        print(f"{summary} is empty"); return
    keys = ["model", "dataset", "type"] if "model" in r.columns else ["dataset", "type"]
    cols = ["dataset", "type", "seed", "status", "n_masked", "n_llm", "n_nan",
            "mae", "rmse"]
    if "model" in r.columns:
        cols.insert(0, "model")
    cols = [c for c in cols if c in r.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n== per-run ==")
        print(r[cols].to_string(index=False))
        g = (r.groupby(keys)
               .agg(runs=("seed", "count"), mae=("mae", "mean"), rmse=("rmse", "mean"),
                    nan_left=("n_nan", "sum")).reset_index())
        print(f"\n== mean over seeds (keyed by {'/'.join(keys)}) ==")
        print(g.to_string(index=False))


# ---------------------------------------------------------------------------
def main(argv=None) -> int:  # argv lets a notebook cell call main([...])
    ap = argparse.ArgumentParser(
        description="Run all ETT imputation experiments (dataset x type x seed).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--types", nargs="+", default=TYPES, choices=TYPES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent runs (default 3)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model id (default {DEFAULT_MODEL}). Any id is passed "
                         f"through; models of the paper: {', '.join(KNOWN_MODELS)} "
                         f"(see --list-models)")
    ap.add_argument("--out-tag", default="",
                    help="subdirectory under imputed_llm_<ds>/ (default: slug of --model)")
    ap.add_argument("--list-models", action="store_true",
                    help="query the endpoint for available models and exit")
    ap.add_argument("--max-passes", type=int, default=0,
                    help="override the number of passes (0 = defaults: 5 long, 3 medium)")
    ap.add_argument("--max-areas", type=int, default=0,
                    help="limit areas per run (quick test); skips the checkpoint shortcut, the completion passes and the observed-values check")
    ap.add_argument("--timeout", type=int, default=300, help="per-request timeout (s)")
    ap.add_argument("--plots", action="store_true", help="also write PNG figures")
    ap.add_argument("--force", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no API calls")
    ap.add_argument("--status", action="store_true", help="print the summary file (--summary) and exit")
    ap.add_argument("--summary", default="results_llm.csv", help="summary file, one row per run")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    if args.list_models:
        try:
            ms = ClientPool(timeout=args.timeout, env_path=HERE / ".env")._client().list()
            for m in (ms.get("models") or []):
                print(" ", m.get("model") or m.get("name"))
        except Exception as ex:
            print(f"could not list models ({type(ex).__name__}: {ex}); known ids:")
            for m in KNOWN_MODELS:
                print(" ", m)
        return 0

    # every model gets its own output folder, so runs never mix
    args.out_tag = args.out_tag or slugify_model(args.model)

    if args.status:
        show_status(args.summary); return 0

    combos = [(d, t, s) for d in args.datasets for t in args.types for s in args.seeds]

    # plan
    rows = [(d, t, s, count_areas(d, t, s)) for d, t, s in combos]
    plan = pd.DataFrame(rows, columns=["dataset", "type", "seed", "areas"])

    print(f"plan: {len(combos)} runs | {int(plan['areas'].sum())} gaps | "
          f"workers={args.workers} | model={args.model}")
    print(f"outputs -> imputed_llm_<dataset>/{args.out_tag}/  (--out-tag to change)")
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(plan.to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    pool = ClientPool(timeout=args.timeout, env_path=HERE / ".env")
    results = []
    sp = HERE / args.summary
    COLS = ["dataset", "type", "seed", "status", "model", "n_masked", "n_llm", "n_nan",
            "mae", "rmse", "areas", "error"]
    wlock = threading.Lock()
    # earlier rows of this summary: kept when a finished run is only re-read ("cached"),
    # and kept for runs not part of this call, so the file is never shrunk or zeroed by a re-run
    KEY = ["dataset", "type", "seed"]
    prev = {}
    if sp.exists():
        for r in pd.read_csv(sp).to_dict("records"):
            prev[(r["dataset"], r["type"], int(r["seed"]))] = r

    def merged():
        cur = {}
        for r in results:
            k = (r["dataset"], r["type"], int(r["seed"]))
            if r.get("error") == "cached" and k in prev:
                cur[k] = prev[k]
            else:
                cur[k] = r
        rows = {**prev, **cur}
        return pd.DataFrame(list(rows.values())).reindex(columns=COLS).sort_values(KEY)

    def write_summary():
        # written after every finished run, so a long sweep survives Ctrl-C
        with wlock:
            merged().to_csv(sp, index=False)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(run_one, d, t, s, args, pool): (d, t, s) for d, t, s in combos}
        for fut in as_completed(futs):
            d, t, s = futs[fut]
            try:
                results.append(fut.result())
            except Exception as ex_:
                log(f"[{d}/{t}/{s}] worker crashed — {ex_}")
                results.append({"dataset": d, "type": t, "seed": s, "status": "crashed",
                                "model": args.model, "n_masked": 0, "n_llm": 0, "n_nan": 0,
                                "mae": np.nan, "rmse": np.nan, "areas": 0, "error": str(ex_)})
            write_summary()

    out = merged()
    out.to_csv(sp, index=False)

    print(f"\ndone — {len(out)} runs | model={args.model}")
    show = ["dataset", "type", "seed", "status", "n_llm", "n_nan", "mae", "rmse"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(out[show].to_string(index=False))
        g = (out.groupby(["dataset", "type"])
                .agg(runs=("seed", "count"), mae=("mae", "mean"),
                     rmse=("rmse", "mean"), nan_left=("n_nan", "sum")).reset_index())
        print(f"\n== mean over seeds ==")
        print(g.to_string(index=False))
    print("summary:", sp)
    bad = out[out["status"] != "ok"]
    if len(bad):
        print(f"\n{len(bad)} run(s) not fully imputed (status != ok). "
              f"Re-run the same command to resume from checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
