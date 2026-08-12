#!/usr/bin/env python3
"""Export tidy CSVs for charting. No plotting here on purpose -- this emits long-
format data so any tool can consume it, and keeps the extraction auditable.

Four outputs in docs/data/, which are versioned in the repository so the README
charts can be regenerated and audited without the logs:

  training_curves.csv   per-log-point training metrics for both runs, with an
                        `instantaneous_s_per_it` column computed by differencing
                        timestamps. Do NOT chart the trainer's printed s/it: it
                        is elapsed/steps-since-start, a cumulative average that
                        climbs all run because the critic-warmup steps are cheap,
                        and it reads as a slowdown that is not happening.
  eval_matrix.csv       every scored evaluation cell, long format, from logs.
  checkpoint_traj.csv   the b64 trajectory aggregated per checkpoint, which is
                        the series that answers "which checkpoint is best".
  step_times.csv        instantaneous step time per interval, for the flatness plot.
"""
import csv, glob, json, os, re, sys

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(WS, 'docs', 'data')
os.makedirs(OUT, exist_ok=True)

LOGRE = re.compile(
    r'\[(\d\d):(\d\d):(\d\d)\] it (\d+)/(\d+) gen (\S+) critic (\S+) '
    r'\|grad\| (\S+) norm (\S+) gn_g (\S+) depth (\d+)')


def secs(h, m, s, prev):
    """Wall-clock seconds, unwrapping midnight using the previous value."""
    t = int(h) * 3600 + int(m) * 60 + int(s)
    while prev is not None and t < prev:
        t += 86400
    return t


def parse_train(path, run):
    rows, prev = [], None
    for line in open(path, errors='ignore'):
        m = LOGRE.match(line)
        if not m:
            continue
        t = secs(m[1], m[2], m[3], prev)
        prev = t
        rows.append(dict(run=run, t=t, step=int(m[4]), total=int(m[5]),
                         loss_gen=m[6], loss_critic=m[7], grad_norm=m[8],
                         normalizer=m[9], gn_gen=m[10], depth=int(m[11])))
    # instantaneous rate: difference consecutive points, not the printed average
    for a, b in zip(rows, rows[1:]):
        b['instantaneous_s_per_it'] = round((b['t'] - a['t']) / (b['step'] - a['step']), 3)
    if rows:
        rows[0]['instantaneous_s_per_it'] = ''
    return rows


def main():
    runs = [('b64_main_pre_crash', 'logs/main_b64_run_to_step770.log'),
            ('b64_resume',         'logs/b64_resume.log'),
            ('b4_2h_proof',        'logs/b4_2h.log'),
            ('b4_ctrl_400',        'logs/ctrl_b4.log'),
            ('b64_dense_350',      'logs/dense_b64.log')]
    train = []
    for name, rel in runs:
        p = os.path.join(WS, rel)
        if os.path.exists(p):
            train += parse_train(p, name)
    cols = ['run', 'step', 'total', 't', 'loss_gen', 'loss_critic', 'grad_norm',
            'normalizer', 'gn_gen', 'depth', 'instantaneous_s_per_it']
    with open(f'{OUT}/training_curves.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in train:
            w.writerow(r)
    print(f'training_curves.csv   {len(train)} rows, {len({r["run"] for r in train})} runs')

    with open(f'{OUT}/step_times.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'step', 'instantaneous_s_per_it'])
        n = 0
        for r in train:
            if r.get('instantaneous_s_per_it') not in ('', None):
                w.writerow([r['run'], r['step'], r['instantaneous_s_per_it']])
                n += 1
    print(f'step_times.csv        {n} rows')

    # ---- evaluation cells, scraped from the eval logs -------------------------
    Q = re.compile(r'sharp ([\d.]+)x\s+contrast ([\d.]+)x\s+motion ([\d.]+)x\s+'
                   r'blockiness ([\d.]+)x\s+latent-std ([\d.]+)x\s+'
                   r'sharp-decay ([\-\d.]+)\s+world-cos-drift ([\-+\d.]+)')
    S = re.compile(r'sustained \(excl\. first\): ([\d.]+) ms/unit -> ([\d.]+) pixel FPS \(([\d.]+)x real time\)')
    B = re.compile(r'real-time budget [\d.]+ ms/unit => (\w+)')
    C = re.compile(r'loaded model from (\S+) \(step (\d+)')

    cells = []
    for p in sorted(glob.glob(os.path.join(WS, 'logs', '*.log'))):
        txt = open(p, errors='ignore').read()
        q = Q.search(txt)
        if not q:
            continue
        name = os.path.basename(p)[:-4]
        s, b, c = S.search(txt), B.search(txt), C.search(txt)
        # Timing is only meaningful when the GPUs were idle. The two mid-run
        # milestone evals were run against training and must be flagged, not
        # silently averaged in with the rest.
        # Timing is only meaningful on idle GPUs:
        #   b64_s1500_*/b64_s2250_*  ran against training (0.90x vs 2.7x idle)
        #   b64_final_*              ran against another session's eval AND was
        #                            killed mid-decode -- 5 of 12 have no score
        #                            line at all. Superseded by FINAL_*.
        contended = (name.startswith('b64_s1500_') or name.startswith('b64_s2250_')
                     or name.startswith('b64_final_'))
        cells.append(dict(
            cell=name, ckpt_file=(c[1] if c else ''), ckpt_step=(c[2] if c else ''),
            sharp=q[1], contrast=q[2], motion=q[3], blockiness=q[4],
            latent_std=q[5], sharp_decay=q[6], world_cos_drift=q[7],
            sustained_ms_unit=(s[1] if s else ''), pixel_fps=(s[2] if s else ''),
            realtime_x=(s[3] if s else ''), rt_budget=(b[1] if b else ''),
            timing_valid=('no_gpu_contended' if contended else 'yes')))
    ecols = ['cell', 'ckpt_file', 'ckpt_step', 'sharp', 'contrast', 'motion',
             'blockiness', 'latent_std', 'sharp_decay', 'world_cos_drift',
             'sustained_ms_unit', 'pixel_fps', 'realtime_x', 'rt_budget', 'timing_valid']
    with open(f'{OUT}/eval_matrix.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ecols)
        w.writeheader()
        for r in cells:
            w.writerow(r)
    print(f'eval_matrix.csv       {len(cells)} cells '
          f'({sum(1 for c in cells if c["timing_valid"] != "yes")} with invalid timing)')

    # ---- the b64 trajectory, aggregated ---------------------------------------
    # Two naming schemes, both measured on idle GPUs, deliberately combined:
    #   FINAL_w<world>_<tag>       base / v5 / step-3000, from the 17:16 sweep
    #   handoff_<tag>_w<world>     the 750/1500/2250 trajectory, which nothing
    #                              else scored on idle hardware
    W = ['0', '44', '60', '82']
    traj = []
    for arm, step, pat in [('base',  None, 'FINAL_w{w}_base'),
                           ('v5',    None, 'FINAL_w{w}_v5'),
                           ('s0750',  750, 'handoff_s0750_w{w}'),
                           ('s1500', 1500, 'handoff_s1500_w{w}'),
                           ('s2250', 2250, 'handoff_s2250_w{w}'),
                           ('s3000', 3000, 'FINAL_w{w}_new')]:
        tag = arm
        got = {}
        for w in W:
            hit = [c for c in cells if c['cell'] == pat.format(w=w)]
            if hit:
                got[w] = hit[0]
        if len(got) != 4:
            continue
        dec = sum(abs(float(got[w]['sharp_decay']) - 1.0) for w in W) / 4
        dri = sum(abs(float(got[w]['world_cos_drift'])) for w in W) / 4
        mot = sum(float(got[w]['motion']) for w in W) / 4
        rt = [float(got[w]['realtime_x']) for w in W if got[w]['realtime_x']]
        traj.append(dict(arm=tag, step=step or '',
                         mean_abs_decay_dev=round(dec, 4), mean_abs_drift=round(dri, 4),
                         mean_motion=round(mot, 4),
                         min_realtime_x=(round(min(rt), 2) if rt else ''),
                         budget_met_all=all(got[w]['rt_budget'] == 'MET' for w in W)))
    if traj:
        with open(f'{OUT}/checkpoint_traj.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(traj[0]))
            w.writeheader()
            for r in traj:
                w.writerow(r)
        print(f'checkpoint_traj.csv   {len(traj)} arms')
    else:
        print('checkpoint_traj.csv   SKIPPED (final_* cells not present yet)')


if __name__ == '__main__':
    main()
