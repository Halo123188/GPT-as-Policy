"""Aggregate pi05-only panel results and compare them with the published RoboDojo tables.

Reads RESULTS/<case_id>/attempt_*/sim/evaluation_outcome.json (written by the native
simulator session). One native-complete attempt per case counts; incomplete or invalid
attempts are reported, never imputed.
"""
import argparse
import csv
import json
from pathlib import Path

from .evaluation import TASKS
from .io import write_json
from .pi05_only import scope_case_ids

ROOT = Path(__file__).resolve().parents[2]


def case_outcome(case_dir):
    attempts = sorted(case_dir.glob('attempt_*'), key=lambda p: int(p.name.split('_')[1]))
    rows = []
    for attempt in attempts:
        outcome_path = attempt/'sim/evaluation_outcome.json'
        outcome = json.loads(outcome_path.read_text()) if outcome_path.is_file() else {}
        result_path = attempt/'controller/result.json'
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        rows.append(dict(attempt=attempt.name, status=outcome.get('status', 'missing'),
            valid=outcome.get('valid_for_success_rate') is True, success=outcome.get('native_success'),
            score=outcome.get('native_score'), steps=outcome.get('native_control_steps'),
            limit=outcome.get('native_step_limit'), inferences=result.get('pi05_inference_calls'),
            wall_seconds=result.get('wall_seconds'), video=str(attempt/'sim/sensors.mp4')))
    valid = [r for r in rows if r['valid']]
    if valid:
        return valid[0], rows
    # A native-invalid layout (e.g. a failed support-arm demonstration) is a final
    # outcome under the panel policy: record and exclude it, never retry it.
    invalid = [r for r in rows if r['status'] == 'invalid_native_layout']
    return (invalid[0] if invalid else None), rows


def published(path):
    """Per-task (score, success rate) of the leaderboard pi0.5 row and the report's methods."""
    suffix = dict(official='(leaderboard)', local='(report, 5 cases)')
    tables = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            if row['model'] in ('π0.5', 'π0.5 + GPT', 'GPT-only'):
                label = f"{row['model']} {suffix[row['origin']]}"
                tables.setdefault(label, {})[row['task']] = (float(row['score_100']), float(row['success_rate_percent']))
    return tables


def summarize(results, scope):
    ids = scope_case_ids(scope)
    cases, invalid, missing = [], [], []
    for case_id in ids:
        chosen, attempts = case_outcome(results/case_id)
        if chosen is None:
            missing.append(dict(case_id=case_id, attempts=attempts))
        elif not chosen['valid']:
            invalid.append(dict(case_id=case_id, **chosen))
        else:
            cases.append(dict(case_id=case_id, task=case_id.split('__')[0], variant=case_id.split('__')[1], **chosen))
    tasks = []
    for task in TASKS:
        rows = [c for c in cases if c['task'] == task]
        planned = sum(1 for i in ids if i.split('__')[0] == task)
        tasks.append(dict(task=task, planned=planned, complete=len(rows),
            invalid=sum(1 for c in invalid if c['case_id'].split('__')[0] == task),
            successes=sum(bool(r['success']) for r in rows),
            # Success rate over all planned cases (report convention); score over valid episodes.
            success_rate_percent=100*sum(bool(r['success']) for r in rows)/planned if rows else None,
            score_100=100*sum(r['score'] for r in rows)/len(rows) if rows else None))
    complete = len(cases)
    return dict(schema='hybrid_rollout.robodojo.pi05_only_summary.v1', results_root=str(results),
        scope=str(scope), planned_cases=len(ids), complete_cases=complete, invalid_cases=len(invalid),
        successes=sum(bool(c['success']) for c in cases),
        success_rate_percent_over_planned=100*sum(bool(c['success']) for c in cases)/len(ids) if complete else None,
        mean_score_100_over_complete=100*sum(c['score'] for c in cases)/complete if complete else None,
        tasks=tasks, cases=cases, invalid=invalid, missing=missing)


def markdown(summary, tables):
    columns = ['π0.5 only (this run)', *tables]
    lines = ['| task | n | ' + ' | '.join(f'{c} score / SR%' for c in columns) + ' |',
             '|' + '---|'*(2+len(columns))]
    fmt = lambda v: '—' if v is None else f'{v:.1f}'
    for row in summary['tasks']:
        cells = [f"{fmt(row['score_100'])} / {fmt(row['success_rate_percent'])}"]
        for label in tables:
            score, rate = tables[label].get(row['task'], (None, None))
            cells.append(f'{fmt(score)} / {fmt(rate)}')
        count = f"{row['complete']}/{row['planned']}" + (f" (+{row['invalid']} invalid)" if row['invalid'] else '')
        lines.append(f"| {row['task']} | {count} | " + ' | '.join(cells) + ' |')
    mean = lambda values: sum(values)/len(values) if values else None
    cells = [f"{fmt(summary['mean_score_100_over_complete'])} / {fmt(summary['success_rate_percent_over_planned'])}"]
    for label in tables:
        values = [tables[label][t] for t in TASKS if t in tables[label]]
        cells.append(f'{fmt(mean([v[0] for v in values]))} / {fmt(mean([v[1] for v in values]))}')
    lines.append(f"| **mean** | {summary['complete_cases']}/{summary['planned_cases']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--scope', type=Path, default=ROOT/'hybrid_rollout/robodojo/eval_panels/robodojo_panel50_scope_v2.json')
    parser.add_argument('--published-scores', type=Path, default=ROOT/'public_results/scores.csv')
    parser.add_argument('--output', type=Path, help='Directory for summary.json, cases.csv and summary.md')
    args = parser.parse_args()
    summary = summarize(args.results.resolve(), args.scope)
    table = markdown(summary, published(args.published_scores))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'summary.json', summary)
        (args.output/'summary.md').write_text(table)
        with open(args.output/'cases.csv', 'w', newline='') as stream:
            fields = ['case_id', 'task', 'variant', 'attempt', 'status', 'success', 'score', 'steps', 'limit',
                      'inferences', 'wall_seconds', 'video']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summary['cases'])
    print(table)
    print(json.dumps({k: summary[k] for k in ('complete_cases', 'planned_cases', 'successes',
        'invalid_cases', 'success_rate_percent_over_planned', 'mean_score_100_over_complete')}))
    if summary['invalid']:
        print('Native-invalid (excluded from score, counted as non-success):', ', '.join(c['case_id'] for c in summary['invalid']))
    if summary['missing']:
        print('Incomplete cases:', ', '.join(m['case_id'] for m in summary['missing']))


if __name__ == '__main__':
    main()
