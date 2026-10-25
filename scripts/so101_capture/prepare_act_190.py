"""Freeze reviewed episode selection and session-disjoint ACT split."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def rows(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, records, fields):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def choose_eval_sessions(items, target):
    """Find session groups closest to 10% without splitting a session."""
    groups = defaultdict(list)
    for item in items:
        groups[item['source']].append(item)
    options = {0: ()}
    for session, episodes in sorted(groups.items()):
        count = len(episodes)
        for total, chosen in list(options.items()):
            next_total = total + count
            if next_total < len(items) and next_total not in options:
                options[next_total] = (*chosen, session)
    valid = [n for n in options if 0 < n < len(items)]
    return set(options[min(valid, key=lambda n: (abs(n - target), n))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--decisions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    inventory = rows(a.episodes)
    decisions = rows(a.decisions)
    by_id = {r['episode_id']: r for r in inventory}
    if len(by_id) != len(inventory) or len(decisions) != 11:
        raise ValueError('Expected unique inventory and 11 review decisions')
    reviewed = set()
    for decision in decisions:
        episode_id = decision['episode_id']
        if episode_id in reviewed or episode_id not in by_id:
            raise ValueError(f'Duplicate or unknown decision: {episode_id}')
        reviewed.add(episode_id)
        item = by_id[episode_id]
        if item['status'] != 'review' or Path(decision['source']) != Path(item['source_folder']) / item['session'] or int(decision['episode_index']) != int(item['episode_index']):
            raise ValueError(f'Review identity mismatch: {episode_id}')
        if decision['decision'] not in ('adopt', 'exclude'):
            raise ValueError(f'Unknown decision: {episode_id}')
        if decision['decision'] == 'adopt' and item['end_reason'] not in ('manual', 'time_limit'):
            raise ValueError(f'Cannot adopt interrupted episode: {episode_id}')
        item['review_decision'] = decision['decision']
    if reviewed != {r['episode_id'] for r in inventory if r['status'] == 'review'}:
        raise ValueError('Missing review decisions')
    adopted, rejected = [], []
    for item in inventory:
        item['source'] = str(Path(item['source_folder']) / item['session'])
        if item['status'] == 'adopted' or item.get('review_decision') == 'adopt':
            adopted.append(item)
        else:
            rejected.append(item)
    if len(adopted) != 190 or len(rejected) != 43:
        raise ValueError(f'Unexpected counts: {len(adopted)} adopted, {len(rejected)} excluded')
    if len({(r['source'], r['episode_index']) for r in adopted}) != 190:
        raise ValueError('Duplicate source episode')
    by_group = defaultdict(list)
    for item in adopted:
        by_group[item['group']].append(item)
    eval_sessions = set()
    for group, items in by_group.items():
        eval_sessions.update(choose_eval_sessions(items, max(1, round(len(items) * .1))))
    train = [r for r in adopted if r['source'] not in eval_sessions]
    eval_rows = [r for r in adopted if r['source'] in eval_sessions]
    if not all(any(r['group'] == group for r in part) for group in by_group for part in (train, eval_rows)):
        raise ValueError('Missing group from train/eval split')
    ordered = train + eval_rows  # LeRobot eval_split selects the last episodes per task.
    a.output.mkdir(parents=True)
    selected = []
    for index, item in enumerate(ordered):
        item['merged_episode_index'] = index
        item['split'] = 'eval' if index >= len(train) else 'train'
        selected.append(dict(source=item['source'], episode_index=int(item['episode_index']),
                             frames=int(item['frames']), group=item['group'], subgroup=item['subgroup'],
                             episode_id=item['episode_id']))
    (a.output / 'selected.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False) + '\n')
    fields = ('merged_episode_index', 'split', 'episode_id', 'group', 'subgroup', 'source', 'episode_index', 'frames', 'duration_s', 'status', 'review_decision')
    write_csv(a.output / 'source_episodes.csv', [{k: r.get(k, '') for k in fields} for r in ordered], fields)
    write_csv(a.output / 'excluded_episodes.csv', [{k: r.get(k, '') for k in ('episode_id', 'group', 'subgroup', 'source', 'episode_index', 'reason', 'review_decision')} for r in rejected],
              ('episode_id', 'group', 'subgroup', 'source', 'episode_index', 'reason', 'review_decision'))
    summary = dict(total=len(inventory), adopted=len(ordered), excluded=len(rejected), train=len(train), eval=len(eval_rows),
                   frames=sum(int(r['frames']) for r in ordered),
                   eval_split=(len(eval_rows) - .5) / len(ordered),
                   groups={g: dict(total=len(items), train=sum(r['group'] == g for r in train), eval=sum(r['group'] == g for r in eval_rows)) for g, items in by_group.items()})
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
