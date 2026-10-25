"""Make small, read-only visual review packages from an existing episode audit."""

import argparse
import csv
import json
from pathlib import Path
import zipfile

import av
from PIL import Image, ImageDraw
import pyarrow.parquet as pq


def write_csv(path, rows, fields):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def video_ranges(source):
    files = sorted((source / 'data').rglob('*.parquet'))
    videos = sorted((source / 'videos/observation.images.iphone').rglob('*.mp4'))
    if len(files) != 1 or len(videos) != 1:
        raise ValueError(f'Expected one Parquet and video: {source}')
    table = pq.read_table(files[0], columns=['episode_index'])
    ranges = {}
    for index, episode in enumerate(table.column('episode_index').to_pylist()):
        episode = int(episode)
        if episode not in ranges:
            ranges[episode] = [index, index + 1]
        else:
            ranges[episode][1] = index + 1
    return videos[0], ranges


def make_media(video, start, end, episode_id, output, make_video, width):
    count = end - start
    targets = sorted(set(round(i * (count - 1) / 7) for i in range(8)))
    frames = []
    encoded = 0
    if make_video:
        container = av.open(str(output / 'videos' / f'{episode_id}.mp4'), 'w')
        stream = container.add_stream('libx264', rate=5)
        stream.width, stream.height = width, width * 3 // 4
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '32', 'preset': 'fast'}
    try:
        with av.open(str(video)) as source:
            for global_index, frame in enumerate(source.decode(video=0)):
                if global_index >= end:
                    break
                if global_index < start:
                    continue
                local = global_index - start
                if local in targets:
                    image = frame.to_image().resize((320, 240))
                    frames.append((local, image))
                if make_video and local % 4 == 0:
                    image = frame.to_image().resize((width, width * 3 // 4))
                    for packet in stream.encode(av.VideoFrame.from_image(image)):
                        container.mux(packet)
                    encoded += 1
        if make_video:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        if make_video:
            container.close()
    if len(frames) != len(targets):
        raise ValueError(f'Video too short: {video} {episode_id}')
    sheet = Image.new('RGB', (1280, 540), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), f'{episode_id}  |  20 fps source  |  {count} frames', fill='black')
    for i, (local, image) in enumerate(frames):
        x, y = (i % 4) * 320, 24 + (i // 4) * 258
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + 240), f'{local / 20:.1f}s  frame {local}', fill='black')
    sheet.save(output / 'sheets' / f'{episode_id}.jpg', quality=82)
    return encoded


def zip_parts(output, max_bytes):
    groups = []
    for sheet in sorted((output / 'sheets').glob('*.jpg')):
        video = output / 'videos' / f'{sheet.stem}.mp4'
        groups.append([sheet, video] if video.exists() else [sheet])
    common = [output / 'episodes.csv', output / 'review_manifest.csv', output / 'README.txt']
    parts, current, size = [], [], 0
    for files in groups:
        group_size = sum(path.stat().st_size for path in files)
        if current and size + group_size > max_bytes:
            parts.append(current)
            current, size = [], 0
        current.extend(files)
        size += group_size
    if current:
        parts.append(current)
    summary = []
    for number, files in enumerate(parts, 1):
        path = output / f'review-part-{number:02d}.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as bundle:
            for item in common + files:
                bundle.write(item, item.relative_to(output))
        episode_ids = {p.stem for p in files}
        summary.append(dict(zip=path.name, bytes=path.stat().st_size,
                            mib=round(path.stat().st_size / 2**20, 2),
                            episode_count=len(episode_ids), episode_ids=';'.join(sorted(episode_ids))))
    write_csv(output / 'zip_parts.csv', summary, ('zip', 'bytes', 'mib', 'episode_count', 'episode_ids'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', action='store_true', help='Encode review episodes at 5 fps')
    parser.add_argument('--width', type=int, choices=(320, 640), default=320)
    parser.add_argument('--zip-mib', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = list(csv.DictReader((args.audit / 'episodes.csv').open()))
    if len({(r['source_folder'], r['session'], r['episode_index']) for r in rows}) != len(rows):
        raise ValueError('Duplicate source episode')
    args.output.mkdir(parents=True)
    (args.output / 'sheets').mkdir()
    (args.output / 'videos').mkdir()
    for i, row in enumerate(rows, 1):
        row['episode_id'] = f'SO101-{i:04d}'
    write_csv(args.output / 'episodes.csv', rows, ('episode_id', *[k for k in rows[0] if k != 'episode_id']))
    review = []
    by_source = {}
    for row in rows:
        if row['status'] != 'review':
            continue
        source = Path(row['source_folder']) / row['session']
        if source not in by_source:
            by_source[source] = video_ranges(source)
        video, ranges = by_source[source]
        episode = int(row['episode_index'])
        if episode not in ranges or ranges[episode][1] - ranges[episode][0] != int(row['frames']):
            raise ValueError(f'Episode/video frame mismatch: {source} ep={episode}')
        encoded = make_media(video, *ranges[episode], row['episode_id'], args.output, args.videos, args.width)
        review.append(dict(episode_id=row['episode_id'], source=str(source), episode_index=episode,
                           reason=row['reason'], frames=row['frames'], duration_s=row['duration_s'],
                           sheet=f'sheets/{row["episode_id"]}.jpg',
                           video=f'videos/{row["episode_id"]}.mp4' if args.videos else '',
                           encoded_frames=encoded))
        print(f'{row["episode_id"]}: {source.name} ep={episode} sheet={len(review)}', flush=True)
    write_csv(args.output / 'review_manifest.csv', review, tuple(review[0]) if review else
              ('episode_id', 'source', 'episode_index', 'reason', 'frames', 'duration_s', 'sheet', 'video', 'encoded_frames'))
    (args.output / 'README.txt').write_text(
        'These images/videos are visual review aids, not success labels.\n'
        'Use episodes.csv to map each SO101 ID to its source session and episode_index.\n'
        'Only status=review was rendered. Original datasets and videos were not changed.\n'
        'Video time matches source time: every fourth 20-fps source frame at 5 fps.\n')
    summary = zip_parts(args.output, args.zip_mib * 2**20)
    (args.output / 'manifest.json').write_text(json.dumps(dict(total=len(rows), review=len(review),
        videos=args.videos, width=args.width, zip_parts=summary), indent=2) + '\n')
    print(f'{len(rows)} episodes; {len(review)} review; {len(summary)} ZIP parts')


if __name__ == '__main__':
    main()
