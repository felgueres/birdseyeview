import json
from pathlib import Path
from collections import defaultdict


class EventVisualizer:
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.events_file = self.session_dir / "events.jsonl"

    def print_timeline(self, event_type_filter: str = None):
        if not self.events_file.exists():
            print(f"No events file found at {self.events_file}")
            return

        print("\nEvent Timeline")
        print("=" * 80)

        with open(self.events_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                frame = entry['frame']
                timestamp = entry['timestamp']
                events = entry['events']

                for event in events:
                    event_type = event['type']

                    if event_type_filter and event_type != event_type_filter:
                        continue

                    print(f"Frame {frame:4d} | {timestamp:.2f}s | {event_type:25s} | "
                          f"conf: {event['confidence']:.2f} | "
                          f"objects: {event['objects']}")

                    if event.get('meta'):
                        for key, value in event['meta'].items():
                            print(f"          └─ {key}: {value}")

    def get_statistics(self):
        if not self.events_file.exists():
            return {}

        stats = defaultdict(int)
        event_types = defaultdict(list)

        with open(self.events_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                for event in entry['events']:
                    event_type = event['type']
                    stats[event_type] += 1
                    event_types[event_type].append(event)

        return {
            'total_events': sum(stats.values()),
            'by_type': dict(stats),
            'events': dict(event_types)
        }

    def print_summary(self):
        stats = self.get_statistics()

        if not stats:
            print("No events found")
            return

        print("\nEvent Summary")
        print("=" * 80)
        print(f"Total Events: {stats['total_events']}")
        print("\nBreakdown by Type:")

        for event_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
            print(f"  {event_type:30s}: {count:4d}")


def visualize_session(session_dir: str, event_filter: str = None):
    viz = EventVisualizer(session_dir)

    viz.print_summary()
    print()
    viz.print_timeline(event_type_filter=event_filter)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m bird.events.visualizer <session_dir> [event_type_filter]")
        sys.exit(1)

    session_dir = sys.argv[1]
    event_filter = sys.argv[2] if len(sys.argv) > 2 else None

    visualize_session(session_dir, event_filter)
