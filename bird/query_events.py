import argparse
from pathlib import Path
from bird.events.writer import EventWriter


def list_sessions():
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        print("No sessions directory found")
        return []

    sessions = sorted([d for d in sessions_dir.iterdir() if d.is_dir()], reverse=True)
    return sessions


def main():
    parser = argparse.ArgumentParser(description='Query events from database')
    parser.add_argument('--session', type=str, help='Session directory (e.g., sessions/2025-12-01T07-40-31)')
    parser.add_argument('--list', action='store_true', help='List all sessions')
    parser.add_argument('--query', type=str, help='Search query (e.g., "child jumping on bed")')
    parser.add_argument('--type', type=str, help='Filter by event type')
    parser.add_argument('--limit', type=int, default=10, help='Number of results (default: 10)')
    parser.add_argument('--stats', action='store_true', help='Show session statistics')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable sessions:")
        print("=" * 60)
        sessions = list_sessions()
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session.name}")
        print()
        return

    if not args.session:
        sessions = list_sessions()
        if not sessions:
            print("No sessions found. Run with --list to see available sessions.")
            return
        args.session = str(sessions[0])
        print(f"Using latest session: {args.session}")

    writer = EventWriter(session_dir=args.session)

    if not writer.database:
        print("Database not available")
        return

    if args.stats:
        stats = writer.database.get_event_statistics()
        print(f"\nSession: {args.session}")
        print("=" * 60)
        print(f"Total events: {stats['total_events']}")
        print(f"\nEvents by type:")
        for event_type, count in stats['by_type'].items():
            print(f"  {event_type}: {count}")
        if stats['time_range']['start'] is not None:
            print(f"\nTime range: {stats['time_range']['start']:.2f}s - {stats['time_range']['end']:.2f}s")
        print()
        return

    if args.query:
        print(f"\nSearching for: '{args.query}'")
        print("=" * 60)

        if not writer.embedder:
            print("Embeddings not available")
            return

        query_embedding = writer.embedder.embed_query(args.query)
        results = writer.database.search_by_embedding(
            query_embedding,
            top_k=args.limit,
            model=writer.embedder.model_name
        )

        if not results:
            print("No results found")
            return

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Frame {result['frame']} | {result['type']} | Similarity: {result['similarity']:.4f}")
            if 'description' in result['meta']:
                print(f"   {result['meta']['description']}")
            elif 'class' in result['meta']:
                print(f"   Class: {result['meta']['class']}")
            if 'object_class' in result['meta']:
                print(f"   Object: {result['meta']['object_class']}")
        print()
        return

    events = writer.database.get_events(
        event_type=args.type,
        limit=args.limit
    )

    print(f"\nEvents in session: {args.session}")
    print("=" * 60)
    print(f"Showing {len(events)} events\n")

    for event in events:
        print(f"Frame {event['frame']} | {event['type']} | Confidence: {event['confidence']:.2f}")
        if 'description' in event['meta']:
            print(f"  {event['meta']['description']}")
        elif 'class' in event['meta']:
            print(f"  Class: {event['meta']['class']}")
        if 'object_class' in event['meta']:
            print(f"  Object: {event['meta']['object_class']}")
        print()


if __name__ == "__main__":
    main()
