Transforms perception into discrete events

```
┌─────────────────────────────────────────────────────────────────┐
│                         DAG                                     |
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frame → Depth → Detection → Tracking → EVENT DETECTION         │
│                                              ↓                  │
│                                         [Motion Events]         │
│                                         [Safety Events]         │
│                                         [Interaction Events]    │
│                                              ↓                  │
│                                    EVENT SERIALIZATION          │
│                                              ↓                  │
│                                    sessions/<timestamp>/        │
│                                      - events.jsonl             │
│                                      - scene_graph.jsonl        │
└─────────────────────────────────────────────────────────────────┘
```
