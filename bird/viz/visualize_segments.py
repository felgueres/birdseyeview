import json
import matplotlib.pyplot as plt
import numpy as np

with open("bedroom_comparison.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
methods = ['threshold', 'derivative', 'adaptive']
titles = ['Threshold (0.96)', 'Derivative', 'Adaptive']

for idx, (method, title) in enumerate(zip(methods, titles)):
    ax = axes[idx]
    result = data[method]

    similarities = np.array(result['similarities'])
    frames = np.arange(len(similarities))

    ax.plot(frames, similarities, linewidth=1, alpha=0.7, label='Similarity')

    for cp in result['change_points']:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

    for i, segment in enumerate(result['segments']):
        start = segment['start_frame']
        end = segment['end_frame']
        ax.axvspan(start, end, alpha=0.15, color=f'C{i % 10}')

    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'{title} - {result["total_segments"]} segments, {len(result["change_points"])} change points')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig('bedroom_segmentation_comparison.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: bedroom_segmentation_comparison.png")
print("\nSegment breakdown:")
print("="*60)

for method, title in zip(methods, titles):
    result = data[method]
    print(f"\n{title}:")
    for i, segment in enumerate(result['segments'], 1):
        duration = segment['duration']
        print(f"  Segment {i:2d}: {segment['start_time']:5.2f}s - {segment['end_time']:5.2f}s ({duration:.2f}s)")
