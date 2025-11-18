#!/usr/bin/env python3
"""
Demo script showing the overlay UI with simulated data.
Useful for testing the overlay without a camera.
"""

import cv2
import numpy as np
import time
from bird.vision.overlay import InfoOverlay


def create_demo_frame():
    """Create a demo frame with some visual content"""
    frame = np.random.randint(50, 150, (720, 1280, 3), dtype=np.uint8)
    
    # Draw some demo objects
    cv2.circle(frame, (300, 200), 50, (0, 255, 0), -1)
    cv2.putText(frame, "DEMO: Camera Feed", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.rectangle(frame, (100, 300), (250, 450), (255, 0, 0), 3)
    cv2.putText(frame, "Person", (110, 290),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.rectangle(frame, (500, 400), (700, 550), (0, 255, 255), 3)
    cv2.putText(frame, "Car", (510, 390),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame


def main():
    """Run the overlay demo"""
    print("🐦 BirdView Overlay Demo")
    print("=" * 50)
    print("This demo shows the overlay UI with simulated data.")
    print("Press 'Q' to quit")
    print("Press 'P' to toggle position (right/left/top/bottom)")
    print("=" * 50)
    
    # Create overlay
    overlay = InfoOverlay(position='right', width=350, alpha=0.75)
    
    positions = ['right', 'left', 'top', 'bottom']
    current_pos_idx = 0
    
    frame_count = 0
    start_time = time.time()
    
    event_triggers = [
        (30, "New person detected (ID:1)"),
        (60, "New car detected (ID:2)"),
        (90, "High motion detected: 120"),
        (120, "VLM analysis complete"),
        (150, "New bottle detected (ID:3)"),
        (180, "Track lost: ID:1"),
    ]
    
    while True:
        # Create demo frame
        frame = create_demo_frame()
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Simulate varying metrics
        detections = 2 + int(np.sin(frame_count * 0.05) * 2)
        tracked = min(detections, 3)
        motion_energy = 50 + np.random.rand() * 100
        frame_time = 20 + np.random.rand() * 15
        
        # Build metrics
        metrics = {
            'Frame': frame_count,
            'FPS': fps,
            'Detections': detections,
            'Tracked Objects': tracked,
            'Active Tracks': tracked,
            '  person': max(1, tracked - 1),
            '  car': 1 if tracked > 1 else 0,
            'Motion Energy': motion_energy,
            'Flow Points': 45 + int(np.random.rand() * 20),
            'Frame Time (ms)': frame_time,
        }
        
        # Check for events
        events = []
        for trigger_frame, event_msg in event_triggers:
            if frame_count == trigger_frame:
                events.append(event_msg)
        
        # Draw overlay
        frame = overlay.draw(frame, metrics, events)
        
        # Add instructions
        cv2.putText(frame, "Press 'Q' to quit, 'P' to change position",
                   (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('BirdView Overlay Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\nExiting demo...")
            break
        elif key == ord('p'):
            current_pos_idx = (current_pos_idx + 1) % len(positions)
            new_position = positions[current_pos_idx]
            overlay = InfoOverlay(position=new_position, width=350, alpha=0.75)
            print(f"Changed position to: {new_position}")
        
        frame_count += 1
    
    cv2.destroyAllWindows()
    print(f"Demo complete! Processed {frame_count} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

