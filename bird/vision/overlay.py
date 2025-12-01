import cv2
import numpy as np
from typing import Dict, List, Any

class InfoOverlay:
    """
    Overlay information panel on video frames for real-time metrics display.
    """
    
    def __init__(self, position='right', width=350, bg_color=(0, 0, 0), 
                 text_color=(255, 255, 255), alpha=0.7):
        """
        Args:
            position: 'right', 'left', 'top', 'bottom'
            width: Width of the panel in pixels (or height for top/bottom)
            bg_color: Background color (BGR)
            text_color: Text color (BGR)
            alpha: Transparency (0.0 = transparent, 1.0 = opaque)
        """
        self.position = position
        self.width = width
        self.bg_color = bg_color
        self.text_color = text_color
        self.alpha = alpha
        self.events_log = []
        self.max_events = 5
    
    def add_event(self, event: str):
        """Add an event to the events log"""
        self.events_log.append(event)
        if len(self.events_log) > self.max_events:
            self.events_log.pop(0)
    
    def draw(self, frame: np.ndarray, metrics: Dict[str, Any],
             events: List = None) -> np.ndarray:
        """
        Draw overlay on frame with metrics and events.

        Args:
            frame: Input frame
            metrics: Dictionary of metrics to display
            events: Optional list of recent events (strings or dicts)

        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()

        if events:
            for event in events:
                if isinstance(event, dict):
                    if event.get('type') == 'vlm_segment_event':
                        description = event.get('meta', {}).get('description', '')
                        if description:
                            self.add_event(description)
                    else:
                        event_str = f"{event.get('type', 'event')} [{','.join(map(str, event.get('objects', [])))}]"
                        self.add_event(event_str)
                else:
                    self.add_event(str(event))
        
        if self.position == 'right':
            return self._draw_side_panel(annotated_frame, metrics, 'right')
        elif self.position == 'left':
            return self._draw_side_panel(annotated_frame, metrics, 'left')
        elif self.position == 'top':
            return self._draw_horizontal_panel(annotated_frame, metrics, 'top')
        elif self.position == 'bottom':
            return self._draw_horizontal_panel(annotated_frame, metrics, 'bottom')
        
        return annotated_frame
    
    def _draw_side_panel(self, frame: np.ndarray, metrics: Dict[str, Any], 
                         side: str) -> np.ndarray:
        """Draw side panel (left or right)"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        if side == 'right':
            x_start = w - self.width
            x_end = w
        else:  # left
            x_start = 0
            x_end = self.width
        
        cv2.rectangle(overlay, (x_start, 0), (x_end, h), self.bg_color, -1)
        frame = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)
        
        # Simple layout
        y_offset = 25
        x_text = x_start + 10
        
        # Draw metrics - simple format
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.1f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(frame, text, (x_text, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1)
            
            y_offset += 22
        
        # Draw events log - latest at top
        if self.events_log and y_offset < h - 100:
            y_offset += 10

            recent_events = self.events_log[-4:][::-1]
            total_events = len(self.events_log)

            for idx, event in enumerate(recent_events):
                event_num = total_events - idx
                max_chars = 32

                if len(event) > max_chars:
                    words = event.split()
                    lines = []
                    current_line = []
                    current_length = 0

                    for word in words:
                        if current_length + len(word) + 1 <= max_chars:
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)

                    if current_line:
                        lines.append(' '.join(current_line))

                    for line_idx, line in enumerate(lines[:3]):
                        prefix = f"{event_num}. " if line_idx == 0 else "   "
                        cv2.putText(frame, prefix + line, (x_text, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
                        y_offset += 18
                else:
                    cv2.putText(frame, f"{event_num}. {event}", (x_text, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
                    y_offset += 18
        
        return frame
    
    def _draw_horizontal_panel(self, frame: np.ndarray, metrics: Dict[str, Any],
                               position: str) -> np.ndarray:
        """Draw horizontal panel (top or bottom)"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        if position == 'top':
            y_start = 0
            y_end = self.width
        else:  # bottom
            y_start = h - self.width
            y_end = h
        
        cv2.rectangle(overlay, (0, y_start), (w, y_end), self.bg_color, -1)
        frame = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)
        
        # Draw metrics horizontally
        x_offset = 20
        y_text = y_start + 30 if position == 'top' else y_start + 20
        
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(frame, text, (x_offset, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            x_offset += 200
            
            if x_offset > w - 200:
                x_offset = 20
                y_text += 25
        
        return frame
    
    
    def draw_fps_counter(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Quick helper to draw FPS in corner"""
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
        return frame

