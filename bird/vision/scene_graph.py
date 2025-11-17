"""
Scene Graph Generation
The VLM identifies objects, their attributes, and relationships between them.
Scene graphs are generated periodically (every N frames) due to VLM latency.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import json
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO

# Optional VLM clients
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    import base64
    import os
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SceneGraphOntology:
    """Define the ontology: object classes, relationships, and attributes"""
    
    # Spatial relationship predicates
    SPATIAL_PREDICATES = [
        'near',           # Close proximity
        'far',            # Distant
        'above',          # Vertically above
        'below',          # Vertically below
        'left_of',        # Horizontally left
        'right_of',       # Horizontally right
    ]
    
    # Semantic relationship predicates (require VLM)
    SEMANTIC_PREDICATES = [
        'holding',
        'carrying',
        'wearing',
        'touching',
        'riding',
        'interacting_with',
        'facing',
        'looking_at',
    ]
    
    # Motion-based attributes
    MOTION_ATTRIBUTES = [
        'stationary',
        'moving',
        'moving_fast',
        'moving_slow',
    ]
    
    # Pose-based attributes (for humans)
    POSE_ATTRIBUTES = [
        'standing',
        'sitting',
        'walking',
        'running',
        'waving',
    ]


class SceneGraphBuilder:
    """
    Builds scene graphs exclusively from VLM analysis.
    
    The VLM analyzes raw frames to identify:
    - Objects and their attributes (e.g., "person, moving, standing")
    - Relationships between objects (e.g., "person near car", "person holding phone")
    
    Runs periodically (every vlm_interval frames) due to VLM latency.
    """
    
    def __init__(
        self,
        use_vlm: bool = False,
        vlm_provider: str = "ollama",  # 'ollama' or 'openai'
        vlm_model: str = "llava:7b",
        vlm_interval: int = 30,  # Run VLM every N frames
        near_threshold: float = 150.0,  # Pixels for "near" relationship
        motion_threshold: float = 5.0,   # Pixels/frame for motion detection
    ):
        self.vlm_provider = vlm_provider.lower()
        self.vlm_model = vlm_model
        self.vlm_interval = vlm_interval
        self.near_threshold = near_threshold
        self.motion_threshold = motion_threshold
        
        # Check if VLM is available
        if self.vlm_provider == "ollama":
            self.use_vlm = use_vlm and OLLAMA_AVAILABLE
            if use_vlm and not OLLAMA_AVAILABLE:
                print("Ollama not available. Install: pip install ollama")
                self.use_vlm = False
        elif self.vlm_provider == "openai":
            self.use_vlm = use_vlm and OPENAI_AVAILABLE
            if use_vlm and not OPENAI_AVAILABLE:
                print("OpenAI not available. Install: pip install openai")
                self.use_vlm = False
            if self.use_vlm:
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                if not os.getenv("OPENAI_API_KEY"):
                    print("OPENAI_API_KEY not set in environment")
                    self.use_vlm = False
        else:
            print(f"Unknown VLM provider: {self.vlm_provider}")
            self.use_vlm = False
        
        self.frame_count = 0
        self.cached_semantic_relationships = []
        self.last_vlm_time = 0
        
        if self.use_vlm:
            self._check_vlm_ready()
    
    def _check_vlm_ready(self):
        """Check if VLM is ready to use"""
        if self.vlm_provider == "ollama":
            try:
                models = ollama.list()
                model_names = [m.model for m in models.get('models', [])]
                if self.vlm_model not in model_names:
                    raise ValueError(f"Ollama model {self.vlm_model} not found. Run: ollama pull {self.vlm_model}")
                print(f"✓ Using Ollama {self.vlm_model}")
            except Exception as e:
                print(f"⚠️  Ollama error: {e}")
                self.use_vlm = False
        elif self.vlm_provider == "openai":
            print(f"✓ Using OpenAI {self.vlm_model}")
    
    def build_graph(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict] = None,  # Unused, kept for API compatibility
    ) -> Optional[Dict]:
        """
        Build scene graph exclusively from VLM analysis.
        Only runs every vlm_interval frames. Returns None on non-VLM frames.
        Returns:
            {
                'frame_id': int,
                'timestamp': float,
                'objects': List[Dict],
                'relationships': List[Dict]
            }
            or None if not a VLM frame
        """
        # Only build graph on VLM frames
        if not self.use_vlm or self.frame_count % self.vlm_interval != 0:
            self.frame_count += 1
            return None
        
        print(f"[Frame {self.frame_count}] Running VLM analysis...")
        
        # Get complete scene understanding from VLM
        vlm_scene = self._analyze_scene_with_vlm(frame)
        
        scene_graph = {
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'objects': vlm_scene.get('objects', []),
            'relationships': vlm_scene.get('relationships', []),
        }
        self.frame_count += 1
        return scene_graph
    
    def _analyze_scene_with_vlm(self, frame: np.ndarray) -> Dict:
        """
        Use VLM to analyze the entire scene: objects, attributes, and relationships.
        
        Returns:
            {
                'objects': List[Dict],  # List of detected objects with attributes
                'relationships': List[Dict]  # List of relationships between objects
            }
        """
        if not self.use_vlm:
            return {'objects': [], 'relationships': []}
        
        if self.vlm_provider == "ollama":
            return self._analyze_with_ollama(frame)
        elif self.vlm_provider == "openai":
            return self._analyze_with_openai(frame)
        else:
            return {'objects': [], 'relationships': []}
    
    def _analyze_with_ollama(self, frame: np.ndarray) -> Dict:
        """Analyze scene using Ollama"""
        _, buffer = cv2.imencode('.jpg', frame)
        
        prompt = """Analyze this image and describe the scene as a structured graph.
            Identify all significant objects, their states/attributes, and relationships between them.
            Respond ONLY with valid JSON in this exact format:
            {
            "objects": [
                {"id": 0, "class": "person", "attributes": ["moving", "standing"]},
                {"id": 1, "class": "car", "attributes": ["stationary", "red"]}
            ],
            "relationships": [
                {"subject": 0, "predicate": "near", "object": 1}
            ]
            }
            Common attributes: moving, stationary, sitting, standing, walking, running, open, closed
            Common relationships: near, holding, carrying, wearing, riding, touching, facing, looking_at
            If the scene is empty, return: {"objects": [], "relationships": []}
        """
        
        try:
            start = time.time()
            response = ollama.chat(
                model=self.vlm_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [buffer.tobytes()]
                }]
            )
            elapsed = time.time() - start
            
            content = response['message']['content']
            
            # Extract JSON (handle markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            data = json.loads(content.strip())
            objects = data.get('objects', [])
            relationships = data.get('relationships', [])
            
            print(f"  Ollama found {len(objects)} objects and {len(relationships)} relationships in {elapsed:.2f}s")
            return {'objects': objects, 'relationships': relationships}
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Failed to parse Ollama response: {e}")
            if 'content' in locals():
                print(f"  Raw response: {content[:200]}")
            return {'objects': [], 'relationships': []}
        except Exception as e:
            print(f"  ⚠️  Ollama error: {e}")
            return {'objects': [], 'relationships': []}
    
    def _analyze_with_openai(self, frame: np.ndarray) -> Dict:
        """Analyze scene using OpenAI GPT-4o"""
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        prompt = """Analyze this image and describe the scene as a structured graph.

Identify all significant objects, their states/attributes, and relationships between them.

Respond ONLY with valid JSON in this exact format:
{
  "objects": [
    {"id": 0, "class": "person", "attributes": ["moving", "standing"]},
    {"id": 1, "class": "car", "attributes": ["stationary", "red"]}
  ],
  "relationships": [
    {"subject": 0, "predicate": "near", "object": 1}
  ]
}

Common attributes: moving, stationary, sitting, standing, walking, running, open, closed
Common relationships: near, holding, carrying, wearing, riding, touching, facing, looking_at

If the scene is empty, return: {"objects": [], "relationships": []}
"""
        
        try:
            start = time.time()
            response = self.openai_client.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=1000
            )
            elapsed = time.time() - start
            
            content = response.choices[0].message.content
            
            # Extract JSON (handle markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            data = json.loads(content.strip())
            objects = data.get('objects', [])
            relationships = data.get('relationships', [])
            
            print(f"  GPT-4o found {len(objects)} objects and {len(relationships)} relationships in {elapsed:.2f}s")
            return {'objects': objects, 'relationships': relationships}
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Failed to parse GPT-4o response: {e}")
            if 'content' in locals():
                print(f"  Raw response: {content[:200]}")
            return {'objects': [], 'relationships': []}
        except Exception as e:
            print(f"  ⚠️  GPT-4o error: {e}")
            return {'objects': [], 'relationships': []}
    
    def format_graph_natural_language(self, scene_graph: Dict) -> str:
        """Convert scene graph to natural language description"""
        if not scene_graph:
            return "No scene graph available."
            
        objects = scene_graph.get('objects', [])
        relationships = scene_graph.get('relationships', [])
        
        if not objects:
            return "Empty scene."
        
        # Build object descriptions
        obj_desc = []
        for obj in objects:
            attrs = ', '.join(obj.get('attributes', [])) if obj.get('attributes') else 'present'
            obj_desc.append(f"Object #{obj['id']} is a {obj['class']} ({attrs})")
        
        # Build relationship descriptions
        rel_desc = []
        obj_map = {obj['id']: obj for obj in objects}
        
        for rel in relationships:
            subj = obj_map.get(rel['subject'])
            obj = obj_map.get(rel['object'])
            if subj and obj:
                rel_desc.append(
                    f"{subj['class']} #{rel['subject']} is {rel['predicate']} "
                    f"{obj['class']} #{rel['object']}"
                )
        
        # Combine
        description = "Scene: " + "; ".join(obj_desc)
        if rel_desc:
            description += ". Relationships: " + "; ".join(rel_desc)
        
        return description
    
    def _create_graph_visualization(self, scene_graph: Dict, width: int = 800, height: int = 600) -> np.ndarray:
        """
        Create a networkx graph visualization from the scene graph.
        
        Returns an image array that can be displayed.
        """
        objects = scene_graph.get('objects', [])
        relationships = scene_graph.get('relationships', [])
        
        if not objects:
            # Return blank image with "Empty scene" text
            fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
            ax.text(0.5, 0.5, 'Empty Scene', ha='center', va='center', 
                   fontsize=20, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            plt.close(fig)
            
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return cv2.resize(img, (width, height))
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes (objects)
        for obj in objects:
            obj_id = obj['id']
            obj_class = obj['class']
            attrs = obj.get('attributes', [])
            
            # Create label with class and attributes
            label = f"{obj_class}\n#{obj_id}"
            if attrs:
                label += f"\n({', '.join(attrs[:2])})"  # Show first 2 attributes
            
            G.add_node(obj_id, label=label, obj_class=obj_class)
        
        # Add edges (relationships)
        edge_labels = {}
        for rel in relationships:
            subj = rel['subject']
            obj = rel['object']
            pred = rel['predicate']
            
            if subj in G.nodes and obj in G.nodes:
                G.add_edge(subj, obj)
                edge_labels[(subj, obj)] = pred
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            # Color code by object class
            obj_class = G.nodes[node].get('obj_class', '')
            if 'person' in obj_class.lower():
                node_colors.append('#FF6B6B')  # Red for people
            elif any(x in obj_class.lower() for x in ['computer', 'monitor', 'screen']):
                node_colors.append('#4ECDC4')  # Cyan for tech
            elif any(x in obj_class.lower() for x in ['chair', 'desk', 'table']):
                node_colors.append('#95E1D3')  # Light green for furniture
            else:
                node_colors.append('#FFE66D')  # Yellow for other
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                               node_size=3000, alpha=0.9, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                               arrows=True, arrowsize=20, 
                               width=2, alpha=0.6, ax=ax)
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=9, 
                                font_weight='bold', ax=ax)
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.axis('off')
        ax.set_title(f"Scene Graph ({len(objects)} objects, {len(relationships)} relationships)", 
                     fontsize=12, pad=20)
        
        plt.tight_layout()
        
        # Convert to opencv image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)
        
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return cv2.resize(img, (width, height))
    
    def draw_scene_graph(
        self,
        frame: np.ndarray,
        scene_graph: Dict,
    ) -> np.ndarray:
        """
        Draw scene graph visualization alongside the frame.
        
        Creates a side-by-side view with the camera frame and graph visualization.
        """
        if not scene_graph:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create graph visualization
        graph_img = self._create_graph_visualization(scene_graph, width=w, height=h)
        
        # Combine frame and graph side by side
        combined = np.hstack([frame, graph_img])
        
        return combined

