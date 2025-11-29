from functools import reduce
import graphviz
import logging

logger = logging.getLogger(__name__)


class Transform:
    def __init__(
        self,
        name: str,
        input_keys: list[str],
        output_keys: list[str],
        run_every_n_frames: int = 1,
        critical: bool = True
    ):
        self.name = name or ""
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.run_every_n_frames = run_every_n_frames
        self.critical = critical

    def should_run(self, state: dict) -> bool:
        frame_count = state.get('frame_count', 0)
        return frame_count % self.run_every_n_frames == 0

    def forward(self, inputs: dict) -> dict:
        raise NotImplementedError


class SumTransform(Transform):
    def __init__(self, input_keys: list[str], output_keys: list[str]):
        assert len(output_keys) == 1
        super().__init__("sumTransform", input_keys, output_keys)
    def forward(self, inputs: dict) -> dict:
        return {self.output_keys[0]: sum([inputs[k] for k in self.input_keys])}


class MulTransform(Transform):
    def __init__(self, input_keys: list[str], output_keys: list[str]):
        assert len(output_keys) == 1
        super().__init__("mulTransform", input_keys, output_keys)

    def forward(self, inputs: dict) -> dict:
        return {
            self.output_keys[0]: reduce(
                lambda a, b: a * b, [inputs[k] for k in self.input_keys]
            )
        }


class DAG:

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms
        self.execution_order = transforms

    def _topo(self):
        adj_list = {}
        for t in self.transforms:
            for k in t.input_keys:
                if k not in adj_list:
                    adj_list[k] = []
                for o in t.output_keys:
                    adj_list[k].append(o)
                    if o not in adj_list:
                        adj_list[o] = []

        visited = set()
        topo = []
        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in adj_list[v]:
                    dfs(child)
                topo.append(v)
        
        for k in adj_list.keys():
            if k not in visited:
                dfs(k)

        print(topo[::-1])
        

    def vis(self):
        assert self.transforms
        dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})
        for t in self.execution_order:
            for k in t.input_keys:
                dot.node(k)
                dot.edge(tail_name=k, head_name=f"{t.output_keys[0]}")
        dot.render(directory="./doctest-output", view=True)

    def forward(self, inputs):
        state = dict(inputs)
        for t in self.execution_order:
            if t.should_run(state):
                in_dict = {k: state.get(k) for k in t.input_keys if k in state}
                try:
                    out = t.forward(in_dict)
                    state.update(out)
                except Exception as e:
                    if t.critical:
                        raise
                    else:
                        logger.warning(f"Transform {t.name} failed: {e}, skipping")
        return state


if __name__ == "__main__":
    inputs = {"a": 2, "b": 3, "c": 4}
    dag = DAG(
        [
            SumTransform(input_keys=["a", "b", "c"], output_keys=["d"]),
            MulTransform(input_keys=["a", "d"], output_keys=["e"]),
            SumTransform(input_keys=["e", "d"], output_keys=["f"]),
        ]
    )
    dag._topo()
    out = dag.forward(inputs=inputs)
    dag.vis()
