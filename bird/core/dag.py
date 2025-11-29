from functools import reduce
import graphviz

class Transform():
	def __init__(self, name: str, input_keys:list[str], output_keys:list[str]):
		self.name = name or ''
		self.input_keys = input_keys or []
		self.output_keys = output_keys or []

	def forward(self, inputs: dict)->dict:
		raise NotImplementedError

class SumTransform(Transform):
	def __init__(self, input_keys:list[str], output_keys:list[str]):
		assert len(output_keys) == 1
		super().__init__('sumTransform', input_keys, output_keys)

	def forward(self, inputs:dict) -> dict:
		return {self.output_keys[0] : sum([inputs[k] for k in self.input_keys])}

class MulTransform(Transform):
	def __init__(self, input_keys:list[str], output_keys:list[str]):
		assert len(output_keys) == 1
		super().__init__('mulTransform', input_keys, output_keys)

	def forward(self, inputs:dict) -> dict:
		return {self.output_keys[0] : reduce(lambda a,b: a*b, [inputs[k] for k in self.input_keys])}

class DAG():

	def __init__(self, transforms: list[Transform]):
		self.transforms = transforms
		self.execution_order = transforms

	def topo(self):
        # for this we need to add to each node ability to know precedents
		pass

	def vis(self):
		assert self.transforms
		dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
		for t in self.execution_order:
			for k in t.input_keys:
				dot.node(k)
				dot.edge(tail_name=k, head_name=f"{t.output_keys[0]}")
		dot.render(directory='./doctest-output', view=True)

	def forward(self, inputs):
		state = dict(inputs)
		for t in self.execution_order:
			in_dict = {k: state[k] for k in t.input_keys}
			out = t.forward(in_dict)
			state.update(out)
		return state

inputs = { 'a' : 2, 'b' : 3, 'c': 4}
dag = DAG([SumTransform(input_keys=['a','b','c'], output_keys=['d']),
	MulTransform(input_keys=['a', 'd'], output_keys=['e']),
	SumTransform(input_keys=['e', 'd'], output_keys=['f'])])

dag.vis()
out = dag.forward(inputs=inputs)
print(out)
