from collections import defaultdict

class Context:
	"""
	Program context holding execution state.
	"""

	def __init__(self, ctx=None):
		# Map from argId -> values
		if ctx == None:
			self.ctx = defaultdict(list)
		else:
			self.ctx = ctx

	def add(self, target, value):
		"""Add the value to the target's current context.
		"""
		self.ctx[target].append(value)

	def set(self, target, value):
		"""Set the target's current context to the value.
		"""
		self.ctx[target] = value

	def set_last_value(self, target, value):
		"""Set the last value in the target's current context to the value.
		"""
		self.ctx[target][-1] = value

	def get_last_value(self, target):
		"""Get the last value in the target's current context.
		"""
		return self.ctx[target][-1]

	def items(self):
		return self.ctx.items()
