import equinox as eqx

class V(eqx.Module):
  u: float = 0.0
  v: float = 0.0
  
  def __add__(self, other):
    return V(u=self.u + other.u, v=self.v + other.v)
  
  def __sub__(self, other):
    return V(u=self.u - other.u, v=self.v - other.v)
  
u = V(u=1.0, v=2.0)
v = V(u=3.0, v=4.0)

print(u + v)