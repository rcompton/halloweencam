import phi.flow as phi
from matplotlib import pyplot as plt

# 1. Define the computational domain as a dictionary
domain = dict(x=64, y=64, bounds=phi.Box(x=100, y=100))

# 2. Create initial state variables
velocity = phi.StaggeredGrid(phi.Noise(), extrapolation=phi.extrapolation.ZERO, **domain)
smoke = phi.CenteredGrid(phi.Box(x=(45, 55), y=(10, 25)), extrapolation=phi.extrapolation.BOUNDARY, **domain)

# 3. Run the simulation and collect the frames
smoke_sequence = [smoke]
for _ in range(100):
    smoke = phi.advect.mac_cormack(smoke, velocity, dt=1.0)
    smoke_sequence.append(smoke)

# 4. Stack the frames into a single Field with a 'time' dimension
smoke_animation = phi.stack(smoke_sequence, dim=phi.batch('time'))

# 5. Show the result as an animation
anim = phi.vis.show(smoke_animation, animate='time') # 👈 Assign to 'anim'
plt.show()