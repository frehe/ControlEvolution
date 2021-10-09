# ControlEvolution
Meta-optimization of evolutionary algorithms using control theory

First iteration: Solve a quadratic optimization problem by formulating it with an LQR controller. Initialize a population of solutions and apply LQR independently on each solution candidate.

### Current State
- State x(k) is array of shape (pop_size, n) flattened to (pop_size x n, 1) in row-major order, where n is the problem dimension
- System is linear: x(k+1) = Ax(k) * Bu(k)
    - A = eye(pop_size x n)
    - B = 0.1 * eye(pop_size x n)
    - A, B are applied independently for each population member
- Objective function is quadratic: J(x, u) = x'Qx + u'Ru
- Access to full state information (no measurement)
- Currently using LQR controller: u(k) = -Fx(k)