# AdaptiveStressTesting.jl

Maintainer: Ritchie Lee, Carnegie Mellon University Silicon Valley, ritchie.lee@sv.cmu.edu

Adaptive Stress Testing is a stress testing tool for finding failure occurrences in multiple time step simulations.  The algorithm uses Monte Carlo tree search, a state-of-the-art planning algorithm, to adapt sampling during search.  This approach leads to efficient best-first exploration of the search space.  The black box method does not make any assumptions about the internal details of the system under test.  Adaptive Stress Testing has been previously applied to stress test airborne collision avoidance systems searching for near mid-air collisions in simulated aircraft encounters [1].   

## Usage

The recommended approach is for the user to define a custom type to contain the simulation state and parameters.  The adaptive stress testing tool then requires the user to expose three functions to the user simulator.
* initialize(sim) - Resets the simulator to the initial state
* update(sim) - Steps the simulator forward a single time step.  The tuple ``(prob, isevent, dist)`` is returned, where prob is the probability of taking that step, isevent indicates whether the failure event has occurred, and dist is an optional distance metric that hints to the optimizer how close the execution was to an event.
* isterminal(sim) - Returns true if the simulation has ended, false otherwise. 

These functions, along with configuration parameters, should be passed to create the adaptive stress test object 
```julia
ast = AdaptiveStressTest(ast_params, sim, MySimType.initialize, MySimType.update, MySimType.isterminal)
```

To draw Monte Carlo samples from the simulator:
```julia
sample(ast)
```
To draw N samples:
```julia
sample(ast, N)
```
Monte Carlo sampling is useful for testing and debugging your simulator before you run the actual stress test. 

When you're ready, run the stess test:
```julia
result = stress_test(ast, mcts_params)
```
where ``mcts_params`` is a ``DPWParams`` object containing the Monte Carlo tree search parameters.
The result object contains the total reward, action sequence, and q-values of the found execution path of the simulator.
```julia
result.reward
result.action_seq
result.q_values
```

For full working examples, see the package's ``examples`` folder.

## References

[1] R. Lee, M. J. Kochenderfer, O. J. Mengshoel, G. P. Brat, and M. P. Owen, "Adaptive Stress Testing of Airborne Collision Avoidance Systems," in Digital Avionics Systems Conference (DASC), Prague, Czech Republic, 2015 
