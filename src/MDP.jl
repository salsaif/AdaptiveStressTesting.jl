module MDP

export TransitionModel, Params, State, Action, Reward, Policy, solve, simulate

type TransitionModel
  getInitialState::Function
  getNextState::Function
  isEndState::Function
  maxSteps::Int64 #maximum number of steps.  Acts as a safety for runaways
  goToState::Function
end
function TransitionModel(getInitialState::Function, getNextState::Function,
                         isEndState::Function, maxSteps::Int64)
  TransitionModel(getInitialState, getNextState, getNextisEndStateAction, maxSteps, identity)
end

typealias Policy Function
typealias Reward Float64
typealias Params Any

abstract State
abstract Action

function simulate(model::TransitionModel,
                  p::Params,
                  policy::Policy,
                  rng::AbstractRNG = MersenneTwister();
                  verbose::Bool = false)

  # This function simulates the model for nSteps using the specified policy and
  # returns the total simulation reward
  cum_reward = 0.0
  actions = Action[]
  r_history = []
  s = model.getInitialState(rng)
  for i = 1:model.maxSteps
    if verbose
      println("Step: $i of $(model.maxSteps)")
    end
    rewards, a = policy(p, s)
    push!(actions, a) #output actions actually taken
    s, r = model.getNextState(s, a, rng)
    cum_reward += r
    append!(r_history, rewards)
    model.isEndState(s) && break
  end

  return cum_reward::Reward, actions::Vector{Action}, r_history
end


end # module
