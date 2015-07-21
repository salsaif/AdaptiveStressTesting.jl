#Makes and deterministic MDP from a Markov model by manipulating the global random seed
#Author: Ritchie Lee

include("RNGTools.jl")
include("MCTSdpw.jl")

module RLESMDPs

# TODO: Consider extracting ObserverImpl into its own package
using SISLES.ObserverImpl #FIXME: remove dependency on SISLES...
import SISLES.addObserver

import MDP: State, Action, TransitionModel
using RNGTools

export RLESMDP, RLESMDP_params, getTransitionModel, uniform_policy,
        actionsToThisState, ESState, ESAction

type RLESMDP_params

  max_steps::Int64 #safety for runaways in sim
  action_counter_reset::Union(Nothing, Uint32) #reset to this action_counter value on initialize()
  action_counter_init::Uint32 #initial value of action_counter on construct

  RLESMDP_params() = new()
end

type RLESMDP

  params::RLESMDP_params

  sim
  sim_hash::Uint64 #hash sim state to match with ESState

  initialize::Function #initialize(sim)
  step::Function #step(sim)
  isEndState::Function #isEndState(sim)
  get_reward::Function #get_reward(sim)

  t_index::Int64 #starts at 1 and counts up in ints

  action_counter::Uint32 #global for new random seed

  observer::Observer

  function RLESMDP(p::RLESMDP_params, sim, initialize_fn::Function, step_fn::Function,
                   isEndState_fn::Function, get_reward_fn::Function)

    mdp = new()

    mdp.params = p
    mdp.sim = sim
    mdp.sim_hash = hash(0)

    mdp.initialize = initialize_fn
    mdp.step = step_fn
    mdp.isEndState = isEndState_fn
    mdp.get_reward = get_reward_fn

    mdp.action_counter = p.action_counter_init

    mdp.observer = Observer()

    return mdp
  end
end

# TODO: Consider renaming
type ESAction <: Action

  seed::Uint32 # Could also be Vector{Uint32}

end

ESAction() = ESAction(uint32(0))

# TODO: Consider renaming
type ESState <: State

  t_index::Int64 #sanity check that at least the time corresponds
  hash::Uint64 #hash sim state to match with ESState
  parent::Union(Nothing, ESState) #parent state, root=nothing
  action::ESAction #action taken from parent, root=0

end

addObserver(mdp::RLESMDP, f::Function) = _addObserver(mdp, f)
addObserver(mdp::RLESMDP, tag::String, f::Function) = _addObserver(mdp, tag, f)

function getTransitionModel(mdp::RLESMDP)

  function getInitialState(rng::AbstractRNG) #rng is unused
    mdp.t_index = 1

    mdp.initialize(mdp.sim)

    if mdp.params.action_counter_reset != nothing #reset if specified
      mdp.action_counter = mdp.params.action_counter_reset
    end

    s = ESState(mdp.t_index, 0, nothing, ESAction())
    s.hash = mdp.sim_hash = hash(s) #overwrites 0

    return s
  end

  function getNextState(s0::ESState, a0::ESAction, rng::AbstractRNG) #rng is unused

    goToState(mdp,s0) #Checks to see if we're in sync with sim, if not, resync

    mdp.t_index += 1

    set_gv_rng_state(a0.seed)
    #saving the entire state of the MersenneTwister would require 770 * 4 bytes.  Instead, for now, just save seed.
    #alternatively, seed can be an array of ints less than size 770 and the rest be generated using hash()
    #would need to reach deep into components to use an RNG that is passed around.  TODO: consider doing this

    notifyObserver(mdp, "action_seq", Any[mdp.t_index, a0]) #piggyback off SISLES.observers

    mdp.step(mdp.sim)

    s1 = ESState(mdp.t_index, 0, s0, a0) #TODO: simplify this two-step hash process
    s1.hash = mdp.sim_hash = hash(s1) #overwrites 0

    r = mdp.get_reward(mdp.sim)

    return (s1, r)
  end

  function isEndState(s::ESState)

    goToState(mdp, s) #Checks to see if s is in sync with sim, if not, resync
    mdp.isEndState(mdp.sim) #avoid naming conflict
  end

  return TransitionModel(getInitialState, getNextState, isEndState, mdp.params.max_steps)
end

function uniform_policy(mdp::RLESMDP, s0::ESState)

  mdp.action_counter = uint32(mdp.action_counter + 1) #return the next random seed

  return ESAction(mdp.action_counter)
end

function actionsToThisState(s::ESState)

  actions = ESAction[]

  #Tracing up the tree relieves us from storing the entire history of actions at each node
  while s.parent != nothing
    prepend!(actions, [s.action])
    s = s.parent
  end

  return actions
end

function goToState(mdp::RLESMDP, targetState::ESState)

  #if sim and state are not sync'ed, retrace in sim
  if mdp.t_index != targetState.t_index ||
      mdp.sim_hash != targetState.hash  #TODO: clean up the hashing by overloading hash()

    model = getTransitionModel(mdp)
    rng = MersenneTwister() #not used

    #Get to state s by traversing starting from initial state
    s = model.getInitialState(rng)
    for a = actionsToThisState(targetState)
      s, r = model.getNextState(s, a, rng)
    end

    @assert s == targetState
  end

  return targetState
end

end #module

include("auxfuncs.jl") #define hash functions for ESState and ESAction
