module MCTSdpw

# This module implements Monte Carlo Tree Search with double progressive widening.
#Both the number of actions considered from a state and the number of transition
#states are restricted and increased throughout the simulation.
#Documentation for this algorithm can be found in
#'Adding double progressive widening to upper confidence trees to cope with
#uncertainty in planning problems' in European Workshop on Reinforcement Learning and
#'A comparison of Monte Carlo tree search and mathematical optimization for large scale
#dynamic resource allocation'

using MDP
using auxfuncs
using CPUTime
import MDP: simulate

export DPWParams, DPWModel, DPW, selectAction

typealias Depth Int64

type DPWParams
  d::Depth                    # search depth
  ec::Float64                 # exploration constant- governs trade-off between exploration and exploitation in MCTS
  n::Int64                    # number of iterations
  k::Float64                  # first constant controlling action generation
  alpha::Float64              # second constant controlling action generation
  kp::Float64                 # first constant controlling transition state generation
  alphap::Float64             # second constant controlling transition state generation
  clear_nodes::Bool           # clear all nodes before selecting next action
  maxtime_s::Float64          # maximum time to iterate, seconds
  rng_seed::Uint64            # random number generator

  DPWParams() = new()
  DPWParams(d::Depth,ec::Float64,n::Int64,k::Float64,alpha::Float64,kp::Float64,alphap::Float64,
            clear_nodes::Bool,maxtime_s::Float64,rng_seed::Uint64) = new(d,ec,n,k,alpha,kp,alphap,
                                                                         clear_nodes,maxtime_s,rng_seed)
end

type DPWModel
  model::TransitionModel      # generative model
  getAction::Function         # returns action for rollout policy
  getNextAction::Function     # generates the next action when widening of the action space is appropriate
end

type StateActionStateNode
  n::Uint64
  r::Float64

  StateActionStateNode() = new(0,0)
end

type StateActionNode
  s::Dict{State,StateActionStateNode}
  n::Uint64
  q::Float64

  StateActionNode() = new(Dict{State,StateActionStateNode}(),0,0)
end

type StateNode
  a::Dict{Action,StateActionNode}
  n::Uint64

  StateNode() = new(Dict{Action,StateActionNode}(),0)
end

type DPW

  s::Dict{State,StateNode}
  p::DPWParams
  f::DPWModel
  rng::AbstractRNG

  DPW(p::DPWParams,f::DPWModel) = new(Dict{State,StateNode}(),p,f,MersenneTwister(p.rng_seed))
end

function saveState(old_d::Dict{State,StateNode},new_d::Dict{State,StateNode},s::State)

  if !haskey(old_d,s)
    return new_d
  else
    new_d[s] = old_d[s]
    for sa in values(old_d[s].a)
      for s1 in keys(sa.s)
        saveState(old_d,new_d,s1)
      end
    end
    return new_d
  end
end

function selectAction(dpw::DPW,s::State; verbose::Bool=false)
  if dpw.p.clear_nodes
    #save s and its successors
    new_dict = saveState(dpw.s,Dict{State,StateNode}(),s)
    empty!(dpw.s) #cleanup
    dpw.s = new_dict
  end

  # This function calls simulate and chooses the approximate best action from the reward approximations
  d = dpw.p.d
  starttime_us = CPUtime_us()
  for i = 1:dpw.p.n
    simulate(dpw,s,d,verbose=verbose)

    if CPUtime_us() - starttime_us > dpw.p.maxtime_s * 1e6
      if verbose
        println("Iterations completed: $i")
      end
      break
    end
  end
  println("Size of sdict: ", length(dpw.s))
  cS = dpw.s[s]
  A = collect(keys(cS.a)) # extract the actions taken in current state
  nA = length(A)
  Q = zeros(Reward,nA) # approximate value for each action
  for i = 1:nA
    Q[i] = cS.a[A[i]].q
  end

  if !isempty(Q)
    return A[indmax(Q)]::Action # choose action with highest approximate value
  else
    return error("This shouldn't be occurring...")
  end
end

function simulate(dpw::DPW,s::State,d::Depth;verbose::Bool=false)
  # This function returns the reward for one iteration of MCTSdpw
  if d == 0 || dpw.f.model.isEndState(s)
    return 0.0::Reward
  end
  if !haskey(dpw.s,s) # if state is not yet explored, add it to the set of states, perform a rollout
    dpw.s[s] = StateNode()
    return rollout(dpw,s,d)::Reward
  end
  dpw.s[s].n += 1
  if length(dpw.s[s].a) <= dpw.p.k*dpw.s[s].n^dpw.p.alpha # criterion for new action generation
    a = dpw.f.getNextAction(s,dpw.rng) # action generation step
    if !haskey(dpw.s[s].a,a) # make sure we haven't already tried this action
      dpw.s[s].a[a] = StateActionNode()
    end
  else # choose an action using UCT criterion
    cS = dpw.s[s] # save current state so we do not have to iterate through map many times
    A = collect(keys(cS.a)) # extract the actions taken in current state
    nA = length(A)
    UCT = zeros(Reward,nA)
    nS = cS.n
    for i = 1:nA
      cA = cS.a[A[i]] # save current action
      UCT[i] = cA.q + dpw.p.ec*sqrt(log(nS)/cA.n)
    end
    a = A[indmax(UCT)] # choose action with highest UCT score
  end
  if length(dpw.s[s].a[a].s) <= dpw.p.kp*dpw.s[s].a[a].n^dpw.p.alphap # criterion for new transition state consideration
    sp,r = dpw.f.model.getNextState(s,a,dpw.rng) # choose a new state and get reward
    if !haskey(dpw.s[s].a[a].s,sp) # if transition state not yet explored, add to set and update reward
      dpw.s[s].a[a].s[sp] = StateActionStateNode()
      dpw.s[s].a[a].s[sp].r = r
    else
      dpw.s[s].a[a].s[sp].n += 1
    end
  else # sample from transition states proportional to their occurence in the past
    cA = dpw.s[s].a[a]
    SP = collect(keys(cA.s))
    rn = rand(dpw.rng)*cA.n
    cnt = 0
    i = 1
    while true
      cnt += cA.s[SP[i]].n
      if rn <= cnt
        sp = SP[i]
        break
      end
      i += 1
    end
    r = dpw.s[s].a[a].s[sp].r
    dpw.s[s].a[a].s[sp].n += 1
  end
  q = r + simulate(dpw,sp,d-1)
  cA = dpw.s[s].a[a]
  cA.n = uint64(cA.n + 1)
  cA.q += (q-cA.q)/cA.n
  dpw.s[s].a[a] = cA
  return q::Reward
end

function rollout(dpw::DPW,s::State,d::Depth)
  # Runs a rollout simulation using the default policy
  if d == 0 || dpw.f.model.isEndState(s)
    return 0.0::Reward
  else
    a = dpw.f.getAction(s,dpw.rng)
    sp,r = dpw.f.model.getNextState(s,a,dpw.rng)
    return (r + rollout(dpw,sp,d-1))::Reward
  end
end

end # module
