# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã 2015, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.  The Reinforcement Learning Encounter Simulator (RLES)
# platform is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You
# may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
# law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# _____________________________________________________________________________
# Reinforcement Learning Encounter Simulator (RLES) includes the following
# third party software. The SISLES.jl package is licensed under the MIT Expat
# License: Copyright (c) 2014: Youngjun Kim.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
# "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# *****************************************************************************

type ActionSequence{A <: Action}
  sequence::Vector{A}
  index::Int64
  ActionSequence{A <: Action}(action_seq::Vector{A}) = new(action_seq, 1)
end

function action_seq_policy(action_seq::ActionSequence, s::State)
  action = action_seq.sequence[action_seq.index]
  action_seq.index += 1
  return action
end

function sample(ast::AdaptiveStressTest; verbose::Bool=true)
  (reward, actions) = simulate(ast.dpw.f.model, ast.rng, uniform_policy, verbose=verbose)
end

function sample(ast::AdaptiveStressTest, nsamples::Int64; verbose::Bool=true)
  #Samples are varied since ast.rng is not reset and sampling is done in series
  #Parallel version will need deterministic splitting of ast.rng
  f() = sample(ast, verbose=verbose) #avoids anonymous
  map(f, 1:nsamples) #returns vector of tuples(reward, actions)
end

function samples_timed(ast::AdaptiveStressTest, maxtime_s::Float64; verbose::Bool=true)
  #Samples are varied since ast.rng is not reset and sampling is done in series
  model = ast.dpw.f.model
  starttime_us = CPUtime_us()
  results = Array((Float64, Vector{Action}), 0)
  while true #this structure guarantees at least 1 sample
    tup = direct_sample(ast, verbose=verbose)
    push!(results, tup)
    if CPUtime_us() - starttime_us > maxtime_s * 1e6
      break
    end
  end
  return results #nsamples = length(results)
end

#Starts MCTS
function stresstest(ast::AdaptiveStressTest; verbose::Bool=true)
  return (mcts_reward, action_seq) = simulate(ast.dpw.f.model, ast.dpw, selectAction, verbose=verbose)
end

function play_sequence{A <: Action}(model::TransitionModel, actions::Vector{A}; verbose::Bool=true)
  reward2, actions2 = direct_sample(model, ActionSequence(actions), policy, verbose=verbose)
  @assert actions == actions2 #check replay
  return (reward2, actions2)
end


