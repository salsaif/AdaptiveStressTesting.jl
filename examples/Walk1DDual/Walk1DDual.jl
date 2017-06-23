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

"""
Dual simulation version of the Walk1D example.  Two simulations are doing
the same random walks. Sim2 has a safety that will kick in on both upper
and lower bounds, whereas Sim1 only has a safety on the lower bound.
The dual simulation should be able to identify this difference.
"""
module Walk1DDual

using Distributions

export SafeWalkParams, SafeWalkSim, initialize, update, isterminal, isevent

type SafeWalkParams
    startx::Float64
    threshx::Float64 #+- thresh
    endtime::Int64
    logging::Bool
    safe::Tuple{Bool,Bool} #lower/upperbound safeties
end
SafeWalkParams() = SafeWalkParams(1.0, 10.0, 20, false, (false,false)) #set some defaults

type SafeWalkSim
    id::Int64
    p::SafeWalkParams #parameters
    x::Float64
    t::Int64 #num steps
    distribution::Distribution
    event::Bool
    mindist::Float64
    log::Vector{Any}
end

#Default to zero-mean Gaussian
function SafeWalkSim(id::Int64, params::SafeWalkParams, sigma::Float64)
    SafeWalkSim(id, params, Normal(0.0, sigma))
end

#Option to set own distribution
function SafeWalkSim(id::Int64, params::SafeWalkParams, distribution::Distribution)
    SafeWalkSim(id, params, params.startx, 0, distribution, false, realmax(Float64), 
        Array(Float64,0))
end

function initialize(sim::SafeWalkSim)
    sim.t = 0
    sim.x = sim.p.startx
    sim.event = false
    sim.mindist = realmax(Float64)
    empty!(sim.log)
    if sim.p.logging
        push!(sim.log, (sim.x, 0))
    end
end

function update(sim::SafeWalkSim)
    sim.t += 1
    r = rand(sim.distribution)
    sim.x += r
    prob = pdf(sim.distribution, r)

    #pull back to center gently if safety is on
    if sim.p.safe[1] && sim.x <= sim.p.startx
        sim.x -= r/2 
    elseif sim.p.safe[2] && sim.x >= sim.p.startx
        sim.x -= r/2 
    end

    dist = max(sim.p.threshx - abs(sim.x), 0.0) #non-negative
    if sim.p.logging
        push!(sim.log, (sim.x, dist))
    end

    sim.event |= isevent(sim)
    sim.mindist = min(sim.mindist, dist)

    return (prob, sim.event, sim.mindist)
end

function isevent(sim::SafeWalkSim)
    abs(sim.x) >= sim.p.threshx #out-of-bounds in +-
end

function isterminal(sim::SafeWalkSim)
    sim.t >= sim.p.endtime
end

end #module
