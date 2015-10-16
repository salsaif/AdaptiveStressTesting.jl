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

module RNGWrapper

export RNG, set_global, next, next!, set_from_seed!, hash, ==, isequal, length

using Iterators
import Base: next, hash, ==, isequal, length

type RNG
  state::Vector{Uint32}
end
function RNG(len::Int64=1, seed::Int64=0)
  return seed_to_state_itr(len, seed) |> collect |> RNG
end

set_from_seed!(rng::RNG, len::Int64, seed::Int64) = copy!(rng.state, seed_to_state_itr(len, seed))
seed_to_state_itr(len::Int64, seed::Int64) = take(iterate(hash_uint32, seed), len)

set_global(rng::RNG) = set_gv_rng_state(rng.state)
function next!(rng::RNG)
  map!(hash_uint32, rng.state)
  return rng
end
function next(rng0::RNG)
  rng1 = deepcopy(rng0)
  next!(rng1)
  return rng1
end

hash_uint32(x) = uint32(hash(x))
set_gv_rng_state(i::Uint32) = set_gv_rng_state([i])
set_gv_rng_state(a::Vector{Uint32}) = Base.dSFMT.dsfmt_gv_init_by_array(a) #not exported, so probably not stable

length(rng::RNG) = length(rng.state)
hash(rng::RNG) = hash(rng.state)
==(rng1::RNG, rng2::RNG) = rng1.state == rng2.state
isequal(rng1::RNG, rng2::RNG) = rng1 == rng2

end
