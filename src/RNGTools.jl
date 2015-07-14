module RNGTools

import Base.dSFMT.dsfmt_gv_init_by_array #not exported, so may not be stable in the future

export set_gv_rng_state

set_gv_rng_state(i::Uint32) = dsfmt_gv_init_by_array([i])

set_gv_rng_state(a::Array{Uint32,1}) = dsfmt_gv_init_by_array(a)

function unitTest()
  println(rand(5))

  #These should be the same
  i = rand(Uint32)
  set_gv_rng_state(i)
  println(rand(5))
  set_gv_rng_state(i)
  println(rand(5))

  #These should be the same
  i = rand(Uint32)
  set_gv_rng_state(i)
  println(rand(5))
  set_gv_rng_state(i)
  println(rand(5))
end

#tic(); unitTest(); toc()

end
