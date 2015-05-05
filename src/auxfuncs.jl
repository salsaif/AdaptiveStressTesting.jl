import Base: hash, isequal
import RLESMDPs: ESState, ESAction

function hash(a::ESAction)
    return hash(a.seed)
end

function hash(s::ESState)
  h = hash(s.t_index)
  h = hash(h,hash(s.parent == nothing ? nothing : s.parent.hash))
  h = hash(h,hash(s.action))

  return h
end

function isequal(w::ESAction,v::ESAction)
    return isequal(w.seed,v.seed)
end

function isequal(w::ESState,v::ESState)
    return hash(w) == hash(v)
end

function ==(w::ESAction,v::ESAction)
    return isequal(w.seed,v.seed)
end

function ==(w::ESState,v::ESState)
    return hash(w) == hash(v)
end
