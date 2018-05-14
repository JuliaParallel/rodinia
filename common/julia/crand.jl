using Random
import Random: rand, srand

const RAND_MAX = typemax(Int32)


struct LibcRNG <: AbstractRNG end

srand(r::LibcRNG, seed::Integer) = ccall(:srand, Cvoid, (Cuint,), seed)
rand(r::LibcRNG, ::Type{Cint})   = ccall(:rand,  Cint, ())
rand(r::LibcRNG)                 = rand(r, Cint)


struct LibcRNG48 <: AbstractRNG end

srand(r::LibcRNG48, seed::Integer) = ccall(:srand48, Cvoid, (Clong,), seed)
rand(r::LibcRNG48, ::Type{Culong}) = ccall(:lrand48, Cint, ())
rand(r::LibcRNG48, ::Type{Clong})  = ccall(:mrand48, Cint, ())
