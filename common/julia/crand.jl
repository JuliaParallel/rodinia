import Base.Random: rand, srand

const RAND_MAX = typemax(Int32)


immutable LibcRNG <: AbstractRNG end

srand(r::LibcRNG, seed::Integer) = ccall((:srand, "libc"), Void, (Cuint,), seed)
rand(r::LibcRNG, ::Type{Cint})   = ccall((:rand, "libc"), Cint, ())
rand(r::LibcRNG)                 = rand(r, Cint)


immutable LibcRNG48 <: AbstractRNG end

srand(r::LibcRNG48, seed::Integer) = ccall((:srand48, "libc"), Void, (Clong,), seed)
rand(r::LibcRNG48, ::Type{Culong}) = ccall((:lrand48, "libc"), Cint, ())
rand(r::LibcRNG48, ::Type{Clong})  = ccall((:mrand48, "libc"), Cint, ())
