# Wrappers for libc functions.
function srand(seed)
    ccall((:srand, "libc"), Void, (Int,), seed)
end

function rand()
    ccall((:rand, "libc"), Int, ())
end

function srand48(seed)
	ccall((:srand48, "libc"), Void, (Int64,), seed)
end

function lrand48()
	ccall((:lrand48, "libc"), Int64, ())
end

function cfree(ptr)
    ccall((:free, "libc"), Void, (Ptr{Void},), ptr)
end

function catan(angle::Float64)
    ccall((:atan, "libm"), Float64, (Float64, ), angle)
end

function catan(angle::Array{Float64})
    ret = similar(angle)
    for i in eachindex(angle)
        ret[i] = catan(angle[i])
    end
    ret
end


function cpow(base::Float64, exp::Float64)
    ccall((:pow, "libm"), Float64, (Float64, Float64), base, exp)
end

function cpow(angle::Array{Float64})
    ret = similar(angle)
    for i in eachindex(angle)
        ret[i] = cpow(angle[i])
    end
    ret
end
