# Wrappers for libc functions.

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
