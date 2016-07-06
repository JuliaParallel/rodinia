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
