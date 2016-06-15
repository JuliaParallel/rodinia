# Wrappers for libc functions.
function srand(seed)
    ccall((:srand, "libc"), Void, (Int,), seed)
end

function rand()
    ccall((:rand, "libc"), Int, ())
end
