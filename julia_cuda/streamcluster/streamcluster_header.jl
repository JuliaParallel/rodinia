const INT_MAX = 0x7fffffff

# this structure represents a point
# these will be passed around to avoid copying coordinates
type Point
    weight::Float32
    coord::Array{Float32,1}
    assign::Int # number of point where this one is assigned
    cost::Float32 # cost of that assignment, weight*distance
end

# this is the array of points
type Points
    dim::Int32 # dimensionality
    num::Int # number of points; may not be N if this is a sample
    p::Array{Point,1} # the array itself

    Points(dim, num, points) = new(dim, num, Array{Point}(points))
end

# synthetic stream
type SimStream
    n::Int64
end

function read(pstream::SimStream, dest, dim, num)
    count = 0
    i = 0

    while i < num && pstream.n > 0
        for k = 1:dim
            dest[i * dim + k] = rand(rng, Culong) / Float32(INT_MAX)
        end

        i += 1
        count += 1
        pstream.n = pstream.n - 1
    end

    return count
end

function feof(pstream::SimStream)
    return pstream.n <= 0
end

function delete(pstream::SimStream)
    # do nothing
end

type FileStream
    fp::IOStream

    FileStream(filename) = new(open(filename, "r"))
end

function read(pstream::FileStream, dest, dim, num)
    offset = 1
    block_size = dim * sizeof(Float32)

    for i = 1:num
        data = read(fp, Float32, dim)
        dest[offset:offset+block_size] = data[1:block_size]
        offset += block_size
    end

    return i - 1
end

function ferror(pstream::FileStream)
    return eof(pstream.fp)
end

function delete(pstream::FileStream)
    println("closing file stream")
    close(fp)
end
