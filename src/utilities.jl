get_indices(d::NamedTuple) = mapreduce(v -> v, vcat, values(d))

function make_traces(ns...)
    return rand(-1:1, ns...)
end
