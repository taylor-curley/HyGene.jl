get_indices(d::AbstractDict, ks) = mapreduce(k -> d[k], vcat, ks)

function make_traces(ns...)
    return rand(-1:1, ns...)
end
