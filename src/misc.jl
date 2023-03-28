"""
    block_header(txt)

Generates blocked headers with centered titles because I got bored.

# Arguments

  - `txt`: String to be included in header

"""
function block_header(txt::String)
    bar = "###############################################################################################"
    if length(txt)+2 > length(bar)
        error("Text cannot be longer than 93 characters")
    end
    start = Int32(round(length(bar)/2) - round(length(txt)/2))
    mid_start = "#" * repeat(" ", start-1) 
    mid_end = repeat(" ", length(bar)-length(mid_start*txt)-1) * "#"
    out = bar * "\n" * mid_start * txt * mid_end * "\n" * bar
    print(out)
end

"""
    line_header(txt)

# Arguments

  - `txt`: String to be included in header. (REQUIRED)
  - `len_char`: Length of the title. (OPTIONAL)

"""
function line_header(txt::String; len_char=93)
    start = Int32(round(len_char/2) - round(length(txt)/2))
    mid_start="#" * repeat("~",start)
    mid_end = repeat("~", len_char-length(mid_start*txt)-1)
    print(mid_start * " " * txt * " " * mid_end * "#")
end