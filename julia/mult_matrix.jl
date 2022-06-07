using Distributed, SharedArrays
addprocs(4)
@everywhere using SharedArrays

n = 10
# create shared arrays and initialize them with random numbers
sa = SharedArray(rand(n,n));
sb = SharedArray(rand(n,n));
sc = SharedArray(rand(n,n));

@everywhere function mymatmul!(n,w,sa,sb,sc)
    # works only for 4 workers and n divisible by 4
    range = 1+(w-2) * div(n,4) : (w-1) * div(n,4)
    sc[:,range] = sa[:,:] * sb[:,range]
end

@time @sync begin
    for w in workers()
        @async remotecall_wait(w, mymatmul!, n, w, sa, sb, sc)
    end
end