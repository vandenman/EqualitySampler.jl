import CpuId, Dates

function log_message(message)
    print("[")
    print(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    print("] ")
    println(message)
end

function get_no_threads_to_use()
    no_physical_cores = CpuId.cpucores()
    no_threads        = CpuId.cputhreads()

    no_threads_to_use = if no_threads == no_physical_cores
        no_physical_cores - 1
    else
        no_threads - (no_threads > 16 ? 4 : 2)
    end
    return no_threads_to_use
end
