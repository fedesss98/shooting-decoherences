using Logging, LoggingExtras, Dates

function setup_logger(log_filepath::String)
    # File Logger: Added standard timestamp formatting (YYYY-MM-DD HH:MM:SS)
    file_logger = FormatLogger(log_filepath; append=true) do io, args
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        println(io, "[$timestamp] [$(args.level)] [$(args._module)] $(args.message)")
    end

    # File gets Debug and above
    filtered_file_logger = MinLevelLogger(file_logger, Logging.Debug)

    # Console Logger: Default is already Info and above, so we don't need 
    # to wrap this one in a MinLevelLogger.
    console_logger = ConsoleLogger(stderr, Logging.Info)

    # Combine them
    demux_logger = TeeLogger(filtered_file_logger, console_logger)

    # Set as global logger
    global_logger(demux_logger)
end