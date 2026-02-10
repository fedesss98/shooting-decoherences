using Logging, LoggingExtras, Dates

# 1. Define the format for the file (Python-style)
file_logger = FormatLogger("debug.log"; append=true) do io, args
    # Format: [Timestamp] [Level] [Module] Message
    println(io, "[$(now())] [$(args.level)] [$(args._module)] $(args.message)")
end

# 2. Define the Console Logger (keep it clean, maybe just warnings/errors)
console_logger = ConsoleLogger(stderr, Logging.Info)

# 3. Combine them (TeeLogger sends logs to both)
# We also filter so the file gets Debug logs, but console only gets Info
demux_logger = TeeLogger(
    MinLevelLogger(file_logger, Logging.Debug),
    MinLevelLogger(console_logger, Logging.Info)
)

# 4. Set it as the global logger
global_logger(demux_logger)