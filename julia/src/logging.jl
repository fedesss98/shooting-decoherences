using Logging, LoggingExtras, Dates

function setup_logger(log_filepath::String; console_level::LogLevel=Logging.Info)

    file_logger = FormatLogger(log_filepath; append=true) do io, args
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

        println(io, "[$timestamp] [$(args.level)] [$(args._module)] $(args.message)")
        println(io, "    @ $(args.file):$(args.line)")

        for (key, value) in pairs(args.kwargs)
            if key === :exception
                err, bt = value

                println(io, "    exception:")
                showerror(io, err, bt)
                println(io)

            else
                println(io, "    $key = $value")
            end
        end
    end

    filtered_file_logger = MinLevelLogger(file_logger, Logging.Debug)

    console_logger = ConsoleLogger(stderr, console_level)

    demux_logger = TeeLogger(filtered_file_logger, console_logger)

    global_logger(demux_logger)
end
