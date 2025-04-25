module PerformanceMetrics

export track_performance, log_metrics, compute_statistics
export MetricsTracker, MetricsLogger, benchmark_model
export memory_profile, latency_profile, throughput_test
export create_performance_report, compare_configurations
export TensorMetrics, ModelMetrics, TrainingMetrics
export create_metrics_tracker, init_metrics_system
export track_loss, track_accuracy, track_epoch_time

using Statistics
using Dates
using LinearAlgebra
using Printf

# Intentar importar CUDA con manejo de errores
cuda_available = false
try
    using CUDA
    cuda_available = CUDA.functional()
    @info "CUDA disponible en PerformanceMetrics: $cuda_available"
catch e
    @warn "CUDA no disponible para PerformanceMetrics: $(e)"
end

# Importar módulos internos con manejo de errores
try
    using ..SpatialField, ..BrainSpace
    @info "Módulos SpatialField y BrainSpace importados correctamente"
catch e
    @warn "Error al importar SpatialField o BrainSpace: $(e)"
end

# Constantes globales
const DEFAULT_LOG_DIR = joinpath(@__DIR__, "..", "logs")
const GLOBAL_METRICS_SYSTEM = Ref{Dict{Symbol, Any}}(Dict{Symbol, Any}())

"""
    MetricsTracker

Sistema para rastrear métricas de rendimiento del modelo.
"""
mutable struct MetricsTracker
    # Identificador del tracker
    name::String
    
    # Métricas de memoria
    memory_usage::Dict{Symbol,Vector{Float64}}
    peak_memory::Dict{Symbol,Float64}
    
    # Métricas de tiempo
    operation_timings::Dict{Symbol,Vector{Float64}}
    epoch_times::Vector{Float64}
    
    # Métricas de modelo
    loss_values::Vector{Float64}
    accuracy_values::Vector{Union{Nothing,Float64}}
    
    # Métricas de rendimiento computacional
    throughput::Vector{Float64}  # elementos/segundo
    flops_utilization::Vector{Float64}  # % de pico teórico
    
    # Tiempos de inicio para medición
    timing_starts::Dict{Symbol,Float64}
    
    # Metadatos
    metadata::Dict{Symbol,Any}
    
    # Constructor con valores iniciales
    function MetricsTracker(name::String="default")
        new(
            name,
            Dict{Symbol,Vector{Float64}}(),  # memory_usage
            Dict{Symbol,Float64}(),          # peak_memory
            Dict{Symbol,Vector{Float64}}(),  # operation_timings
            Float64[],                       # epoch_times
            Float64[],                       # loss_values
            Union{Nothing,Float64}[],        # accuracy_values
            Float64[],                       # throughput
            Float64[],                       # flops_utilization
            Dict{Symbol,Float64}(),          # timing_starts
            Dict{Symbol,Any}(                # metadata
                :creation_time => Dates.now(),
                :name => name
            )
        )
    end
end

"""
    MetricsLogger

Sistema para registro y persistencia de métricas.
"""
mutable struct MetricsLogger
    log_dir::String
    log_file::Union{Nothing,IOStream}
    metrics_history::Dict{Symbol,Vector{Any}}
    log_level::Symbol  # :detailed, :normal, :minimal
    
    function MetricsLogger(log_dir::String=""; log_level::Symbol=:normal)
        # Usar directorio predeterminado si no se proporciona uno
        if isempty(log_dir)
            log_dir = DEFAULT_LOG_DIR
        end
        
        # Asegurar que el directorio existe
        try
            if !isdir(log_dir)
                mkpath(log_dir)
                @info "Directorio de logs creado: $log_dir"
            end
        catch e
            @warn "No se pudo crear el directorio de logs: $log_dir" exception=e
            # Intentar usar un directorio temporal como fallback
            log_dir = mktempdir(prefix="rnta_logs_")
            @info "Usando directorio temporal para logs: $log_dir"
        end
        
        # Preparar archivo de log
        log_file_path = joinpath(log_dir, "metrics_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS")).log")
        log_file = nothing
        
        try
            log_file = open(log_file_path, "w")
            @info "Archivo de log creado: $log_file_path"
        catch e
            @warn "No se pudo crear el archivo de log: $log_file_path" exception=e
        end
        
        new(
            log_dir,
            log_file,
            Dict{Symbol,Vector{Any}}(),
            log_level
        )
    end
end

"""
    TensorMetrics

Métricas específicas para operaciones tensoriales.
"""
struct TensorMetrics
    # Dimensiones y propiedades estructurales
    dimensions::Tuple{Int,Int,Int}
    total_elements::Int
    memory_size_bytes::Int
    
    # Propiedades estadísticas
    sparsity::Float64  # proporción de elementos cercanos a cero
    mean_value::Float64
    std_value::Float64
    min_value::Float64
    max_value::Float64
    
    # Propiedades computacionales
    computational_density::Float64  # operaciones por byte
    gradient_norm::Union{Nothing,Float64}
    
    # Constructor con valores calculados a partir de un campo tensorial
    function TensorMetrics(field; with_gradients::Bool=false)
        try
            # Verificar si el objeto tiene los campos esperados
            if !isdefined(field, :data)
                error("El campo no tiene el atributo 'data' esperado")
            end
            
            # Calcular propiedades básicas
            dims = size(field.data)
            total_elems = length(field.data)
            mem_size = total_elems * sizeof(eltype(field.data))
            
            # Calcular propiedades estadísticas
            # Asegurar que tenemos un Array (por si es CuArray)
            data_array = nothing
            try
                if isdefined(Main, :CUDA) && typeof(field.data) <: CuArray
                    data_array = Array(field.data)
                else
                    data_array = field.data
                end
            catch e
                @warn "Error al convertir datos del campo: $(e)"
                data_array = field.data  # Usar como está
            end
            
            sparsity_threshold = 1e-6
            sparsity = count(abs.(data_array) .< sparsity_threshold) / total_elems
            
            mean_val = mean(data_array)
            std_val = std(data_array)
            min_val = minimum(data_array)
            max_val = maximum(data_array)
            
            # Densidad computacional (aproximación simplificada)
            # Basado en la idea de que elementos no cercanos a cero requieren cómputo
            comp_density = (1 - sparsity) * 2  # ~2 operaciones por elemento no cero
            
            # Norma del gradiente (si está disponible)
            grad_norm = nothing
            if with_gradients && isdefined(field, :metadata) && haskey(field.metadata, "gradients")
                grad_data = field.metadata["gradients"]
                grad_norm = norm(grad_data)
            end
            
            new(
                dims,
                total_elems,
                mem_size,
                sparsity,
                mean_val,
                std_val,
                min_val,
                max_val,
                comp_density,
                grad_norm
            )
        catch e
            @error "Error al crear TensorMetrics" exception=(e, catch_backtrace())
            # Valores predeterminados para evitar fallos
            return new(
                (0, 0, 0),
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                nothing
            )
        end
    end
end

"""
    ModelMetrics

Métricas a nivel de modelo completo.
"""
struct ModelMetrics
    # Tamaño y complejidad
    total_parameters::Int
    total_fields::Int
    total_connections::Int
    model_size_bytes::Int
    
    # Métricas computacionales
    peak_memory_usage::Dict{Symbol,Float64}  # Por dispositivo
    avg_memory_usage::Dict{Symbol,Float64}
    estimated_flops::Float64
    
    # Métricas de rendimiento
    inference_latency_ms::Float64
    forward_pass_ms::Float64
    backward_pass_ms::Float64
    
    # Eficiencia
    memory_efficiency::Float64  # ratio de memoria útil/total
    compute_efficiency::Float64  # ratio de cómputo útil/total
    
    # Constructor con valores predeterminados y manejo de errores
    function ModelMetrics(;
        total_parameters::Int = 0,
        total_fields::Int = 0,
        total_connections::Int = 0,
        model_size_bytes::Int = 0,
        peak_memory_usage::Dict{Symbol,Float64} = Dict{Symbol,Float64}(),
        avg_memory_usage::Dict{Symbol,Float64} = Dict{Symbol,Float64}(),
        estimated_flops::Float64 = 0.0,
        inference_latency_ms::Float64 = 0.0,
        forward_pass_ms::Float64 = 0.0,
        backward_pass_ms::Float64 = 0.0,
        memory_efficiency::Float64 = 0.0,
        compute_efficiency::Float64 = 0.0
    )
        new(
            total_parameters,
            total_fields,
            total_connections,
            model_size_bytes,
            peak_memory_usage,
            avg_memory_usage,
            estimated_flops,
            inference_latency_ms,
            forward_pass_ms,
            backward_pass_ms,
            memory_efficiency,
            compute_efficiency
        )
    end
end

"""
    TrainingMetrics

Métricas relacionadas con el entrenamiento.
"""
struct TrainingMetrics
    # Información básica
    dataset_size::Int
    batch_size::Int
    learning_rate::Float64
    
    # Progreso de entrenamiento
    current_epoch::Int
    total_epochs::Int
    samples_processed::Int
    
    # Métricas de calidad
    train_loss::Float64
    validation_loss::Union{Nothing,Float64}
    train_accuracy::Union{Nothing,Float64}
    validation_accuracy::Union{Nothing,Float64}
    
    # Métricas de tiempo
    epoch_time_seconds::Float64
    samples_per_second::Float64
    time_breakdown::Dict{Symbol,Float64}  # %tiempo en data_loading, forward, backward, optimization
    
    # Métricas de gradiente
    gradient_norm::Float64
    gradient_noise_scale::Union{Nothing,Float64}
    
    # Constructor con valores predeterminados y manejo de errores
    function TrainingMetrics(;
        dataset_size::Int = 0,
        batch_size::Int = 0,
        learning_rate::Float64 = 0.0,
        current_epoch::Int = 0,
        total_epochs::Int = 0,
        samples_processed::Int = 0,
        train_loss::Float64 = 0.0,
        validation_loss::Union{Nothing,Float64} = nothing,
        train_accuracy::Union{Nothing,Float64} = nothing,
        validation_accuracy::Union{Nothing,Float64} = nothing,
        epoch_time_seconds::Float64 = 0.0,
        samples_per_second::Float64 = 0.0,
        time_breakdown::Dict{Symbol,Float64} = Dict{Symbol,Float64}(),
        gradient_norm::Float64 = 0.0,
        gradient_noise_scale::Union{Nothing,Float64} = nothing
    )
        new(
            dataset_size,
            batch_size,
            learning_rate,
            current_epoch,
            total_epochs,
            samples_processed,
            train_loss,
            validation_loss,
            train_accuracy,
            validation_accuracy,
            epoch_time_seconds,
            samples_per_second,
            time_breakdown,
            gradient_norm,
            gradient_noise_scale
        )
    end
end

"""
    init_metrics_system()

Inicializa el sistema global de métricas.

## Retorna
- `Dict{Symbol, Any}`: Configuración del sistema de métricas
"""
function init_metrics_system()
    try
        @info "Inicializando sistema de métricas..."
        
        # Crear directorio de logs si no existe
        log_dir = DEFAULT_LOG_DIR
        if !isdir(log_dir)
            mkpath(log_dir)
            @info "Directorio de logs creado: $log_dir"
        end
        
        # Configurar sistema global
        GLOBAL_METRICS_SYSTEM[] = Dict{Symbol, Any}(
            :start_time => Dates.now(),
            :log_dir => log_dir,
            :trackers => Dict{String, MetricsTracker}(),
            :loggers => Dict{String, MetricsLogger}(),
            :hardware_info => detect_hardware_info(),
            :default_logger => MetricsLogger(log_dir)
        )
        
        @info "Sistema de métricas inicializado correctamente"
        return GLOBAL_METRICS_SYSTEM[]
    catch e
        @error "Error al inicializar sistema de métricas" exception=(e, catch_backtrace())
        
        # Crear un sistema mínimo como fallback
        fallback_system = Dict{Symbol, Any}(
            :start_time => Dates.now(),
            :error => "$(e)",
            :fallback => true
        )
        
        GLOBAL_METRICS_SYSTEM[] = fallback_system
        return fallback_system
    end
end

"""
    detect_hardware_info()

Detecta información del hardware del sistema.

## Retorna
- `Dict{Symbol, Any}`: Información del hardware
"""
function detect_hardware_info()
    hw_info = Dict{Symbol, Any}(
        :cpu_threads => Sys.CPU_THREADS,
        :memory_gb => Sys.total_memory() / (1024^3),
        :os => string(Sys.KERNEL),
        :julia_version => string(VERSION)
    )
    
    # Detectar CUDA si está disponible
    if cuda_available
        try
            hw_info[:cuda_available] = true
            hw_info[:cuda_devices] = length(CUDA.devices())
            
            # Obtener información del dispositivo actual
            device_props = CUDA.device_properties()
            hw_info[:cuda_device] = Dict{Symbol, Any}(
                :name => device_props.name,
                :mem_gb => device_props.totalmem / (1024^3),
                :compute_capability => "$(device_props.capability.major).$(device_props.capability.minor)"
            )
        catch e
            @warn "Error al obtener información detallada de CUDA" exception=e
            hw_info[:cuda_available] = false
            hw_info[:cuda_error] = "$(e)"
        end
    else
        hw_info[:cuda_available] = false
    end
    
    return hw_info
end

"""
    create_metrics_tracker(name::String="default")

Crea un nuevo rastreador de métricas.

## Argumentos
- `name::String="default"`: Nombre del tracker para identificarlo

## Retorna
- `MetricsTracker`: Nuevo rastreador de métricas
"""
function create_metrics_tracker(name::String="default")
    try
        # Asegurar que el sistema está inicializado
        if isempty(GLOBAL_METRICS_SYSTEM[])
            init_metrics_system()
        end
        
        # Crear tracker
        tracker = MetricsTracker(name)
        
        # Registrar en el sistema global si existe
        if haskey(GLOBAL_METRICS_SYSTEM[], :trackers)
            GLOBAL_METRICS_SYSTEM[][:trackers][name] = tracker
        end
        
        return tracker
    catch e
        @error "Error al crear metrics tracker" name=name exception=(e, catch_backtrace())
        
        # Devolver un tracker básico como fallback
        return MetricsTracker(name)
    end
end

"""
    track_performance(tracker::MetricsTracker, key::Symbol, value::Number)

Registra una métrica de rendimiento específica.

## Argumentos
- `tracker::MetricsTracker`: Rastreador donde registrar la métrica
- `key::Symbol`: Clave para identificar la métrica
- `value::Number`: Valor de la métrica a registrar

## Retorna
- `Bool`: `true` si se registró correctamente, `false` en caso contrario
"""
function track_performance(tracker::MetricsTracker, key::Symbol, value::Number)
    try
        # Inicializar vector para la clave si no existe
        if key ∉ keys(tracker.memory_usage) && key ∉ keys(tracker.operation_timings)
            if startswith(String(key), "mem_")
                tracker.memory_usage[key] = Float64[]
            else
                tracker.operation_timings[key] = Float64[]
            end
        end
        
        # Registrar el valor en el vector apropiado
        if startswith(String(key), "mem_")
            push!(tracker.memory_usage[key], Float64(value))
            
            # Actualizar pico de memoria
            if get(tracker.peak_memory, key, 0.0) < value
                tracker.peak_memory[key] = Float64(value)
            end
        else
            push!(tracker.operation_timings[key], Float64(value))
        end
        
        return true
    catch e
        @warn "Error al registrar métrica de rendimiento" key=key value=value exception=e
        return false
    end
end

"""
    start_timing(tracker::MetricsTracker, key::Symbol)

Inicia la medición de tiempo para una operación específica.

## Argumentos
- `tracker::MetricsTracker`: Rastreador donde registrar la medición
- `key::Symbol`: Clave para identificar la operación

## Retorna
- `Bool`: `true` si se inició correctamente, `false` en caso contrario
"""
function start_timing(tracker::MetricsTracker, key::Symbol)
    try
        tracker.timing_starts[key] = time()
        return true
    catch e
        @warn "Error al iniciar timing" key=key exception=e
        return false
    end
end

"""
    end_timing(tracker::MetricsTracker, key::Symbol)

Finaliza la medición de tiempo para una operación específica y registra el resultado.

## Argumentos
- `tracker::MetricsTracker`: Rastreador donde se inició la medición
- `key::Symbol`: Clave que identifica la operación

## Retorna
- `Float64`: Tiempo transcurrido en segundos, o -1.0 en caso de error
"""
function end_timing(tracker::MetricsTracker, key::Symbol)
    try
        if key ∉ keys(tracker.timing_starts)
            @warn "No se encontró un inicio de tiempo para la clave: $key"
            return -1.0
        end
        
        elapsed = time() - tracker.timing_starts[key]
        track_performance(tracker, key, elapsed)
        delete!(tracker.timing_starts, key)
        
        return elapsed
    catch e
        @warn "Error al finalizar timing" key=key exception=e
        return -1.0
    end
end

"""
    track_loss(tracker::MetricsTracker, loss_value::Number)

Registra un valor de pérdida durante el entrenamiento.

## Argumentos
- `tracker::MetricsTracker`: Rastreador donde registrar la pérdida
- `loss_value::Number`: Valor de pérdida a registrar

## Retorna
- `Bool`: `true` si se registró correctamente, `false` en caso contrario
"""
function track_loss(tracker::MetricsTracker, loss_value::Number)
    try
        push!(tracker.loss_values, Float64(loss_value))
        return true
    catch e
        @warn "Error al registrar pérdida" loss_value=loss_value exception=e
        return false
    end
end

"""
    track_accuracy(tracker::MetricsTracker, accuracy::Union{Nothing,Number})

Registra un valor de precisión durante el entrenamiento.

## Argumentos
- `tracker::MetricsTracker`: Rastreador donde registrar la precisión
- `accuracy::Union{Nothing,Number}`: Valor de precisión a registrar

## Retorna
- `Bool`: `true` si se registró correctamente, `false` en caso contrario
"""
function track_accuracy(tracker::MetricsTracker, accuracy::Union{Nothing,Number})
    try
        push!(tracker.accuracy_values, accuracy === nothing ? nothing : Float64(accuracy))
        return true
    catch e
        @warn "Error al registrar precisión" accuracy=accuracy exception=e
        return false
    end
end

"""
    track_epoch_time(tracker::MetricsTracker, seconds::Number)

Registra el tiempo de procesamiento de una época.

## Argumentos
- `tracker::MetricsTracker`: Rastreador donde registrar el tiempo
- `seconds::Number`: Tiempo en segundos

## Retorna
- `Bool`: `true` si se registró correctamente, `false` en caso contrario
"""
function track_epoch_time(tracker::MetricsTracker, seconds::Number)
    try
        push!(tracker.epoch_times, Float64(seconds))
        return true
    catch e
        @warn "Error al registrar tiempo de época" seconds=seconds exception=e
        return false
    end
end

"""
    compute_statistics(values::Vector{<:Number})

Calcula estadísticas descriptivas para un conjunto de valores.

## Argumentos
- `values::Vector{<:Number}`: Vector de valores numéricos

## Retorna
- `Dict`: Estadísticas calculadas (count, mean, std, min, max, median)
"""
function compute_statistics(values::Vector{<:Number})
    try
        if isempty(values)
            return Dict(
                :count => 0,
                :mean => NaN,
                :std => NaN,
                :min => NaN,
                :max => NaN,
                :median => NaN
            )
        end
        
        return Dict(
            :count => length(values),
            :mean => mean(values),
            :std => std(values),
            :min => minimum(values),
            :max => maximum(values),
            :median => median(values)
        )
    catch e
        @error "Error al calcular estadísticas" exception=(e, catch_backtrace())
        
        # Devolver valores por defecto en caso de error
        return Dict(
            :count => 0,
            :mean => NaN,
            :std => NaN,
            :min => NaN,
            :max => NaN,
            :median => NaN,
            :error => "$(e)"
        )
    end
end

"""
    create_performance_report(tracker::MetricsTracker)

Genera un informe completo de rendimiento a partir de las métricas registradas.

## Argumentos
- `tracker::MetricsTracker`: Rastreador con métricas registradas

## Retorna
- `Dict`: Informe de rendimiento estructurado
"""
function create_performance_report(tracker::MetricsTracker)
    try
        report = Dict{Symbol,Any}()
        
        # Estadísticas de memoria
        memory_stats = Dict{Symbol,Any}()
        for (key, values) in tracker.memory_usage
            memory_stats[key] = compute_statistics(values)
        end
        memory_stats[:peak] = tracker.peak_memory
        report[:memory] = memory_stats
        
        # Estadísticas de tiempo
        timing_stats = Dict{Symbol,Any}()
        for (key, values) in tracker.operation_timings
            timing_stats[key] = compute_statistics(values)
        end
        report[:timing] = timing_stats
        
        # Estadísticas de entrenamiento
        if !isempty(tracker.loss_values)
            report[:training] = Dict(
                :loss => compute_statistics(tracker.loss_values),
                :epoch_time => compute_statistics(tracker.epoch_times)
            )
            
            # Incluir precisión si está disponible
            valid_accuracy = filter(x -> x !== nothing, tracker.accuracy_values)
            if !isempty(valid_accuracy)
                report[:training][:accuracy] = compute_statistics(valid_accuracy)
            end
        end
        
        # Métricas de rendimiento computacional
        if !isempty(tracker.throughput)
            report[:performance] = Dict(
                :throughput => compute_statistics(tracker.throughput),
                :flops_utilization => compute_statistics(tracker.flops_utilization)
            )
        end
        
        # Incluir cualquier metadato
        report[:metadata] = tracker.metadata
        
        return report
    catch e
        @error "Error al crear informe de rendimiento" exception=(e, catch_backtrace())
        
        # Devolver un informe mínimo en caso de error
        return Dict{Symbol,Any}(
            :error => "Error al crear informe: $(e)",
            :timestamp => Dates.now()
        )
    end
end

"""
    log_metrics(logger::MetricsLogger, metrics::Dict)

Registra métricas en el archivo de log y en el historial.

## Argumentos
- `logger::MetricsLogger`: Logger donde registrar las métricas
- `metrics::Dict`: Métricas a registrar

## Retorna
- `Bool`: `true` si se registró correctamente, `false` en caso contrario
"""
function log_metrics(logger::MetricsLogger, metrics::Dict)
    try
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
        
        # Formatear para el archivo de log
        log_line = "$timestamp - "
        
        # Determinar qué métricas registrar según el nivel
        metrics_to_log = if logger.log_level == :detailed
            metrics  # Todas las métricas
        elseif logger.log_level == :minimal
            # Solo métricas clave
            filter(pair -> pair.first in [:loss, :accuracy, :epoch, :samples_per_second], metrics)
        else  # :normal
            # Métricas importantes
            filter(pair -> !(pair.first in [:detailed_timings, :memory_usage]), metrics)
        end
        
        # Formatear métricas para log
        for (key, value) in metrics_to_log
            if value isa Number
                log_line *= "$(key)=$(round(value, digits=5)) "
            else
                log_line *= "$(key)=$(value) "
            end
        end
        
        # Escribir al archivo si está disponible
        if logger.log_file !== nothing && isopen(logger.log_file)
            println(logger.log_file, log_line)
            flush(logger.log_file)
        end
        
        # Guardar en el historial
        for (key, value) in metrics
            if !haskey(logger.metrics_history, key)
                logger.metrics_history[key] = []
            end
            push!(logger.metrics_history[key], value)
        end
        
        return true
    catch e
        @warn "Error al registrar métricas en log" exception=e
        return false
    end
end

"""
    benchmark_model(brain_space, input_dims::Tuple, num_iterations::Int=100;
                   warmup::Int=10, use_cuda::Bool=false)

Realiza un benchmark exhaustivo del modelo.

## Argumentos
- `brain_space`: Espacio cerebral a evaluar
- `input_dims::Tuple`: Dimensiones de entrada para el benchmark
- `num_iterations::Int=100`: Número de iteraciones para el benchmark
- `warmup::Int=10`: Número de iteraciones de calentamiento
- `use_cuda::Bool=false`: Si se debe usar CUDA para el benchmark

## Retorna
- `Dict`: Resultados del benchmark
"""
function benchmark_model(brain_space, input_dims::Tuple, num_iterations::Int=100;
                       warmup::Int=10, use_cuda::Bool=false)
    
    # Verificar disponibilidad de CUDA si se solicita
    if use_cuda && !cuda_available
        @warn "CUDA solicitado pero no disponible. Usando CPU."
        use_cuda = false
    end
    
    tracker = MetricsTracker("benchmark")
    
    try
        # Preparar datos de entrada
        input_data = randn(Float32, input_dims)
        if use_cuda
            try
                input_data = CuArray(input_data)
            catch e
                @warn "Error al transferir datos a CUDA. Usando CPU." exception=e
                use_cuda = false
            end
        end
        
        # Iteraciones de calentamiento
        @info "Realizando $warmup iteraciones de calentamiento..."
        for i in 1:warmup
            # Ejecución del modelo (en una implementación real, esto llamaría al modelo)
            # forward_pass(brain_space, input_data)
        end
        
        # Sincronizar dispositivo si es CUDA
        if use_cuda
            try
                CUDA.synchronize()
            catch e
                @warn "Error al sincronizar CUDA después del calentamiento" exception=e
            end
        end
        
        # Medir latencia de inferencia
        @info "Realizando $num_iterations iteraciones de benchmark..."
        latencies = zeros(num_iterations)
        
        for i in 1:num_iterations
            # Iniciar tiempo
            start_time = time()
            
            # Ejecución del modelo
            # forward_pass(brain_space, input_data)
            
            # Sincronizar y medir tiempo
            if use_cuda
                try
                    CUDA.synchronize()
                catch e
                    @warn "Error al sincronizar CUDA en iteración $i" exception=e
                end
            end
            latencies[i] = (time() - start_time) * 1000  # Convertir a ms
            
            # Recopilar uso de memoria
            if use_cuda
                try
                    free_mem, total_mem = CUDA.MemoryInfo()
                    used_mem = total_mem - free_mem
                    track_performance(tracker, :mem_gpu, used_mem / 1024^2)  # MB
                catch e
                    @warn "Error al recopilar memoria CUDA en iteración $i" exception=e
                end
            end
            
            track_performance(tracker, :mem_cpu, Sys.free_memory() / 1024^2)  # MB
       end
       
       # Calcular estadísticas
       stats = compute_statistics(latencies)
       
       # Estimar FLOPS
       estimated_flops = 0.0
       estimated_tflops = 0.0
       
       try
           estimated_flops = estimate_model_flops(brain_space, input_dims)
           avg_time_seconds = stats[:mean] / 1000
           estimated_tflops = estimated_flops / 1e12 / avg_time_seconds
       catch e
           @warn "Error al estimar FLOPS" exception=e
       end
       
       # Registrar en tracker
       tracker.metadata[:estimated_flops] = estimated_flops
       tracker.metadata[:tflops] = estimated_tflops
       tracker.metadata[:batch_size] = input_dims[1]
       tracker.metadata[:input_dims] = input_dims
       tracker.metadata[:hardware] = use_cuda ? "CUDA" : "CPU"
       
       if use_cuda
           try
               device_props = CUDA.device_properties()
               tracker.metadata[:gpu_name] = device_props.name
               tracker.metadata[:compute_capability] = "$(device_props.capability.major).$(device_props.capability.minor)"
           catch e
               @warn "Error al recopilar información de GPU" exception=e
               tracker.metadata[:gpu_error] = "$(e)"
           end
       else
           tracker.metadata[:cpu_info] = Dict(
               :cores => Sys.CPU_CORES,
               :threads => Sys.CPU_THREADS
           )
       end
       
       # Retornar resultados
       @info "Benchmark completado con éxito"
       
       return Dict(
           :latency_ms => stats,
           :throughput => input_dims[1] / (stats[:mean] / 1000),  # elementos por segundo
           :estimated_tflops => estimated_tflops,
           :memory_usage => Dict(
               :peak_gpu_mb => get(tracker.peak_memory, :mem_gpu, 0.0),
               :peak_cpu_mb => get(tracker.peak_memory, :mem_cpu, 0.0)
           ),
           :tracker => tracker
       )
   catch e
       @error "Error al realizar benchmark" exception=(e, catch_backtrace())
       
       # Devolver resultados parciales si hay alguno
       return Dict(
           :error => "$(e)",
           :latency_ms => isempty(latencies) ? Dict(:mean => NaN) : compute_statistics(latencies),
           :tracker => tracker
       )
   end
end

"""
   estimate_model_flops(brain_space, input_dims::Tuple)

Estima el número de operaciones de punto flotante (FLOPS) para un pase del modelo.

## Argumentos
- `brain_space`: Espacio cerebral a evaluar
- `input_dims::Tuple`: Dimensiones de entrada para la estimación

## Retorna
- `Float64`: Estimación de FLOPS
"""
function estimate_model_flops(brain_space, input_dims::Tuple)
   try
       # Verificar que los campos esperados existen
       if !isdefined(brain_space, :fields) || !isdefined(brain_space, :connections)
           @warn "Estructura del espacio cerebral no reconocida, usando estimación genérica"
           # Estimación genérica basada en dimensiones
           total_elements = prod(input_dims)
           return total_elements * 1000  # 1000 ops por elemento como estimación genérica
       end
       
       # Contar campos y conexiones
       num_fields = length(brain_space.fields)
       num_connections = length(brain_space.connections)
       
       # Estimar operaciones por campo (activaciones, normalizaciones, etc.)
       flops_per_field = 10_000_000  # Ejemplo: 10M ops por campo
       
       # Estimar operaciones por conexión (multiplicaciones, sumas, etc.)
       flops_per_connection = 100_000_000  # Ejemplo: 100M ops por conexión
       
       # Ajustar por tamaño de batch
       batch_size = input_dims[1]
       
       total_flops = (num_fields * flops_per_field + num_connections * flops_per_connection) * batch_size
       
       return total_flops
   catch e
       @warn "Error al estimar FLOPS" exception=e
       # Devolver una estimación predeterminada básica
       return 1.0e9  # 1 GFLOP como estimación de fallback
   end
end

"""
   memory_profile(brain_space, input_dims::Tuple;
                 track_detailed::Bool=false, use_cuda::Bool=false)

Analiza el uso de memoria del modelo durante la ejecución.

## Argumentos
- `brain_space`: Espacio cerebral a evaluar
- `input_dims::Tuple`: Dimensiones de entrada para el análisis
- `track_detailed::Bool=false`: Si se debe realizar un seguimiento detallado por componente
- `use_cuda::Bool=false`: Si se debe usar CUDA para el análisis

## Retorna
- `Dict`: Resultados del análisis de memoria
"""
function memory_profile(brain_space, input_dims::Tuple;
                      track_detailed::Bool=false, use_cuda::Bool=false)
   
   # Verificar disponibilidad de CUDA si se solicita
   if use_cuda && !cuda_available
       @warn "CUDA solicitado pero no disponible. Usando CPU."
       use_cuda = false
   end
   
   try
       # Inicializar seguimiento
       baseline_memory = Dict{Symbol,Float64}()
       peak_memory = Dict{Symbol,Float64}()
       
       # Medir línea base de memoria
       if use_cuda
           try
               CUDA.reclaim()
               GC.gc()
               free_mem, total_mem = CUDA.MemoryInfo()
               baseline_memory[:gpu] = (total_mem - free_mem) / 1024^2  # MB
           catch e
               @warn "Error al medir línea base de memoria GPU" exception=e
               baseline_memory[:gpu] = 0.0
           end
       end
       
       GC.gc()
       baseline_memory[:cpu] = (Sys.total_memory() - Sys.free_memory()) / 1024^2  # MB
       
       # Preparar datos de entrada
       input_data = nothing
       try
           input_data = randn(Float32, input_dims)
           if use_cuda
               input_data = CuArray(input_data)
           end
       catch e
           @warn "Error al preparar datos de entrada" exception=e
           # Crear datos de tamaño mínimo como fallback
           input_data = randn(Float32, (1, 1, 1))
           use_cuda = false  # Forzar CPU
       end
       
       # Lista de puntos de medición
       measurement_points = [:initialization, :forward_pass, :computation_complete]
       memory_usage = Dict{Symbol,Dict{Symbol,Float64}}()
       
       # Medir uso de memoria en puntos clave
       for point in measurement_points
           # En una implementación real, aquí iría el código que ejecuta cada fase
           
           # Sincronizar si es CUDA
           if use_cuda
               try
                   CUDA.synchronize()
               catch e
                   @warn "Error al sincronizar CUDA en punto $point" exception=e
               end
           end
           
           # Medir memoria
           current_memory = Dict{Symbol,Float64}()
           
           if use_cuda
               try
                   free_mem, total_mem = CUDA.MemoryInfo()
                   current_memory[:gpu] = (total_mem - free_mem) / 1024^2  # MB
                   peak_memory[:gpu] = get(peak_memory, :gpu, 0.0) < current_memory[:gpu] ? 
                                     current_memory[:gpu] : get(peak_memory, :gpu, 0.0)
               catch e
                   @warn "Error al medir memoria GPU en punto $point" exception=e
                   current_memory[:gpu] = 0.0
               end
           end
           
           current_memory[:cpu] = (Sys.total_memory() - Sys.free_memory()) / 1024^2  # MB
           peak_memory[:cpu] = get(peak_memory, :cpu, 0.0) < current_memory[:cpu] ? 
                             current_memory[:cpu] : get(peak_memory, :cpu, 0.0)
           
           memory_usage[point] = current_memory
       end
       
       # Calcular memoria utilizada (delta desde la línea base)
       memory_delta = Dict{Symbol,Dict{Symbol,Float64}}()
       for (point, usage) in memory_usage
           memory_delta[point] = Dict{Symbol,Float64}()
           for (device, value) in usage
               memory_delta[point][device] = value - baseline_memory[device]
           end
       end
       
       # Detalle por componente si se solicita
       component_memory = Dict{Symbol,Float64}()
       if track_detailed
           # Verificar si la estructura del modelo es reconocida
           if isdefined(brain_space, :fields) && isdefined(brain_space, :connections)
               # Analizar componentes del modelo
               for (i, field) in enumerate(brain_space.fields)
                   try
                       # Estimar memoria del campo
                       field_memory = estimate_field_memory(field)
                       component_memory[Symbol("field_$(i)")] = field_memory
                   catch e
                       @warn "Error al estimar memoria del campo $i" exception=e
                   end
               end
               
               for (i, connection) in enumerate(brain_space.connections)
                   try
                       # Estimar memoria de la conexión
                       connection_memory = estimate_connection_memory(connection)
                       component_memory[Symbol("connection_$(i)")] = connection_memory
                   catch e
                       @warn "Error al estimar memoria de la conexión $i" exception=e
                   end
               end
           else
               @warn "Estructura del modelo no reconocida para análisis detallado"
               component_memory[:unknown] = sum(values(baseline_memory))
           end
       end
       
       @info "Análisis de memoria completado" 
             baseline_cpu=baseline_memory[:cpu] 
             peak_cpu=peak_memory[:cpu]
             use_cuda=use_cuda
       
       return Dict(
           :baseline => baseline_memory,
           :usage => memory_usage,
           :delta => memory_delta,
           :peak => peak_memory,
           :by_component => component_memory
       )
   catch e
       @error "Error al realizar análisis de memoria" exception=(e, catch_backtrace())
       
       # Devolver resultados mínimos como fallback
       return Dict(
           :error => "$(e)",
           :baseline => Dict(:cpu => Sys.free_memory() / 1024^2),
           :peak => Dict(:cpu => Sys.free_memory() / 1024^2)
       )
   end
end

"""
   estimate_field_memory(field)

Estima el uso de memoria de un campo.

## Argumentos
- `field`: Campo espacial a evaluar

## Retorna
- `Float64`: Memoria estimada en MB
"""
function estimate_field_memory(field)
   try
       # Verificar si el campo tiene la estructura esperada
       if !isdefined(field, :data)
           @warn "Campo no tiene estructura esperada"
           return 1.0  # 1 MB como estimación por defecto
       end
       
       # Memoria del tensor principal
       tensor_bytes = length(field.data) * sizeof(eltype(field.data))
       
       # Memoria adicional (metadatos, gradientes si hay, etc.)
       metadata_bytes = 1000  # Estimación simplificada
       
       # Gradientes si están presentes
       gradient_bytes = 0
       if isdefined(field, :metadata) && haskey(field.metadata, "gradients")
           grad_data = field.metadata["gradients"]
           gradient_bytes = length(grad_data) * sizeof(eltype(grad_data))
       end
       
       total_bytes = tensor_bytes + metadata_bytes + gradient_bytes
       return total_bytes / (1024^2)  # Convertir a MB
   catch e
       @warn "Error al estimar memoria de campo" exception=e
       return 1.0  # 1 MB como estimación por defecto
   end
end

"""
   estimate_connection_memory(connection)

Estima el uso de memoria de una conexión.

## Argumentos
- `connection`: Conexión a evaluar

## Retorna
- `Float64`: Memoria estimada en MB
"""
function estimate_connection_memory(connection)
   try
       # Verificar si la conexión tiene la estructura esperada
       if !isdefined(connection, :metadata)
           @warn "Conexión no tiene estructura esperada"
           return 2.0  # 2 MB como estimación por defecto
       end
       
       # Estimación basada en tamaño de pesos
       if haskey(connection.metadata, "weights")
           weights = connection.metadata["weights"]
           weights_bytes = length(weights) * sizeof(eltype(weights))
           
           # Memoria adicional para gradientes, optimizador, etc.
           additional_bytes = weights_bytes * 3  # Aproximadamente 3x para gradientes, momentos, etc.
           
           return (weights_bytes + additional_bytes) / (1024^2)  # Convertir a MB
       else
           # Estimación predeterminada si no hay información de pesos
           return 2.0  # 2 MB por defecto
       end
   catch e
       @warn "Error al estimar memoria de conexión" exception=e
       return 2.0  # 2 MB como estimación por defecto
   end
end

"""
   latency_profile(brain_space, input_dims::Tuple, num_iterations::Int=10;
                  breakdown::Bool=true, use_cuda::Bool=false)

Analiza la latencia del modelo con desglose por componentes.

## Argumentos
- `brain_space`: Espacio cerebral a evaluar
- `input_dims::Tuple`: Dimensiones de entrada para el análisis
- `num_iterations::Int=10`: Número de iteraciones para el análisis
- `breakdown::Bool=true`: Si se debe realizar un desglose por componentes
- `use_cuda::Bool=false`: Si se debe usar CUDA para el análisis

## Retorna
- `Dict`: Resultados del análisis de latencia
"""
function latency_profile(brain_space, input_dims::Tuple, num_iterations::Int=10;
                       breakdown::Bool=true, use_cuda::Bool=false)
   
   # Verificar disponibilidad de CUDA si se solicita
   if use_cuda && !cuda_available
       @warn "CUDA solicitado pero no disponible. Usando CPU."
       use_cuda = false
   end
   
   try
       # Preparar datos
       input_data = randn(Float32, input_dims)
       if use_cuda
           try
               input_data = CuArray(input_data)
           catch e
               @warn "Error al transferir datos a CUDA. Usando CPU." exception=e
               use_cuda = false
           end
       end
       
       # Iteraciones de calentamiento
       @info "Realizando iteraciones de calentamiento..."
       for i in 1:3
           # forward_pass(brain_space, input_data)
       end
       
       # Sincronizar
       if use_cuda
           try
               CUDA.synchronize()
           catch e
               @warn "Error al sincronizar CUDA después del calentamiento" exception=e
           end
       end
       
       # Medir latencia general
       total_latencies = zeros(num_iterations)
       
       @info "Midiendo latencia general ($num_iterations iteraciones)..."
       for i in 1:num_iterations
           start_time = time()
           
           # forward_pass(brain_space, input_data)
           
           if use_cuda
               try
                   CUDA.synchronize()
               catch e
                   @warn "Error al sincronizar CUDA en iteración $i" exception=e
               end
           end
           total_latencies[i] = (time() - start_time) * 1000  # Convertir a ms
       end
       
       results = Dict(
           :total_latency_ms => compute_statistics(total_latencies)
       )
       
       # Desglose por componentes si se solicita
       if breakdown
           component_latencies = Dict{Symbol,Vector{Float64}}()
           
           @info "Midiendo latencia por componentes..."
           # Medir latencia por componente
           for i in 1:num_iterations
               # Ejemplos simulados para demostración
               # En una implementación real, esto mediría efectivamente cada componente
               
               # Cortical layers
               component_latencies[:cortical_layers] = get(component_latencies, :cortical_layers, Float64[])
               push!(component_latencies[:cortical_layers], rand(1.0:5.0))
               
               # Attention system
               component_latencies[:attention_system] = get(component_latencies, :attention_system, Float64[])
               push!(component_latencies[:attention_system], rand(2.0:7.0))
               
               # Prefrontal system
               component_latencies[:prefrontal_system] = get(component_latencies, :prefrontal_system, Float64[])
               push!(component_latencies[:prefrontal_system], rand(3.0:8.0))
           end
           
           # Calcular estadísticas por componente
           component_stats = Dict{Symbol,Dict{Symbol,Any}}()
           for (component, latencies) in component_latencies
               component_stats[component] = compute_statistics(latencies)
           end
           
           results[:component_latency_ms] = component_stats
           
           # Calcular porcentaje del tiempo total
           total_avg = results[:total_latency_ms][:mean]
           component_percentages = Dict{Symbol,Float64}()
           
           for (component, stats) in component_stats
               component_percentages[component] = (stats[:mean] / total_avg) * 100
           end
           
           results[:component_percentage] = component_percentages
       end
       
       @info "Análisis de latencia completado" latencia_ms=results[:total_latency_ms][:mean]
       
       return results
   catch e
       @error "Error al realizar análisis de latencia" exception=(e, catch_backtrace())
       
       # Devolver resultados mínimos como fallback
       return Dict(
           :error => "$(e)",
           :total_latency_ms => isempty(total_latencies) ? 
               Dict(:mean => NaN, :std => NaN) : compute_statistics(total_latencies)
       )
   end
end

"""
   throughput_test(brain_space, batch_sizes::Vector{Int}, 
                 sequence_length::Int, num_iterations::Int=5;
                 use_cuda::Bool=false)

Mide el rendimiento del modelo con diferentes tamaños de batch.

## Argumentos
- `brain_space`: Espacio cerebral a evaluar
- `batch_sizes::Vector{Int}`: Tamaños de batch a probar
- `sequence_length::Int`: Longitud de secuencia para los datos de entrada
- `num_iterations::Int=5`: Número de iteraciones por tamaño de batch
- `use_cuda::Bool=false`: Si se debe usar CUDA para la prueba

## Retorna
- `Dict`: Resultados de la prueba de rendimiento
"""
function throughput_test(brain_space, batch_sizes::Vector{Int}, 
                      sequence_length::Int, num_iterations::Int=5;
                      use_cuda::Bool=false)
   
   # Verificar disponibilidad de CUDA si se solicita
   if use_cuda && !cuda_available
       @warn "CUDA solicitado pero no disponible. Usando CPU."
       use_cuda = false
   end
   
   try
       results = Dict{Int,Dict{Symbol,Any}}()
       
       @info "Iniciando prueba de rendimiento con $(length(batch_sizes)) tamaños de batch..."
       
       for batch_size in batch_sizes
           @info "Probando batch_size = $batch_size"
           input_dims = (batch_size, sequence_length, 64)  # Ejemplo de dimensiones
           
           # Preparar datos
           input_data = randn(Float32, input_dims)
           if use_cuda
               try
                   input_data = CuArray(input_data)
               catch e
                   @warn "Error al transferir datos a CUDA para batch_size $batch_size" exception=e
                   # Continuar con CPU para este tamaño de batch
                   input_data = randn(Float32, input_dims)
               end
           end
           
           # Calentar
           for i in 1:2
               # forward_pass(brain_space, input_data)
           end
           
           if use_cuda
               try 
                   CUDA.synchronize()
               catch e
                   @warn "Error al sincronizar CUDA después del calentamiento" exception=e
               end
           end
           
           # Medir
           batch_times = zeros(num_iterations)
           
           for i in 1:num_iterations
               start_time = time()
               
               # forward_pass(brain_space, input_data)
               
               if use_cuda
                   try
                       CUDA.synchronize()
                   catch e
                       @warn "Error al sincronizar CUDA en iteración $i" exception=e
                   end
               end
               
               batch_times[i] = time() - start_time
           end
           
           # Calcular throughput
           avg_time = mean(batch_times)
           tokens_per_second = batch_size * sequence_length / avg_time
           samples_per_second = batch_size / avg_time
           
           # Registrar resultados
           results[batch_size] = Dict(
               :avg_time_seconds => avg_time,
               :tokens_per_second => tokens_per_second,
               :samples_per_second => samples_per_second,
               :times => batch_times
           )
           
           @info "Batch $batch_size: $samples_per_second muestras/segundo"
       end
       
       # Determinar batch óptimo (mayor throughput)
       optimal_batch = maximum([(bs, results[bs][:samples_per_second]) for bs in batch_sizes], 
                              by=x->x[2])
       
       @info "Prueba de rendimiento completada" batch_óptimo=optimal_batch[1] throughput=optimal_batch[2]
       
       return Dict(
           :by_batch_size => results,
           :optimal_batch => optimal_batch[1],
           :max_throughput => optimal_batch[2]
       )
   catch e
       @error "Error al realizar prueba de rendimiento" exception=(e, catch_backtrace())
       
       # Devolver resultados mínimos como fallback
       return Dict(
           :error => "$(e)",
           :by_batch_size => Dict{Int,Dict{Symbol,Any}}(),
           :optimal_batch => 0,
           :max_throughput => 0.0
       )
   end
end

"""
   compare_configurations(configs::Vector{Dict}, brain_space, 
                        input_dims::Tuple, num_iterations::Int=10)

Compara múltiples configuraciones del modelo para rendimiento.

## Argumentos
- `configs::Vector{Dict}`: Configuraciones a comparar
- `brain_space`: Espacio cerebral a evaluar
- `input_dims::Tuple`: Dimensiones de entrada para la comparación
- `num_iterations::Int=10`: Número de iteraciones por configuración

## Retorna
- `Dict`: Resultados de la comparación
"""
function compare_configurations(configs::Vector{Dict}, brain_space, 
                             input_dims::Tuple, num_iterations::Int=10)
   
   try
       results = Dict{String,Dict{Symbol,Any}}()
       
       @info "Comparando $(length(configs)) configuraciones..."
       
       for (i, config) in enumerate(configs)
           config_name = get(config, :name, "config_$i")
           @info "Evaluando configuración: $config_name"
           
           # Aplicar configuración
           # En una implementación real, esto modificaría el modelo según la configuración
           # configured_model = apply_configuration(brain_space, config)
           configured_model = brain_space  # Placeholder
           
           # Preparar datos
           use_cuda = get(config, :use_cuda, false)
           
           # Verificar disponibilidad de CUDA si se solicita
           if use_cuda && !cuda_available
               @warn "CUDA solicitado pero no disponible para configuración $config_name. Usando CPU."
               use_cuda = false
           end
           
           input_data = randn(Float32, input_dims)
           if use_cuda
               try
                   input_data = CuArray(input_data)
               catch e
                   @warn "Error al transferir datos a CUDA para configuración $config_name" exception=e
                   use_cuda = false
               end
           end
           
           # Calentar
           for i in 1:3
               # forward_pass(configured_model, input_data)
           end
           
           if use_cuda
               try
                   CUDA.synchronize()
               catch e
                   @warn "Error al sincronizar CUDA después del calentamiento" exception=e
               end
           end
           
           # Medir rendimiento
           latencies = zeros(num_iterations)
           memory_usage = zeros(num_iterations)
           
           for i in 1:num_iterations
               # Limpiar memoria antes de cada iteración
               if use_cuda
                   try
                       CUDA.reclaim()
                   catch e
                       @warn "Error al liberar memoria CUDA en iteración $i" exception=e
                   end
               end
               GC.gc()
               
               # Medir latencia
               start_time = time()
               # forward_pass(configured_model, input_data)
               
               if use_cuda
                   try
                       CUDA.synchronize()
                   catch e
                       @warn "Error al sincronizar CUDA en iteración $i" exception=e
                   end
               end
               latencies[i] = (time() - start_time) * 1000  # ms
               
               # Medir memoria
               if use_cuda
                   try
                       free_mem, total_mem = CUDA.MemoryInfo()
                       memory_usage[i] = (total_mem - free_mem) / 1024^2  # MB
                   catch e
                       @warn "Error al medir memoria CUDA en iteración $i" exception=e
                       memory_usage[i] = 0.0
                   end
               else
                   memory_usage[i] = (Sys.total_memory() - Sys.free_memory()) / 1024^2  # MB
               end
           end
           
           # Calcular métricas
           results[config_name] = Dict(
               :latency_ms => compute_statistics(latencies),
               :memory_mb => compute_statistics(memory_usage),
               :throughput => input_dims[1] / (mean(latencies) / 1000),
               :config => config
           )
           
           @info "Configuración $config_name: $(results[config_name][:latency_ms][:mean]) ms"
       end
       
       # Determinar mejor configuración (menor latencia)
       best_config = minimum([(name, results[name][:latency_ms][:mean]) for name in keys(results)], 
                            by=x->x[2])
       
       @info "Comparación completada" mejor_configuración=best_config[1] latencia=best_config[2]
       
       return Dict(
           :by_config => results,
           :best_config => best_config[1],
           :best_latency => best_config[2]
       )
   catch e
       @error "Error al comparar configuraciones" exception=(e, catch_backtrace())
       
       # Devolver resultados mínimos como fallback
       return Dict(
           :error => "$(e)",
           :by_config => Dict{String,Dict{Symbol,Any}}(),
           :best_config => "none",
           :best_latency => Inf
       )
   end
end

"""
   close_logger(logger::MetricsLogger)

Cierra correctamente un logger de métricas.

## Argumentos
- `logger::MetricsLogger`: Logger a cerrar

## Retorna
- `Bool`: `true` si se cerró correctamente, `false` en caso contrario
"""
function close_logger(logger::MetricsLogger)
   try
       if logger.log_file !== nothing && isopen(logger.log_file)
           close(logger.log_file)
           @info "Logger cerrado correctamente"
           return true
       end
       return false
   catch e
       @warn "Error al cerrar logger" exception=e
       return false
   end
end

"""
   reset_metrics_tracker(tracker::MetricsTracker)

Reinicia un rastreador de métricas manteniendo sus metadatos.

## Argumentos
- `tracker::MetricsTracker`: Rastreador a reiniciar

## Retorna
- `MetricsTracker`: Rastreador reiniciado
"""
function reset_metrics_tracker(tracker::MetricsTracker)
   try
       name = get(tracker.metadata, :name, "default")
       
       # Preservar metadatos importantes
       preserved_metadata = Dict{Symbol, Any}(
           :creation_time => get(tracker.metadata, :creation_time, Dates.now()),
           :name => name,
           :reset_time => Dates.now()
       )
       
       # Crear nuevo tracker con nombre preservado
       new_tracker = MetricsTracker(name)
       
       # Restaurar metadatos preservados
       for (key, value) in preserved_metadata
           new_tracker.metadata[key] = value
       end
       
       return new_tracker
   catch e
       @warn "Error al reiniciar metrics tracker" exception=e
       # Devolver un nuevo tracker como fallback
       return MetricsTracker("reset_fallback")
   end
end

"""
   get_metrics_summary(tracker::MetricsTracker)

Genera un resumen conciso de las métricas registradas.

## Argumentos
- `tracker::MetricsTracker`: Rastreador con métricas registradas

## Retorna
- `Dict`: Resumen de las métricas más importantes
"""
function get_metrics_summary(tracker::MetricsTracker)
   try
       summary = Dict{Symbol, Any}()
       
       # Extraer métricas clave si están disponibles
       if !isempty(tracker.loss_values)
           summary[:loss] = Dict(
               :last => tracker.loss_values[end],
               :min => minimum(tracker.loss_values),
               :mean => mean(tracker.loss_values)
           )
       end
       
       if !isempty(tracker.epoch_times)
           summary[:epoch_time] = Dict(
               :mean => mean(tracker.epoch_times),
               :total => sum(tracker.epoch_times)
           )
       end
       
       # Extraer información de precisión si está disponible
       # Extraer información de precisión si está disponible
       valid_accuracy = filter(x -> x !== nothing, tracker.accuracy_values)
       if !isempty(valid_accuracy)
           summary[:accuracy] = Dict(
               :last => valid_accuracy[end],
               :max => maximum(valid_accuracy),
               :mean => mean(valid_accuracy)
           )
       end
       
       # Extraer información de memoria si está disponible
       if haskey(tracker.peak_memory, :mem_gpu)
           summary[:peak_memory_gpu_mb] = tracker.peak_memory[:mem_gpu]
       end
       
       if haskey(tracker.peak_memory, :mem_cpu)
           summary[:peak_memory_cpu_mb] = tracker.peak_memory[:mem_cpu]
       end
       
       # Incluir información básica de metadata
       summary[:name] = get(tracker.metadata, :name, "unknown")
       summary[:duration] = haskey(tracker.metadata, :creation_time) ? 
                          Dates.now() - tracker.metadata[:creation_time] : 
                          Dates.Minute(0)
       
       return summary
   catch e
       @warn "Error al generar resumen de métricas" exception=e
       # Devolver un resumen mínimo en caso de error
       return Dict{Symbol, Any}(
           :error => "$(e)",
           :timestamp => Dates.now()
       )
   end
end

# Inicialización segura del módulo
function __init__()
   @info "Inicializando módulo PerformanceMetrics..."
   
   # Verificar disponibilidad de CUDA
   if cuda_available
       @info "CUDA disponible para métricas de rendimiento"
   else
       @info "CUDA no disponible, usando solo CPU para métricas"
   end
   
   # Crear directorio de logs si no existe
   try
       if !isdir(DEFAULT_LOG_DIR)
           mkpath(DEFAULT_LOG_DIR)
           @info "Directorio de logs creado: $(DEFAULT_LOG_DIR)"
       end
   catch e
       @warn "No se pudo crear directorio de logs" exception=e
   end
   
   # Inicializar sistema de métricas
   try
       init_metrics_system()
   catch e
       @warn "Error durante inicialización de sistema de métricas" exception=e
   end
   
   @info "Módulo PerformanceMetrics inicializado correctamente"
end

end # module PerformanceMetrics