"""
RNTA.jl - Red Neuronal Tensorial Adaptativa
===========================================

Una biblioteca de Julia que implementa un nuevo paradigma de deep learning basado en 
representaciones tensoriales tridimensionales y dinámicas inspiradas en estructuras cerebrales.

Esta biblioteca implementa un enfoque revolucionario para la IA basado en:
1. Arquitectura neuronal volumétrica (3D) que supera las limitaciones de las redes tradicionales
2. Mecanismos de adaptación dinámica inspirados en la neuroplasticidad cerebral
3. Sistemas de deliberación interna y diálogo para razonamiento profundo
4. Representaciones tensoriales ricas que capturan relaciones semánticas complejas
5. Optimización automática de la arquitectura según tareas y datos

Autores: [Equipo de Desarrollo RNTA]
"""

# Función auxiliar para carga segura de módulos
function include_safe(path)
    try
        include(path)
        @info "✅ Cargado con éxito: $path"
        return true
    catch e
        @error "❌ Error al cargar: $path" exception=(e, catch_backtrace())
        return false
    end
end

# Función para verificar que un módulo se cargó correctamente
function verify_module(module_name, functions_to_check)
    module_exists = isdefined(Main, Symbol(module_name))
    
    if !module_exists
        @warn "⚠️ El módulo $module_name no está definido"
        return false
    end
    
    module_obj = getfield(Main, Symbol(module_name))
    all_functions_available = true
    
    for func in functions_to_check
        if !isdefined(module_obj, func)
            @warn "⚠️ Función $func no definida en módulo $module_name"
            all_functions_available = false
        end
    end
    
    if all_functions_available
        @info "✅ Módulo $module_name verificado correctamente"
    else
        @warn "⚠️ Módulo $module_name cargó pero falta alguna función"
    end
    
    return all_functions_available
end

# Cargar dependencias principales
# Cargar dependencias principales
using LinearAlgebra
using Statistics
using Random
using Distributed
using Dates
using Logging  # Añade esta línea

# Intentar cargar dependencias opcionales con manejo de errores
function load_optional_dependency(pkg_name)
    try
        @eval using $pkg_name
        @info "✅ Dependencia cargada con éxito: $pkg_name"
        return true
    catch e
        @warn "⚠️ Dependencia no disponible: $pkg_name" exception=e
        return false
    end
end

# Cargar dependencias opcionales
fileio_available = load_optional_dependency(:FileIO)
bson_available = load_optional_dependency(:BSON)
uuids_available = load_optional_dependency(:UUIDs)
cuda_available = load_optional_dependency(:CUDA)
colors_available = load_optional_dependency(:Colors)
makie_available = load_optional_dependency(:Makie)
glmakie_available = load_optional_dependency(:GLMakie)

@info "Estado de carga de dependencias opcionales:" fileio_available bson_available uuids_available cuda_available colors_available makie_available glmakie_available

# Configurar registro de errores a archivo
log_dir = joinpath(@__DIR__, "..", "logs")
try
    isdir(log_dir) || mkdir(log_dir)
    log_file = joinpath(log_dir, "rnta_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log")
    open(log_file, "w") do io
        global_logger(ConsoleLogger(io, Logging.Debug))
        @info "Iniciando registro de errores en: $log_file"
    end
catch e
    @warn "No se pudo configurar registro de errores a archivo" exception=e
end

# Restaurar logger a consola
global_logger(ConsoleLogger(stderr, Logging.Info))

@info "Iniciando carga de módulos RNTA..."

# Registrar módulos cargados para depuración
loaded_modules = Dict{String, Bool}()

# 0 dependencias
@info "Cargando módulos base (0 dependencias)..."
loaded_modules["PlasticityRules"] = include_safe("adaptation/PlasticityRules.jl")
loaded_modules["SpatialField"] = include_safe("core/SpatialField.jl")
loaded_modules["TensorTransformations"] = include_safe("operations/TensorTransformations.jl")
loaded_modules["VolumetricActivations"] = include_safe("operations/VolumetricActivations.jl")
loaded_modules["MultidimensionalLoss"] = include_safe("training/MultidimensionalLoss.jl")
loaded_modules["TensorialTokenizer"] = include_safe("nlp/TensorialTokenizer.jl")
loaded_modules["CUDATensors"] = include_safe("acceleration/CUDATensors.jl")

# necesarios antes 
@info "Cargando módulos core..."
loaded_modules["TensorNeuron"] = include_safe("core/TensorNeuron.jl")
loaded_modules["Connections"] = include_safe("core/Connections.jl")
loaded_modules["BrainSpace"] = include_safe("core/BrainSpace.jl")

# 1 dependencia
@info "Cargando módulos con 1 dependencia..."
loaded_modules["SpatialAttention"] = include_safe("operations/SpatialAttention.jl")
loaded_modules["DynamicExpansion"] = include_safe("adaptation/DynamicExpansion.jl")
loaded_modules["TensorParallelism"] = include_safe("acceleration/TensorParallelism.jl")
loaded_modules["ConfigurationSystem"] = include_safe("utils/ConfigurationSystem.jl")

# 2 dependencias
@info "Cargando módulos con 2 dependencias..."
loaded_modules["HardwareAdaptation"] = include_safe("acceleration/HardwareAdaptation.jl")
loaded_modules["MemoryOptimization"] = include_safe("acceleration/MemoryOptimization.jl")
loaded_modules["TrainingMonitor"] = include_safe("visualization/TrainingMonitor.jl")
loaded_modules["PerformanceMetrics"] = include_safe("utils/PerformanceMetrics.jl")

# 3 dependencias
@info "Cargando módulos con 3 dependencias..."
loaded_modules["SelfPruning"] = include_safe("adaptation/SelfPruning.jl")
loaded_modules["SpatialOptimizers"] = include_safe("training/SpatialOptimizers.jl")
loaded_modules["BrainSpaceVisualizer"] = include_safe("visualization/BrainSpaceVisualizer.jl")
loaded_modules["ActivityMapper"] = include_safe("visualization/ActivityMapper.jl")
loaded_modules["ConnectionVisualizer"] = include_safe("visualization/ConnectionVisualizer.jl")
loaded_modules["TensorIO"] = include_safe("utils/TensorIO.jl")

# 4 dependencias
@info "Cargando módulos con 4 dependencias..."
loaded_modules["PropagationDynamics"] = include_safe("operations/PropagationDynamics.jl")
loaded_modules["Specialization"] = include_safe("adaptation/Specialization.jl")
loaded_modules["GradientPropagation"] = include_safe("training/GradientPropagation.jl")
loaded_modules["SemanticSpace"] = include_safe("nlp/SemanticSpace.jl")

# antes de la inferencia
@info "Cargando módulos pre-inferencia..."
loaded_modules["ModelCloning"] = include_safe("training/ModelCloning.jl")
loaded_modules["ContextualMapping"] = include_safe("nlp/ContextualMapping.jl")
loaded_modules["InternalDialogue"] = include_safe("inference/InternalDialogue.jl")
loaded_modules["ReasoningPathways"] = include_safe("inference/ReasoningPathways.jl")
loaded_modules["UncertaintyEstimation"] = include_safe("inference/UncertaintyEstimation.jl")
loaded_modules["MultimodalIntegration"] = include_safe("inference/MultimodalIntegration.jl")

# 5 dependencias
@info "Cargando módulos con 5 dependencias..."
loaded_modules["HippocampalMemory"] = include_safe("architecture/HippocampalMemory.jl")
loaded_modules["Serialization"] = include_safe("utils/Serialization.jl")

# 6 dependencias
@info "Cargando módulos con 6 dependencias..."
loaded_modules["LanguageGeneration"] = include_safe("nlp/LanguageGeneration.jl")
loaded_modules["CorticalLayers"] = include_safe("architecture/CorticalLayers.jl")
loaded_modules["AttentionalSystem"] = include_safe("architecture/AttentionalSystem.jl")
loaded_modules["PrefrontalSystem"] = include_safe("architecture/PrefrontalSystem.jl")

# Verificar estado de módulos críticos
@info "Verificando estado de módulos críticos..."
critical_modules_status = Dict{String, Bool}()
critical_modules_status["CUDATensors"] = verify_module("CUDATensors", [:detect_hardware, :init_cuda_tensors, :use_cuda_tensors])
critical_modules_status["MemoryOptimization"] = verify_module("MemoryOptimization", [:get_global_memory_pool])
critical_modules_status["BrainSpace"] = verify_module("BrainSpace", [:forward_propagation, :update_attention_map!])

@info "Resumen de estado de módulos:" all_loaded=all(values(loaded_modules)) all_critical=all(values(critical_modules_status))

module RNTA
import Dates: now



@info "Configurando módulo RNTA..."

# Importar todas las dependencias necesarias con manejo de errores
function import_dependency(pkg_name)
    try
        @eval import $pkg_name
        return true
    catch e
        @warn "No se pudo importar $pkg_name: $(sprint(showerror, e))"
        return false
    end
end

# Cargar dependencias básicas
dependencies = [:LinearAlgebra, :Statistics, :Random, :Distributed, :Dates]
optional_dependencies = [:FileIO, :BSON, :UUIDs, :CUDA, :Colors, :Makie, :GLMakie]

# Importar dependencias básicas
basic_deps_status = Dict(dep => import_dependency(dep) for dep in dependencies)
@info "Estado de dependencias básicas:" basic_deps_status

# Importar dependencias opcionales
optional_deps_status = Dict(dep => import_dependency(dep) for dep in optional_dependencies)
@info "Estado de dependencias opcionales:" optional_deps_status

# Importación de módulos RNTA con manejo de errores
function use_module(mod_name, mod_symbol)
    try
        @eval using ..$mod_symbol
        return true
    catch e
        @warn "No se pudo cargar el módulo $mod_name: $(sprint(showerror, e))"
        return false
    end
end

# Lista de todos los módulos a importar
modules_to_import = [
    ("PlasticityRules", :PlasticityRules),
    ("SpatialField", :SpatialField),
    ("TensorOperations", :TensorOperations),
    ("VolumetricActivations", :VolumetricActivations),
    ("MultidimensionalLoss", :dimensionalLoss),
    ("TensorialTokenizer", :Tokenizer),
    ("CUDATensors", :CUDATensors),
    ("TensorNeuron", :TensorNeuron),
    ("Connections", :Connections),
    ("BrainSpace", :BrainSpace),
    ("SpatialAttention", :SpatialAttention),
    ("DynamicExpansion", :DynamicExpansion),
    ("TensorParallelism", :TensorParallelism),
    ("ConfigurationSystem", :ConfigurationSystem),
    ("HardwareAdaptation", :HardwareAdaptation),
    ("MemoryOptimization", :MemoryOptimization),
    ("PerformanceMetrics", :PerformanceMetrics),
    ("SelfPruning", :SelfPruning),
    ("SpatialOptimizers", :SpatialOptimizers),
    ("TensorIO", :TensorIO),
    ("PropagationDynamics", :PropagationDynamics),
    ("Specialization", :Specialization),
    ("GradientPropagation", :GradientPropagation),
    ("SemanticSpace", :SemanticSpace),
    ("ModelCloning", :ModelCloning),
    ("ContextualMapping", :ContextualMapping),
    ("InternalDialogue", :InternalDialogue),
    ("ReasoningPathways", :ReasoningPathways),
    ("UncertaintyEstimation", :UncertaintyEstimation),
    ("MultimodalIntegration", :MultimodalIntegration),
    ("HippocampalMemory", :Hippocampal_Memory),
    ("Serialization", :Serialization),
    ("LanguageGeneration", :LanguageGeneration),
    ("CorticalLayers", :CorticalLayers),
    ("AttentionalSystem", :Attentional_System),
    ("PrefrontalSystem", :Prefrontal_System),
    ("ActivityMapper", :ActivityMapper),
    ("BrainSpaceVisualizer", :BrainSpaceVisualizer),
    ("ConnectionVisualizer", :ConnectionVisualizer),
    ("TrainingMonitor", :TrainingMonitor)
]

# Importar todos los módulos
module_import_status = Dict(name => use_module(name, symbol) for (name, symbol) in modules_to_import)
@info "Estado de importación de módulos RNTA:" module_import_status

# Verificar disponibilidad de CUDA para operaciones aceleradas
cuda_available = false
try
    cuda_available = CUDA.functional()
    @info "CUDA disponible: $cuda_available"
catch e
    @warn "Error al verificar estado de CUDA: $(sprint(showerror, e))"
end

# Constantes globales
const VERSION = v"0.1.0"
const DEFAULT_DIMENSIONS = (32, 32, 32)

# Decidir qué exportar basado en los módulos cargados correctamente
to_export = Symbol[]

# Funciones core para exportar siempre
core_exports = [:BrainSpace, :TensorNeuron, :SpatialField, :TensorConnection]
append!(to_export, core_exports)

# Funciones principales
if module_import_status["BrainSpace"] && module_import_status["TensorNeuron"]
    append!(to_export, [
        :process_input, :train!, :forward_propagation, :reason,
        :configure_brain_space, :optimize_for_hardware, :detect_hardware,
        :load_brain, :save_brain, :brain_summary, :visualize_activity
    ])
end

# Exportar los símbolos disponibles
for sym in to_export
    try
        @eval export $sym
    catch e
        @warn "No se puede exportar $sym: $(sprint(showerror, e))"
    end
end

"""
    initialize(;config_file=nothing, hardware_profile=:auto, use_cuda=false)

Inicializa el sistema RNTA con la configuración especificada.

## Argumentos
- `config_file`: Ruta a un archivo de configuración (opcional)
- `hardware_profile`: Perfil de hardware a utilizar (:auto, :cpu, :gpu, :multi_gpu, etc.)
- `use_cuda`: Si se debe utilizar aceleración CUDA cuando esté disponible

## Retorna
- `Dict`: Configuración inicializada del sistema
"""
function initialize(;config_file=nothing, hardware_profile=:auto, use_cuda=false)
    # Inicializar sistema
    @info "Inicializando RNTA.jl - Red Neuronal Tensorial Adaptativa"
    
    # Configuración del sistema
    config = Dict{Symbol, Any}()
    
    # Cargar configuración desde archivo si se proporciona
    if config_file !== nothing
        try
            if isfile(config_file)
                if module_import_status["ConfigurationSystem"]
                    config = ConfigurationSystem.load_configuration(config_file)
                    @info "Configuración cargada desde: $config_file"
                else
                    @warn "No se puede cargar configuración: Módulo ConfigurationSystem no disponible"
                    config = default_configuration()
                end
            else
                @warn "Archivo de configuración no encontrado: $config_file"
                config = default_configuration()
                @info "Utilizando configuración por defecto"
            end
        catch e
            @error "Error al cargar configuración desde archivo" exception=(e, catch_backtrace())
            config = default_configuration()
            @info "Utilizando configuración por defecto debido a error"
        end
    else
        config = default_configuration()
        @info "Utilizando configuración por defecto"
    end
    
    # Detectar y configurar hardware
    hw_profile = Dict{Symbol, Any}()
    
    try
        # Verificar que el módulo CUDATensors está disponible
        if !module_import_status["CUDATensors"]
            error("Módulo CUDATensors no disponible para detección de hardware")
        end
        
        # Verificar que la función nprocs está disponible en Distributed
        if !isdefined(Distributed, :nprocs)
            @warn "Función nprocs no disponible en Distributed, usando fallback"
            # Definir una implementación fallback de nprocs
            Distributed.nprocs = () -> 1
        end
        
        if hardware_profile == :auto
            hw_profile = CUDATensors.detect_hardware()
            @info "Hardware detectado: $(get(hw_profile, :description, "Desconocido"))"
        else
            hw_profile = CUDATensors.get_hardware_profile(hardware_profile)
            @info "Utilizando perfil de hardware: $hardware_profile"
        end
    catch e
        @error "Error al detectar hardware" exception=(e, catch_backtrace())
        @info "Usando configuración básica de CPU"
        
        # Configuración básica como fallback
        hw_profile = Dict{Symbol, Any}(
            :hardware_type => :CPU,
            :cpu_threads => Sys.CPU_THREADS,
            :memory_gb => Sys.total_memory() / (1024^3),
            :has_cuda_gpu => false,
            :supports_distributed => false,
            :description => "CPU básica (fallback)"
        )
    end
    
    # Configurar aceleración CUDA si está disponible y solicitada
    cuda_initialized = false
    if use_cuda && get(hw_profile, :has_cuda_gpu, false)
        try
            # Verificar que el módulo CUDATensors está disponible
            if !module_import_status["CUDATensors"]
                error("Módulo CUDATensors no disponible para inicialización CUDA")
            end
            
            # Intentar inicializar CUDA
            @info "Intentando inicializar CUDA..."
            cuda_initialized = CUDATensors.init_cuda_tensors()
            
            if cuda_initialized
                # Activar uso de CUDA para tensores
                CUDATensors.use_cuda_tensors(true)
                
                # Optimizar para GPU
                config[:compute_precision] = :float32
                config[:memory_optimization] = :gpu
                config[:batch_size] = CUDATensors.determine_optimal_batch_size(nothing, hw_profile)
                @info "Aceleración CUDA activada con éxito"
            else
                @warn "No se pudo inicializar CUDA. Usando computación en CPU."
            end
        catch e
            @error "Error al configurar CUDA" exception=(e, catch_backtrace())
            @info "Usando computación en CPU como alternativa"
        end
    else
        # Asegurar que CUDA está desactivado
        try
            if module_import_status["CUDATensors"]
                CUDATensors.use_cuda_tensors(false)
                @info "CUDA desactivado explícitamente"
            end
        catch e
            @warn "Error al desactivar CUDA" exception=e
        end
        @info "Utilizando computación en CPU"
    end
    
    # Configurar el pool global de memoria si está disponible
    try
        if module_import_status["MemoryOptimization"]
            memory_mb = Int(floor(min(get(hw_profile, :memory_gb, 4.0) * 1024 * 0.7, 16 * 1024)))  # Convertir a Int  # Usar 70% de la memoria disponible o máximo 16GB
            MemoryOptimization.get_global_memory_pool(memory_mb)
            @info "Pool de memoria configurado: $(round(memory_mb/1024, digits=2)) GB"
        else
            @warn "Módulo MemoryOptimization no disponible, no se configurará el pool de memoria"
        end
    catch e
        @error "No se pudo configurar el pool de memoria" exception=(e, catch_backtrace())
    end
    
    # Inicializar sistema de métricas
    try
        if module_import_status["PerformanceMetrics"]
            # Verificar si la función específica existe
            if isdefined(PerformanceMetrics, :init_metrics_system)
                PerformanceMetrics.init_metrics_system()
                @info "Sistema de métricas inicializado"
            else
                @warn "Función init_metrics_system no definida en módulo PerformanceMetrics"
            end
        else
            @warn "Módulo PerformanceMetrics no disponible, no se inicializará el sistema de métricas"
        end
    catch e
        @error "No se pudo inicializar el sistema de métricas" exception=(e, catch_backtrace())
    end
    
    # Devolver la configuración inicializada
    return config
end

"""
    default_configuration()

Crea una configuración por defecto para el sistema RNTA.
"""
function default_configuration()
    @info "Creando configuración por defecto"
    
    config = Dict{Symbol, Any}()
    
    # Dimensiones del espacio cerebral
    config[:brain_dimensions] = (32, 32, 32)
    
    # Configuración de neuronas
    config[:initial_neurons] = 1000
    config[:neuron_density] = 0.05
    config[:activation_function] = :adaptive_tanh
    
    # Configuración de conexiones
    config[:connection_probability] = 0.1
    config[:initial_weight_scale] = 0.1
    config[:max_connection_distance] = 10.0
    
    # Configuración de aprendizaje
    config[:learning_rate] = 0.001
    config[:optimizer] = :adam
    config[:batch_size] = 32
    config[:gradient_clip] = 5.0
    
    # Configuración de adaptación
    config[:plasticity_type] = :hebbian
    config[:specialization_threshold] = 0.6
    config[:pruning_threshold] = 0.2
    
    # Configuración de procesamiento
    config[:attention_focus_factor] = 2.0
    config[:attention_radius] = 5.0
    config[:propagation_type] = :wave
    
    # Configuración de recursos
    config[:compute_precision] = :float32
    config[:memory_optimization] = :balanced
    config[:parallelism_strategy] = :auto
    
    return config
end

"""
    create_brain_space(dimensions=(32, 32, 32); config=Dict())

Crea un nuevo espacio cerebral con las dimensiones y configuración especificadas.

## Argumentos
- `dimensions`: Dimensiones del espacio cerebral 3D
- `config`: Configuración adicional para el espacio cerebral

## Retorna
- `BrainSpace`: Nuevo espacio cerebral inicializado
"""
function create_brain_space(dimensions=(32, 32, 32); config=Dict())
    @info "Creando espacio cerebral con dimensiones $dimensions"
    
    # Combinar con configuración por defecto si es necesario
    if isempty(config)
        config = default_configuration()
        @info "Usando configuración por defecto para el espacio cerebral"
    end
    
    try
        # Verificar disponibilidad de módulos necesarios
        if !module_import_status["BrainSpace"]
            error("Módulo BrainSpace no disponible para crear espacio cerebral")
        end
        
        # Crear espacio cerebral
        @info "Inicializando espacio cerebral..."
        
        # Extraer dimensiones individuales para pasarlas como argumentos posicionales
        dim_x, dim_y, dim_z = dimensions
        
        # Intentar crear el objeto BrainSpaceConfig si está disponible
        brain_config = nothing
        if isdefined(Main.Connections, :BrainSpaceConfig)
            try
                brain_config = Main.Connections.BrainSpaceConfig(
                initial_density = convert(Float32, get(config, :neuron_density, 0.05)),
                init_scale = convert(Float32, get(config, :initial_weight_scale, 0.1)),
                max_connection_radius = convert(Float32, get(config, :max_connection_distance, 10.0)),
                base_connection_probability = convert(Float32, get(config, :connection_probability, 0.1)),
                expansion_factor = convert(Float32, 1.5),
                propagation_layers = 3
                )
            catch cfg_err
                @warn "Error al crear BrainSpaceConfig" exception=cfg_err
            end
        else
            @debug "BrainSpaceConfig no disponible en Main.Connections"
        end
        
        # Crear el espacio cerebral usando el constructor correcto
        brain = if brain_config !== nothing
            BrainSpace.Brain_Space(dim_x, dim_y, dim_z, config=brain_config)
        else
            BrainSpace.Brain_Space(dim_x, dim_y, dim_z)
        end
        
        # Configurar el espacio cerebral
        if module_import_status["ConfigurationSystem"]
            @info "Configurando espacio cerebral..."
            ConfigurationSystem.configure_brain_space(brain, config)
        else
            @warn "Módulo ConfigurationSystem no disponible, usando configuración básica"
        end
        
        # Aplicar optimizaciones de hardware si están disponibles
        if get(config, :optimize_hardware, true) && module_import_status["HardwareAdaptation"]
            @info "Optimizando para hardware..."
            
            hw_profile = Dict{Symbol, Any}()
            if module_import_status["CUDATensors"]
                hw_profile = CUDATensors.detect_hardware()
            else
                hw_profile = Dict{Symbol, Any}(
                    :hardware_type => :CPU,
                    :cpu_threads => Sys.CPU_THREADS,
                    :memory_gb => Sys.total_memory() / (1024^3),
                    :has_cuda_gpu => false
                )
            end
            
            HardwareAdaptation.optimize_for_hardware(brain, hw_profile)
        else
            @info "Omitiendo optimización de hardware"
        end
        
        # Inicializar neuronas iniciales
        @info "Poblando neuronas iniciales..."
        neurons_config = Dict(
            :initial_count => get(config, :initial_neurons, 1000),
            :connection_probability => get(config, :connection_probability, 0.1),
            :weight_scale => get(config, :initial_weight_scale, 0.1)
        )
        
        try
            BrainSpace.populate_initial_neurons!(
                brain.neurons, 
                dimensions, 
                neurons_config
            )
            @info "Neuronas inicializadas: $(length(brain.neurons))"
        catch e
            @error "Error al poblar neuronas iniciales" exception=(e, catch_backtrace())
        end
        
        # Establecer conexiones iniciales
        @info "Estableciendo conexiones iniciales..."
        try
            BrainSpace.establish_connections!(brain)
            @info "Conexiones establecidas: $(length(brain.connections))"
        catch e
            @error "Error al establecer conexiones iniciales" exception=(e, catch_backtrace())
        end
        
        @info "Espacio cerebral creado con éxito"
        return brain
    catch e
        @error "Error crítico al crear espacio cerebral" exception=(e, catch_backtrace())
        
        # Si todo lo anterior falla, crear un espacio cerebral simulado como fallback
        @warn "Intentando crear espacio cerebral simulado como fallback"
        try
            # Crear una estructura simulada para pruebas
            brain = Dict{Symbol, Any}(
                :dimensions => dimensions,
                :neurons => Dict{Int, Dict{Symbol, Any}}(),
                :connections => Vector{Dict{Symbol, Any}}(),
                :global_state => zeros(Float32, dimensions),
                :attention_map => ones(Float32, dimensions),
                :creation_time => Dates.now()
            )
            
            # Poblar con algunas neuronas aleatorias para simulación
            for i in 1:100
                pos = (rand(1:dimensions[1]), rand(1:dimensions[2]), rand(1:dimensions[3]))
                brain[:neurons][i] = Dict{Symbol, Any}(
                    :position => pos,
                    :id => isdefined(Main, :UUIDs) ? UUIDs.uuid4() : i,
                    :activation => 0.0,
                    :specialization_level => rand(),
                    :functional_type => rand([:excitatory, :inhibitory, :modulatory])
                )
            end
            
            # Establecer algunas conexiones aleatorias
            for i in 1:200
                source = rand(1:100)
                target = rand(1:100)
                if source != target
                    connection = Dict{Symbol, Any}(
                        :source_id => brain[:neurons][source][:id],
                        :target_id => brain[:neurons][target][:id],
                        :weight => 0.1 * randn(),
                        :strength => abs(0.1 * randn()),
                        :connection_type => rand() < 0.8 ? :excitatory : :inhibitory
                    )
                    push!(brain[:connections], connection)
                end
            end
            
            @info "Espacio cerebral simulado creado como fallback: $(length(brain[:neurons])) neuronas, $(length(brain[:connections])) conexiones"
            return brain
        catch fallback_err
            error("No se pudo crear el espacio cerebral: $(e). Fallback también falló: $(fallback_err)")
        end
    end
end

"""
    train!(brain::BrainSpace, input_data, target_data; 
           epochs=10, 
           learning_rate=0.001, 
           batch_size=32,
           loss_function=default_loss(),
           optimizer=default_optimizer(brain),
           verbose=true)

Entrena el espacio cerebral con los datos proporcionados.

## Argumentos
- `brain`: Espacio cerebral a entrenar
- `input_data`: Datos de entrada (vector de tensores 3D)
- `target_data`: Datos objetivo (vector de tensores 3D)
- `epochs`: Número de épocas de entrenamiento
- `learning_rate`: Tasa de aprendizaje
- `batch_size`: Tamaño del lote para entrenamiento
- `loss_function`: Función de pérdida a utilizar
- `optimizer`: Optimizador a utilizar
- `verbose`: Si se debe mostrar progreso durante el entrenamiento

## Retorna
- `Dict`: Métricas de entrenamiento
"""
function train!(brain, input_data, target_data; 
                epochs=10, 
                learning_rate=0.001, 
                batch_size=32,
                loss_function=nothing,
                optimizer=nothing,
                verbose=true)
    
    @info "Iniciando entrenamiento..." epochs=epochs batch_size=batch_size learning_rate=learning_rate
    
    # Verificar que los módulos necesarios estén disponibles
    required_modules = ["MultidimensionalLoss", "SpatialOptimizers", "GradientPropagation", "PlasticityRules"]
    for mod in required_modules
        if !module_import_status[mod]
            error("Módulo $mod requerido para entrenamiento no está disponible")
        end
    end
    
    # Validar datos
    try
        n_samples = length(input_data)
        if n_samples != length(target_data)
            error("El número de muestras de entrada ($(length(input_data))) y objetivo ($(length(target_data))) debe ser igual")
        end
        
        if n_samples == 0
            error("No hay datos de entrenamiento")
        end
        
        @info "Conjunto de datos validado" n_samples=n_samples
    catch e
        @error "Error al validar datos de entrenamiento" exception=(e, catch_backtrace())
        rethrow(e)
    end
    
    # Inicializar función de pérdida y optimizador si no se proporcionan
    try
        if loss_function === nothing
            @info "Inicializando función de pérdida por defecto"
            # Crear una función de pérdida simple como fallback
            loss_function = (predicted, target) -> begin
                diff = predicted .- target
                return sum(diff .* diff) / length(diff)  # MSE básico
            end
        end
        
        if optimizer === nothing
            @info "Inicializando optimizador por defecto"
            # Crear un optimizador simple como fallback
            optimizer = Dict{Symbol, Any}(
                :type => :sgd,
                :beta1 => 0.9f0,
                :beta2 => 0.999f0,
                :lambda => 1.0f0
            )
        end
    catch e
        @error "Error al inicializar función de pérdida u optimizador" exception=(e, catch_backtrace())
        rethrow(e)
    end
    
    # Inicializar métricas
    metrics = Dict{Symbol, Vector{Float64}}(
        :loss => Float64[],
        :epoch_time => Float64[]
    )
    
    # Inicializar tracker de métricas si se solicita visualización
    # Inicializar tracker de métricas si se solicita visualización
    tracker = nothing
    if verbose && module_import_status["PerformanceMetrics"]
        try
            @info "Inicializando tracker de métricas"
            tracker = PerformanceMetrics.create_metrics_tracker("training")
        catch e
            @warn "No se pudo inicializar tracker de métricas" exception=e
            tracker = nothing
        end
    end
    
    # Configuración de gradientes
    try
        @info "Configurando sistema de gradientes"
        gradient_config = GradientPropagation.GradientConfig(
            backprop_factor=1.0f0,
            lateral_factor=0.3f0,
            update_method=optimizer.type,
            momentum=optimizer.beta1,
            gradient_clip=optimizer.lambda
        )
    catch e
        @error "Error al configurar sistema de gradientes" exception=(e, catch_backtrace())
        rethrow(e)
    end
    
    # Bucle principal de entrenamiento
    @info "Iniciando bucle de entrenamiento para $epochs épocas"
    for epoch in 1:epochs
        epoch_start_time = time()
        epoch_losses = Float64[]
        
        # Mezclar datos para cada época
        try
            indices = shuffle(1:n_samples)
            @debug "Datos mezclados para época $epoch"
        catch e
            @warn "Error al mezclar datos, usando orden secuencial" exception=e
            indices = 1:n_samples
        end
        
        # Procesar por lotes
        n_batches = 0
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_indices = indices[batch_start:batch_end]
            n_batches += 1
            
            # Preparar lotes
            try
                @debug "Preparando lote $n_batches (muestras $batch_start:$batch_end)"
                input_batch = [input_data[i] for i in batch_indices]
                target_batch = [target_data[i] for i in batch_indices]
            catch e
                @error "Error al preparar lote" batch_num=n_batches exception=(e, catch_backtrace())
                continue
            end
            
            # Procesar lote
            try
                batch_loss = GradientPropagation.process_batch!(
                    brain,
                    input_batch,
                    target_batch,
                    loss_function,
                    learning_rate,
                    gradient_config
                )
                
                push!(epoch_losses, batch_loss)
                @debug "Lote $n_batches procesado con pérdida $batch_loss"
            catch e
                @error "Error al procesar lote" batch_num=n_batches exception=(e, catch_backtrace())
            end
        end
        
        # Verificar si se procesó algún lote correctamente
        if isempty(epoch_losses)
            @warn "No se procesó ningún lote correctamente en época $epoch"
            continue
        end
        
        # Aplicar neuroplasticidad después de cada época
        try
            if module_import_status["PlasticityRules"]
                @debug "Aplicando reglas de plasticidad"
                PlasticityRules.apply_brain_plasticity!(brain, learning_rate * 0.1)
            end
        catch e
            @warn "Error al aplicar plasticidad" exception=e
        end
        
        # Realizar auto-poda si es necesario
        if epoch % 5 == 0 && module_import_status["SelfPruning"]
            try
                @debug "Aplicando auto-poda"
                SelfPruning.self_prune!(brain)
            catch e
                @warn "Error al aplicar auto-poda" exception=e
            end
        end
        
        # Realizar especialización si es necesario
        if epoch % 10 == 0 && module_import_status["Specialization"]
            try
                @debug "Aplicando especialización de neuronas"
                Specialization.specialize_neurons!(brain)
            catch e
                @warn "Error al aplicar especialización" exception=e
            end
        end
        
        # Expandir el espacio si es necesario
        if epoch % 20 == 0 && module_import_status["DynamicExpansion"]
            try
                @debug "Verificando si se debe expandir el espacio"
                if DynamicExpansion.should_expand_space(brain)
                    @info "Expandiendo espacio cerebral"
                    DynamicExpansion.expand_space!(brain)
                end
            catch e
                @warn "Error al expandir espacio" exception=e
            end
        end
        
        # Actualizar métricas
        epoch_time = time() - epoch_start_time
        mean_loss = mean(epoch_losses)
        push!(metrics[:loss], mean_loss)
        push!(metrics[:epoch_time], epoch_time)
        
        # Registrar métricas si está habilitado
        if verbose && tracker !== nothing && module_import_status["PerformanceMetrics"]
            try
                PerformanceMetrics.track_loss(tracker, mean_loss)
                PerformanceMetrics.track_epoch_time(tracker, epoch_time)
            catch e
                @warn "Error al registrar métricas" exception=e
            end
        end
        
        # Mostrar progreso si se solicita
        if verbose
            @info "Progreso de entrenamiento" 
                  epoca="$epoch/$epochs" 
                  perdida=round(mean_loss, digits=6) 
                  tiempo_epoca="$(round(epoch_time, digits=2))s"
        end
    end
    
    # Generar visualizaciones si se solicita
    if verbose && tracker !== nothing && 
       module_import_status["PerformanceMetrics"] && 
       module_import_status["TrainingMonitor"]
        try
            @info "Generando reporte de entrenamiento"
            training_report = PerformanceMetrics.create_performance_report(tracker)
            TrainingMonitor.visualize_training_progress(training_report, "training_progress.html")
            @info "Reporte de entrenamiento guardado en 'training_progress.html'"
        catch e
            @warn "Error al generar visualizaciones de entrenamiento" exception=e
        end
    end
    
    @info "Entrenamiento completado" 
          epocas=epochs 
          perdida_final=isempty(metrics[:loss]) ? "N/A" : round(metrics[:loss][end], digits=6)
    
    return metrics
end

"""
    process_input(brain, input_tensor; use_attention=true)

Procesa un tensor de entrada a través del espacio cerebral.

## Argumentos
- `brain`: Espacio cerebral a utilizar
- `input_tensor`: Tensor de entrada 3D
- `use_attention`: Si se debe utilizar mecanismo de atención durante el procesamiento

## Retorna
- Tensor de salida procesado
"""
function process_input(brain, input_tensor; use_attention=true)
    @info "Procesando tensor de entrada" dim=size(input_tensor) use_attention=use_attention
    
    try
        # Verificar módulos necesarios
        if !module_import_status["BrainSpace"]
            error("Módulo BrainSpace no disponible para procesar entrada")
        end
        
        # Preparar el tensor de entrada para las dimensiones correctas
        @debug "Preparando tensor de entrada"
        prepared_input = BrainSpace.prepare_input_tensor(input_tensor, brain.dimensions)
        @debug "Tensor preparado" dim_original=size(input_tensor) dim_preparado=size(prepared_input)
        
        # Crear/actualizar mapa de atención si se solicita
        if use_attention && module_import_status["SpatialAttention"]
            @debug "Creando mapa de atención"
            # Crear mapa de atención basado en saliencia del tensor de entrada
            attention_map = SpatialAttention.create_attention_from_activity(prepared_input)
            BrainSpace.update_attention_map!(brain, attention_map)
            @debug "Mapa de atención actualizado"
        end
        
        # Procesar a través del espacio cerebral
        @debug "Ejecutando propagación hacia adelante"
        result = BrainSpace.forward_propagation(brain, prepared_input)
        @debug "Propagación completada" dim_resultado=size(result)
        
        @info "Procesamiento completado con éxito"
        return result
    catch e
        @error "Error al procesar tensor de entrada" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    reason(brain, input; 
           reasoning_type=:analysis, 
           use_dialogue=false, 
           uncertainty_estimation=false)

Realiza razonamiento sobre un tensor de entrada utilizando el sistema de inferencia.

## Argumentos
- `brain`: Espacio cerebral a utilizar
- `input`: Tensor de entrada 3D o texto
- `reasoning_type`: Tipo de razonamiento a realizar (:analysis, :creative, :critical, etc.)
- `use_dialogue`: Si se debe utilizar diálogo interno para refinar el razonamiento
- `uncertainty_estimation`: Si se debe estimar la incertidumbre del resultado

## Retorna
- `Dict`: Resultado del razonamiento, incluyendo tensor de salida, trayectoria y metadatos
"""
function reason(brain, input; 
                reasoning_type=:analysis, 
                use_dialogue=false, 
                uncertainty_estimation=false)
    
    @info "Iniciando razonamiento" tipo=reasoning_type dialogo=use_dialogue incertidumbre=uncertainty_estimation
    
    # Verificar módulos necesarios
    required_modules = ["ReasoningPathways"]
    if use_dialogue
        push!(required_modules, "InternalDialogue")
    end
    if uncertainty_estimation
        push!(required_modules, "UncertaintyEstimation")
    end
    
    for mod in required_modules
        if !module_import_status[mod]
            error("Módulo $mod requerido para razonamiento no está disponible")
        end
    end
    
    # Convertir texto a tensor si es necesario
    input_tensor = nothing
    try
        if isa(input, String)
            @info "Entrada es texto, convirtiendo a tensor"
            
            if !module_import_status["TensorialTokenizer"]
                error("Módulo TensorialTokenizer no disponible para procesar texto")
            end
            
            # Crear tokenizador si no existe
            @debug "Creando tokenizador"
            tokenizer = TensorialTokenizer.create_default_tokenizer()
            
            # Procesar texto a tensor
            @debug "Procesando texto a tensor"
            input_tensor = TensorialTokenizer.process_text(tokenizer, input)
            @debug "Texto convertido a tensor" dim=size(input_tensor)
        else
            input_tensor = input
            @debug "Usando tensor proporcionado directamente" dim=size(input_tensor)
        end
        
        # Preparar el tensor de entrada
        @debug "Preparando tensor para dimensiones del cerebro"
        prepared_input = BrainSpace.prepare_input_tensor(input_tensor, brain.dimensions)
        @debug "Tensor preparado" dim=size(prepared_input)
    catch e
        @error "Error al preparar entrada para razonamiento" exception=(e, catch_backtrace())
        rethrow(e)
    end
    
    # Crear motor de razonamiento
    try
        @debug "Creando motor de razonamiento"
        engine = ReasoningPathways.ReasoningEngine(brain)
        
        # Crear trayectoria de razonamiento basada en el tipo especificado
        @info "Creando trayectoria de razonamiento tipo $reasoning_type"
        pathway = ReasoningPathways.create_pathway(engine, reasoning_type, prepared_input)
        
        # Ejecutar trayectoria de razonamiento
        @info "Ejecutando trayectoria de razonamiento"
        ReasoningPathways.run_pathway!(engine, pathway)
        
        # Obtener resultado inicial
        @debug "Obteniendo resultado de trayectoria"
        result_tensor = ReasoningPathways.get_pathway_result(pathway)
        pathway_confidence = ReasoningPathways.calculate_pathway_confidence(pathway)
        @debug "Confianza de la trayectoria: $pathway_confidence"
    catch e
        @error "Error en proceso de razonamiento" exception=(e, catch_backtrace())
        rethrow(e)
    end
    
    # Refinar mediante diálogo interno si se solicita
    dialogue_analysis = nothing
    if use_dialogue
        try
            @info "Iniciando refinamiento por diálogo interno"
            dialogue_system = InternalDialogue.DialogueSystem(brain, num_agents=3)
            dialogue_context = InternalDialogue.start_dialogue!(dialogue_system, result_tensor)
            
            # Ejecutar diálogo hasta convergencia o número máximo de pasos
            @debug "Ejecutando diálogo interno"
            InternalDialogue.run_dialogue!(dialogue_system, max_steps=5)
            
            # Obtener resultado refinado
            @debug "Obteniendo resultado refinado del diálogo"
            result_tensor = InternalDialogue.get_dialogue_result(dialogue_system)
            
            # Analizar diálogo para metadatos
            @debug "Analizando diálogo para metadatos"
            dialogue_analysis = InternalDialogue.analyze_dialogue(dialogue_context)
            @info "Diálogo interno completado"
        catch e
            @error "Error en diálogo interno" exception=(e, catch_backtrace())
            @warn "Continuando con resultado sin refinar por diálogo"
        end
    end
    
    # Estimar incertidumbre si se solicita
    uncertainty = nothing
    if uncertainty_estimation
        try
            @info "Estimando incertidumbre del resultado"
            uncertainty_estimator = UncertaintyEstimation.UncertaintyEstimator(brain)
            uncertainty = UncertaintyEstimation.estimate_uncertainty(uncertainty_estimator, result_tensor)
            @debug "Incertidumbre estimada" valor=uncertainty
        catch e
            @error "Error al estimar incertidumbre" exception=(e, catch_backtrace())
            @warn "Continuando sin estimación de incertidumbre"
        end
    end
    
    # Construir resultado
    @info "Construyendo resultado final"
    result = Dict{Symbol, Any}(
        :output_tensor => result_tensor,
        :reasoning_type => reasoning_type,
        :confidence => pathway_confidence,
        :reasoning_pathway => pathway
    )
    
    # Añadir análisis de diálogo si está disponible
    if dialogue_analysis !== nothing
        result[:dialogue_analysis] = dialogue_analysis
    end
    
    # Añadir incertidumbre si está disponible
    if uncertainty !== nothing
        result[:uncertainty] = uncertainty
    end
    
    @info "Razonamiento completado con éxito"
    return result
end

"""
    generate_text(brain, input; max_length=100, temperature=1.0f0)

Genera texto a partir de un tensor o texto de entrada utilizando el sistema NLP.

## Argumentos
- `brain`: Espacio cerebral a utilizar
- `input`: Tensor de entrada o texto inicial
- `max_length`: Longitud máxima del texto generado
- `temperature`: Temperatura para controlar la creatividad (1.0 = normal, <1 = más determinista, >1 = más aleatorio)

## Retorna
- `String`: Texto generado
"""
function generate_text(brain, input; max_length=100, temperature=1.0f0)
    @info "Iniciando generación de texto" max_length=max_length temperature=temperature
    
    # Verificar módulos necesarios
    required_modules = ["TensorialTokenizer", "SemanticSpace", "ContextualMapping", "LanguageGeneration"]
    for mod in required_modules
        if !module_import_status[mod]
            error("Módulo $mod requerido para generación de texto no está disponible")
        end
    end
    
    # Convertir texto a tensor si es necesario
    input_tensor = nothing
    try
        if isa(input, String)
            @info "Entrada es texto, convirtiendo a tensor"
            
            # Crear tokenizador si no existe
            @debug "Creando tokenizador"
            tokenizer = TensorialTokenizer.create_default_tokenizer()
            
            # Procesar texto a tensor
            @debug "Procesando texto a tensor"
            input_tensor = TensorialTokenizer.process_text(tokenizer, input)
            @debug "Texto convertido a tensor" dim=size(input_tensor)
        else
            input_tensor = input
            @debug "Usando tensor proporcionado directamente" dim=size(input_tensor)
        end
    catch e
        @error "Error al preparar entrada para generación de texto" exception=(e, catch_backtrace())
        rethrow(e)
    end
    
    try
        # Crear espacio semántico y mapeador contextual
        @debug "Creando espacio semántico"
        semantic_space = SemanticSpace.Semantic3DSpace(brain.dimensions, brain=brain, tokenizer=tokenizer)
        
        @debug "Creando mapeador contextual"
        context_mapper = ContextualMapping.ContextMapper(brain.dimensions, semantic_space=semantic_space, brain=brain)
        
        # Actualizar contexto con el tensor de entrada
        @debug "Actualizando contexto con tensor de entrada"
        ContextualMapping.process_tensor(context_mapper, input_tensor)
        
        # Configurar decodificador
        @debug "Configurando decodificador de lenguaje"
        decoder_config = LanguageGeneration.DecoderConfig(
            temperature=temperature,
            threshold=0.05f0,
            repetition_penalty=1.2f0,
            context_window=50,
            max_tokens=max_length,
            decoding_strategy=:sampling
        )
        
        # Crear decodificador de lenguaje
        @debug "Creando decodificador de lenguaje"
        decoder = LanguageGeneration.LanguageDecoder(
            tokenizer,
            brain.dimensions,
            semantic_space=semantic_space,
            context_mapper=context_mapper,
            config=decoder_config
        )
        
        # Procesar tensor de entrada a través del cerebro
        @info "Procesando tensor a través del cerebro"
        processed_tensor = BrainSpace.forward_propagation(brain, input_tensor)
        
        # Generar texto a partir del tensor procesado
        @info "Generando texto"
        generated_text = LanguageGeneration.generate_text(decoder, processed_tensor, max_length=max_length)
        
        @info "Texto generado con éxito" length=length(generated_text)
        return generated_text
    catch e
        @error "Error durante la generación de texto" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    visualize_brain(brain; options...)

Genera una visualización del espacio cerebral actual.

## Argumentos
- `brain`: Espacio cerebral a visualizar
- `options...`: Opciones adicionales para la visualización

## Retorna
- Objeto de visualización (el tipo depende del backend de visualización)
"""
function visualize_brain(brain; 
                         show_neurons=true, 
                         show_connections=true, 
                         show_activity=true,
                         highlight_active=true,
                         connection_threshold=0.2f0,
                         activity_threshold=0.3f0,
                         file_path=nothing)
    
    @info "Iniciando visualización del espacio cerebral" show_neurons=show_neurons show_connections=show_connections show_activity=show_activity
    
    # Verificar módulos necesarios
    if !module_import_status["BrainSpaceVisualizer"]
        error("Módulo BrainSpaceVisualizer no disponible para visualizar espacio cerebral")
    end
    
    try
        # Crear visualizador
        @debug "Creando visualizador"
        vis = BrainSpaceVisualizer.create_brain_visualizer(brain.dimensions)
        
        # Configurar visualización
        @debug "Configurando parámetros de visualización"
        config = Dict{Symbol, Any}(
            :show_neurons => show_neurons,
            :show_connections => show_connections,
            :show_activity => show_activity,
            :highlight_active => highlight_active,
            :connection_threshold => connection_threshold,
            :activity_threshold => activity_threshold
        )
        
        # Visualizar neuronas si se solicita
        if show_neurons
            @debug "Visualizando neuronas"
            BrainSpaceVisualizer.add_neurons_to_visualization!(vis, brain, config)
        end
        
        # Visualizar conexiones si se solicita
        if show_connections
            @debug "Visualizando conexiones"
            BrainSpaceVisualizer.add_connections_to_visualization!(vis, brain, config)
        end
        
        # Visualizar actividad si se solicita
        if show_activity
            @debug "Visualizando actividad"
            BrainSpaceVisualizer.add_activity_to_visualization!(vis, brain, config)
        end
        
        # Guardar visualización si se proporciona ruta
        if file_path !== nothing
            @info "Guardando visualización en: $file_path"
            BrainSpaceVisualizer.save_visualization(vis, file_path)
        end
        
        @info "Visualización completada con éxito"
        return vis
    catch e
        @error "Error al visualizar espacio cerebral" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    visualize_tensor(tensor::Array{T,3}; options...)

Genera una visualización de un tensor 3D utilizando las herramientas de visualización disponibles.

## Argumentos
- `tensor`: Tensor 3D a visualizar
- `options...`: Opciones adicionales para la visualización

## Retorna
- Objeto de visualización
"""
function visualize_tensor(tensor::Array{T,3}; 
                          colormap=:viridis, 
                          threshold=0.1f0, 
                          slice_mode=:orthogonal,
                          interactive=true,
                          file_path=nothing) where T
    
    @info "Iniciando visualización de tensor" dim=size(tensor) mode=slice_mode
    
    # Verificar dependencias necesarias
    if !makie_available
        error("Makie no está disponible para visualización de tensor")
    end
    
    try
        # Importar en este scope para evitar errores si no está disponible
        @eval using Makie, GLMakie
        
        # Crear figura
        @debug "Creando figura"
        fig = Figure(resolution=(1200, 800))
        
        # Panel para vista de proyecciones 2D
        if slice_mode == :orthogonal
            @debug "Usando modo de visualización ortogonal"
            # Crear paneles para tres vistas ortogonales
            ax_xy = Axis(fig[1, 1], aspect=:equal, title="Proyección XY")
            ax_xz = Axis(fig[1, 2], aspect=:equal, title="Proyección XZ")
            ax_yz = Axis(fig[2, 1], aspect=:equal, title="Proyección YZ")
            
            # Calcular proyecciones máximas
            @debug "Calculando proyecciones máximas"
            projection_xy = dropdims(maximum(tensor, dims=3), dims=3)
            projection_xz = dropdims(maximum(tensor, dims=2), dims=2)
            projection_yz = dropdims(maximum(tensor, dims=1), dims=1)
            
            # Visualizar proyecciones como mapas de calor
            @debug "Creando mapas de calor para proyecciones"
            hm_xy = heatmap!(ax_xy, projection_xy, colormap=colormap)
            hm_xz = heatmap!(ax_xz, projection_xz, colormap=colormap)
            hm_yz = heatmap!(ax_yz, projection_yz, colormap=colormap)
            
            # Añadir barra de color
            Colorbar(fig[2, 2], hm_xy, label="Valor")
        else
            @debug "Usando modo de visualización por cortes"
            # Visualizar cortes centrales
            dims = size(tensor)
            center_x = div(dims[1], 2)
            center_y = div(dims[2], 2)
            center_z = div(dims[3], 2)
            
            ax_xy = Axis(fig[1, 1], aspect=:equal, title="Corte XY (z=$center_z)")
            ax_xz = Axis(fig[1, 2], aspect=:equal, title="Corte XZ (y=$center_y)")
            ax_yz = Axis(fig[2, 1], aspect=:equal, title="Corte YZ (x=$center_x)")
            
            # Obtener cortes
            @debug "Obteniendo cortes centrales"
            slice_xy = tensor[:, :, center_z]
            slice_xz = tensor[:, center_y, :]
            slice_yz = tensor[center_x, :, :]
            
            # Visualizar cortes
            @debug "Creando mapas de calor para cortes"
            hm_xy = heatmap!(ax_xy, slice_xy, colormap=colormap)
            hm_xz = heatmap!(ax_xz, slice_xz, colormap=colormap)
            hm_yz = heatmap!(ax_yz, slice_yz, colormap=colormap)
            
            # Añadir barra de color
            Colorbar(fig[2, 2], hm_xy, label="Valor")
        end
        
        # Panel de estadísticas
        stats_panel = fig[3, 1:2]
        
        # Calcular estadísticas básicas
        @debug "Calculando estadísticas del tensor"
        min_val, max_val = extrema(tensor)
        mean_val = mean(tensor)
        std_val = std(tensor)
        
        # Umbralizar para estadísticas de regiones activas
        active_mask = tensor .>= threshold
        num_active = count(active_mask)
        active_ratio = num_active / length(tensor)
        
        # Crear texto con estadísticas
        stats_text = """
        Estadísticas del Tensor:
        • Dimensiones: $(size(tensor))
        • Total elementos: $(length(tensor))
        • Valor mínimo: $(round(min_val, digits=4))
        • Valor máximo: $(round(max_val, digits=4))
        • Media: $(round(mean_val, digits=4))
        • Desviación estándar: $(round(std_val, digits=4))
        • Elementos activos (>=$(threshold)): $num_active ($(round(active_ratio*100, digits=2))%)
        """
        
        Label(stats_panel, stats_text, tellwidth=false, fontsize=14)
        
        # Guardar visualización si se proporciona ruta
        if file_path !== nothing
            @info "Guardando visualización en: $file_path"
            save(file_path, fig)
        end
        
        @info "Visualización de tensor completada con éxito"
        return fig
    catch e
        @error "Error al visualizar tensor" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    save_brain(brain, filename::String)

Guarda un espacio cerebral completo en un archivo.

## Argumentos
- `brain`: Espacio cerebral a guardar
- `filename`: Ruta del archivo donde guardar el cerebro
"""
function save_brain(brain, filename::String)
    @info "Iniciando guardado de espacio cerebral en: $filename"
    
    if !module_import_status["Serialization"]
        error("Módulo Serialization no disponible para guardar espacio cerebral")
    end
    
    try
        Serialization.save_brain(brain, filename)
        @info "Espacio cerebral guardado con éxito en: $filename"
    catch e
        @error "Error al guardar espacio cerebral" archivo=filename exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    load_brain(filename::String)

Carga un espacio cerebral desde un archivo.

## Argumentos
- `filename`: Ruta del archivo del cerebro

## Retorna
- `BrainSpace`: Espacio cerebral cargado
"""
function load_brain(filename::String)
    @info "Iniciando carga de espacio cerebral desde: $filename"
    
    if !module_import_status["Serialization"]
        error("Módulo Serialization no disponible para cargar espacio cerebral")
    end
    
    try
        # Verificar que el archivo existe
        if !isfile(filename)
            error("El archivo especificado no existe: $filename")
        end
        
        brain = Serialization.load_brain(filename)
        @info "Espacio cerebral cargado con éxito desde: $filename"
        return brain
    catch e
        @error "Error al cargar espacio cerebral" archivo=filename exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    brain_summary(brain)

Genera un resumen estadístico del estado actual del espacio cerebral.

## Argumentos
- `brain`: Espacio cerebral a analizar

## Retorna
- `Dict`: Resumen estadístico
"""
function brain_summary(brain)
    @info "Generando resumen del espacio cerebral"
    
    try
        # Verificar el tipo de brain (normal o fallback)
        if isa(brain, Dict)
            # Caso de cerebro fallback
            n_neurons = length(brain[:neurons])
            n_connections = length(brain[:connections])
            
            @debug "Estadísticas básicas (fallback)" neuronas=n_neurons conexiones=n_connections
            
            # Crear resumen para el cerebro fallback
            return Dict{Symbol, Any}(
                :dimensions => brain[:dimensions],
                :neurons => n_neurons,
                :connections => n_connections,
                :creation_time => get(brain, :creation_time, Dates.now()),
                :age_days => 0,
                :is_fallback => true
            )
        else
            # Caso de cerebro normal
            n_neurons = length(brain.neurons)
            n_connections = length(brain.connections)
            
            @debug "Estadísticas básicas" neuronas=n_neurons conexiones=n_connections
            
            # Calcular densidad de conexiones
            max_possible_connections = n_neurons * (n_neurons - 1)
            connection_density = max_possible_connections > 0 ? n_connections / max_possible_connections : 0.0
            
            # Crear resumen
            summary = Dict{Symbol, Any}(
                :dimensions => brain.dimensions,
                :neurons => n_neurons,
                :connections => n_connections,
                :connection_density => connection_density,
                :creation_time => isdefined(brain, :creation_time) ? brain.creation_time : Dates.now(),
                :age_days => 0,
                :is_fallback => false
            )
            
            # Calcular edad si es posible
            if isdefined(brain, :creation_time)
                summary[:age_days] = (Dates.now() - brain.creation_time).value / (1000 * 60 * 60 * 24)
            end
            
            @info "Resumen generado con éxito" 
                  neuronas=n_neurons 
                  conexiones=n_connections 
                  densidad=round(connection_density*100, digits=2)
            
            return summary
        end
    catch e
        @error "Error al generar resumen del espacio cerebral" exception=(e, catch_backtrace())
        
        # Intentar devolver un resumen mínimo en caso de error
        try
            if isa(brain, Dict)
                return Dict{Symbol, Any}(
                    :error => "Error al generar resumen completo: $(e)",
                    :dimensions => brain[:dimensions],
                    :neurons => length(brain[:neurons]),
                    :connections => length(brain[:connections]),
                    :is_fallback => true
                )
            else
                return Dict{Symbol, Any}(
                    :error => "Error al generar resumen completo: $(e)",
                    :dimensions => isdefined(brain, :dimensions) ? brain.dimensions : (0, 0, 0),
                    :neurons => isdefined(brain, :neurons) ? length(brain.neurons) : 0,
                    :connections => isdefined(brain, :connections) ? length(brain.connections) : 0,
                    :is_fallback => false
                )
            end
        catch
            return Dict{Symbol, Any}(:error => "Error crítico al generar resumen: $(e)")
        end
    end
end

# Función de seguridad para inicializar el sistema de métricas
function init_metrics_system()
   @debug "Inicializando sistema de métricas de forma segura"
   
   try
       if module_import_status["PerformanceMetrics"] && 
           isdefined(PerformanceMetrics, :init_metrics_system)
           return PerformanceMetrics.init_metrics_system()
       else
           @warn "Módulo PerformanceMetrics no disponible o función init_metrics_system no definida"
           # Implementación de respaldo para evitar errores
           return Dict{Symbol, Any}()
       end
   catch e
       @warn "Error al inicializar sistema de métricas" exception=e
       # Implementación de respaldo para evitar errores
       return Dict{Symbol, Any}()
   end
end

"""
   verify_cuda_availability()

Verifica la disponibilidad de CUDA en el sistema.

## Retorna
- `Tuple{Bool, String}`: Estado de disponibilidad y mensaje descriptivo
"""
function verify_cuda_availability()
   @debug "Verificando disponibilidad de CUDA"
   
   if !module_import_status["CUDATensors"]
       return false, "Módulo CUDATensors no disponible"
   end
   
   try
       cuda_available = CUDATensors.is_cuda_available()
       if cuda_available
           return true, "CUDA está disponible en el sistema"
       else
           return false, "CUDA no está disponible o no es funcional"
       end
   catch e
       @warn "Error al verificar disponibilidad de CUDA" exception=e
       return false, "Error al verificar CUDA: $(e)"
   end
end

"""
   system_diagnostic()

Realiza un diagnóstico completo del sistema RNTA.

## Retorna
- `Dict`: Informe de diagnóstico del sistema
"""
function system_diagnostic()
   @info "Iniciando diagnóstico del sistema RNTA"
   
   diagnostic = Dict{Symbol, Any}(
       :version => VERSION,
       :timestamp => Dates.now(),
       :julia_version => string(VERSION),
       :modules_status => Dict{String, Bool}(),
       :cuda_status => Dict{Symbol, Any}(),
       :system_info => Dict{Symbol, Any}()
   )
   
   # Verificar estado de módulos
   for (name, symbol) in modules_to_import
       diagnostic[:modules_status][name] = module_import_status[name]
   end
   
   # Verificar CUDA
   available, msg = verify_cuda_availability()
   diagnostic[:cuda_status][:available] = available
   diagnostic[:cuda_status][:message] = msg
   
   if available && module_import_status["CUDATensors"]
       try
           hw_profile = CUDATensors.detect_hardware()
           diagnostic[:cuda_status][:hardware_profile] = hw_profile
       catch e
           @warn "Error al obtener perfil de hardware CUDA" exception=e
           diagnostic[:cuda_status][:error] = "$(e)"
       end
   end
   
   # Información del sistema
   diagnostic[:system_info][:cpu_threads] = Sys.CPU_THREADS
   diagnostic[:system_info][:memory_gb] = Sys.total_memory() / (1024^3)
   diagnostic[:system_info][:os] = string(Sys.KERNEL)
   
   @info "Diagnóstico completado" 
         modulos_ok=count(values(diagnostic[:modules_status])) 
         total_modulos=length(diagnostic[:modules_status])
         cuda=diagnostic[:cuda_status][:available]
   
   return diagnostic
end

# Inicialización segura del módulo
let
   @info "Iniciando inicialización del módulo RNTA"
   
   # Realizar diagnóstico inicial
   diagnostic_result = system_diagnostic()
   @info "Diagnóstico inicial completado" 
         modulos_cargados="$(count(values(diagnostic_result[:modules_status])))/$(length(diagnostic_result[:modules_status]))"
         cuda_disponible=diagnostic_result[:cuda_status][:available]
   
   try
       # Configuración inicial con CUDA desactivado por defecto
       # Para activar CUDA, el usuario debe llamar explícitamente a initialize(use_cuda=true)
       @info "Inicializando configuración global"
       global GLOBAL_CONFIG = initialize(use_cuda=false)
       @info "RNTA inicializado correctamente"
   catch e
       @error "Error durante la inicialización de RNTA" exception=(e, catch_backtrace())
       @warn "Se utilizará configuración mínima de respaldo"
       
       # Configuración de respaldo para permitir que el módulo cargue
       global GLOBAL_CONFIG = default_configuration()
   end
end

# Mensaje de inicialización completada
@info "Módulo RNTA cargado y configurado. Versión $(VERSION)"

end # module RNTA