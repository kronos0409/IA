module CUDATensors

using CUDA
using LinearAlgebra
using Distributed  # Añadir esta línea

export CUDATensorState,
       init_cuda_tensors,
       use_cuda_tensors,
       is_cuda_active,
       is_cuda_available,
       to_cuda,
       to_host,
       cuda_tensor_convolution,
       cuda_adaptive_pooling,
       cuda_volumetric_activation,
       cuda_spatial_attention_transform,
       cuda_zero_pad,
       cuda_adaptive_tanh,
       cuda_tensor_relu,
       detect_hardware,
       get_hardware_profile,
       determine_optimal_batch_size

# Variable global para controlar si CUDA está activo
global USE_CUDA_TENSORS = false
global CUDA_INITIALIZED = false

"""
    CUDATensorState

Estado global para la aceleración CUDA.
"""
struct CUDATensorState
    # Indica si CUDA está disponible y activo
    cuda_active::Bool
    
    # Dispositivo seleccionado
    device_id::Int
    
    # Información del dispositivo
    device_properties::Dict{String,Any}
    
    # Caché para kernels compilados
    kernel_cache::Dict{String,Any}
    
    # Máximo tamaño de bloque preferido
    preferred_block_size::NTuple{3,Int}
end

# Estado global
global CUDA_STATE = nothing

"""
    debug_log(msg)

Función interna para registro de mensajes de depuración.
"""
function debug_log(msg; level=:info)
    # Nivel de depuración (puede configurarse como variable de entorno)
    debug_level = get(ENV, "RNTA_DEBUG", "info")
    
    if level == :error || debug_level ∈ ["debug", "info", "error"]
        if level == :error
            @error msg
        elseif level == :warn && debug_level ∈ ["debug", "info"]
            @warn msg
        elseif level == :info && debug_level ∈ ["debug", "info"]
            @info msg
        elseif debug_level == "debug"
            println("DEBUG: $msg")
        end
    end
end

"""
    is_cuda_available()

Verifica si CUDA está disponible en el sistema.
"""
function is_cuda_available()
    try
        return CUDA.functional()
    catch e
        debug_log("Error al verificar disponibilidad de CUDA: $e", level=:warn)
        return false
    end
end

"""
    init_cuda_tensors()

Inicializa el sistema de aceleración CUDA para RNTA.

# Retorna
- `Bool`: `true` si CUDA se inicializó correctamente, `false` en caso contrario
"""
function init_cuda_tensors()
    debug_log("Intentando inicializar CUDA...")
    
    # Si ya está inicializado, devolvemos el estado actual
    if CUDA_INITIALIZED
        debug_log("CUDA ya estaba inicializado. Estado actual: $(is_cuda_active() ? "activo" : "inactivo")")
        return is_cuda_active()
    end
    
    # Verificar disponibilidad de CUDA
    if !is_cuda_available()
        debug_log("CUDA no está disponible o no es funcional", level=:warn)
        
        # Inicializar estado para CPU
        global CUDA_STATE = CUDATensorState(
            false,
            0,
            Dict{String,Any}(),
            Dict{String,Any}(),
            (8, 8, 8)
        )
        global CUDA_INITIALIZED = true
        global USE_CUDA_TENSORS = false
        
        return false
    end
    
    try
        # Intentar obtener lista de dispositivos CUDA
        num_devices = 0
        try
            num_devices = length(CUDA.devices())
            debug_log("Detectados $(num_devices) dispositivos CUDA")
        catch e
            debug_log("Error al enumerar dispositivos CUDA: $e", level=:warn)
            debug_log("Asumiendo un solo dispositivo CUDA")
            num_devices = 1
        end
        
        if num_devices == 0
            debug_log("No se encontraron dispositivos CUDA", level=:warn)
            throw(ErrorException("No se encontraron dispositivos CUDA aunque CUDA.functional() == true"))
        end
        
        # Seleccionar primer dispositivo (índice 0)
        device_id = 0
        debug_log("Seleccionando dispositivo CUDA #$device_id")
        
        # Cambiar al dispositivo seleccionado
        try
            CUDA.device!(device_id)
            debug_log("Dispositivo CUDA #$device_id seleccionado correctamente")
        catch e
            debug_log("Error al seleccionar dispositivo CUDA #$device_id: $e", level=:error)
            throw(e)
        end
        
        # Obtener propiedades del dispositivo de manera segura
        device_props = Dict{String,Any}()
        
        # Nombre del dispositivo
        try
            device_props["name"] = CUDA.name()
            debug_log("Nombre del dispositivo: $(device_props["name"])")
        catch e
            debug_log("No se pudo obtener el nombre del dispositivo: $e", level=:warn)
            device_props["name"] = "Unknown CUDA Device"
        end
        
        # Memoria total
        try
            device_props["memory"] = CUDA.totalmem()
            debug_log("Memoria total: $(round(device_props["memory"] / 1024^3, digits=2)) GB")
        catch e
            debug_log("No se pudo obtener la memoria total: $e", level=:warn)
            device_props["memory"] = 4 * 1024^3  # Asumir 4GB
        end
        
        # Capacidad de cómputo
        try
            cc = CUDA.capability()
            device_props["compute_capability"] = cc
            debug_log("Capacidad de cómputo: $(cc[1]).$(cc[2])")
        catch e
            debug_log("No se pudo obtener la capacidad de cómputo: $e", level=:warn)
            device_props["compute_capability"] = (7, 0)  # Asumir 7.0
        end
        
        # Otros atributos (usando valores seguros por defecto)
        device_props["num_multiprocessors"] = 1
        device_props["max_threads_per_block"] = 1024
        
        # Determinar tamaño de bloque óptimo para tensores 3D
        max_threads = device_props["max_threads_per_block"]
        dim = Int(floor(∛(max_threads)))
        preferred_block_size = (dim, dim, dim)
        
        debug_log("Tamaño de bloque preferido: $preferred_block_size")
        
        # Inicializar estado global CUDA
        global CUDA_STATE = CUDATensorState(
            true,
            device_id,
            device_props,
            Dict{String,Any}(),
            preferred_block_size
        )
        
        global CUDA_INITIALIZED = true
        global USE_CUDA_TENSORS = true
        
        # Precalentar sistema CUDA
        _warmup_cuda()
        
        debug_log("CUDA inicializado con éxito")
        return true
    catch e
        debug_log("Error al inicializar CUDA: $e", level=:error)
        debug_log(sprint(showerror, e, catch_backtrace()), level=:error)
        
        # Inicializar estado para CPU como fallback
        global CUDA_STATE = CUDATensorState(
            false,
            0,
            Dict{String,Any}(),
            Dict{String,Any}(),
            (8, 8, 8)
        )
        
        global CUDA_INITIALIZED = true
        global USE_CUDA_TENSORS = false
        
        return false
    end
end

"""
    _warmup_cuda()

Precalienta el sistema CUDA para reducir latencia en operaciones iniciales.
"""
function _warmup_cuda()
    debug_log("Precalentando sistema CUDA...")
    
    try
        # Usar un tensor pequeño para evitar problemas de memoria
        dim = 32  # Dimensión menor para evitar problemas de memoria
        
        debug_log("Creando tensores de prueba de $(dim)×$(dim)×$(dim)...")
        a = CUDA.zeros(Float32, dim, dim, dim)
        b = CUDA.ones(Float32, dim, dim, dim)
        
        debug_log("Ejecutando operaciones de prueba...")
        c = a .+ b
        d = a .* b
        
        debug_log("Sincronizando dispositivo...")
        CUDA.synchronize()
        
        debug_log("Precalentamiento completado con éxito")
    catch e
        debug_log("Error durante el precalentamiento de CUDA: $e", level=:warn)
        debug_log(sprint(showerror, e, catch_backtrace()), level=:warn)
    end
end

"""
    use_cuda_tensors(active=true)

Activa o desactiva el uso de CUDA para operaciones tensoriales.

# Argumentos
- `active::Bool=true`: Si se debe activar CUDA

# Retorna
- `Bool`: `true` si CUDA está activo después de la llamada, `false` en caso contrario
"""
function use_cuda_tensors(active::Bool=true)
    debug_log("Configurando uso de CUDA: $(active ? "activar" : "desactivar")")
    
    # Si CUDA no está inicializado, inicializarlo primero
    if !CUDA_INITIALIZED
        init_cuda_tensors()
    end
    
    # Si queremos activar CUDA
    if active
        # Verificar si CUDA está disponible y funcional
        if !is_cuda_available()
            debug_log("No se puede activar CUDA porque no está disponible", level=:warn)
            global USE_CUDA_TENSORS = false
            return false
        end
        
        # Si CUDA_STATE no está activo, activarlo
        if CUDA_STATE !== nothing && !CUDA_STATE.cuda_active
            debug_log("Reactivando estado CUDA existente")
            # Crear nuevo estado con cuda_active = true
            global CUDA_STATE = CUDATensorState(
                true,
                CUDA_STATE.device_id,
                CUDA_STATE.device_properties,
                CUDA_STATE.kernel_cache,
                CUDA_STATE.preferred_block_size
            )
        end
        
        global USE_CUDA_TENSORS = true
        debug_log("CUDA activado con éxito")
    else
        # Desactivar CUDA
        if CUDA_STATE !== nothing && CUDA_STATE.cuda_active
            debug_log("Desactivando CUDA")
            # Crear nuevo estado con cuda_active = false
            global CUDA_STATE = CUDATensorState(
                false,
                CUDA_STATE.device_id,
                CUDA_STATE.device_properties,
                CUDA_STATE.kernel_cache,
                CUDA_STATE.preferred_block_size
            )
        end
        
        global USE_CUDA_TENSORS = false
        debug_log("CUDA desactivado")
    end
    
    return is_cuda_active()
end

"""
    is_cuda_active()

Verifica si la aceleración CUDA está activa.

# Retorna
- `Bool`: `true` si CUDA está activo, `false` en caso contrario
"""
function is_cuda_active()
    return CUDA_STATE !== nothing && CUDA_STATE.cuda_active && USE_CUDA_TENSORS
end

"""
    to_cuda(tensor)

Transfiere un tensor a la GPU si CUDA está activo.

# Argumentos
- `tensor::Array{T,N}`: Tensor a transferir

# Retorna
- Tensor en GPU si CUDA está activo, o el mismo tensor si no
"""
function to_cuda(tensor::Array{T,N}) where {T <: AbstractFloat, N}
    if !is_cuda_active()
        return tensor
    end
    
    try
        # Convertir a Float32 para optimización en GPU
        tensor_f32 = convert(Array{Float32}, tensor)
        return CuArray{Float32}(tensor_f32)
    catch e
        debug_log("Error al transferir tensor a GPU: $e", level=:warn)
        return tensor  # Devolver tensor original en caso de error
    end
end

# Versión para ya en la GPU
function to_cuda(tensor::CuArray{T,N}) where {T <: AbstractFloat, N}
    return tensor  # Ya está en la GPU
end

"""
    to_host(tensor)

Transfiere un tensor de la GPU a la CPU.

# Argumentos
- `tensor::CuArray{T,N}`: Tensor en GPU

# Retorna
- `Array{T,N}`: Tensor en CPU
"""
function to_host(tensor::CuArray{T,N}) where {T <: AbstractFloat, N}
    try
        return Array(tensor)
    catch e
        debug_log("Error al transferir tensor a CPU: $e", level=:error)
        # Intentar crear un tensor vacío del mismo tamaño como fallback
        return zeros(T, size(tensor))
    end
end

# Versión para tensores ya en CPU
function to_host(tensor::Array{T,N}) where {T <: AbstractFloat, N}
    return tensor  # Ya está en CPU
end

"""
    detect_hardware()

Detecta las capacidades de hardware del sistema.

# Retorna
- `Dict{Symbol,Any}`: Perfil de hardware con información del sistema
"""
function detect_hardware()
    debug_log("Detectando hardware del sistema...")
    
    # Información de CPU
    cpu_threads = Sys.CPU_THREADS
    debug_log("CPU threads: $cpu_threads")
    
    cpu_memory_gb = Sys.total_memory() / (1024^3)
    debug_log("Memoria del sistema: $(round(cpu_memory_gb, digits=2)) GB")
    
    # Valores predeterminados para hardware
    hardware_type = :CPU
    has_cuda = false
    
    # Verificar disponibilidad de CUDA
    cuda_available = is_cuda_available()
    debug_log("CUDA disponible: $cuda_available")
    
    if cuda_available
        hardware_type = :CUDA_GPU
        has_cuda = true
        
        # Intentar obtener información adicional
        if CUDA_INITIALIZED && CUDA_STATE !== nothing
            debug_log("Obteniendo información adicional de CUDA...")
            
            try
                num_devices = length(CUDA.devices())
                debug_log("Número de dispositivos CUDA: $num_devices")
                
                # Si tenemos información de dispositivo, mostrarla
                if !isempty(CUDA_STATE.device_properties)
                    props = CUDA_STATE.device_properties
                    if haskey(props, "name")
                        debug_log("Dispositivo CUDA activo: $(props["name"])")
                    end
                    if haskey(props, "memory")
                        memory_gb = props["memory"] / (1024^3)
                        debug_log("Memoria CUDA: $(round(memory_gb, digits=2)) GB")
                    end
                end
            catch e
                debug_log("Error al obtener detalles adicionales de CUDA: $e", level=:warn)
            end
        end
    end
    
    # Comprobar si se está ejecutando en modo distribuido
    distributed = nprocs() > 1
    debug_log("Ejecución distribuida: $distributed")
    
    # Crear perfil de hardware
    hw_profile = Dict{Symbol, Any}(
        :hardware_type => hardware_type,
        :cpu_threads => cpu_threads,
        :memory_gb => cpu_memory_gb,
        :has_cuda_gpu => has_cuda,
        :supports_distributed => distributed,
        :description => "CPU: $cpu_threads hilos, $(round(cpu_memory_gb, digits=1)) GB RAM"
    )
    
    # Añadir información adicional si CUDA está disponible
    if has_cuda
        hw_profile[:description] *= ", CUDA GPU disponible"
        
        # Si CUDA está inicializado, incluir información detallada
        if CUDA_INITIALIZED && CUDA_STATE !== nothing && !isempty(CUDA_STATE.device_properties)
            props = CUDA_STATE.device_properties
            
            if haskey(props, "name")
                hw_profile[:gpu_name] = props["name"]
                hw_profile[:description] *= " ($(props["name"]))"
            end
            
            if haskey(props, "memory")
                memory_gb = props["memory"] / (1024^3)
                hw_profile[:gpu_memory_gb] = memory_gb
                hw_profile[:description] *= ", $(round(memory_gb, digits=1)) GB VRAM"
            end
            
            if haskey(props, "compute_capability")
                cc = props["compute_capability"]
                hw_profile[:compute_capability] = cc
                hw_profile[:description] *= ", CC $(cc[1]).$(cc[2])"
            end
        end
    end
    
    debug_log("Perfil de hardware: $(hw_profile[:description])")
    return hw_profile
end

"""
    get_hardware_profile(profile::Symbol)

Obtiene un perfil de hardware predefinido.

# Argumentos
- `profile::Symbol`: Tipo de perfil (:auto, :cpu, :gpu)

# Retorna
- `Dict{Symbol,Any}`: Perfil de hardware configurado
"""
function get_hardware_profile(profile::Symbol)
    debug_log("Obteniendo perfil de hardware: $profile")
    
    if profile == :cpu
        hw_profile = Dict{Symbol, Any}(
            :hardware_type => :CPU,
            :cpu_threads => Sys.CPU_THREADS,
            :memory_gb => Sys.total_memory() / (1024^3),
            :has_cuda_gpu => false,
            :supports_distributed => nprocs() > 1,
            :description => "Perfil CPU optimizado ($(Sys.CPU_THREADS) hilos)"
        )
        debug_log("Usando perfil CPU forzado")
        return hw_profile
    elseif profile == :gpu
        hw_profile = Dict{Symbol, Any}(
            :hardware_type => :CUDA_GPU,
            :cpu_threads => Sys.CPU_THREADS,
            :memory_gb => Sys.total_memory() / (1024^3),
            :has_cuda_gpu => true,
            :supports_distributed => nprocs() > 1,
            :description => "Perfil GPU CUDA optimizado"
        )
        debug_log("Usando perfil GPU forzado")
        return hw_profile
    else
        # Para :auto u otros perfiles, detectar automáticamente
        debug_log("Detectando hardware automáticamente")
        return detect_hardware()
    end
end

"""
    determine_optimal_batch_size(brain, hw_profile)

Determina el tamaño de batch óptimo según el perfil de hardware.

# Argumentos
- `brain`: Estado del cerebro RNTA (puede ser `nothing`)
- `hw_profile`: Perfil de hardware

# Retorna
- `Int`: Tamaño de batch recomendado
"""
function determine_optimal_batch_size(brain, hw_profile)
    debug_log("Determinando tamaño de batch óptimo...")
    
    # Valores por defecto según tipo de hardware
    if get(hw_profile, :has_cuda_gpu, false)
        # Ajustar según memoria GPU disponible
        gpu_memory_gb = get(hw_profile, :gpu_memory_gb, 4.0)
        
        # Estrategia básica para ajustar batch según memoria disponible
        if gpu_memory_gb >= 16.0
            batch_size = 128
        elseif gpu_memory_gb >= 8.0
            batch_size = 64
        elseif gpu_memory_gb >= 4.0
            batch_size = 32
        else
            batch_size = 16
        end
        
        debug_log("Tamaño de batch para GPU: $batch_size (memoria: $(round(gpu_memory_gb, digits=1)) GB)")
        return batch_size
    else
        # Para CPU, ajustar según número de núcleos
        cpu_threads = get(hw_profile, :cpu_threads, 4)
        
        if cpu_threads >= 32
            batch_size = 32
        elseif cpu_threads >= 16
            batch_size = 24
        elseif cpu_threads >= 8
            batch_size = 16
        elseif cpu_threads >= 4
            batch_size = 8
        else
            batch_size = 4
        end
        
        debug_log("Tamaño de batch para CPU: $batch_size (hilos: $cpu_threads)")
        return batch_size
    end
end

### Operaciones Tensoriales aceleradas por CUDA ###

"""
    cuda_tensor_convolution(input, kernel; stride=(1,1,1), padding=0)

Implementación acelerada por CUDA de la convolución tensorial 3D.

# Argumentos
- `input::Array{T,3}`: Tensor de entrada
- `kernel::Array{S,3}`: Kernel de convolución
- `stride::NTuple{3,Int}=(1,1,1)`: Paso de la convolución en cada dimensión
- `padding::Int=0`: Padding a aplicar alrededor del tensor de entrada

# Retorna
- `Array{Float32,3}`: Resultado de la convolución
"""
function cuda_tensor_convolution(
    input::Array{T,3}, 
    kernel::Array{S,3}; 
    stride::NTuple{3,Int}=(1,1,1), 
    padding::Int=0
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Verificar si CUDA está activo
    if !is_cuda_active()
        debug_log("cuda_tensor_convolution: CUDA no está activo, usando implementación CPU", level=:warn)
        # Aquí debería llamar a la implementación CPU, pero no la tenemos definida en este módulo
        # Implementación mínima para evitar errores
        error("Implementación CPU de tensor_convolution no disponible en este módulo")
    end
    
    debug_log("Ejecutando convolución tensorial en CUDA...")
    debug_log("Dimensiones de entrada: $(size(input)), Kernel: $(size(kernel)), Stride: $stride, Padding: $padding")
    
    try
        # Convertir a Float32 para optimización
        input_f32 = convert(Array{Float32}, input)
        kernel_f32 = convert(Array{Float32}, kernel)
        
        # Transferir a GPU
        debug_log("Transfiriendo tensores a GPU...")
        cu_input = CuArray(input_f32)
        cu_kernel = CuArray(kernel_f32)
        
        # Aplicar padding si es necesario
        if padding > 0
            debug_log("Aplicando padding de $padding...")
            cu_input = cuda_zero_pad(cu_input, padding)
            debug_log("Dimensiones después de padding: $(size(cu_input))")
        end
        
        # Calcular dimensiones de salida
        in_dim_x, in_dim_y, in_dim_z = size(cu_input)
        k_dim_x, k_dim_y, k_dim_z = size(cu_kernel)
        
        out_dim_x = div(in_dim_x - k_dim_x + 1, stride[1])
        out_dim_y = div(in_dim_y - k_dim_y + 1, stride[2])
        out_dim_z = div(in_dim_z - k_dim_z + 1, stride[3])
        
        debug_log("Dimensiones de salida: ($out_dim_x, $out_dim_y, $out_dim_z)")
        
        # Preparar salida
        cu_output = CUDA.zeros(Float32, out_dim_x, out_dim_y, out_dim_z)
        
        # Definir kernel CUDA
        function conv3d_kernel!(output, input, kernel, stride)
            # Obtener índices
            x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
            
            # Dimensiones
            out_dim_x, out_dim_y, out_dim_z = size(output)
            k_dim_x, k_dim_y, k_dim_z = size(kernel)
            
            # Verificar límites
            if x <= out_dim_x && y <= out_dim_y && z <= out_dim_z
                # Calcular posición en input
                in_x = (x - 1) * stride[1] + 1
                in_y = (y - 1) * stride[2] + 1
                in_z = (z - 1) * stride[3] + 1
                
                # Realizar convolución
                sum_val = 0.0f0
                
                for kx in 1:k_dim_x
                    for ky in 1:k_dim_y
                        for kz in 1:k_dim_z
                            sum_val += input[in_x+kx-1, in_y+ky-1, in_z+kz-1] * kernel[kx, ky, kz]
                        end
                    end
                end
                
                # Escribir resultado
                output[x, y, z] = sum_val
            end
            
            return nothing
        end
        
        # Configurar ejecución del kernel
        if CUDA_STATE === nothing
            error("Estado CUDA no inicializado")
        end
        
        debug_log("Configurando ejecución del kernel CUDA...")
        threads = min.(CUDA_STATE.preferred_block_size, (out_dim_x, out_dim_y, out_dim_z))
        blocks = ceil.(Int, (out_dim_x, out_dim_y, out_dim_z) ./ threads)
        
        debug_log("Configuración: Threads=$threads, Blocks=$blocks")
        
        # Ejecutar kernel
        debug_log("Ejecutando kernel de convolución...")
        @cuda threads=threads blocks=blocks conv3d_kernel!(cu_output, cu_input, cu_kernel, stride)
        
        # Sincronizar para asegurar que la computación haya terminado
        CUDA.synchronize()
        
        # Transferir resultado de vuelta a CPU
        debug_log("Transfiriendo resultado a CPU...")
        output = Array(cu_output)
        
        debug_log("Convolución tensorial completada con éxito")
        return output
    catch e
        debug_log("Error en cuda_tensor_convolution: $e", level=:error)
        debug_log(sprint(showerror, e, catch_backtrace()), level=:error)
        
        # Como fallback, intentar implementar una convolución muy básica en CPU
        debug_log("Intentando implementación fallback en CPU...")
        
        # Aplicar padding si es necesario
        if padding > 0
            padded_input = zeros(T, size(input) .+ (2*padding, 2*padding, 2*padding))
            padded_input[padding+1:end-padding, padding+1:end-padding, padding+1:end-padding] = input
            input = padded_input
        end
        
        # Dimensiones
        in_dims = size(input)
        k_dims = size(kernel)
        
        # Calcular dimensiones de salida
        out_dims = (
            div(in_dims[1] - k_dims[1] + 1, stride[1]),
            div(in_dims[2] - k_dims[2] + 1, stride[2]),
            div(in_dims[3] - k_dims[3] + 1, stride[3])
        )
        
        # Inicializar salida
        output = zeros(Float32, out_dims)
        
        # Convolución ingenua (muy ineficiente, solo para fallback)
        for x in 1:out_dims[1]
            for y in 1:out_dims[2]
                for z in 1:out_dims[3]
                    # Calcular posición en input
                    in_x = (x - 1) * stride[1] + 1
                    in_y = (y - 1) * stride[2] + 1
                    in_z = (z - 1) * stride[3] + 1
                    
                    # Realizar convolución
                    sum_val = 0.0f0
                    
                    for kx in 1:k_dims[1]
                        for ky in 1:k_dims[2]
                            for kz in 1:k_dims[3]
                                sum_val += input[in_x+kx-1, in_y+ky-1, in_z+kz-1] * kernel[kx, ky, kz]
                            end
                        end
                    end
                    
                    output[x, y, z] = sum_val
                end
            end
        end
        
        debug_log("Convolución fallback completada")
        return output
    end
end

"""
    cuda_zero_pad(tensor, padding)

Implementación acelerada por CUDA para añadir padding de ceros a un tensor.

# Argumentos
- `tensor::CuArray{T,3}`: Tensor en GPU a padear
- `padding::Int`: Cantidad de padding a añadir en todas las dimensiones

# Retorna
- `CuArray{T,3}`: Tensor con padding en GPU
"""
function cuda_zero_pad(tensor::CuArray{T,3}, padding::Int) where T <: AbstractFloat
    debug_log("Aplicando zero padding en CUDA...")
    
    try
        dim_x, dim_y, dim_z = size(tensor)
        debug_log("Dimensiones originales: ($dim_x, $dim_y, $dim_z)")
        
        # Dimensiones con padding
        padded_dims = (dim_x + 2*padding, dim_y + 2*padding, dim_z + 2*padding)
        debug_log("Dimensiones con padding: $padded_dims")
        
        # Crear tensor con padding
        # Crear tensor con padding
        padded = CUDA.zeros(T, padded_dims)
        
        # Copiar tensor original al centro
        padded[padding+1:padding+dim_x, 
               padding+1:padding+dim_y, 
               padding+1:padding+dim_z] = tensor
        
        return padded
    catch e
        debug_log("Error en cuda_zero_pad: $e", level=:error)
        
        # Implementación fallback en CPU (ineficiente pero segura)
        host_tensor = Array(tensor)
        
        dim_x, dim_y, dim_z = size(host_tensor)
        padded = zeros(T, dim_x + 2*padding, dim_y + 2*padding, dim_z + 2*padding)
        
        padded[padding+1:padding+dim_x, 
               padding+1:padding+dim_y, 
               padding+1:padding+dim_z] = host_tensor
        
        # Volver a transferir a GPU
        return CuArray(padded)
    end
end

"""
    cuda_zero_pad(tensor, padding)

Variante de zero_pad para tensores en CPU.

# Argumentos
- `tensor::Array{T,3}`: Tensor en CPU
- `padding::Int`: Cantidad de padding a añadir

# Retorna
- `Array{T,3}`: Tensor con padding
"""
function cuda_zero_pad(tensor::Array{T,3}, padding::Int) where T <: AbstractFloat
    if !is_cuda_active()
        # Implementación en CPU
        dim_x, dim_y, dim_z = size(tensor)
        padded = zeros(T, dim_x + 2*padding, dim_y + 2*padding, dim_z + 2*padding)
        padded[padding+1:padding+dim_x, 
               padding+1:padding+dim_y, 
               padding+1:padding+dim_z] = tensor
        return padded
    end
    
    # Si CUDA está activo, transferir a GPU, procesar y volver a CPU
    cu_tensor = CuArray{Float32}(tensor)
    cu_padded = cuda_zero_pad(cu_tensor, padding)
    return Array(cu_padded)
end

"""
    cuda_adaptive_pooling(input, output_size; mode=:max)

Implementación acelerada por CUDA del pooling adaptativo.

# Argumentos
- `input::Array{T,3}`: Tensor de entrada
- `output_size::NTuple{3,Int}`: Dimensiones del tensor de salida
- `mode::Symbol=:max`: Modo de pooling (:max o :avg)

# Retorna
- `Array{Float32,3}`: Resultado del pooling
"""
function cuda_adaptive_pooling(
    input::Array{T,3}, 
    output_size::NTuple{3,Int}; 
    mode::Symbol=:max
) where T <: AbstractFloat
    if !is_cuda_active()
        debug_log("cuda_adaptive_pooling: CUDA no está activo, usando implementación CPU", level=:warn)
        # Aquí debería llamar a la implementación CPU, pero no la tenemos definida en este módulo
        error("Implementación CPU de adaptive_pooling no disponible en este módulo")
    end
    
    debug_log("Ejecutando pooling adaptativo en CUDA...")
    debug_log("Dimensiones de entrada: $(size(input)), Salida: $output_size, Modo: $mode")
    
    try
        # Convertir a Float32 para optimización
        input_f32 = convert(Array{Float32}, input)
        
        # Transferir a GPU
        debug_log("Transfiriendo tensor a GPU...")
        cu_input = CuArray(input_f32)
        
        # Calcular dimensiones
        in_dim_x, in_dim_y, in_dim_z = size(cu_input)
        out_dim_x, out_dim_y, out_dim_z = output_size
        
        # Calcular tamaños de ventana de pooling
        window_x = div(in_dim_x, out_dim_x)
        window_y = div(in_dim_y, out_dim_y)
        window_z = div(in_dim_z, out_dim_z)
        
        # Asegurar que los tamaños de ventana son al menos 1
        window_x = max(1, window_x)
        window_y = max(1, window_y)
        window_z = max(1, window_z)
        
        debug_log("Tamaños de ventana de pooling: ($window_x, $window_y, $window_z)")
        
        # Preparar salida
        cu_output = CUDA.zeros(Float32, output_size)
        
        # Definir kernel CUDA para max pooling
        function max_pool_kernel!(output, input, in_dims, out_dims, window_sizes)
            # Obtener índices
            x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
            
            # Verificar límites
            if x <= out_dims[1] && y <= out_dims[2] && z <= out_dims[3]
                # Calcular índices en input
                start_x = (x - 1) * window_sizes[1] + 1
                start_y = (y - 1) * window_sizes[2] + 1
                start_z = (z - 1) * window_sizes[3] + 1
                
                end_x = min(start_x + window_sizes[1] - 1, in_dims[1])
                end_y = min(start_y + window_sizes[2] - 1, in_dims[2])
                end_z = min(start_z + window_sizes[3] - 1, in_dims[3])
                
                # Inicializar con valor mínimo
                max_val = -1.0e38f0
                
                # Encontrar máximo en ventana
                for ix in start_x:end_x
                    for iy in start_y:end_y
                        for iz in start_z:end_z
                            max_val = max(max_val, input[ix, iy, iz])
                        end
                    end
                end
                
                # Escribir resultado
                output[x, y, z] = max_val
            end
            
            return nothing
        end
        
        # Definir kernel CUDA para average pooling
        function avg_pool_kernel!(output, input, in_dims, out_dims, window_sizes)
            # Obtener índices
            x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
            
            # Verificar límites
            if x <= out_dims[1] && y <= out_dims[2] && z <= out_dims[3]
                # Calcular índices en input
                start_x = (x - 1) * window_sizes[1] + 1
                start_y = (y - 1) * window_sizes[2] + 1
                start_z = (z - 1) * window_sizes[3] + 1
                
                end_x = min(start_x + window_sizes[1] - 1, in_dims[1])
                end_y = min(start_y + window_sizes[2] - 1, in_dims[2])
                end_z = min(start_z + window_sizes[3] - 1, in_dims[3])
                
                # Calcular promedio
                sum_val = 0.0f0
                count = 0
                
                for ix in start_x:end_x
                    for iy in start_y:end_y
                        for iz in start_z:end_z
                            sum_val += input[ix, iy, iz]
                            count += 1
                        end
                    end
                end
                
                # Escribir resultado
                output[x, y, z] = sum_val / count
            end
            
            return nothing
        end
        
        # Configurar ejecución del kernel
        if CUDA_STATE === nothing
            error("Estado CUDA no inicializado")
        end
        
        debug_log("Configurando ejecución del kernel CUDA...")
        threads = min.(CUDA_STATE.preferred_block_size, output_size)
        blocks = ceil.(Int, output_size ./ threads)
        
        debug_log("Configuración: Threads=$threads, Blocks=$blocks")
        
        # Ejecutar kernel según modo
        debug_log("Ejecutando kernel de pooling ($mode)...")
        if mode == :max
            @cuda threads=threads blocks=blocks max_pool_kernel!(
                cu_output, cu_input, 
                (in_dim_x, in_dim_y, in_dim_z),
                (out_dim_x, out_dim_y, out_dim_z),
                (window_x, window_y, window_z)
            )
        else  # :avg o cualquier otro
            @cuda threads=threads blocks=blocks avg_pool_kernel!(
                cu_output, cu_input, 
                (in_dim_x, in_dim_y, in_dim_z),
                (out_dim_x, out_dim_y, out_dim_z),
                (window_x, window_y, window_z)
            )
        end
        
        # Sincronizar para asegurar que la computación haya terminado
        CUDA.synchronize()
        
        # Transferir resultado de vuelta a CPU
        debug_log("Transfiriendo resultado a CPU...")
        output = Array(cu_output)
        
        debug_log("Pooling adaptativo completado con éxito")
        return output
    catch e
        debug_log("Error en cuda_adaptive_pooling: $e", level=:error)
        debug_log(sprint(showerror, e, catch_backtrace()), level=:error)
        
        # Implementación fallback de CPU muy básica
        debug_log("Usando implementación fallback en CPU...")
        
        # Dimensiones
        in_dims = size(input)
        
        # Inicializar salida
        output = zeros(Float32, output_size)
        
        # Calcular tamaños de ventana
        window_sizes = (
            max(1, div(in_dims[1], output_size[1])),
            max(1, div(in_dims[2], output_size[2])),
            max(1, div(in_dims[3], output_size[3]))
        )
        
        # Implementación básica de pooling
        for x in 1:output_size[1]
            for y in 1:output_size[2]
                for z in 1:output_size[3]
                    # Calcular región en input
                    start_x = (x - 1) * window_sizes[1] + 1
                    start_y = (y - 1) * window_sizes[2] + 1
                    start_z = (z - 1) * window_sizes[3] + 1
                    
                    end_x = min(start_x + window_sizes[1] - 1, in_dims[1])
                    end_y = min(start_y + window_sizes[2] - 1, in_dims[2])
                    end_z = min(start_z + window_sizes[3] - 1, in_dims[3])
                    
                    if mode == :max
                        # Max pooling
                        max_val = -Inf
                        for ix in start_x:end_x
                            for iy in start_y:end_y
                                for iz in start_z:end_z
                                    max_val = max(max_val, input[ix, iy, iz])
                                end
                            end
                        end
                        output[x, y, z] = max_val
                    else
                        # Avg pooling
                        sum_val = 0.0
                        count = 0
                        for ix in start_x:end_x
                            for iy in start_y:end_y
                                for iz in start_z:end_z
                                    sum_val += input[ix, iy, iz]
                                    count += 1
                                end
                            end
                        end
                        output[x, y, z] = sum_val / count
                    end
                end
            end
        end
        
        debug_log("Pooling adaptativo fallback completado")
        return output
    end
end

"""
    cuda_volumetric_activation(tensor; type=:adaptive_tanh, parameters=nothing)

Implementación acelerada por CUDA de las activaciones volumétricas.

# Argumentos
- `tensor::Array{T,3}`: Tensor de entrada
- `type::Symbol=:adaptive_tanh`: Tipo de activación a aplicar
- `parameters=nothing`: Parámetros adicionales para la activación

# Retorna
- `Array{Float32,3}`: Tensor activado
"""
function cuda_volumetric_activation(
    tensor::Array{T,3}; 
    type::Symbol=:adaptive_tanh, 
    parameters=nothing
) where T <: AbstractFloat
    if !is_cuda_active()
        debug_log("cuda_volumetric_activation: CUDA no está activo, usando implementación CPU", level=:warn)
        # Aquí debería llamar a la implementación CPU, pero no la tenemos definida en este módulo
        error("Implementación CPU de volumetric_activation no disponible en este módulo")
    end
    
    debug_log("Ejecutando activación volumétrica en CUDA...")
    debug_log("Tipo de activación: $type")
    
    try
        # Convertir a Float32 para optimización
        tensor_f32 = convert(Array{Float32}, tensor)
        
        # Transferir a GPU
        debug_log("Transfiriendo tensor a GPU...")
        cu_tensor = CuArray(tensor_f32)
        
        # Preparar parámetros según el tipo de activación
        if type == :adaptive_tanh
            debug_log("Usando activación adaptive_tanh...")
            
            # Extraer parámetro de pendiente
            slope_factor = 0.1f0
            if !isnothing(parameters) && isa(parameters, Dict) && haskey(parameters, :slope_factor)
                slope_factor = convert(Float32, parameters[:slope_factor])
            end
            
            debug_log("Parámetro slope_factor: $slope_factor")
            
            # Aplicar activación
            cu_result = cuda_adaptive_tanh(cu_tensor, slope_factor)
            
        elseif type == :tensor_relu
            debug_log("Usando activación tensor_relu...")
            
            # Extraer parámetros
            alpha = 0.01f0
            sine_factor = 0.05f0
            
            if !isnothing(parameters) && isa(parameters, Dict)
                if haskey(parameters, :alpha)
                    alpha = convert(Float32, parameters[:alpha])
                end
                if haskey(parameters, :sine_factor)
                    sine_factor = convert(Float32, parameters[:sine_factor])
                end
            end
            
            debug_log("Parámetros - alpha: $alpha, sine_factor: $sine_factor")
            
            # Aplicar activación
            cu_result = cuda_tensor_relu(cu_tensor, alpha, sine_factor)
            
        else
            debug_log("Tipo de activación no implementado en CUDA: $type", level=:warn)
            # Para otros tipos, transferir de vuelta a CPU y usar implementación estándar
            return volumetric_activation(Array(cu_tensor), type=type, parameters=parameters)
        end
        
        # Sincronizar para asegurar que la computación haya terminado
        CUDA.synchronize()
        
        # Transferir resultado de vuelta a CPU
        debug_log("Transfiriendo resultado a CPU...")
        result = Array(cu_result)
        
        debug_log("Activación volumétrica completada con éxito")
        return result
    catch e
        debug_log("Error en cuda_volumetric_activation: $e", level=:error)
        debug_log(sprint(showerror, e, catch_backtrace()), level=:error)
        
        # Implementación fallback para los casos soportados
        debug_log("Usando implementación fallback en CPU...")
        
        if type == :adaptive_tanh
            # Extraer parámetro de pendiente
            slope_factor = 0.1f0
            if !isnothing(parameters) && isa(parameters, Dict) && haskey(parameters, :slope_factor)
                slope_factor = convert(Float32, parameters[:slope_factor])
            end
            
            # Implementación CPU de adaptive_tanh
            result = similar(tensor, Float32)
            for i in eachindex(tensor)
                val = tensor[i]
                adaptive_slope = 1.0f0 + slope_factor * abs(val)
                result[i] = tanh(val * adaptive_slope)
            end
            
            return result
            
        elseif type == :tensor_relu
            # Extraer parámetros
            alpha = 0.01f0
            sine_factor = 0.05f0
            
            if !isnothing(parameters) && isa(parameters, Dict)
                if haskey(parameters, :alpha)
                    alpha = convert(Float32, parameters[:alpha])
                end
                if haskey(parameters, :sine_factor)
                    sine_factor = convert(Float32, parameters[:sine_factor])
                end
            end
            
            # Implementación CPU de tensor_relu
            result = similar(tensor, Float32)
            for i in eachindex(tensor)
                val = tensor[i]
                if val > 0
                    result[i] = val + sine_factor * sin(val)
                else
                    result[i] = alpha * val
                end
            end
            
            return result
            
        else
            # Para otros tipos, usar implementación estándar
            error("Tipo de activación no implementado y sin fallback disponible: $type")
        end
    end
end

"""
    cuda_adaptive_tanh(tensor, slope_factor)

Implementación CUDA de la activación adaptive_tanh.

# Argumentos
- `tensor::CuArray{T,3}`: Tensor de entrada en GPU
- `slope_factor::Float32`: Factor de pendiente adaptativa

# Retorna
- `CuArray{T,3}`: Tensor activado en GPU
"""
function cuda_adaptive_tanh(tensor::CuArray{T,3}, slope_factor::Float32) where T <: AbstractFloat
    debug_log("Ejecutando kernel cuda_adaptive_tanh...")
    
    try
        # Definir kernel CUDA
        function adaptive_tanh_kernel!(output, input, slope_factor)
            # Obtener índices
            x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
            
            dim_x, dim_y, dim_z = size(input)
            
            # Verificar límites
            if x <= dim_x && y <= dim_y && z <= dim_z
                # Obtener valor
                val = input[x, y, z]
                
                # Aplicar activación adaptativa
                adaptive_slope = 1.0f0 + slope_factor * abs(val)
                output[x, y, z] = tanh(val * adaptive_slope)
            end
            
            return nothing
        end
        
        # Preparar salida
        output = similar(tensor)
        
        # Dimensiones
        dim_x, dim_y, dim_z = size(tensor)
        debug_log("Dimensiones del tensor: ($dim_x, $dim_y, $dim_z)")
        
        # Configurar ejecución del kernel
        if CUDA_STATE === nothing
            error("Estado CUDA no inicializado")
        end
        
        threads = min.(CUDA_STATE.preferred_block_size, (dim_x, dim_y, dim_z))
        blocks = ceil.(Int, (dim_x, dim_y, dim_z) ./ threads)
        
        debug_log("Configuración: Threads=$threads, Blocks=$blocks")
        
        # Ejecutar kernel
        @cuda threads=threads blocks=blocks adaptive_tanh_kernel!(output, tensor, slope_factor)
        
        # Sincronizar para asegurar que la computación haya terminado
        CUDA.synchronize()
        
        debug_log("Kernel cuda_adaptive_tanh completado con éxito")
        return output
    catch e
        debug_log("Error en cuda_adaptive_tanh: $e", level=:error)
        
        # Implementación alternativa en GPU usando broadcasting
        debug_log("Intentando implementación alternativa usando broadcasting...")
        try
            # Crear función de activación vectorizada
            function adaptive_tanh(x, slope_factor)
                adaptive_slope = 1.0f0 + slope_factor * abs(x)
                return tanh(x * adaptive_slope)
            end
            
            # Aplicar usando broadcasting
            output = adaptive_tanh.(tensor, slope_factor)
            
            CUDA.synchronize()
            return output
        catch e2
            debug_log("Error en implementación alternativa: $e2", level=:error)
            
            # Fallback final: transferir a CPU, procesar, y volver a GPU
            debug_log("Utilizando fallback CPU...")
            host_tensor = Array(tensor)
            
            # Procesar en CPU
            result = similar(host_tensor)
            for i in eachindex(host_tensor)
                val = host_tensor[i]
                adaptive_slope = 1.0f0 + slope_factor * abs(val)
                result[i] = tanh(val * adaptive_slope)
            end
            
            # Volver a GPU
            return CuArray(result)
        end
    end
end

"""
    cuda_tensor_relu(tensor, alpha, sine_factor)

Implementación CUDA de la activación tensor_relu.

# Argumentos
- `tensor::CuArray{T,3}`: Tensor de entrada en GPU
- `alpha::Float32`: Factor de pendiente para valores negativos
- `sine_factor::Float32`: Factor para el término sinusoidal

# Retorna
- `CuArray{T,3}`: Tensor activado en GPU
"""
function cuda_tensor_relu(tensor::CuArray{T,3}, alpha::Float32, sine_factor::Float32) where T <: AbstractFloat
    debug_log("Ejecutando kernel cuda_tensor_relu...")
    
    try
        # Definir kernel CUDA
        function tensor_relu_kernel!(output, input, alpha, sine_factor)
            # Obtener índices
            x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
            
            dim_x, dim_y, dim_z = size(input)
            
            # Verificar límites
            if x <= dim_x && y <= dim_y && z <= dim_z
                # Obtener valor
                val = input[x, y, z]
                
                # Aplicar ReLU tensorial
                if val > 0
                    output[x, y, z] = val + sine_factor * sin(val)
                else
                    output[x, y, z] = alpha * val
                end
            end
            
            return nothing
        end
        
        # Preparar salida
        output = similar(tensor)
        
        # Dimensiones
        dim_x, dim_y, dim_z = size(tensor)
        debug_log("Dimensiones del tensor: ($dim_x, $dim_y, $dim_z)")
        
        # Configurar ejecución del kernel
        if CUDA_STATE === nothing
            error("Estado CUDA no inicializado")
        end
        
        threads = min.(CUDA_STATE.preferred_block_size, (dim_x, dim_y, dim_z))
        blocks = ceil.(Int, (dim_x, dim_y, dim_z) ./ threads)
        
        debug_log("Configuración: Threads=$threads, Blocks=$blocks")
        
        # Ejecutar kernel
        @cuda threads=threads blocks=blocks tensor_relu_kernel!(output, tensor, alpha, sine_factor)
        
        # Sincronizar para asegurar que la computación haya terminado
        CUDA.synchronize()
        
        debug_log("Kernel cuda_tensor_relu completado con éxito")
        return output
    catch e
        debug_log("Error en cuda_tensor_relu: $e", level=:error)
        
        # Implementación alternativa en GPU usando broadcasting
        debug_log("Intentando implementación alternativa usando broadcasting...")
        try
            # Crear función de activación vectorizada
            function tensor_relu(x, alpha, sine_factor)
                if x > 0
                    return x + sine_factor * sin(x)
                else
                    return alpha * x
                end
            end
            
            # Aplicar usando broadcasting
            output = tensor_relu.(tensor, alpha, sine_factor)
            
            CUDA.synchronize()
            return output
        catch e2
            debug_log("Error en implementación alternativa: $e2", level=:error)
            
            # Fallback final: transferir a CPU, procesar, y volver a GPU
            debug_log("Utilizando fallback CPU...")
            host_tensor = Array(tensor)
            
            # Procesar en CPU
            result = similar(host_tensor)
            for i in eachindex(host_tensor)
                val = host_tensor[i]
                if val > 0
                    result[i] = val + sine_factor * sin(val)
                else
                    result[i] = alpha * val
                end
            end
            
            # Volver a GPU
            return CuArray(result)
        end
    end
end

"""
    cuda_spatial_attention_transform(input, attention_map)

Implementación acelerada por CUDA de la transformación atencional.

# Argumentos
- `input::Array{T,3}`: Tensor de entrada
- `attention_map::Array{S,3}`: Mapa de atención

# Retorna
- `Array{Float32,3}`: Tensor transformado
"""
function cuda_spatial_attention_transform(
    input::Array{T,3}, 
    attention_map::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    if !is_cuda_active()
        debug_log("cuda_spatial_attention_transform: CUDA no está activo, usando implementación CPU", level=:warn)
        # Implementación básica en CPU
        if size(input) != size(attention_map)
            error("Las dimensiones del tensor y el mapa de atención deben coincidir")
        end
        return input .* attention_map
    end
    
    debug_log("Ejecutando transformación atencional en CUDA...")
    debug_log("Dimensiones - Input: $(size(input)), Mapa de atención: $(size(attention_map))")
    
    try
        # Asegurar que las dimensiones coincidan
        if size(input) != size(attention_map)
            debug_log("Dimensiones no coincidentes, interpolando mapa de atención...")
            # Necesitamos implementar o llamar a tensor_interpolation aquí
            # Por ahora lanzamos un error
            error("Las dimensiones del tensor y el mapa de atención deben coincidir")
        end
        
        # Convertir a Float32 para optimización
        input_f32 = convert(Array{Float32}, input)
        attention_map_f32 = convert(Array{Float32}, attention_map)
        
        # Transferir a GPU
        debug_log("Transfiriendo tensores a GPU...")
        cu_input = CuArray(input_f32)
        cu_attention = CuArray(attention_map_f32)
        
        # Multiplicación elemento a elemento (optimizada en CUDA)
        debug_log("Aplicando transformación atencional...")
        cu_output = cu_input .* cu_attention
        
        # Sincronizar para asegurar que la computación haya terminado
        CUDA.synchronize()
        
        # Transferir resultado de vuelta a CPU
        debug_log("Transfiriendo resultado a CPU...")
        output = Array(cu_output)
        
        debug_log("Transformación atencional completada con éxito")
        return output
    catch e
        debug_log("Error en cuda_spatial_attention_transform: $e", level=:error)
        
        # Implementación fallback en CPU
        debug_log("Usando implementación fallback en CPU...")
        return input .* attention_map
    end
end

end # module CUDATensors