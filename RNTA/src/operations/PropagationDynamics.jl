# operations/PropagationDynamics.jl
# Implementa dinámica de propagación de activación en el espacio neuronal 3D

module PropagationDynamics

using LinearAlgebra
using Statistics
using UUIDs

# Importaciones de otros módulos de RNTA
using ..TensorOperations
using ..BrainSpace
using ..TensorNeuron
using ..Connections

"""
    PropagationParameters

Parámetros que controlan cómo se propaga la activación en el espacio cerebral.
"""
struct PropagationParameters
    # Factor de decaimiento temporal (0-1)
    temporal_decay::Float32
    
    # Factor de atenuación espacial
    spatial_attenuation::Float32
    
    # Velocidad de propagación (unidades espaciales por paso)
    propagation_speed::Float32
    
    # Umbral de activación para disparo neuronal
    activation_threshold::Float32
    
    # Período refractario (pasos temporales)
    refractory_period::Int
    
    # Factor de integración para combinar señales entrantes
    integration_factor::Float32
    
    # Tipo de propagación
    propagation_type::Symbol
end

# Constructor con valores por defecto
function PropagationParameters(;
    temporal_decay::Float32=0.9f0,
    spatial_attenuation::Float32=0.2f0,
    propagation_speed::Float32=1.0f0,
    activation_threshold::Float32=0.3f0,
    refractory_period::Int=2,
    integration_factor::Float32=0.5f0,
    propagation_type::Symbol=:wave
)
    return PropagationParameters(
        temporal_decay,
        spatial_attenuation,
        propagation_speed,
        activation_threshold,
        refractory_period,
        integration_factor,
        propagation_type
    )
end

"""
    propagate_activation(brain, input_activation; params=PropagationParameters())

Propaga la activación a través del espacio cerebral según los parámetros especificados.
"""
function propagate_activation(
    brain::Brain_Space,
    input_activation::Array{T,3};
    params::PropagationParameters=PropagationParameters()
) where T <: AbstractFloat
    # Verificar que las dimensiones coincidan
    if size(input_activation) != brain.dimensions
        input_activation = tensor_interpolation(input_activation, brain.dimensions)
    end
    
    # Inicializar campo de activación con la entrada
    activation_field = copy(input_activation)
    
    # Estado de neurona refractaria (marcar como verdadero si está en período refractario)
    refractory_state = falses(brain.dimensions)
    
    # Contador de tiempo refractario para cada posición
    refractory_counter = zeros(Int, brain.dimensions)
    
    # Historial de activación para trazado de dinámica temporal
    activation_history = [copy(activation_field)]
    
    # Simular propagación por pasos de tiempo
    num_steps = 10  # Ajustar según la dinámica deseada
    
    for step in 1:num_steps
        # Aplicar decaimiento temporal
        activation_field .*= params.temporal_decay
        
        # Propagar activación según tipo de propagación
        if params.propagation_type == :wave
            activation_field = propagate_wave(
                activation_field,
                refractory_state,
                refractory_counter,
                params
            )
        elseif params.propagation_type == :diffusion
            activation_field = propagate_diffusion(
                activation_field,
                params
            )
        elseif params.propagation_type == :saltatory
            activation_field = propagate_saltatory(
                activation_field,
                brain.neurons,
                brain.connections,
                params
            )
        else
            # Por defecto, usar propagación por onda
            activation_field = propagate_wave(
                activation_field,
                refractory_state,
                refractory_counter,
                params
            )
        end
        
        # Actualizar estado refractario
        for idx in CartesianIndices(activation_field)
            if activation_field[idx] > params.activation_threshold && !refractory_state[idx]
                # Neurona disparó, entrar en período refractario
                refractory_state[idx] = true
                refractory_counter[idx] = params.refractory_period
            elseif refractory_state[idx]
                # Decrementar contador refractario
                refractory_counter[idx] -= 1
                
                # Si el contador llega a cero, salir del estado refractario
                if refractory_counter[idx] <= 0
                    refractory_state[idx] = false
                end
            end
        end
        
        # Guardar estado actual en historial
        push!(activation_history, copy(activation_field))
    end
    
    # Actualizar el estado global del cerebro
    brain.global_state = activation_field
    
    return activation_field, activation_history
end

"""
    propagate_wave(activation_field, refractory_state, refractory_counter, params)

Implementa propagación de activación tipo onda.
"""
function propagate_wave(
    activation_field::Array{T,3},
    refractory_state::BitArray{3},
    refractory_counter::Array{Int,3},
    params::PropagationParameters
) where T <: AbstractFloat
    # Crear campo para la siguiente iteración
    next_field = copy(activation_field)
    
    # Dimensiones del campo
    dim_x, dim_y, dim_z = size(activation_field)
    
    # Radio de propagación para este paso
    prop_radius = max(1, round(Int, params.propagation_speed))
    
    # Para cada posición en el espacio
    for x in 1:dim_x
        for y in 1:dim_y
            for z in 1:dim_z
                # Omitir si la posición está en período refractario
                if refractory_state[x, y, z]
                    continue
                end
                
                # Valor de activación actual
                current_value = activation_field[x, y, z]
                
                # Si supera el umbral, propagar a vecinos
                if current_value > params.activation_threshold
                    # Propagar a vecinos dentro del radio
                    for dx in -prop_radius:prop_radius
                        for dy in -prop_radius:prop_radius
                            for dz in -prop_radius:prop_radius
                                # Omitir posición central
                                if dx == 0 && dy == 0 && dz == 0
                                    continue
                                end
                                
                                # Calcular posición del vecino
                                nx = x + dx
                                ny = y + dy
                                nz = z + dz
                                
                                # Verificar límites
                                if 1 <= nx <= dim_x && 1 <= ny <= dim_y && 1 <= nz <= dim_z
                                    # Omitir si vecino está en período refractario
                                    if refractory_state[nx, ny, nz]
                                        continue
                                    end
                                    
                                    # Calcular distancia
                                    distance = sqrt(dx^2 + dy^2 + dz^2)
                                    
                                    # Calcular atenuación basada en distancia
                                    attenuation = exp(-distance * params.spatial_attenuation)
                                    
                                    # Propagar activación al vecino
                                    propagated_value = current_value * attenuation
                                    
                                    # Integrar con valor actual del vecino
                                    integrated_value = next_field[nx, ny, nz] * (1 - params.integration_factor) +
                                                      propagated_value * params.integration_factor
                                    
                                    # Actualizar valor del vecino (solo si aumenta)
                                    next_field[nx, ny, nz] = max(next_field[nx, ny, nz], integrated_value)
                                end
                            end
                        end
                    end
                    
                    # Reducir ligeramente el valor actual tras propagar
                    next_field[x, y, z] *= 0.95f0
                end
            end
        end
    end
    
    return next_field
end

"""
    propagate_diffusion(activation_field, params)

Implementa propagación de activación por difusión.
"""
function propagate_diffusion(
    activation_field::Array{T,3},
    params::PropagationParameters
) where T <: AbstractFloat
    # Dimensiones del campo
    dim_x, dim_y, dim_z = size(activation_field)
    
    # Crear campo para la siguiente iteración
    next_field = copy(activation_field)
    
    # Kernel de difusión 3D
    # El tamaño y forma del kernel determina cómo se difunde la activación
    kernel_size = 3
    kernel = zeros(Float32, kernel_size, kernel_size, kernel_size)
    
    # Llenar kernel con valores que disminuyen con la distancia al centro
    center = div(kernel_size, 2) + 1
    for i in 1:kernel_size
        for j in 1:kernel_size
            for k in 1:kernel_size
                # Distancia al centro
                dist = sqrt((i - center)^2 + (j - center)^2 + (k - center)^2)
                
                # Valor basado en distancia
                kernel[i, j, k] = exp(-dist * params.spatial_attenuation)
            end
        end
    end
    
    # Valor central - determina cuánto se retiene vs. se difunde
    kernel[center, center, center] = params.temporal_decay
    
    # Normalizar kernel
    kernel ./= sum(kernel)
    
    # Aplicar convolución para simular difusión
    diffused = tensor_convolution(activation_field, kernel, padding=1)
    
    # Combinar según factor de integración
    next_field = activation_field * (1 - params.integration_factor) +
                diffused * params.integration_factor
    
    return next_field
end

"""
    propagate_saltatory(activation_field, neurons, connections, params)

Implementa propagación de activación saltatorio a través de conexiones.
"""
function propagate_saltatory(
    activation_field::Array{T,3},
    neurons::Dict{NTuple{3,Int}, Tensor_Neuron},
    connections::Vector{Tensor_Connection},
    params::PropagationParameters
) where T <: AbstractFloat
    # Crear campo para la siguiente iteración
    next_field = copy(activation_field)
    
    # Para cada neurona, verificar si supera umbral de activación
    active_neurons = Dict{UUID, Float32}()
    
    for (pos, neuron) in neurons
        # Obtener activación actual en esta posición
        activation = activation_field[pos...]
        
        # Si supera umbral, marcar como activa
        if activation > params.activation_threshold
            active_neurons[neuron.id] = activation
        end
    end
    
    # Si hay neuronas activas, propagar a través de conexiones
    if !isempty(active_neurons)
        for connection in connections
            # Verificar si neurona origen está activa
            if haskey(active_neurons, connection.source_id)
                # Obtener activación de neurona origen
                source_activation = active_neurons[connection.source_id]
                
                # Encontrar neurona destino
                target_pos = nothing
                for (pos, neuron) in neurons
                    if neuron.id == connection.target_id
                        target_pos = pos
                        break
                    end
                end
                
                if !isnothing(target_pos)
                    # Calcular activación propagada
                    propagated_activation = source_activation * connection.strength
                    
                    if connection.connection_type == :inhibitory
                        # Conexión inhibitoria - disminuir activación
                        propagated_activation = -propagated_activation
                    end
                    
                    # Integrar con activación actual
                    current_activation = next_field[target_pos...]
                    next_field[target_pos...] = current_activation * (1 - params.integration_factor) +
                                              propagated_activation * params.integration_factor
                end
            end
        end
    end
    
    return next_field
end

"""
    extract_temporal_dynamics(activation_history, position)

Extrae la dinámica temporal de activación en una posición específica.
"""
function extract_temporal_dynamics(
    activation_history::Vector{Array{T,3}},
    position::NTuple{3,Int}
) where T <: AbstractFloat
    # Extraer serie temporal para la posición dada
    time_series = [history[position...] for history in activation_history]
    
    return time_series
end

"""
    calculate_temporal_features(time_series)

Calcula características temporales de una serie de activación.
"""
function calculate_temporal_features(time_series::Vector{T}) where T <: AbstractFloat
    # Calcular características básicas
    features = Dict{Symbol, Float32}()
    
    # Valor medio
    features[:mean] = mean(time_series)
    
    # Desviación estándar
    features[:std] = std(time_series)
    
    # Valor máximo
    features[:max] = maximum(time_series)
    
    # Índice del máximo (latencia)
    _, max_idx = findmax(time_series)
    features[:latency] = Float32(max_idx)
    
    # Duración sobre umbral (persistencia)
    threshold = 0.5 * features[:max]
    features[:persistence] = count(x -> x > threshold, time_series) / length(time_series)
    
    # Calcular oscilación (diferencia media entre puntos consecutivos)
    diffs = [abs(time_series[i] - time_series[i-1]) for i in 2:length(time_series)]
    features[:oscillation] = mean(diffs)
    
    return features
end

"""
    visualize_propagation(activation_history)

Genera una visualización de la propagación de activación a través del tiempo.
"""
function visualize_propagation(activation_history::Vector{Array{T,3}}) where T <: AbstractFloat
    # Esta función sería una interfaz para el módulo de visualización
    # Por ahora, devolvemos un resumen textual
    
    num_steps = length(activation_history)
    
    # Calcular estadísticas por paso
    step_stats = []
    
    for (step, activation) in enumerate(activation_history)
        # Calcular estadísticas para este paso
        stats = (
            step = step,
            mean = mean(activation),
            max = maximum(activation),
            active_voxels = count(x -> x > 0.1, activation)
        )
        
        push!(step_stats, stats)
    end
    
    return step_stats
end

# Exportar tipos y funciones principales
export PropagationParameters, 
       propagate_activation,
       propagate_wave, propagate_diffusion, propagate_saltatory,
       extract_temporal_dynamics, calculate_temporal_features,
       visualize_propagation

end # module PropagationDynamics