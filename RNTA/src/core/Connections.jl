# core/Connections.jl
# Define las conexiones entre neuronas tensoriales en el espacio 3D

"""
    TensorConnection

Representa una conexión entre dos neuronas tensoriales en el espacio cerebral.
"""
# Añadir al inicio del archivo:
module Connections

using UUIDs
using DataStructures
using ..TensorNeuron  # Asumiendo que TensorNeuron es un módulo hermano

export Tensor_Connection, transmit, update_weight!, should_prune, connection_distance, connection_probability, establish_connections!, find_connection, get_outgoing_connections, get_incoming_connections, prune_connections!, BrainSpaceConfig
mutable struct Tensor_Connection
    # Identificador único
    id::UUID
    
    # Neurona origen
    source_id::UUID
    
    # Neurona destino
    target_id::UUID
    
    # Peso de conexión (tensor que modula la transferencia de información)
    weight::Array{Float32,3}
    
    # Fuerza de la conexión (escalar para optimización)
    strength::Float32
    
    # Historial de actividad (para reglas de plasticidad)
    activity_history::CircularBuffer{Float32}
    
    # Tiempo de creación
    creation_time::Float64
    
    # Tipo de conexión
    connection_type::Symbol
end

"""
Constructor principal para TensorConnection
"""
function Tensor_Connection(
    source::Tensor_Neuron,
    target::Tensor_Neuron;
    weight_size::NTuple{3,Int}=(3,3,3),
    init_scale::Float32=0.1f0,
    history_length::Int=10,
    connection_type::Symbol=:excitatory
)
    id = uuid4()
    
    # Inicializar peso con valores aleatorios pequeños
    weight = randn(Float32, weight_size) * init_scale
    
    # Si el tipo es inhibitorio, invertir signos
    if connection_type == :inhibitory
        weight .*= -1.0f0
    end
    
    # Inicializar fuerza
    strength = init_scale
    
    # Inicializar buffer circular para historial de actividad
    activity_history = CircularBuffer{Float32}(history_length)
    for _ in 1:history_length
        push!(activity_history, 0.0f0)
    end
    
    # Tiempo actual
    creation_time = time()
    
    return Tensor_Connection(
        id,
        source.id,
        target.id,
        weight,
        strength,
        activity_history,
        creation_time,
        connection_type
    )
end
"""
    BrainSpaceConfig

Configuración para el espacio cerebral que define parámetros para
las conexiones entre neuronas tensoriales.
"""
struct BrainSpaceConfig
    # Densidad inicial de neuronas en el espacio
    initial_density::Float32
    
    # Escala de inicialización para pesos
    init_scale::Float32
    
    # Radio máximo para establecer conexiones entre neuronas
    max_connection_radius::Float32
    
    # Probabilidad base para establecer conexiones
    base_connection_probability::Float32
    
    # Factor de expansión para crecimiento de regiones
    expansion_factor::Float32
    
    # Número de capas para propagación
    propagation_layers::Int
    
    # Constructor con valores por defecto
    function BrainSpaceConfig(;
        initial_density::Float32 = 0.05f0,
        init_scale::Float32 = 0.1f0,
        max_connection_radius::Float32 = 10.0f0,
        base_connection_probability::Float32 = 0.3f0,
        expansion_factor::Float32 = 1.5f0,
        propagation_layers::Int = 3
    )
        return new(
            initial_density,
            init_scale,
            max_connection_radius,
            base_connection_probability,
            expansion_factor,
            propagation_layers
        )
    end
end
"""
    transmit(connection, source_state)

Transmite información desde la neurona origen a la destino a través de la conexión.
"""
function transmit(connection::Tensor_Connection, source_state::Array{T,3}) where T <: AbstractFloat
    # Si las dimensiones no coinciden, redimensionar
    if size(source_state) != size(connection.weight)
        source_state = tensor_interpolation(source_state, size(connection.weight))
    end
    
    # Modular el estado de origen con el peso de la conexión
    modulated = source_state .* connection.weight
    
    # Registrar actividad de esta transmisión
    activity = sum(abs.(modulated)) / length(modulated)
    push!(connection.activity_history, activity)
    
    return modulated
end

"""
    update_weight!(connection, pre_activation, post_activation, learning_rate)

Actualiza el peso de la conexión basado en actividad pre y post sináptica.
"""
function update_weight!(
    connection::Tensor_Connection,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    learning_rate::Float32
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Redimensionar si es necesario
    if size(pre_activation) != size(connection.weight)
        pre_activation = tensor_interpolation(pre_activation, size(connection.weight))
    end
    
    if size(post_activation) != size(connection.weight)
        post_activation = tensor_interpolation(post_activation, size(connection.weight))
    end
    
    # Regla de aprendizaje Hebbiano tensorial
    # "Las neuronas que se activan juntas, se conectan juntas"
    hebbian_update = pre_activation .* post_activation
    
    # Aplicar actualización
    connection.weight .+= learning_rate .* hebbian_update
    
    # Actualizar fuerza de conexión
    connection.strength = mean(abs.(connection.weight))
    
    return connection
end

"""
    should_prune(connection, threshold)

Determina si la conexión debería ser eliminada basado en su actividad reciente.
"""
function should_prune(connection::Tensor_Connection, threshold::Float32=0.01f0)
    # Calcular actividad media reciente
    recent_activity = mean(connection.activity_history)
    
    # Conexiones con poca actividad reciente son candidatas para eliminación
    return recent_activity < threshold
end

"""
    connection_distance(source_neuron, target_neuron)

Calcula la distancia entre dos neuronas conectadas.
"""
function connection_distance(source_neuron::Tensor_Neuron, target_neuron::Tensor_Neuron)
    return sqrt(
        (source_neuron.position[1] - target_neuron.position[1])^2 +
        (source_neuron.position[2] - target_neuron.position[2])^2 +
        (source_neuron.position[3] - target_neuron.position[3])^2
    )
end

"""
    connection_probability(source_neuron, target_neuron, max_distance, base_probability)

Calcula la probabilidad de conexión entre dos neuronas basada en su distancia.
"""
function connection_probability(
    source_neuron::Tensor_Neuron,
    target_neuron::Tensor_Neuron,
    max_distance::Float32,
    base_probability::Float32
)
    # Calcular distancia
    dist = connection_distance(source_neuron, target_neuron)
    
    # Si la distancia excede el máximo, probabilidad cero
    if dist > max_distance
        return 0.0f0
    end
    
    # Probabilidad decae con la distancia
    return base_probability * (1.0f0 - dist / max_distance)
end

"""
    establish_connections!(connections, neurons, config)

Establece conexiones iniciales entre neuronas.
"""
function establish_connections!(
    connections::Vector{Tensor_Connection},
    neurons::Dict{NTuple{3,Int}, TensorNeuron},
    config::BrainSpaceConfig
)
    # Para cada par de neuronas, intentar establecer conexión
    neuron_list = collect(values(neurons))
    
    for i in 1:length(neuron_list)
        for j in 1:length(neuron_list)
            # No conectar una neurona consigo misma
            if i == j
                continue
            end
            
            source_neuron = neuron_list[i]
            target_neuron = neuron_list[j]
            
            # Calcular probabilidad de conexión
            prob = connection_probability(
                source_neuron,
                target_neuron,
                config.max_connection_radius,
                config.base_connection_probability
            )
            
            # Decidir si establecer conexión
            if rand() < prob
                # Determinar tipo de conexión (80% excitatorias, 20% inhibitorias)
                conn_type = rand() < 0.8 ? :excitatory : :inhibitory
                
                # Crear conexión
                connection = TensorConnection(
                    source_neuron,
                    target_neuron,
                    connection_type=conn_type
                )
                
                # Añadir a la lista
                push!(connections, connection)
            end
        end
    end
    
    return connections
end

"""
    find_connection(connections, source_id, target_id)

Encuentra una conexión específica entre dos neuronas.
"""
function find_connection(connections::Vector{Tensor_Connection}, source_id::UUID, target_id::UUID)
    for connection in connections
        if connection.source_id == source_id && connection.target_id == target_id
            return connection
        end
    end
    
    return nothing
end

"""
    get_outgoing_connections(connections, neuron_id)

Obtiene todas las conexiones salientes de una neurona.
"""
function get_outgoing_connections(connections::Vector{Tensor_Connection}, neuron_id::UUID)
    outgoing = TensorConnection[]
    
    for connection in connections
        if connection.source_id == neuron_id
            push!(outgoing, connection)
        end
    end
    
    return outgoing
end

"""
    get_incoming_connections(connections, neuron_id)

Obtiene todas las conexiones entrantes a una neurona.
"""
function get_incoming_connections(connections::Vector{Tensor_Connection}, neuron_id::UUID)
    incoming = TensorConnection[]
    
    for connection in connections
        if connection.target_id == neuron_id
            push!(incoming, connection)
        end
    end
    
    return incoming
end

"""
    prune_connections!(connections, threshold)

Elimina conexiones débiles basado en un umbral de actividad.
"""
function prune_connections!(connections::Vector{Tensor_Connection}, threshold::Float32=0.01f0)
    # Identificar conexiones a eliminar
    to_remove = Int[]
    
    for (i, connection) in enumerate(connections)
        if should_prune(connection, threshold)
            push!(to_remove, i)
        end
    end
    
    # Eliminar conexiones en orden inverso (para mantener índices válidos)
    sort!(to_remove, rev=true)
    
    for i in to_remove
        deleteat!(connections, i)
    end
    
    return connections
end
end