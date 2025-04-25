# architecture/CorticalLayers.jl
# Implementa capas neuronales inspiradas en la corteza cerebral

module CorticalLayers

using LinearAlgebra
using Statistics
using Random

# Importaciones de otros módulos de RNTA
using .. BrainSpace
using .. TensorNeuron
using .. Connections
using .. TensorOperations
using .. VolumetricActivations
using .. SpatialAttention
using UUIDs

"""
    CorticalLayerConfig

Configuración para una capa cortical.
"""
struct CorticalLayerConfig
    # Número de neuronas en la capa
    neurons_count::Int
    
    # Dimensiones de la capa
    dimensions::NTuple{3,Int}
    
    # Probabilidad de conexiones laterales entre neuronas de la misma capa
    lateral_connectivity::Float32
    
    # Tipo de neuronas en la capa
    neuron_type::Symbol
    
    # Tipo de activación para la capa
    activation_type::Symbol
    
    # Factor de plasticidad (0.0-1.0)
    plasticity_factor::Float32
    
    # Densidad de conexiones feed-forward
    feedforward_density::Float32
    
    # Densidad de conexiones feedback
    feedback_density::Float32
    
    # Factor de especialización inicial
    initial_specialization::Float32
end

# Constructor con valores por defecto
function CorticalLayerConfig(;
    neurons_count::Int=100,
    dimensions::NTuple{3,Int}=(10, 10, 1),
    lateral_connectivity::Float32=0.1f0,
    neuron_type::Symbol=:general,
    activation_type::Symbol=:adaptive_tanh,
    plasticity_factor::Float32=0.5f0,
    feedforward_density::Float32=0.3f0,
    feedback_density::Float32=0.1f0,
    initial_specialization::Float32=0.0f0
)
    return CorticalLayerConfig(
        neurons_count,
        dimensions,
        lateral_connectivity,
        neuron_type,
        activation_type,
        plasticity_factor,
        feedforward_density,
        feedback_density,
        initial_specialization
    )
end

"""
    CorticalLayer

Implementa una capa cortical compuesta por neuronas tensoriales.
"""
mutable struct CorticalLayer
    # Identificador único
    id::UUID
    
    # Nombre de la capa
    name::String
    
    # Configuración de la capa
    config::CorticalLayerConfig
    
    # Neuronas que componen la capa
    neurons::Dict{NTuple{3,Int}, Tensor_Neuron}
    
    # Conexiones entre neuronas de la capa (laterales)
    lateral_connections::Vector{Tensor_Connection}
    
    # Mapas de atención para la capa
    attention_maps::Dict{Symbol, SpatialAttentionMap}
    
    # Estado de activación de la capa
    activation_state::Array{Float32,3}
    
    # Capa anterior (feed-forward input)
    previous_layer::Union{CorticalLayer, Nothing}
    
    # Capa siguiente (feed-forward output)
    next_layer::Union{CorticalLayer, Nothing}
    
    # Conexiones feed-forward (hacia adelante)
    feedforward_connections::Vector{Tensor_Connection}
    
    # Conexiones feedback (hacia atrás)
    feedback_connections::Vector{Tensor_Connection}
    
    # Historial de activaciones
    activation_history::Vector{Array{Float32,3}}
    
    # Metadatos adicionales
    metadata::Dict{Symbol, Any}
end

"""
Constructor principal para CorticalLayer
"""
function CorticalLayer(
    name::String,
    config::CorticalLayerConfig;
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    # Crear identificador único
    id = uuid4()
    
    # Inicializar diccionario de neuronas
    neurons = Dict{NTuple{3,Int}, Tensor_Neuron}()
    
    # Inicializar estado de activación
    activation_state = zeros(Float32, config.dimensions)
    
    # Inicializar conexiones
    lateral_connections = Vector{Tensor_Connection}()
    feedforward_connections = Vector{Tensor_Connection}()
    feedback_connections = Vector{Tensor_Connection}()
    
    # Inicializar mapas de atención
    attention_maps = Dict{Symbol, SpatialAttentionMap}(
        :input => SpatialAttentionMap(config.dimensions),
        :output => SpatialAttentionMap(config.dimensions),
        :lateral => SpatialAttentionMap(config.dimensions)
    )
    
    # Inicializar historial de activaciones
    activation_history = Vector{Array{Float32,3}}()
    
    return CorticalLayer(
        id,
        name,
        config,
        neurons,
        lateral_connections,
        attention_maps,
        activation_state,
        nothing,
        nothing,
        feedforward_connections,
        feedback_connections,
        activation_history,
        metadata
    )
end

"""
    initialize_neurons!(layer)

Inicializa las neuronas de la capa cortical.
"""
function initialize_neurons!(layer::CorticalLayer)
    # Calcular número de neuronas en cada dimensión
    dim_x, dim_y, dim_z = layer.config.dimensions
    
    # Distribuir neuronas en el espacio 3D
    positions = distribute_neurons(
        layer.config.neurons_count, 
        layer.config.dimensions
    )
    
    # Crear neuronas en las posiciones
    for pos in positions
        # Configurar tamaño del campo receptivo
        receptive_field_size = determine_receptive_field_size(layer.config.neuron_type)
        
        # Crear neurona
        neuron = TensorNeuron(
            pos, 
            receptive_field_size,
            functional_type=layer.config.neuron_type,
            specialization=layer.config.initial_specialization
        )
        
        # Añadir a diccionario
        layer.neurons[pos] = neuron
    end
    
    return layer
end

"""
    determine_receptive_field_size(neuron_type)

Determina el tamaño del campo receptivo según el tipo de neurona.
"""
function determine_receptive_field_size(neuron_type::Symbol)
    # Diferentes tipos de neuronas tienen diferentes tamaños de campo receptivo
    if neuron_type == :spatial
        return (5, 5, 3)
    elseif neuron_type == :temporal
        return (3, 3, 5)
    elseif neuron_type == :feature
        return (4, 4, 4)
    elseif neuron_type == :contextual
        return (7, 7, 3)
    else
        # Tipo general
        return (3, 3, 3)
    end
end

"""
    distribute_neurons(count, dimensions)

Distribuye el número especificado de neuronas en el espacio tridimensional.
"""
function distribute_neurons(
    count::Int,
    dimensions::NTuple{3,Int}
)
    dim_x, dim_y, dim_z = dimensions
    positions = Set{NTuple{3,Int}}()
    
    # Calcular volumen total
    total_volume = dim_x * dim_y * dim_z
    
    # Si el número de neuronas es mayor que el volumen total,
    # limitar al volumen total (una neurona por posición)
    count = min(count, total_volume)
    
    # Distribuir neuronas uniformemente
    if count >= total_volume * 0.7
        # Alta densidad: distribuir en grid regular
        spacing_x = max(1, div(dim_x, ceil(Int, cbrt(count))))
        spacing_y = max(1, div(dim_y, ceil(Int, cbrt(count))))
        spacing_z = max(1, div(dim_z, ceil(Int, cbrt(count))))
        
        for x in 1:spacing_x:dim_x
            for y in 1:spacing_y:dim_y
                for z in 1:spacing_z:dim_z
                    push!(positions, (x, y, z))
                    
                    if length(positions) >= count
                        return collect(positions)
                    end
                end
            end
        end
    else
        # Baja densidad: distribución aleatoria
        while length(positions) < count
            x = rand(1:dim_x)
            y = rand(1:dim_y)
            z = rand(1:dim_z)
            
            push!(positions, (x, y, z))
        end
    end
    
    return collect(positions)
end

"""
    create_lateral_connections!(layer)

Crea conexiones laterales entre neuronas de la misma capa.
"""
function create_lateral_connections!(layer::CorticalLayer)
    # Lista de neuronas
    neurons = collect(values(layer.neurons))
    
    # Para cada par de neuronas
    for i in 1:length(neurons)
        for j in (i+1):length(neurons)
            # Calcular probabilidad basada en distancia
            dist = neuron_distance(neurons[i], neurons[j])
            prob = compute_connection_probability(
                dist, 
                layer.config.lateral_connectivity
            )
            
            # Decidir si crear conexión
            if rand() < prob
                # Crear conexión bidireccional
                create_lateral_connection!(layer, neurons[i], neurons[j])
                create_lateral_connection!(layer, neurons[j], neurons[i])
            end
        end
    end
    
    return layer
end

"""
    create_lateral_connection!(layer, source, target)

Crea una conexión lateral entre dos neuronas.
"""
function create_lateral_connection!(
    layer::CorticalLayer,
    source::Tensor_Neuron,
    target::Tensor_Neuron
)
    # Determinar tipo de conexión (excitatorio o inhibitorio)
    # Las conexiones laterales tienden a ser inhibitorias a mayor distancia
    dist = neuron_distance(source, target)
    max_dist = sqrt(sum(layer.config.dimensions .^ 2))
    rel_dist = dist / max_dist
    
    if rand() < 0.2 + 0.6 * rel_dist  # 20% a 80% de probabilidad de ser inhibitoria según distancia
        connection_type = :inhibitory
    else
        connection_type = :excitatory
    end
    
    # Crear conexión
    connection = TensorConnection(
        source,
        target,
        connection_type=connection_type
    )
    
    # Añadir a conexiones laterales
    push!(layer.lateral_connections, connection)
    
    return connection
end

"""
    neuron_distance(neuron1, neuron2)

Calcula la distancia euclidiana entre dos neuronas.
"""
function neuron_distance(neuron1::Tensor_Neuron, neuron2::Tensor_Neuron)
    return sqrt(
        (neuron1.position[1] - neuron2.position[1])^2 +
        (neuron1.position[2] - neuron2.position[2])^2 +
        (neuron1.position[3] - neuron2.position[3])^2
    )
end

"""
    compute_connection_probability(distance, base_probability)

Calcula la probabilidad de conexión basada en la distancia.
"""
function compute_connection_probability(
    distance::Float64,
    base_probability::Float32
)
    # Probabilidad decae exponencialmente con la distancia
    return base_probability * exp(-distance / 5.0)
end

"""
    connect_feedforward!(source_layer, target_layer)

Establece conexiones feed-forward entre dos capas.
"""
function connect_feedforward!(
    source_layer::CorticalLayer,
    target_layer::CorticalLayer
)
    # Establecer referencias de capas
    source_layer.next_layer = target_layer
    target_layer.previous_layer = source_layer
    
    # Obtener densidad de conexiones
    density = target_layer.config.feedforward_density
    
    # Para cada neurona en la capa destino
    for target_neuron in values(target_layer.neurons)
        # Seleccionar un subconjunto de neuronas fuente basado en la densidad
        source_neurons = collect(values(source_layer.neurons))
        num_connections = max(1, round(Int, length(source_neurons) * density))
        
        # Seleccionar neuronas fuente con sesgo hacia la misma posición relativa
        source_probs = compute_positional_bias(target_neuron, source_neurons, source_layer.config.dimensions)
        
        # Muestrear según probabilidades
        selected_indices = sample_with_weights(source_probs, num_connections)
        
        # Crear conexiones
        for idx in selected_indices
            source_neuron = source_neurons[idx]
            
            # Crear conexión feed-forward (mayor probabilidad de ser excitatoria)
            if rand() < 0.8  # 80% probabilidad de conexión excitatoria
                connection_type = :excitatory
            else
                connection_type = :inhibitory
            end
            
            connection = TensorConnection(
                source_neuron,
                target_neuron,
                connection_type=connection_type
            )
            
            push!(target_layer.feedforward_connections, connection)
        end
    end
    
    return target_layer
end

"""
    connect_feedback!(source_layer, target_layer)

Establece conexiones de retroalimentación entre dos capas.
"""
function connect_feedback!(
    source_layer::CorticalLayer,
    target_layer::CorticalLayer
)
    # Obtener densidad de conexiones
    density = source_layer.config.feedback_density
    
    # Para cada neurona en la capa fuente
    for source_neuron in values(source_layer.neurons)
        # Seleccionar un subconjunto de neuronas destino basado en la densidad
        target_neurons = collect(values(target_layer.neurons))
        num_connections = max(1, round(Int, length(target_neurons) * density))
        
        # Seleccionar neuronas destino con sesgo hacia la misma posición relativa
        target_probs = compute_positional_bias(source_neuron, target_neurons, target_layer.config.dimensions)
        
        # Muestrear según probabilidades
        selected_indices = sample_with_weights(target_probs, num_connections)
        
        # Crear conexiones
        for idx in selected_indices
            target_neuron = target_neurons[idx]
            
            # Crear conexión feedback (mayor probabilidad de ser modulatoria/inhibitoria)
            if rand() < 0.6  # 60% probabilidad de conexión inhibitoria
                connection_type = :inhibitory
            else
                connection_type = :excitatory
            end
            
            connection = TensorConnection(
                source_neuron,
                target_neuron,
                connection_type=connection_type
            )
            
            push!(source_layer.feedback_connections, connection)
        end
    end
    
    return source_layer
end

"""
    compute_positional_bias(reference_neuron, neurons, dimensions)

Calcula un sesgo posicional para conexiones basado en la posición relativa.
"""
function compute_positional_bias(
    reference_neuron::Tensor_Neuron,
    neurons::Vector{Tensor_Neuron},
    dimensions::NTuple{3,Int}
)
    # Calcular probabilidades basadas en posición relativa
    probabilities = Float64[]
    
    for neuron in neurons
        # Calcular distancia relativa normalizada
        rel_dist = sqrt(
            ((reference_neuron.position[1] / dimensions[1]) - (neuron.position[1] / dimensions[1]))^2 +
            ((reference_neuron.position[2] / dimensions[2]) - (neuron.position[2] / dimensions[2]))^2 +
            ((reference_neuron.position[3] / dimensions[3]) - (neuron.position[3] / dimensions[3]))^2
        )
        
        # Probabilidad inversamente proporcional a la distancia
        prob = exp(-5.0 * rel_dist)
        push!(probabilities, prob)
    end
    
    # Normalizar probabilidades
    if sum(probabilities) > 0
        probabilities ./= sum(probabilities)
    else
        # Si todas las probabilidades son cero, usar distribución uniforme
        probabilities = fill(1.0 / length(neurons), length(neurons))
    end
    
    return probabilities
end

"""
    sample_with_weights(weights, count)

Muestrea índices con probabilidades proporcionales a los pesos.
"""
function sample_with_weights(weights::Vector{Float64}, count::Int)
    # Número total de elementos
    n = length(weights)
    
    if count >= n
        # Devolver todos los índices
        return collect(1:n)
    end
    
    # Muestreo con reemplazo
    indices = Set{Int}()
    
    # Convertir pesos a distribución acumulativa
    cum_weights = cumsum(weights)
    
    while length(indices) < count
        # Generar número aleatorio uniforme
        r = rand() * cum_weights[end]
        
        # Encontrar índice correspondiente
        idx = findfirst(w -> w >= r, cum_weights)
        
        if !isnothing(idx)
            push!(indices, idx)
        end
    end
    
    return collect(indices)
end

"""
    forward_pass!(layer, input_tensor)

Realiza una pasada hacia adelante a través de la capa.
"""
function forward_pass!(
    layer::CorticalLayer,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Asegurar dimensiones compatibles
    if size(input_tensor) != layer.config.dimensions
        input_tensor = tensor_interpolation(input_tensor, layer.config.dimensions)
    end
    
    # Aplicar mapa de atención de entrada
    attended_input = apply_attention(input_tensor, layer.attention_maps[:input])
    
    # Inicializar estado de activación
    layer.activation_state = zeros(Float32, layer.config.dimensions)
    
    # Para cada neurona, procesar input
    for (pos, neuron) in layer.neurons
        # Procesar input para esta neurona
        activation = process_input(neuron, attended_input)
        
        # Actualizar estado de activación en la posición de la neurona
        layer.activation_state[pos...] = mean(activation)
    end
    
    # Procesar conexiones laterales
    process_lateral_connections!(layer)
    
    # Aplicar función de activación global
    layer.activation_state = volumetric_activation(
        layer.activation_state, 
        type=layer.config.activation_type
    )
    
    # Guardar en historial
    push!(layer.activation_history, copy(layer.activation_state))
    
    # Actualizar mapa de atención de salida
    update_output_attention!(layer)
    
    return layer.activation_state
end

"""
    process_lateral_connections!(layer)

Procesa conexiones laterales dentro de la capa.
"""
function process_lateral_connections!(layer::CorticalLayer)
    # No procesar si no hay conexiones laterales
    if isempty(layer.lateral_connections)
        return
    end
    
    # Copia del estado actual para evitar actualizaciones simultáneas
    current_state = copy(layer.activation_state)
    
    # Para cada conexión lateral
    for connection in layer.lateral_connections
        # Encontrar neuronas origen y destino
        source_pos = nothing
        target_pos = nothing
        
        for (pos, neuron) in layer.neurons
            if neuron.id == connection.source_id
                source_pos = pos
            elseif neuron.id == connection.target_id
                target_pos = pos
            end
            
            if !isnothing(source_pos) && !isnothing(target_pos)
                break
            end
        end
        
        if !isnothing(source_pos) && !isnothing(target_pos)
            # Obtener activación de origen
            source_activation = current_state[source_pos...]
            
            # Calcular influencia
            influence = source_activation * connection.strength
            
            # Invertir para conexiones inhibitorias
            if connection.connection_type == :inhibitory
                influence = -influence
            end
            
            # Aplicar a neurona destino
            layer.activation_state[target_pos...] += influence
        end
    end
    
    return layer.activation_state
end

"""
    update_output_attention!(layer)

Actualiza el mapa de atención de salida basado en el estado de activación.
"""
function update_output_attention!(layer::CorticalLayer)
    # Crear mapa de atención basado en activación
    layer.attention_maps[:output] = create_attention_from_activity(
        layer.activation_state,
        threshold=0.3f0
    )
    
    return layer.attention_maps[:output]
end

"""
    backward_pass!(layer, output_gradient)

Realiza una pasada hacia atrás a través de la capa.
"""
function backward_pass!(
    layer::CorticalLayer,
    output_gradient::Array{T,3}
) where T <: AbstractFloat
    # Asegurar dimensiones compatibles
    if size(output_gradient) != layer.config.dimensions
        output_gradient = tensor_interpolation(output_gradient, layer.config.dimensions)
    end
    
    # Para cada neurona, calcular gradiente
    neuron_gradients = Dict{UUID, Array{Float32,3}}()
    
    for (pos, neuron) in layer.neurons
        # Extraer gradiente local para esta neurona
        local_gradient = extract_local_gradient(output_gradient, pos)
        
        # Calcular gradiente para la neurona
        neuron_gradient = compute_neuron_gradient(neuron, local_gradient, layer.config.plasticity_factor)
        
        # Guardar gradiente
        neuron_gradients[neuron.id] = neuron_gradient
    end
    
    # Calcular gradientes para conexiones laterales
    connection_gradients = Dict{UUID, Float32}()
    
    for connection in layer.lateral_connections
        # Encontrar gradientes de neuronas origen y destino
        source_gradient = get(neuron_gradients, connection.source_id, nothing)
        target_gradient = get(neuron_gradients, connection.target_id, nothing)
        
        if !isnothing(source_gradient) && !isnothing(target_gradient)
            # Calcular gradiente de conexión
            conn_gradient = compute_connection_gradient(
                connection, 
                source_gradient, 
                target_gradient,
                layer.config.plasticity_factor
            )
            
            # Guardar gradiente
            connection_gradients[connection.id] = conn_gradient
        end
    end
    
    return neuron_gradients, connection_gradients
end

"""
    extract_local_gradient(global_gradient, position)

Extrae gradiente local para una neurona en la posición dada.
"""
function extract_local_gradient(
    global_gradient::Array{T,3},
    position::NTuple{3,Int}
) where T <: AbstractFloat
    # Radio para extracción local
    radius = 1
    dim_x, dim_y, dim_z = size(global_gradient)
    
    # Calcular límites de la región
    x_min = max(1, position[1] - radius)
    y_min = max(1, position[2] - radius)
    z_min = max(1, position[3] - radius)
    
    x_max = min(dim_x, position[1] + radius)
    y_max = min(dim_y, position[2] + radius)
    z_max = min(dim_z, position[3] + radius)
    
    # Extraer región local del gradiente
    local_gradient = global_gradient[x_min:x_max, y_min:y_max, z_min:z_max]
    
    return local_gradient
end

"""
    compute_neuron_gradient(neuron, local_gradient, plasticity_factor)

Computa gradiente para una neurona específica.
"""
function compute_neuron_gradient(
    neuron::Tensor_Neuron,
    local_gradient::Array{T,3},
    plasticity_factor::Float32
) where T <: AbstractFloat
    # Modificar gradiente según plasticidad y especialización
    effective_plasticity = plasticity_factor * (1.0f0 - neuron.specialization)
    
    # Combinar con gradiente local
    neuron_gradient = local_gradient .* effective_plasticity
    
    return neuron_gradient
end

"""
    compute_connection_gradient(connection, source_gradient, target_gradient, plasticity_factor)

Computa gradiente para una conexión específica.
"""
function compute_connection_gradient(
    connection::Tensor_Connection,
    source_gradient::Array{T,3},
    target_gradient::Array{S,3},
    plasticity_factor::Float32
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Calcular gradiente como producto escalar de gradientes origen y destino
    dot_product = sum(source_gradient .* target_gradient) / (length(source_gradient) * length(target_gradient))
    
    # Aplicar factor de plasticidad
    connection_gradient = dot_product * plasticity_factor
    
    # Para conexiones inhibitorias, invertir signo
    if connection.connection_type == :inhibitory
        connection_gradient *= -1
    end
    
    return connection_gradient
end

"""
    propagate_to_next_layer!(layer, input_tensor=nothing)

Propaga activación a la siguiente capa.
"""
function propagate_to_next_layer!(
    layer::CorticalLayer,
    input_tensor::Union{Array{T,3}, Nothing}=nothing
) where T <: AbstractFloat
    # Si se proporciona entrada, procesarla primero
    if !isnothing(input_tensor)
        forward_pass!(layer, input_tensor)
    end
    
    # Verificar que hay una capa siguiente
    if isnothing(layer.next_layer)
        return nothing
    end
    
    # Aplicar atención de salida
    attended_output = apply_attention(layer.activation_state, layer.attention_maps[:output])
    
    # Propagar a siguiente capa
    return forward_pass!(layer.next_layer, attended_output)
end

"""
    create_cortical_column(layer_configs)

Crea una columna cortical con múltiples capas.
"""
function create_cortical_column(layer_configs::Vector{CorticalLayerConfig})
    layers = CorticalLayer[]
    
    # Crear cada capa
    for (i, config) in enumerate(layer_configs)
        layer = CorticalLayer("Layer_$i", config)
        initialize_neurons!(layer)
        create_lateral_connections!(layer)
        
        push!(layers, layer)
    end
    
    # Conectar capas
    for i in 1:length(layers)-1
        connect_feedforward!(layers[i], layers[i+1])
        connect_feedback!(layers[i+1], layers[i])
    end
    
    return layers
end

# Exportar tipos y funciones principales
export CorticalLayerConfig, CorticalLayer,
       initialize_neurons!, create_lateral_connections!,
       connect_feedforward!, connect_feedback!,
       forward_pass!, backward_pass!, propagate_to_next_layer!,
       create_cortical_column

end # module CorticalLayers