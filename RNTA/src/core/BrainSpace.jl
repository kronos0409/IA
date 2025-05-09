# core/BrainSpace.jl
# Define el espacio tridimensional computacional

"""
    BrainSpace

Representa el espacio tridimensional que contiene y organiza las neuronas tensoriales.
"""
# Añadir al inicio del archivo:
module BrainSpace

using UUIDs
using ..TensorNeuron
using ..Connections

export Brain_Space, forward_propagation, propagate_layer!, update_attention_map!, update_global_state!, expand_space!, expand_region!, establish_new_connections!, identify_expansion_regions, find_activity_clusters, count_neurons_in_region, visualize_activity, prepare_input_tensor, self_prune!, should_expand_space, clone_brain

struct ExpansionEvent
    # Timestamp del evento de expansión
    timestamp::Float64
    
    # Región que fue expandida
    region::NTuple{3,UnitRange{Int}}
    
    # Dimensiones antes de la expansión
    previous_dimensions::NTuple{3,Int}
    
    # Dimensiones después de la expansión
    new_dimensions::NTuple{3,Int}
    
    # Número de neuronas añadidas
    neurons_added::Int
    
    # Razón de la expansión (por ejemplo, :high_activity, :saturation, etc.)
    reason::Symbol
    
    # Métricas adicionales sobre el evento
    metrics::Dict{Symbol, Any}
end # Fin del módulo BrainSpace

mutable struct Brain_Space
    # Dimensiones del espacio
    dimensions::NTuple{3,Int}
    
    # Matriz 3D de neuronas tensoriales (posiblemente sparse)
    neurons::Dict{NTuple{3,Int}, Tensor_Neuron}
    
    # Conexiones entre neuronas
    connections::Vector{Tensor_Connection}
    
    # Regiones funcionales (áreas especializadas que emergen)
    functional_regions::Dict{Symbol, Vector{NTuple{3,Int}}}
    
    # Estado global del espacio (tensor que representa la actividad)
    global_state::Array{Float32,3}
    
    # Sistema atencional que modula la actividad
    attention_map::Array{Float32,3}
    
    # Historial de expansiones para análisis
    expansion_history::Vector{ExpansionEvent}
    
    # Configuración del espacio
    config::BrainSpaceConfig
end

"""
Constructor principal para BrainSpace
"""
function Brain_Space(
    dim_x::Int, 
    dim_y::Int, 
    dim_z::Int;
    config::BrainSpaceConfig=BrainSpaceConfig()
)
    dimensions = (dim_x, dim_y, dim_z)
    
    # Inicializar diccionario vacío de neuronas
    neurons = Dict{NTuple{3,Int}, Tensor_Neuron}()
    
    # Poblar el espacio con neuronas iniciales (según la densidad configurada)
    populate_initial_neurons!(neurons, dimensions, config)
    
    # Inicializar conexiones vacías
    connections = Vector{Tensor_Connection}()
    
    # Establecer conexiones iniciales
    establish_connections!(connections, neurons, config)
    
    # Inicializar regiones funcionales vacías
    functional_regions = Dict{Symbol, Vector{NTuple{3,Int}}}()
    
    # Inicializar estado global
    global_state = zeros(Float32, dimensions)
    
    # Inicializar mapa de atención (inicialmente uniforme)
    attention_map = ones(Float32, dimensions)
    
    # Inicializar historial de expansiones
    expansion_history = Vector{ExpansionEvent}()
    
    return Brain_Space(
        dimensions,
        neurons,
        connections,
        functional_regions,
        global_state,
        attention_map,
        expansion_history,
        config
    )
end

"""
    populate_initial_neurons!(neurons, dimensions, config)

Puebla el espacio cerebral con neuronas iniciales.
"""
function populate_initial_neurons!(
    neurons::Dict{NTuple{3,Int}, Tensor_Neuron},
    dimensions::NTuple{3,Int},
    config::BrainSpaceConfig
)
    # Calcular número esperado de neuronas iniciales
    total_volume = dimensions[1] * dimensions[2] * dimensions[3]
    target_neurons = round(Int, total_volume * config.initial_density)
    
    # Distribución espacial de neuronas
    positions = sample_positions(dimensions, target_neurons)
    
    # Crear neuronas en las posiciones seleccionadas
    for pos in positions
        # Tamaño de campo receptivo inicial
        receptive_field_size = (3, 3, 3)
        
        # Crear neurona
        neuron = TensorNeuron.Tensor_Neuron(
        pos, 
        receptive_field_size,
        init_scale=config.init_scale
        )
        
        # Añadir a diccionario
        neurons[pos] = neuron
    end
    
    return neurons
end

"""
    sample_positions(dimensions, n)

Muestrea n posiciones únicas dentro de las dimensiones dadas.
"""
function sample_positions(dimensions::NTuple{3,Int}, n::Int)
    dim_x, dim_y, dim_z = dimensions
    
    # Conjunto para asegurar posiciones únicas
    positions = Set{NTuple{3,Int}}()
    
    while length(positions) < n
        x = rand(1:dim_x)
        y = rand(1:dim_y)
        z = rand(1:dim_z)
        
        push!(positions, (x, y, z))
    end
    
    return collect(positions)
end

"""
    forward_propagation(brain, input_tensor)

Propaga un tensor de entrada a través del espacio cerebral.
"""
function forward_propagation(brain::Brain_Space, input_tensor::Array{T,3}) where T <: AbstractFloat
    # Asegurarse de que el tensor de entrada tiene dimensiones compatibles
    input_tensor = prepare_input_tensor(input_tensor, brain.dimensions)
    
    # Aplicar atención de entrada
    attended_input = input_tensor .* brain.attention_map
    
    # Crear un mapa de activaciones para cada neurona
    neuron_activations = Dict{UUID, Array{Float32,3}}()
    
    # Procesar capas en orden secuencial
    for layer in 1:brain.config.propagation_layers
        # Propagar a través de conexiones
        propagate_layer!(brain, neuron_activations, attended_input, layer)
        
        # Actualizar mapa de atención basado en activaciones actuales
        update_attention_map!(brain, neuron_activations)
    end
    
    # Actualizar estado global del cerebro
    update_global_state!(brain, neuron_activations)
    
    return brain.global_state
end

"""
    propagate_layer!(brain, neuron_activations, input_tensor, layer)

Propaga activaciones a través de una capa específica de neuronas.
"""
function propagate_layer!(
    brain::Brain_Space,
    neuron_activations::Dict{UUID, Array{Float32,3}},
    input_tensor::Array{T,3},
    layer::Int
) where T <: AbstractFloat
    # Obtener neuronas para esta capa (simplificado, en realidad dependería de la topología)
    layer_neurons = values(brain.neurons)
    
    # Procesar cada neurona
    for neuron in layer_neurons
        # Procesar input
        activation = process_input(neuron, input_tensor)
        
        # Guardar activación
        neuron_activations[neuron.id] = activation
    end
    
    # Propagar a través de conexiones
    for connection in brain.connections
        # Obtener activación de neurona origen
        if !haskey(neuron_activations, connection.source_id)
            continue
        end
        
        source_activation = neuron_activations[connection.source_id]
        
        # Transmitir a través de la conexión
        transmitted = transmit(connection, source_activation)
        
        # Sumar a la activación de neurona destino
        if !haskey(neuron_activations, connection.target_id)
            neuron_activations[connection.target_id] = transmitted
        else
            neuron_activations[connection.target_id] .+= transmitted
        end
    end
end

"""
    update_attention_map!(brain, neuron_activations)

Actualiza el mapa de atención basado en activaciones actuales.
"""
function update_attention_map!(
    brain::Brain_Space,
    neuron_activations::Dict{UUID, Array{Float32,3}}
)
    # Reiniciar mapa de atención
    brain.attention_map .= 0.1f0  # Base mínima de atención
    
    # Para cada neurona, contribuir al mapa de atención
    for (neuron_id, activation) in neuron_activations
        # Encontrar neurona correspondiente
        neuron_pos = nothing
        for (pos, neuron) in brain.neurons
            if neuron.id == neuron_id
                neuron_pos = pos
                break
            end
        end
        
        if neuron_pos === nothing
            continue
        end
        
        # Calcular contribución de esta neurona al mapa de atención
        activity_level = sum(abs.(activation)) / length(activation)
        
        # Añadir contribución en posición de neurona y vecindario
        update_attention_region!(brain.attention_map, neuron_pos, activity_level)
    end
    
    # Normalizar mapa de atención
    max_attention = maximum(brain.attention_map)
    if max_attention > 0
        brain.attention_map ./= max_attention
    end
end

"""
    update_attention_region!(attention_map, position, activity_level)

Actualiza una región del mapa de atención alrededor de una posición.
"""
function update_attention_region!(
    attention_map::Array{Float32,3},
    position::NTuple{3,Int},
    activity_level::Float32,
    radius::Int=2
)
    dim_x, dim_y, dim_z = size(attention_map)
    
    # Iterar sobre región
    for dx in -radius:radius
        for dy in -radius:radius
            for dz in -radius:radius
                # Calcular posición
                x = position[1] + dx
                y = position[2] + dy
                z = position[3] + dz
                
                # Verificar límites
                if 1 <= x <= dim_x && 1 <= y <= dim_y && 1 <= z <= dim_z
                    # Distancia al centro
                    distance = sqrt(dx^2 + dy^2 + dz^2)
                    
                    # Factor de atenuación por distancia
                    attenuation = exp(-distance)
                    
                    # Actualizar mapa de atención
                    attention_map[x, y, z] += activity_level * attenuation
                end
            end
        end
    end
end

"""
    update_global_state!(brain, neuron_activations)

Actualiza el estado global del cerebro basado en activaciones de neuronas.
"""
function update_global_state!(
    brain::Brain_Space,
    neuron_activations::Dict{UUID, Array{Float32,3}}
)
    # Reiniciar estado global
    brain.global_state .= 0.0f0
    
    # Para cada neurona, contribuir al estado global
    for (neuron_id, activation) in neuron_activations
        # Encontrar neurona correspondiente
        neuron_pos = nothing
        for (pos, neuron) in brain.neurons
            if neuron.id == neuron_id
                neuron_pos = pos
                break
            end
        end
        
        if neuron_pos === nothing
            continue
        end
        
        # Añadir contribución al estado global
        update_global_state_region!(brain.global_state, neuron_pos, activation)
    end
end

"""
    update_global_state_region!(global_state, position, activation)

Actualiza una región del estado global alrededor de una posición.
"""
function update_global_state_region!(
    global_state::Array{Float32,3},
    position::NTuple{3,Int},
    activation::Array{Float32,3}
)
    dim_x, dim_y, dim_z = size(global_state)
    act_dim_x, act_dim_y, act_dim_z = size(activation)
    
    # Calcular regiones de inicio y fin
    half_x = div(act_dim_x, 2)
    half_y = div(act_dim_y, 2)
    half_z = div(act_dim_z, 2)
    
    start_x = max(1, position[1] - half_x)
    start_y = max(1, position[2] - half_y)
    start_z = max(1, position[3] - half_z)
    
    end_x = min(dim_x, position[1] + half_x)
    end_y = min(dim_y, position[2] + half_y)
    end_z = min(dim_z, position[3] + half_z)
    
    # Calcular regiones correspondientes en tensor de activación
    act_start_x = 1 + max(0, start_x - (position[1] - half_x))
    act_start_y = 1 + max(0, start_y - (position[2] - half_y))
    act_start_z = 1 + max(0, start_z - (position[3] - half_z))
    
    act_end_x = act_dim_x - max(0, (position[1] + half_x) - end_x)
    act_end_y = act_dim_y - max(0, (position[2] + half_y) - end_y)
    act_end_z = act_dim_z - max(0, (position[3] + half_z) - end_z)
    
    # Actualizar estado global
    global_state[start_x:end_x, start_y:end_y, start_z:end_z] .+= 
        activation[act_start_x:act_end_x, act_start_y:act_end_y, act_start_z:act_end_z]
end

"""
    expand_space!(brain, regions=nothing)

Expande el espacio cerebral, añadiendo nuevas neuronas y conexiones.
"""
function expand_space!(brain::Brain_Space, regions=nothing)
    # Si no se especifican regiones, determinar automáticamente
    if isnothing(regions)
        regions = identify_expansion_regions(brain)
    end
    
    # Para cada región a expandir
    for region in regions
        # Guardar dimensiones previas
        previous_dimensions = brain.dimensions
        
        # Expandir región
        neurons_before = length(brain.neurons)
        expand_region!(brain, region)
        neurons_after = length(brain.neurons)
        neurons_added = neurons_after - neurons_before
        
        # Registrar evento de expansión
        push!(brain.expansion_history, ExpansionEvent(
            time(),                # timestamp
            region,                # region
            previous_dimensions,   # previous_dimensions 
            brain.dimensions,      # new_dimensions
            neurons_added,         # neurons_added
            :high_activity,        # reason (podría ser más específico según el caso)
            Dict{Symbol, Any}()    # metrics
        ))
    end
    
    return brain
end

"""
    expand_region!(brain, region)

Expande una región específica del espacio cerebral.
"""
function expand_region!(brain::Brain_Space, region::NTuple{3,UnitRange{Int}})
    # Calcular dimensiones actuales de la región
    region_dim_x = length(region[1])
    region_dim_y = length(region[2])
    region_dim_z = length(region[3])
    
    # Calcular nuevas dimensiones expandidas
    new_dim_x = ceil(Int, region_dim_x * brain.config.expansion_factor)
    new_dim_y = ceil(Int, region_dim_y * brain.config.expansion_factor)
    new_dim_z = ceil(Int, region_dim_z * brain.config.expansion_factor)
    
    # Calcular posiciones para nuevas neuronas
    new_positions = Set{NTuple{3,Int}}()
    
    # Añadir nuevas posiciones hasta alcanzar la densidad deseada
    target_neurons = round(Int, new_dim_x * new_dim_y * new_dim_z * brain.config.initial_density)
    
    while length(new_positions) < target_neurons
        # Generar posición aleatoria dentro de la región expandida
        x = rand(region[1].start:region[1].stop)
        y = rand(region[2].start:region[2].stop)
        z = rand(region[3].start:region[3].stop)
        
        pos = (x, y, z)
        
        # Verificar que la posición no esté ya ocupada
        if !haskey(brain.neurons, pos)
            push!(new_positions, pos)
        end
    end
    
    # Crear nuevas neuronas en las posiciones seleccionadas
    for pos in new_positions
        # Crear neurona
        neuron = Tensor_Neuron(
            pos, 
            (3, 3, 3),
            init_scale=brain.config.init_scale
        )
        
        # Añadir a diccionario
        brain.neurons[pos] = neuron
    end
    
    # Establecer conexiones para las nuevas neuronas
    establish_new_connections!(brain)
end

"""
    establish_new_connections!(brain)

Establece conexiones para las neuronas recién añadidas.
"""
function establish_new_connections!(brain::Brain_Space)
    # Identificar neuronas sin conexiones salientes
    unconnected_neurons = Tensor_Neuron[]
    
    for neuron in values(brain.neurons)
        # Verificar si tiene conexiones salientes
        has_outgoing = false
        
        for connection in brain.connections
            if connection.source_id == neuron.id
                has_outgoing = true
                break
            end
        end
        
        if !has_outgoing
            push!(unconnected_neurons, neuron)
        end
    end
    
    # Para cada neurona no conectada, intentar establecer conexiones
    for source_neuron in unconnected_neurons
        for target_neuron in values(brain.neurons)
            # No conectar consigo misma
            if source_neuron.id == target_neuron.id
                continue
            end
            
            # Calcular probabilidad de conexión
            prob = connection_probability(
                source_neuron,
                target_neuron,
                brain.config.max_connection_radius,
                brain.config.base_connection_probability
            )
            
            # Decidir si establecer conexión
            if rand() < prob
                # Determinar tipo de conexión
                conn_type = rand() < 0.8 ? :excitatory : :inhibitory
                
                # Crear conexión
                connection = Tensor_Connection(
                    source_neuron,
                    target_neuron,
                    connection_type=conn_type
                )
                
                # Añadir a la lista
                push!(brain.connections, connection)
            end
        end
    end
end

"""
    identify_expansion_regions(brain)

Identifica regiones del espacio cerebral que deberían expandirse.
"""
function identify_expansion_regions(brain::Brain_Space)
    # Calcular mapa de actividad
    activity_map = zeros(Float32, brain.dimensions)
    
    # Para cada neurona, contribuir al mapa de actividad
    for (pos, neuron) in brain.neurons
        if should_expand(neuron)
            # Actualizar región alrededor de la neurona
            update_activity_region!(activity_map, pos, 1.0f0)
        end
    end
    
    # Identificar clusters de alta actividad
    regions = find_activity_clusters(activity_map, threshold=0.5f0)
    
    return regions
end

"""
    update_activity_region!(activity_map, position, value)

Actualiza una región del mapa de actividad alrededor de una posición.
"""
function update_activity_region!(
    activity_map::Array{Float32,3},
    position::NTuple{3,Int},
    value::Float32,
    radius::Int=2
)
    dim_x, dim_y, dim_z = size(activity_map)
    
    # Iterar sobre región
    for dx in -radius:radius
        for dy in -radius:radius
            for dz in -radius:radius
                # Calcular posición
                x = position[1] + dx
                y = position[2] + dy
                z = position[3] + dz
                
                # Verificar límites
                if 1 <= x <= dim_x && 1 <= y <= dim_y && 1 <= z <= dim_z
                    # Distancia al centro
                    distance = sqrt(dx^2 + dy^2 + dz^2)
                    
                    # Factor de atenuación por distancia
                    attenuation = exp(-distance)
                    
                    # Actualizar mapa de actividad
                    activity_map[x, y, z] += value * attenuation
                end
            end
        end
    end
end

"""
    find_activity_clusters(activity_map, threshold)

Encuentra clusters de alta actividad en el mapa.
"""
function find_activity_clusters(activity_map::Array{Float32,3}; threshold::Float32=0.5f0)
    # Normalizar mapa
    max_activity = maximum(activity_map)
    if max_activity > 0
        activity_map ./= max_activity
    end
    
    # Crear mapa binario
    binary_map = activity_map .> threshold
    
    # Encontrar componentes conectados (simplificado)
    # En una implementación real usaríamos un algoritmo de clustering apropiado
    regions = NTuple{3,UnitRange{Int}}[]
    
    # Recorrer el mapa en busca de regiones activas
    dim_x, dim_y, dim_z = size(activity_map)
    visited = falses(size(binary_map))
    
    for x in 1:dim_x
        for y in 1:dim_y
            for z in 1:dim_z
                if binary_map[x, y, z] && !visited[x, y, z]
                    # Encontrar región conectada
                    region_bounds = find_connected_region(binary_map, visited, (x, y, z))
                    push!(regions, region_bounds)
                end
            end
        end
    end
    
    return regions
end

"""
    find_connected_region(binary_map, visited, start_pos)

Encuentra una región conectada en el mapa binario.
"""
function find_connected_region(
    binary_map::BitArray{3},
    visited::BitArray{3},
    start_pos::NTuple{3,Int}
)
    # Dimensiones del mapa
    dim_x, dim_y, dim_z = size(binary_map)
    
    # Inicializar límites de la región
    min_x, max_x = start_pos[1], start_pos[1]
    min_y, max_y = start_pos[2], start_pos[2]
    min_z, max_z = start_pos[3], start_pos[3]
    
    # Pila para búsqueda en profundidad
    stack = [start_pos]
    
    # Marcar posición inicial como visitada
    visited[start_pos...] = true
    
    # Direcciones vecinas (6-conectividad)
    neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    # Búsqueda en profundidad
    while !isempty(stack)
        current = pop!(stack)
        x, y, z = current
        
        # Actualizar límites
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)
        
        # Explorar vecinos
        for (dx, dy, dz) in neighbors
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Verificar límites
            if 1 <= nx <= dim_x && 1 <= ny <= dim_y && 1 <= nz <= dim_z
                # Verificar si es parte de la región y no visitado
                if binary_map[nx, ny, nz] && !visited[nx, ny, nz]
                    visited[nx, ny, nz] = true
                    push!(stack, (nx, ny, nz))
                end
            end
        end
    end
    
    # Crear rangos para la región
    region_x = min_x:max_x
    region_y = min_y:max_y
    region_z = min_z:max_z
    
    return (region_x, region_y, region_z)
end

"""
    count_neurons_in_region(brain, region)

Cuenta el número de neuronas en una región específica.
"""
function count_neurons_in_region(
    brain::Brain_Space, 
    region::NTuple{3,UnitRange{Int}}
)
    count = 0
    
    for (pos, _) in brain.neurons
        if pos[1] in region[1] && pos[2] in region[2] && pos[3] in region[3]
            count += 1
        end
    end
    
    return count
end

"""
    visualize_activity(brain; options...)

Genera una visualización del estado actual del espacio cerebral.
"""
function visualize_activity(brain::Brain_Space; options...)
    # Esta función sería una interfaz para el módulo de visualización
    # Por ahora simplemente devolvemos un resumen textual
    
    # Contar neuronas por tipo
    neuron_types = Dict{Symbol, Int}()
    
    for (_, neuron) in brain.neurons
        if !haskey(neuron_types, neuron.functional_type)
            neuron_types[neuron.functional_type] = 0
        end
        
        neuron_types[neuron.functional_type] += 1
    end
    
    # Calcular estadísticas de actividad
    activity_stats = (
        mean = mean(abs.(brain.global_state)),
        max = maximum(abs.(brain.global_state)),
        active_regions = count(x -> abs(x) > 0.5, brain.global_state)
    )
    
    # Calcular estadísticas de conexiones
    connection_stats = (
        total = length(brain.connections),
        excitatory = count(c -> c.connection_type == :excitatory, brain.connections),
        inhibitory = count(c -> c.connection_type == :inhibitory, brain.connections)
    )
    
    return (
        dimensions = brain.dimensions,
        num_neurons = length(brain.neurons),
        neuron_types = neuron_types,
        activity = activity_stats,
        connections = connection_stats,
        expansions = length(brain.expansion_history)
    )
end

"""
    prepare_input_tensor(input_tensor, dimensions)

Prepara un tensor de entrada para que tenga las dimensiones correctas.
"""
function prepare_input_tensor(input_tensor::Array{T}, dimensions::NTuple{3,Int}) where T <: AbstractFloat
    # Si ya tiene las dimensiones correctas, devolver como está
    if size(input_tensor) == dimensions
        return convert(Array{Float32,3}, input_tensor)
    end
    
    # Si es un tensor 2D, expandir a 3D
    if ndims(input_tensor) == 2
        expanded = zeros(Float32, dimensions)
        expanded[:, :, 1] = input_tensor
        return expanded
    end
    
    # Si es un tensor 1D, expandir a 3D
    if ndims(input_tensor) == 1
        expanded = zeros(Float32, dimensions)
        expanded[:, 1, 1] = input_tensor
        return expanded
    end
    
    # Si es 3D pero con dimensiones diferentes, redimensionar
    return tensor_interpolation(convert(Array{Float32,3}, input_tensor), dimensions)
end

"""
    self_prune!(brain)

Realiza auto-optimización eliminando conexiones innecesarias.
"""
function self_prune!(brain::Brain_Space)
    # Umbral de poda basado en actividad
    threshold = 0.01f0
    
    # Eliminar conexiones poco activas
    prune_connections!(brain.connections, threshold)
    
    return brain
end

"""
    should_expand_space(brain)

Determina si el espacio cerebral debería expandirse.
"""
function should_expand_space(brain::Brain_Space)
    # Contar neuronas candidatas para expansión
    expansion_candidates = 0
    
    for (_, neuron) in brain.neurons
        if should_expand(neuron)
            expansion_candidates += 1
        end
    end
    
    # Calcular porcentaje
    percentage = expansion_candidates / length(brain.neurons)
    
    # Expandir si más del 20% de neuronas están saturadas
    return percentage > 0.2
end

"""
    clone_brain(brain)

Crea una copia profunda del espacio cerebral para deliberación interna.
"""
function clone_brain(brain::Brain_Space)
    # Clonar neuronas
    cloned_neurons = Dict{NTuple{3,Int}, Tensor_Neuron}()
    
    for (pos, neuron) in brain.neurons
        cloned_neurons[pos] = clone(neuron)
    end
    
    # Clonar conexiones
    cloned_connections = Vector{Tensor_Connection}(undef, length(brain.connections))
    
    for (i, connection) in enumerate(brain.connections)
        # Encontrar neuronas correspondientes en el clon
        source_neuron = nothing
        target_neuron = nothing
        
        for (_, neuron) in cloned_neurons
            if neuron.id == connection.source_id
                source_neuron = neuron
            elseif neuron.id == connection.target_id
                target_neuron = neuron
            end
            
            if !isnothing(source_neuron) && !isnothing(target_neuron)
                break
            end
        end
        
        if !isnothing(source_neuron) && !isnothing(target_neuron)
            # Crear conexión clonada
            cloned_connection = Tensor_Connection(
                source_neuron,
                target_neuron,
                weight_size=size(connection.weight),
                connection_type=connection.connection_type
            )
            
            # Copiar peso y otros atributos
            cloned_connection.weight = copy(connection.weight)
            cloned_connection.strength = connection.strength
            cloned_connection.activity_history = deepcopy(connection.activity_history)
            
            cloned_connections[i] = cloned_connection
        end
    end
    
    # Clonar regiones funcionales
    cloned_regions = Dict{Symbol, Vector{NTuple{3,Int}}}()
    
    for (type, positions) in brain.functional_regions
        cloned_regions[type] = copy(positions)
    end
    
    # Crear cerebro clonado
    return Brain_Space(
        brain.dimensions,
        cloned_neurons,
        cloned_connections,
        cloned_regions,
        copy(brain.global_state),
        copy(brain.attention_map),
        copy(brain.expansion_history),
        brain.config  # La configuración es inmutable, no necesita copiarse
    )
end
end # Fin del módulo BrainSpace