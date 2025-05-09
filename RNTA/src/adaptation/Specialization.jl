# adaptation/Specialization.jl
# Implementa mecanismos de especialización de neuronas tensoriales

module Specialization

using Statistics
using LinearAlgebra
using DataStructures
using Random

# Importaciones de otros módulos de RNTA
using ..PlasticityRules
using ..TensorNeuron
using ..BrainSpace
using ..Connections
"""
    SpecializationParameters

Parámetros que controlan el proceso de especialización neuronal.
"""
struct SpecializationParameters
    # Tasa base de especialización
    base_rate::Float32
    
    # Factor de decaimiento con la edad
    age_decay::Float32
    
    # Umbral para cambio de tipo funcional
    type_change_threshold::Float32
    
    # Factor de impacto de actividad consistente
    consistency_factor::Float32
    
    # Peso para patrones espaciales
    spatial_weight::Float32
    
    # Peso para patrones temporales
    temporal_weight::Float32
    
    # Peso para patrones de características
    feature_weight::Float32
    
    # Peso para patrones contextuales
    context_weight::Float32
end

# Constructor con valores por defecto
function SpecializationParameters(;
    base_rate::Float32=0.001f0,
    age_decay::Float32=0.2f0,
    type_change_threshold::Float32=0.7f0,
    consistency_factor::Float32=2.0f0,
    spatial_weight::Float32=1.0f0,
    temporal_weight::Float32=1.0f0,
    feature_weight::Float32=1.0f0,
    context_weight::Float32=1.0f0
)
    return SpecializationParameters(
        base_rate,
        age_decay,
        type_change_threshold,
        consistency_factor,
        spatial_weight,
        temporal_weight,
        feature_weight,
        context_weight
    )
end

"""
    specialize_neurons!(brain, params=SpecializationParameters())

Aplica el proceso de especialización a todas las neuronas del cerebro.
"""
function specialize_neurons!(
    brain::Brain_Space,
    params::SpecializationParameters=SpecializationParameters()
)
    # Contador de cambios de tipo
    type_changes = 0
    
    # Para cada neurona, evaluar y actualizar especialización
    for (_, neuron) in brain.neurons
        # Analizar patrones de activación
        updated = update_specialization!(neuron, params)
        
        # Contar cambios de tipo
        if updated
            type_changes += 1
        end
    end
    
    # Registrar métricas de especialización para el cerebro
    specialization_stats = analyze_brain_specialization(brain)
    
    return type_changes, specialization_stats
end

"""
    update_specialization!(neuron, params)

Actualiza el nivel de especialización y posiblemente el tipo funcional de una neurona.
"""
function update_specialization!(
    neuron::Tensor_Neuron,
    params::SpecializationParameters
)
    # Analizar patrones recientes de activación
    patterns = analyze_activation_patterns(neuron.activation_history)
    
    # Calcular afinidad para cada tipo funcional basada en patrones
    affinities = Dict{Symbol, Float32}()
    
    affinities[:spatial] = patterns.spatial_sensitivity * params.spatial_weight
    affinities[:temporal] = patterns.temporal_sensitivity * params.temporal_weight
    affinities[:feature] = patterns.feature_sensitivity * params.feature_weight
    affinities[:contextual] = patterns.context_sensitivity * params.context_weight
    affinities[:general] = 0.5f0  # Valor base para tipo general
    
    # Ajustar por consistencia de patrones
    consistency = calculate_activation_consistency(neuron.activation_history)
    consistency_multiplier = 1.0f0 + (consistency * params.consistency_factor)
    
    # Tipo funcional actual tiene ventaja (inercia)
    if haskey(affinities, neuron.functional_type)
        affinities[neuron.functional_type] *= 1.2f0
    end
    
    # Encontrar tipo con mayor afinidad
    best_type, best_affinity = :general, affinities[:general]
    
    for (type, affinity) in affinities
        if affinity > best_affinity
            best_type = type
            best_affinity = affinity
        end
    end
    
    # Decidir si cambiar tipo funcional
    type_changed = false
    
    # Si la afinidad supera el umbral y es diferente del tipo actual
    if best_affinity > params.type_change_threshold && best_type != neuron.functional_type
        # Probabilidad de cambio inversamente proporcional al nivel de especialización
        change_probability = 1.0f0 - neuron.specialization
        
        if rand() < change_probability
            neuron.functional_type = best_type
            type_changed = true
        end
    end
    
    # Incrementar especialización basado en consistencia y afinidad
    specialization_increment = params.base_rate * consistency_multiplier * best_affinity
    
    # Aplicar decaimiento basado en nivel actual (más difícil especializarse más)
    decay_factor = 1.0f0 - (neuron.specialization * params.age_decay)
    specialization_increment *= decay_factor
    
    # Actualizar especialización
    neuron.specialization = min(1.0f0, neuron.specialization + specialization_increment)
    
    return type_changed
end

"""
    calculate_activation_consistency(activation_history)

Calcula la consistencia de los patrones de activación recientes.
"""
function calculate_activation_consistency(activation_history)
    # Si hay menos de 2 estados en el historial, no se puede calcular consistencia
    if length(activation_history) < 2
        return 0.0f0
    end
    
    # Calcular correlaciones entre estados consecutivos
    correlations = Float32[]
    
    for i in 2:length(activation_history)
        prev_state = vec(activation_history[i-1])
        curr_state = vec(activation_history[i])
        
        # Normalizar vectores
        prev_norm = norm(prev_state)
        curr_norm = norm(curr_state)
        
        # Evitar división por cero
        if prev_norm > 1e-8 && curr_norm > 1e-8
            # Calcular correlación normalizada
            correlation = dot(prev_state, curr_state) / (prev_norm * curr_norm)
            push!(correlations, abs(correlation))
        else
            push!(correlations, 0.0f0)
        end
    end
    
    # Consistencia media
    return mean(correlations)
end

"""
    analyze_brain_specialization(brain)

Analiza las estadísticas de especialización para todo el cerebro.
"""
function analyze_brain_specialization(brain::Brain_Space)
    # Extraer niveles de especialización
    specialization_levels = [neuron.specialization for (_, neuron) in brain.neurons]
    
    # Contar neuronas por tipo
    type_counts = Dict{Symbol, Int}()
    
    for (_, neuron) in brain.neurons
        if !haskey(type_counts, neuron.functional_type)
            type_counts[neuron.functional_type] = 0
        end
        
        type_counts[neuron.functional_type] += 1
    end
    
    # Calcular estadísticas
    stats = Dict{Symbol, Any}(
        :mean_specialization => mean(specialization_levels),
        :max_specialization => maximum(specialization_levels),
        :min_specialization => minimum(specialization_levels),
        :std_specialization => std(specialization_levels),
        :type_distribution => type_counts
    )
    
    return stats
end

"""
    identify_specialized_regions(brain, threshold=0.6)

Identifica regiones de neuronas altamente especializadas.
"""
function identify_specialized_regions(
    brain::Brain_Space,
    threshold::Float32=0.6f0
)
    # Mapa de especialización 3D
    specialization_map = zeros(Float32, brain.dimensions)
    
    # Llenar mapa con valores de especialización
    for (pos, neuron) in brain.neurons
        specialization_map[pos...] = neuron.specialization
    end
    
    # Encontrar clusters de alta especialización (regiones)
    specialized_regions = find_specialized_clusters(specialization_map, threshold)
    
    # Identificar tipo dominante para cada región
    region_types = Dict{Int, Symbol}()
    region_specialization = Dict{Int, Float32}()
    
    for (region_id, region_positions) in specialized_regions
        # Contar tipos de neurona en esta región
        type_counts = Dict{Symbol, Int}()
        total_specialization = 0.0f0
        
        for pos in region_positions
            if haskey(brain.neurons, pos)
                neuron = brain.neurons[pos]
                
                if !haskey(type_counts, neuron.functional_type)
                    type_counts[neuron.functional_type] = 0
                end
                
                type_counts[neuron.functional_type] += 1
                total_specialization += neuron.specialization
            end
        end
        
        # Tipo dominante para esta región
        if !isempty(type_counts)
            dominant_type = sort(collect(type_counts), by=x->x[2], rev=true)[1][1]
            region_types[region_id] = dominant_type
            
            # Especialización media para la región
            region_specialization[region_id] = total_specialization / length(region_positions)
        end
    end
    
    return specialized_regions, region_types, region_specialization
end

"""
    find_specialized_clusters(specialization_map, threshold)

Encuentra clusters de alta especialización en el mapa 3D.
"""
function find_specialized_clusters(
    specialization_map::Array{Float32,3},
    threshold::Float32
)
    # Dimensiones del mapa
    dim_x, dim_y, dim_z = size(specialization_map)
    
    # Mapa binario de posiciones especializadas
    binary_map = specialization_map .>= threshold
    
    # Mapa de etiquetas para clusters (0 = no asignado)
    label_map = zeros(Int, dim_x, dim_y, dim_z)
    
    # Contador de clusters
    current_label = 0
    
    # Diccionario para almacenar posiciones por cluster
    clusters = Dict{Int, Vector{NTuple{3,Int}}}()
    
    # Para cada posición en el mapa
    for x in 1:dim_x
        for y in 1:dim_y
            for z in 1:dim_z
                # Si esta posición es especializada y no está etiquetada
                if binary_map[x, y, z] && label_map[x, y, z] == 0
                    # Incrementar contador de clusters
                    current_label += 1
                    
                    # Inicializar lista de posiciones para este cluster
                    clusters[current_label] = NTuple{3,Int}[]
                    
                    # Realizar búsqueda en anchura para encontrar cluster completo
                    grow_cluster!(binary_map, label_map, (x, y, z), current_label, clusters[current_label])
                end
            end
        end
    end
    
    return clusters
end

"""
    grow_cluster!(binary_map, label_map, start_pos, label, positions)

Expande un cluster desde una posición inicial usando búsqueda en anchura.
"""
function grow_cluster!(
    binary_map::BitArray{3},
    label_map::Array{Int,3},
    start_pos::NTuple{3,Int},
    label::Int,
    positions::Vector{NTuple{3,Int}}
)
    # Dimensiones del mapa
    dim_x, dim_y, dim_z = size(binary_map)
    
    # Cola para búsqueda en anchura
    queue = [start_pos]
    
    # Marcar posición inicial
    label_map[start_pos...] = label
    push!(positions, start_pos)
    
    # Direcciones de los 6 vecinos directos (conectividad de cara)
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    # Mientras haya posiciones en la cola
    while !isempty(queue)
        # Extraer posición actual
        current = popfirst!(queue)
        x, y, z = current
        
        # Explorar vecinos
        for (dx, dy, dz) in directions
            # Calcular coordenadas del vecino
            nx = x + dx
            ny = y + dy
            nz = z + dz
            
            # Verificar límites
            if 1 <= nx <= dim_x && 1 <= ny <= dim_y && 1 <= nz <= dim_z
                # Si el vecino es especializado y no está etiquetado
                if binary_map[nx, ny, nz] && label_map[nx, ny, nz] == 0
                    # Marcar vecino
                    label_map[nx, ny, nz] = label
                    
                    # Añadir a lista de posiciones
                    push!(positions, (nx, ny, nz))
                    
                    # Añadir a cola para explorar sus vecinos
                    push!(queue, (nx, ny, nz))
                end
            end
        end
    end
end

"""
    register_functional_regions!(brain, threshold=0.7)

Registra regiones funcionales en el cerebro basadas en especialización.
"""
function register_functional_regions!(
    brain::Brain_Space,
    threshold::Float32=0.7f0
)
    # Identificar regiones especializadas
    regions, region_types, region_specialization = identify_specialized_regions(brain, threshold)
    
    # Limpiar regiones funcionales existentes
    empty!(brain.functional_regions)
    
    # Registrar nuevas regiones funcionales
    for (region_id, positions) in regions
        if haskey(region_types, region_id)
            region_type = region_types[region_id]
            
            # Solo registrar si el tipo no es general
            if region_type != :general
                # Inicializar lista para este tipo si no existe
                if !haskey(brain.functional_regions, region_type)
                    brain.functional_regions[region_type] = Vector{NTuple{3,Int}}()
                end
                
                # Añadir posiciones a esta región funcional
                append!(brain.functional_regions[region_type], positions)
            end
        end
    end
    
    # Devolver número de regiones registradas
    return length(regions)
end

"""
    specialize_connections!(brain, connections_threshold=0.5)

Especializa las conexiones basadas en los tipos funcionales de las neuronas.
"""
function specialize_connections!(
    brain::Brain_Space,
    connections_threshold::Float32=0.5f0
)
    # Contador de conexiones modificadas
    modified_connections = 0
    
    # Para cada conexión, ajustar según tipos funcionales
    for connection in brain.connections
        # Encontrar neurona origen y destino
        source_neuron = nothing
        target_neuron = nothing
        
        for (_, neuron) in brain.neurons
            if neuron.id == connection.source_id
                source_neuron = neuron
            elseif neuron.id == connection.target_id
                target_neuron = neuron
            end
            
            # Si encontramos ambas, salir del bucle
            if !isnothing(source_neuron) && !isnothing(target_neuron)
                break
            end
        end
        
        # Si encontramos ambas neuronas
        if !isnothing(source_neuron) && !isnothing(target_neuron)
            # Verificar nivel mínimo de especialización
            if source_neuron.specialization > connections_threshold && 
               target_neuron.specialization > connections_threshold
                
                # Modificar conexión según tipos funcionales
                modified = specialize_connection!(
                    connection, 
                    source_neuron.functional_type,
                    target_neuron.functional_type
                )
                
                if modified
                    modified_connections += 1
                end
            end
        end
    end
    
    return modified_connections
end

"""
    specialize_connection!(connection, source_type, target_type)

Modifica una conexión según los tipos funcionales de las neuronas conectadas.
"""
function specialize_connection!(
    connection::Tensor_Connection,
    source_type::Symbol,
    target_type::Symbol
)
    # Flag para indicar si se modificó la conexión
    modified = false
    
    # Factor de escalado base para la fuerza
    scale_factor = 1.0f0
    
    # Ajustar según combinación de tipos
    if source_type == :temporal && target_type == :temporal
        # Conexiones temporales preferentes - reforzar
        scale_factor = 1.5f0
        modified = true
        
    elseif source_type == :spatial && target_type == :spatial
        # Conexiones espaciales locales - reforzar
        scale_factor = 1.3f0
        modified = true
        
    elseif source_type == :feature && target_type == :feature
        # Conexiones especializadas - reforzar
        scale_factor = 1.4f0
        modified = true
        
    elseif source_type == :contextual && target_type == :contextual
        # Conexiones contextuales - reforzar mucho
        scale_factor = 1.6f0
        modified = true
        
    elseif source_type == :contextual && target_type != :contextual
        # Conexiones contextuales a otras - modular según tipo de conexión
        if connection.connection_type == :excitatory
            # Reforzar excitatorias
            scale_factor = 1.2f0
        else
            # Reforzar inhibitorias aún más
            scale_factor = 1.4f0
        end
        modified = true
        
    elseif source_type == :general || target_type == :general
        # Conexiones con general - debilitar ligeramente
        scale_factor = 0.9f0
        modified = true
    end
    
    # Si se debe modificar, aplicar cambios
    if modified
        # Ajustar fuerza de conexión
        connection.strength *= scale_factor
        
        # Escalar tensor de peso
        connection.weight .*= scale_factor
    end
    
    return modified
end

# Exportar tipos y funciones principales
export SpecializationParameters, 
       specialize_neurons!, update_specialization!,
       identify_specialized_regions, register_functional_regions!,
       specialize_connections!

end # module Specialization