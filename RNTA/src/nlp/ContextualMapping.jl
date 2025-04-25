# nlp/ContextualMapping.jl
# Implementa mapeo contextual para procesamiento de lenguaje natural

module ContextualMapping

using LinearAlgebra
using Statistics
using Random
using DataStructures

# Importaciones de otros módulos de RNTA
using ..BrainSpace
using ..SemanticSpace
using ..TensorOperations
using ..SpatialAttention
using ..Tokenizer

"""
    ContextState

Estado de contexto para el procesamiento de lenguaje natural.
"""
mutable struct ContextState
    # Tensor de estado contextual
    tensor::Array{Float32,3}
    
    # Historial de estados de contexto
    history::CircularBuffer{Array{Float32,3}}
    
    # Factor de decaimiento para contexto antiguo
    decay_factor::Float32
    
    # Metadatos contextuales
    metadata::Dict{Symbol, Any}
    
    # Mapa de atención contextual
    attention_map::SpatialAttentionMap
    
    # Diccionario de referencias a entidades activas en el contexto
    active_entities::Dict{String, Float32}
    
    # Factor de inercia (resistencia al cambio)
    inertia_factor::Float32
end

"""
Constructor para ContextState
"""
function ContextState(
    dimensions::NTuple{3,Int};
    history_length::Int=5,
    decay_factor::Float32=0.8f0,
    inertia_factor::Float32=0.3f0
)
    # Inicializar tensor de contexto vacío
    tensor = zeros(Float32, dimensions)
    
    # Inicializar historial
    history = CircularBuffer{Array{Float32,3}}(history_length)
    for _ in 1:history_length
        push!(history, zeros(Float32, dimensions))
    end
    
    # Crear mapa de atención
    attention_map = SpatialAttentionMap(dimensions)
    
    return ContextState(
        tensor,
        history,
        decay_factor,
        Dict{Symbol, Any}(),
        attention_map,
        Dict{String, Float32}(),
        inertia_factor
    )
end

"""
    ContextMapper

Sistema para mapeo contextual de lenguaje a representaciones tensoriales.
"""
mutable struct ContextMapper
    # Dimensiones del espacio contextual
    dimensions::NTuple{3,Int}
    
    # Estado de contexto actual
    state::ContextState
    
    # Espacio semántico para conceptos
    semantic_space::Union{Semantic3DSpace, Nothing}
    
    # Tokenizador para entrada de texto
    tokenizer::Union{TensorialTokenizer, Nothing}
    
    # Cerebro base (opcional, para integración con el resto del sistema)
    brain::Union{Brain_Space, Nothing}
    
    # Diccionario de operadores contextuales (funciones que transforman el contexto)
    context_operators::Dict{Symbol, Function}
    
    # Parámetros de configuración
    config::Dict{Symbol, Any}
end

"""
Constructor para ContextMapper
"""
function ContextMapper(
    dimensions::NTuple{3,Int};
    semantic_space::Union{Semantic3DSpace, Nothing}=nothing,
    tokenizer::Union{TensorialTokenizer, Nothing}=nothing,
    brain::Union{Brain_Space, Nothing}=nothing,
    config::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    # Crear estado de contexto inicial
    state = ContextState(dimensions)
    
    # Inicializar operadores contextuales
    context_operators = initialize_context_operators()
    
    return ContextMapper(
        dimensions,
        state,
        semantic_space,
        tokenizer,
        brain,
        context_operators,
        config
    )
end

"""
    initialize_context_operators()

Inicializa los operadores contextuales predefinidos.
"""
function initialize_context_operators()
    operators = Dict{Symbol, Function}()
    
    # Operador de adición de contexto
    operators[:add] = (current, new) -> current + new
    
    # Operador de sustracción de contexto
    operators[:subtract] = (current, new) -> current - new
    
    # Operador de intersección de contexto
    operators[:intersect] = (current, new) -> min.(current, new)
    
    # Operador de unión de contexto
    operators[:union] = (current, new) -> max.(current, new)
    
    # Operador de modulación de contexto
    operators[:modulate] = (current, new) -> current .* new
    
    # Operador de desplazamiento de contexto
    operators[:shift] = (current, new) -> 0.5f0 * current + 0.5f0 * new
    
    # Operador de reemplazo de contexto
    operators[:replace] = (current, new) -> new
    
    # Operador de normalización de contexto
    operators[:normalize] = (current, _) -> normalize_tensor(current)
    
    return operators
end

"""
    process_text(mapper, text; operation=:add)

Procesa texto y actualiza el estado de contexto.
"""
function process_text(
    mapper::ContextMapper,
    text::String;
    operation::Symbol=:add,
    weight::Float32=1.0f0
)
    # Verificar tokenizador
    if isnothing(mapper.tokenizer)
        error("No hay tokenizador disponible para procesar texto")
    end
    
    # Convertir texto a tensor
    input_tensor = process_text(mapper.tokenizer, text)
    
    # Redimensionar al tamaño del contexto
    if size(input_tensor) != mapper.dimensions
        input_tensor = tensor_interpolation(input_tensor, mapper.dimensions)
    end
    
    # Actualizar contexto con el nuevo tensor
    update_context!(mapper.state, input_tensor, operation=operation, weight=weight)
    
    # Extraer entidades del texto (si el espacio semántico está disponible)
    if !isnothing(mapper.semantic_space)
        extract_entities!(mapper, text)
    end
    
    return mapper.state.tensor
end

"""
    process_tensor(mapper, tensor; operation=:add)

Procesa un tensor de entrada y actualiza el estado de contexto.
"""
function process_tensor(
    mapper::ContextMapper,
    tensor::Array{T,3};
    operation::Symbol=:add,
    weight::Float32=1.0f0
) where T <: AbstractFloat
    # Redimensionar al tamaño del contexto
    if size(tensor) != mapper.dimensions
        tensor = tensor_interpolation(tensor, mapper.dimensions)
    end
    
    # Actualizar contexto con el nuevo tensor
    update_context!(mapper.state, tensor, operation=operation, weight=weight)
    
    return mapper.state.tensor
end

"""
    update_context!(state, input_tensor; operation=:add, weight=1.0)

Actualiza el estado de contexto con un tensor de entrada.
"""
function update_context!(
    state::ContextState,
    input_tensor::Array{T,3};
    operation::Symbol=:add,
    weight::Float32=1.0f0
) where T <: AbstractFloat
    # Guardar estado anterior en historial
    push!(state.history, copy(state.tensor))
    
    # Aplicar factor de inercia
    inertia_tensor = state.tensor * state.inertia_factor
    
    # Buscar operador contextual
    if haskey(state.context_operators, operation)
        operator = state.context_operators[operation]
        
        # Aplicar operador con peso
        new_tensor = operator(state.tensor, input_tensor * weight)
    else
        # Operador por defecto: adición ponderada
        new_tensor = state.tensor + input_tensor * weight
    end
    
    # Combinar con inercia
    state.tensor = inertia_tensor + new_tensor * (1.0f0 - state.inertia_factor)
    
    # Actualizar mapa de atención
    update_attention!(state, input_tensor)
    
    return state.tensor
end

"""
    update_attention!(state, input_tensor)

Actualiza el mapa de atención del contexto basado en el tensor de entrada.
"""
function update_attention!(
    state::ContextState,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Encontrar regiones significativas en el tensor de entrada
    attention_regions = find_significant_regions(input_tensor, threshold=0.3f0)
    
    # Si hay regiones significativas, actualizar mapa de atención
    if !isempty(attention_regions)
        # Obtener centro de la primera región significativa
        region = attention_regions[1]
        center_x = (region[1].start + region[1].stop) / 2
        center_y = (region[2].start + region[2].stop) / 2
        center_z = (region[3].start + region[3].stop) / 2
        
        # Actualizar centro de atención
        shift_attention!(state.attention_map, (center_x, center_y, center_z))
    end
    
    return state.attention_map
end

"""
    find_significant_regions(tensor; threshold=0.5)

Encuentra regiones con valores significativos en un tensor.
"""
function find_significant_regions(
    tensor::Array{T,3};
    threshold::Float32=0.5f0
) where T <: AbstractFloat
    # Normalizar tensor
    normalized = tensor ./ max(maximum(abs.(tensor)), 1e-8f0)
    
    # Máscara binaria de regiones significativas
    significant_mask = abs.(normalized) .> threshold
    
    # Encontrar componentes conectados
    regions = find_connected_regions(significant_mask)
    
    return regions
end

"""
    find_connected_regions(mask)

Encuentra regiones conectadas en una máscara binaria 3D.
"""
function find_connected_regions(mask::BitArray{3})
    dim_x, dim_y, dim_z = size(mask)
    
    # Matriz de etiquetas (0 = no visitado)
    labels = zeros(Int, dim_x, dim_y, dim_z)
    
    # Vector de regiones (coords_x, coords_y, coords_z)
    regions = Vector{NTuple{3, UnitRange{Int}}}()
    
    # Contador de regiones
    current_label = 0
    
    # Para cada punto en el tensor
    for x in 1:dim_x
        for y in 1:dim_y
            for z in 1:dim_z
                # Si es significativo y no visitado
                if mask[x, y, z] && labels[x, y, z] == 0
                    # Nueva región
                    current_label += 1
                    
                    # Explorar región usando BFS
                    x_coords, y_coords, z_coords = explore_region!(mask, labels, x, y, z, current_label)
                    
                    # Crear rangos para la región
                    x_range = minimum(x_coords):maximum(x_coords)
                    y_range = minimum(y_coords):maximum(y_coords)
                    z_range = minimum(z_coords):maximum(z_coords)
                    
                    # Añadir región
                    push!(regions, (x_range, y_range, z_range))
                end
            end
        end
    end
    
    return regions
end

"""
    explore_region!(mask, labels, start_x, start_y, start_z, label)

Explora una región conectada usando búsqueda en anchura.
"""
function explore_region!(
    mask::BitArray{3},
    labels::Array{Int,3},
    start_x::Int,
    start_y::Int,
    start_z::Int,
    label::Int
)
    dim_x, dim_y, dim_z = size(mask)
    
    # Coordenadas de los puntos en esta región
    x_coords = Int[]
    y_coords = Int[]
    z_coords = Int[]
    
    # Cola para BFS
    queue = [(start_x, start_y, start_z)]
    
    # Marcar punto inicial
    labels[start_x, start_y, start_z] = label
    push!(x_coords, start_x)
    push!(y_coords, start_y)
    push!(z_coords, start_z)
    
    # Direcciones para vecinos (6-conectividad)
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    # Mientras haya puntos en la cola
    while !isempty(queue)
        # Extraer punto actual
        x, y, z = popfirst!(queue)
        
        # Explorar vecinos
        for (dx, dy, dz) in directions
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Verificar límites
            if 1 <= nx <= dim_x && 1 <= ny <= dim_y && 1 <= nz <= dim_z
                # Si es significativo y no visitado
                if mask[nx, ny, nz] && labels[nx, ny, nz] == 0
                    # Marcar como visitado
                    labels[nx, ny, nz] = label
                    
                    # Añadir a cola
                    push!(queue, (nx, ny, nz))
                    
                    # Guardar coordenadas
                    push!(x_coords, nx)
                    push!(y_coords, ny)
                    push!(z_coords, nz)
                end
            end
        end
    end
    
    return x_coords, y_coords, z_coords
end

"""
    extract_entities!(mapper, text)

Extrae entidades del texto y las añade al contexto.
"""
function extract_entities!(
    mapper::ContextMapper,
    text::String
)
    # Verificar espacio semántico
    if isnothing(mapper.semantic_space)
        return Dict{String, Float32}()
    end
    
    # Buscar entidades en el espacio semántico
    results = query(mapper.semantic_space, text, top_k=5)
    
    # Actualizar entidades activas
    for (concept, similarity) in results
        if similarity > 0.3f0  # Umbral de similitud
            # Añadir o actualizar entidad
            current_activation = get(mapper.state.active_entities, concept.id, 0.0f0)
            mapper.state.active_entities[concept.id] = max(current_activation, similarity)
            
            # Añadir representación al contexto con peso basado en similitud
            process_tensor(
                mapper, 
                concept.tensor, 
                operation=:add, 
                weight=similarity * 0.5f0
            )
        end
    end
    
    # Aplicar decaimiento a entidades antiguas
    decay_entities!(mapper.state)
    
    return mapper.state.active_entities
end

"""
    decay_entities!(state)

Aplica decaimiento a entidades activas en el contexto.
"""
function decay_entities!(state::ContextState)
    # Aplicar decaimiento a todas las entidades activas
    for entity_id in keys(state.active_entities)
        state.active_entities[entity_id] *= state.decay_factor
        
        # Eliminar entidades con activación muy baja
        if state.active_entities[entity_id] < 0.1f0
            delete!(state.active_entities, entity_id)
        end
    end
    
    return state.active_entities
end

"""
    normalize_tensor(tensor)

Normaliza un tensor para evitar valores extremos.
"""
function normalize_tensor(tensor::Array{T,3}) where T <: AbstractFloat
    # Normalizar por norma euclídea
    norm_val = norm(vec(tensor))
    
    # Evitar división por cero
    if norm_val > 1e-8f0
        return tensor ./ norm_val
    else
        return tensor
    end
end

"""
    get_context_vector(mapper)

Obtiene una representación vectorial del contexto actual.
"""
function get_context_vector(mapper::ContextMapper)
    # Aplicar mapa de atención al contexto
    attended_context = apply_attention(mapper.state.tensor, mapper.state.attention_map)
    
    # Reducir a vector (colapsar por dimensión Z)
    dim_x, dim_y, dim_z = size(attended_context)
    
    # Promediar a lo largo del eje Z
    vector = mean(attended_context, dims=3)[:, :, 1]
    
    # Aplanar a vector unidimensional
    return vec(vector)
end

"""
    reset_context!(mapper)

Reinicia el estado de contexto a cero.
"""
function reset_context!(mapper::ContextMapper)
    # Reiniciar tensor de contexto
    mapper.state.tensor .= 0.0f0
    
    # Reiniciar historial
    for i in 1:length(mapper.state.history)
        mapper.state.history[i] .= 0.0f0
    end
    
    # Reiniciar entidades activas
    empty!(mapper.state.active_entities)
    
    # Reiniciar metadatos
    empty!(mapper.state.metadata)
    
    return mapper.state
end

"""
    get_active_entities(mapper; threshold=0.1)

Obtiene las entidades actualmente activas en el contexto.
"""
function get_active_entities(
    mapper::ContextMapper;
    threshold::Float32=0.1f0
)
    # Filtrar entidades por umbral de activación
    active = Dict{String, Float32}()
    
    for (entity_id, activation) in mapper.state.active_entities
        if activation >= threshold
            active[entity_id] = activation
        end
    end
    
    return active
end

"""
    get_context_history(mapper; n=3)

Obtiene los últimos n estados de contexto.
"""
function get_context_history(
    mapper::ContextMapper;
    n::Int=3
)
    # Limitar a número disponible
    n = min(n, length(mapper.state.history))
    
    # Extraer últimos n estados
    history = Vector{Array{Float32,3}}()
    
    for i in 1:n
        idx = length(mapper.state.history) - n + i
        if idx > 0
            push!(history, mapper.state.history[idx])
        end
    end
    
    return history
end

"""
    compare_contexts(context1, context2)

Compara dos estados de contexto y calcula su similitud.
"""
function compare_contexts(
    context1::Array{T,3},
    context2::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(context1) != size(context2)
        context2 = tensor_interpolation(context2, size(context1))
    end
    
    # Calcular similitud coseno
    dot_product = sum(context1 .* context2)
    norm1 = norm(vec(context1))
    norm2 = norm(vec(context2))
    
    # Evitar división por cero
    if norm1 < 1e-8f0 || norm2 < 1e-8f0
        return 0.0f0
    end
    
    return dot_product / (norm1 * norm2)
end

"""
    save_context_state(state, filename)

Guarda el estado de contexto en un archivo.
"""
function save_context_state(
    state::ContextState,
    filename::String
)
    # Preparar datos para guardar
    data = Dict{String, Any}()
    
    # Guardar tensor de contexto
    data["tensor"] = state.tensor
    
    # Guardar historial
    data["history"] = collect(state.history)
    
    # Guardar otros parámetros
    data["decay_factor"] = state.decay_factor
    data["inertia_factor"] = state.inertia_factor
    
    # Guardar entidades activas
    data["active_entities"] = state.active_entities
    
    # Guardar metadatos
    data["metadata"] = state.metadata
    
    # Guardar en archivo
    save(filename, data)
    
    return filename
end

"""
    load_context_state(filename, dimensions)

Carga un estado de contexto desde un archivo.
"""
function load_context_state(
    filename::String,
    dimensions::NTuple{3,Int}
)
    # Cargar datos
    data = load(filename)
    
    # Crear estado de contexto
    state = ContextState(dimensions)
    
    # Cargar tensor de contexto
    state.tensor = data["tensor"]
    
    # Cargar historial
    history_data = data["history"]
    for tensor in history_data
        push!(state.history, tensor)
    end
    
    # Cargar otros parámetros
    state.decay_factor = data["decay_factor"]
    state.inertia_factor = data["inertia_factor"]
    
    # Cargar entidades activas
    state.active_entities = data["active_entities"]
    
    # Cargar metadatos
    state.metadata = data["metadata"]
    
    return state
end

# Exportar tipos y funciones principales
export ContextState, ContextMapper,
       process_text, process_tensor, reset_context!,
       get_context_vector, get_active_entities, get_context_history,
       compare_contexts, save_context_state, load_context_state

end # module ContextualMapping