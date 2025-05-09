# nlp/SemanticSpace.jl
# Implementa un espacio semántico tridimensional para representación de lenguaje

module SemanticSpace

using LinearAlgebra
using Statistics
using Random
using SparseArrays

# Importaciones de otros módulos de RNTA
using ..BrainSpace
using ..TensorOperations
using ..SpatialAttention
using ..Tokenizer

"""
    SemanticRepresentation

Representa un concepto o elemento semántico en el espacio 3D.
"""
struct SemanticRepresentation
    # Identificador único
    id::String
    
    # Etiqueta o nombre del concepto
    label::String
    
    # Tensor de representación volumétrica
    tensor::Array{Float32,3}
    
    # Metadatos asociados
    metadata::Dict{Symbol, Any}
    
    # Medida de confianza de esta representación
    confidence::Float32
end

"""
Constructor para SemanticRepresentation
"""
function SemanticRepresentation(
    label::String,
    tensor::Array{T,3};
    id::String="",
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}(),
    confidence::Float32=1.0f0
) where T <: AbstractFloat
    # Generar ID si no se proporciona
    if isempty(id)
        id = string(hash(label), base=16)
    end
    
    return SemanticRepresentation(
        id,
        label,
        convert(Array{Float32,3}, tensor),
        metadata,
        confidence
    )
end

"""
    Semantic3DSpace

Espacio tridimensional para representación semántica.
"""
mutable struct Semantic3DSpace
    # Dimensiones del espacio
    dimensions::NTuple{3,Int}
    
    # Diccionario de conceptos o elementos semánticos
    concepts::Dict{String, SemanticRepresentation}
    
    # Índice espacial para búsqueda eficiente (ubicación → concepto)
    spatial_index::Dict{NTuple{3,Int}, Vector{String}}
    
    # Mapa de atención semántica
    attention_map::SpatialAttentionMap
    
    # Cerebro base (opcional, para integración con el resto del sistema)
    brain::Union{Brain_Space, Nothing}
    
    # Tokenizador tensorial para entrada de texto
    tokenizer::Union{TensorialTokenizer, Nothing}
    
    # Histórico de consultas
    query_history::Vector{Tuple{String, Array{Float32,3}}}
    
    # Pesos de importancia para dimensiones semánticas
    dimension_weights::NTuple{3,Float32}
end

"""
Constructor para Semantic3DSpace
"""
function Semantic3DSpace(
    dimensions::NTuple{3,Int};
    brain::Union{Brain_Space, Nothing}=nothing,
    tokenizer::Union{TensorialTokenizer, Nothing}=nothing,
    dimension_weights::NTuple{3,Float32}=(1.0f0, 1.0f0, 1.0f0)
)
    # Crear mapa de atención
    attention_map = SpatialAttentionMap(dimensions)
    
    return Semantic3DSpace(
        dimensions,
        Dict{String, SemanticRepresentation}(),
        Dict{NTuple{3,Int}, Vector{String}}(),
        attention_map,
        brain,
        tokenizer,
        Vector{Tuple{String, Array{Float32,3}}}(),
        dimension_weights
    )
end

"""
    add_concept!(space, representation)

Añade un concepto al espacio semántico.
"""
function add_concept!(
    space::Semantic3DSpace,
    representation::SemanticRepresentation
)
    # Asegurar que el tensor tiene las dimensiones correctas
    if size(representation.tensor) != space.dimensions
        resized_tensor = tensor_interpolation(representation.tensor, space.dimensions)
        representation = SemanticRepresentation(
            representation.label,
            resized_tensor,
            id=representation.id,
            metadata=representation.metadata,
            confidence=representation.confidence
        )
    end
    
    # Añadir al diccionario de conceptos
    space.concepts[representation.id] = representation
    
    # Actualizar índice espacial
    update_spatial_index!(space, representation)
    
    return representation
end

"""
    remove_concept!(space, concept_id)

Elimina un concepto del espacio semántico.
"""
function remove_concept!(
    space::Semantic3DSpace,
    concept_id::String
)
    if !haskey(space.concepts, concept_id)
        return false
    end
    
    # Obtener representación
    representation = space.concepts[concept_id]
    
    # Eliminar de índice espacial
    remove_from_spatial_index!(space, representation)
    
    # Eliminar del diccionario de conceptos
    delete!(space.concepts, concept_id)
    
    return true
end

"""
    update_spatial_index!(space, representation)

Actualiza el índice espacial para una representación.
"""
function update_spatial_index!(
    space::Semantic3DSpace,
    representation::SemanticRepresentation
)
    # Limpiar entradas anteriores para este concepto
    remove_from_spatial_index!(space, representation)
    
    # Encontrar regiones significativas en el tensor
    significant_regions = find_significant_regions(representation.tensor)
    
    # Añadir al índice espacial
    for region in significant_regions
        for x in region[1]
            for y in region[2]
                for z in region[3]
                    position = (x, y, z)
                    
                    # Inicializar vector si es necesario
                    if !haskey(space.spatial_index, position)
                        space.spatial_index[position] = String[]
                    end
                    
                    # Añadir ID de concepto
                    push!(space.spatial_index[position], representation.id)
                end
            end
        end
    end
    
    return space
end

"""
    remove_from_spatial_index!(space, representation)

Elimina una representación del índice espacial.
"""
function remove_from_spatial_index!(
    space::Semantic3DSpace,
    representation::SemanticRepresentation
)
    # Para cada posición en el índice
    for (position, concept_ids) in space.spatial_index
        # Filtrar referencias a este concepto
        filter!(id -> id != representation.id, concept_ids)
        
        # Eliminar entrada si queda vacía
        if isempty(concept_ids)
            delete!(space.spatial_index, position)
        end
    end
    
    return space
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
    query(space, text_query; top_k=5)

Realiza una consulta semántica basada en texto.
"""
function query(
    space::Semantic3DSpace,
    text_query::String;
    top_k::Int=5
)
    # Procesar texto a tensor
    if isnothing(space.tokenizer)
        error("No hay tokenizador disponible para consultas de texto")
    end
    
    # Convertir texto a tensor
    query_tensor = process_text(space.tokenizer, text_query)
    
    # Redimensionar al tamaño del espacio
    if size(query_tensor) != space.dimensions
        query_tensor = tensor_interpolation(query_tensor, space.dimensions)
    end
    
    # Guardar consulta en historial
    push!(space.query_history, (text_query, query_tensor))
    
    # Actualizar mapa de atención basado en consulta
    update_attention!(space, query_tensor)
    
    # Realizar búsqueda semántica
    results = semantic_search(space, query_tensor, top_k=top_k)
    
    return results
end

"""
    query(space, tensor_query; top_k=5)

Realiza una consulta semántica basada en tensor.
"""
function query(
    space::Semantic3DSpace,
    tensor_query::Array{T,3};
    top_k::Int=5
) where T <: AbstractFloat
    # Redimensionar al tamaño del espacio
    if size(tensor_query) != space.dimensions
        tensor_query = tensor_interpolation(tensor_query, space.dimensions)
    end
    
    # Guardar consulta en historial
    push!(space.query_history, ("tensor_query", tensor_query))
    
    # Actualizar mapa de atención basado en consulta
    update_attention!(space, tensor_query)
    
    # Realizar búsqueda semántica
    results = semantic_search(space, tensor_query, top_k=top_k)
    
    return results
end

"""
    update_attention!(space, query_tensor)

Actualiza el mapa de atención basado en una consulta.
"""
function update_attention!(
    space::Semantic3DSpace,
    query_tensor::Array{T,3}
) where T <: AbstractFloat
    # Crear mapa de atención basado en valores significativos de la consulta
    space.attention_map = create_attention_from_activity(
        query_tensor,
        threshold=0.3f0,
        radius=3.0f0,
        focus_factor=2.0f0
    )
    
    return space.attention_map
end

"""
    semantic_search(space, query_tensor; top_k=5)

Realiza una búsqueda semántica en el espacio.
"""
function semantic_search(
    space::Semantic3DSpace,
    query_tensor::Array{T,3};
    top_k::Int=5
) where T <: AbstractFloat
    # Si no hay conceptos, devolver vacío
    if isempty(space.concepts)
        return Vector{Tuple{SemanticRepresentation, Float32}}()
    end
    
    # Normalizar tensor de consulta
    norm_query = query_tensor ./ max(norm(vec(query_tensor)), 1e-8f0)
    
    # Aplicar ponderación por dimensiones
    weighted_query = apply_dimension_weights(norm_query, space.dimension_weights)
    
    # Calcular similitud con cada concepto
    similarities = Tuple{SemanticRepresentation, Float32}[]
    
    for (_, concept) in space.concepts
        # Normalizar tensor de concepto
        norm_concept = concept.tensor ./ max(norm(vec(concept.tensor)), 1e-8f0)
        
        # Aplicar ponderación por dimensiones
        weighted_concept = apply_dimension_weights(norm_concept, space.dimension_weights)
        
        # Calcular similitud coseno ponderada por atención
        similarity = compute_similarity(weighted_query, weighted_concept, space.attention_map)
        
        # Ajustar por confianza del concepto
        similarity *= concept.confidence
        
        push!(similarities, (concept, similarity))
    end
    
    # Ordenar por similitud descendente
    sort!(similarities, by=x -> x[2], rev=true)
    
    # Devolver los top_k resultados
    return similarities[1:min(top_k, length(similarities))]
end

"""
    apply_dimension_weights(tensor, weights)

Aplica ponderación por dimensiones a un tensor.
"""
function apply_dimension_weights(
    tensor::Array{T,3},
    weights::NTuple{3,Float32}
) where T <: AbstractFloat
    dim_x, dim_y, dim_z = size(tensor)
    weighted = copy(tensor)
    
    # Aplicar pesos por dimensión
    for x in 1:dim_x
        # Peso para dimensión x
        x_factor = 1.0f0 + (x / dim_x - 0.5f0) * weights[1]
        for y in 1:dim_y
            # Peso para dimensión y
            y_factor = 1.0f0 + (y / dim_y - 0.5f0) * weights[2]
            for z in 1:dim_z
                # Peso para dimensión z
                z_factor = 1.0f0 + (z / dim_z - 0.5f0) * weights[3]
                
                # Factor combinado
                combined_factor = x_factor * y_factor * z_factor
                
                # Aplicar ponderación
                weighted[x, y, z] *= combined_factor
            end
        end
    end
    
    return weighted
end

"""
    compute_similarity(tensor1, tensor2, attention_map)

Calcula similitud entre dos tensores, ponderada por atención.
"""
function compute_similarity(
    tensor1::Array{T,3},
    tensor2::Array{S,3},
    attention_map::SpatialAttentionMap
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(tensor1) != size(tensor2)
        tensor2 = tensor_interpolation(tensor2, size(tensor1))
    end
    
    # Obtener mapa de atención
    attention_field = attention_map.attention_field
    if size(attention_field) != size(tensor1)
        attention_field = tensor_interpolation(attention_field, size(tensor1))
    end
    
    # Calcular producto escalar ponderado por atención
    weighted_dot = sum(tensor1 .* tensor2 .* attention_field)
    
    # Normalizar por las normas
    norm1 = sqrt(sum((tensor1 .* attention_field).^2))
    norm2 = sqrt(sum((tensor2 .* attention_field).^2))
    
    # Evitar división por cero
    if norm1 < 1e-8 || norm2 < 1e-8
        return 0.0f0
    end
    
    # Similitud coseno
    similarity = weighted_dot / (norm1 * norm2)
    
    return similarity
end

"""
    create_concept_from_text(text, tokenizer; label="", metadata=Dict{Symbol,Any}())

Crea una representación semántica a partir de texto.
"""
function create_concept_from_text(
    text::String,
    tokenizer::TensorialTokenizer;
    label::String="",
    metadata::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    confidence::Float32=1.0f0
)
    # Si no se proporciona etiqueta, usar el texto
    if isempty(label)
        label = text
    end
    
    # Procesar texto a tensor
    tensor = process_text(tokenizer, text)
    
    # Crear representación semántica
    representation = SemanticRepresentation(
        label,
        tensor,
        metadata=metadata,
        confidence=confidence
    )
    
    return representation
end

"""
    merge_representations(rep1, rep2; weight1=0.5, weight2=0.5)

Combina dos representaciones semánticas.
"""
function merge_representations(
    rep1::SemanticRepresentation,
    rep2::SemanticRepresentation;
    weight1::Float32=0.5f0,
    weight2::Float32=0.5f0
)
    # Asegurar que los tensores tienen el mismo tamaño
    if size(rep1.tensor) != size(rep2.tensor)
        tensor2 = tensor_interpolation(rep2.tensor, size(rep1.tensor))
    else
        tensor2 = rep2.tensor
    end
    
    # Combinar tensores según pesos
    combined_tensor = weight1 * rep1.tensor + weight2 * tensor2
    
    # Generar etiqueta combinada
    combined_label = "$(rep1.label)_$(rep2.label)"
    
    # Combinar metadatos
    combined_metadata = merge(rep1.metadata, rep2.metadata)
    
    # Calcular confianza combinada
    combined_confidence = weight1 * rep1.confidence + weight2 * rep2.confidence
    
    # Crear nueva representación
    combined_rep = SemanticRepresentation(
        combined_label,
        combined_tensor,
        metadata=combined_metadata,
        confidence=combined_confidence
    )
    
    return combined_rep
end

"""
    extract_concept_field(space, concept_id)

Extrae la representación de un concepto como campo espacial.
"""
function extract_concept_field(
    space::Semantic3DSpace,
    concept_id::String
)
    if !haskey(space.concepts, concept_id)
        error("Concepto no encontrado: $concept_id")
    end
    
    # Obtener representación
    representation = space.concepts[concept_id]
    
    # Convertir a campo espacial
    return representation.tensor
end

"""
    find_related_concepts(space, concept_id; top_k=5)

Encuentra conceptos relacionados con un concepto dado.
"""
function find_related_concepts(
    space::Semantic3DSpace,
    concept_id::String;
    top_k::Int=5
)
    if !haskey(space.concepts, concept_id)
        error("Concepto no encontrado: $concept_id")
    end
    
    # Obtener representación
    representation = space.concepts[concept_id]
    
    # Usar como consulta
    results = semantic_search(space, representation.tensor, top_k=top_k+1)
    
    # Filtrar el propio concepto
    filtered_results = filter(x -> x[1].id != concept_id, results)
    
    # Limitar a top_k
    return filtered_results[1:min(top_k, length(filtered_results))]
end

"""
    process_query_in_brain!(space, query)

Procesa una consulta semántica en el cerebro asociado.
"""
function process_query_in_brain!(
    space::Semantic3DSpace,
    query::String
)
    if isnothing(space.brain)
        error("No hay cerebro asociado al espacio semántico")
    end
    
    # Procesar texto a tensor
    if isnothing(space.tokenizer)
        error("No hay tokenizador disponible para consultas de texto")
    end
    
    # Convertir texto a tensor
    query_tensor = process_text(space.tokenizer, query)
    
    # Propagar a través del cerebro
    result = process(space.brain, query_tensor)
    
    # Actualizar mapa de atención
    update_attention!(space, result)
    
    # Guardar consulta en historial
    push!(space.query_history, (query, result))
    
    return result
end

"""
    visualize_semantic_space(space; options...)

Genera una visualización del espacio semántico.
"""
function visualize_semantic_space(
    space::Semantic3DSpace;
    show_concepts::Bool=true,
    show_attention::Bool=true,
    highlight_concept::String=""
)
    # Esta función sería una interfaz para el módulo de visualización
    # Por ahora devolvemos un resumen textual
    
    # Estadísticas básicas
    stats = Dict{Symbol, Any}(
        :dimensions => space.dimensions,
        :num_concepts => length(space.concepts),
        :num_queries => length(space.query_history)
    )
    
    # Lista de conceptos
    concept_list = []
    
    if show_concepts
        for (id, concept) in space.concepts
            # Destacar concepto si es solicitado
            is_highlighted = (id == highlight_concept)
            
            # Calcular estadísticas del tensor
            tensor_stats = (
                mean = mean(concept.tensor),
                max = maximum(concept.tensor),
                norm = norm(vec(concept.tensor))
            )
            
            # Información del concepto
            concept_info = (
                id = id,
                label = concept.label,
                confidence = concept.confidence,
                highlighted = is_highlighted,
                tensor_stats = tensor_stats
            )
            
            push!(concept_list, concept_info)
        end
    end
    
    # Información sobre atención
    attention_info = nothing
    
    if show_attention
        attention_info = (
            focus_center = space.attention_map.focus_center,
            focus_factor = space.attention_map.focus_factor,
            effective_radius = space.attention_map.effective_radius
        )
    end
    
    # Devolver resumen
    return (
        stats = stats,
        concepts = concept_list,
        attention = attention_info,
        dimension_weights = space.dimension_weights
    )
end

"""
    save_semantic_space(space, filename)

Guarda el espacio semántico en un archivo.
"""
function save_semantic_space(
    space::Semantic3DSpace,
    filename::String
)
    # Preparar datos para guardar
    data = Dict{String, Any}()
    
    # Guardar dimensiones
    data["dimensions"] = space.dimensions
    
    # Guardar conceptos
    concepts_data = Dict{String, Any}()
    
    for (id, concept) in space.concepts
        concept_data = Dict{String, Any}(
            "id" => concept.id,
            "label" => concept.label,
            "tensor" => concept.tensor,
            "metadata" => concept.metadata,
            "confidence" => concept.confidence
        )
        
        concepts_data[id] = concept_data
    end
    
    data["concepts"] = concepts_data
    
    # Guardar mapa de atención
    data["attention_map"] = Dict{String, Any}(
        "attention_field" => space.attention_map.attention_field,
        "focus_factor" => space.attention_map.focus_factor,
        "effective_radius" => space.attention_map.effective_radius,
        "focus_center" => space.attention_map.focus_center,
        "decay_type" => string(space.attention_map.decay_type)
    )
    
    # Guardar dimensión weights
    data["dimension_weights"] = space.dimension_weights
    
    # Guardar en archivo
    save(filename, data)
    
    return filename
end

"""
    load_semantic_space(filename)

Carga un espacio semántico desde un archivo.
"""
function load_semantic_space(
    filename::String;
    brain::Union{Brain_Space, Nothing}=nothing,
    tokenizer::Union{TensorialTokenizer, Nothing}=nothing
)
    # Cargar datos
    data = load(filename)
    
    # Obtener dimensiones
    dimensions = data["dimensions"]
    
    # Obtener pesos de dimensiones
    dimension_weights = data["dimension_weights"]
    
    # Crear espacio
    space = Semantic3DSpace(
        dimensions,
        brain=brain,
        tokenizer=tokenizer,
        dimension_weights=dimension_weights
    )
    
    # Cargar mapa de atención
    attention_data = data["attention_map"]
    attention_field = attention_data["attention_field"]
    focus_factor = attention_data["focus_factor"]
    effective_radius = attention_data["effective_radius"]
    focus_center = attention_data["focus_center"]
    decay_type = Symbol(attention_data["decay_type"])
    
    space.attention_map = SpatialAttentionMap(
        attention_field,
        focus_factor,
        effective_radius,
        focus_center,
        decay_type
    )
    
    # Cargar conceptos
    concepts_data = data["concepts"]
    
    for (id, concept_data) in concepts_data
        # Crear representación
        representation = SemanticRepresentation(
            concept_data["label"],
            concept_data["tensor"],
            id=concept_data["id"],
            metadata=concept_data["metadata"],
            confidence=concept_data["confidence"]
        )
        
        # Añadir a espacio
        add_concept!(space, representation)
    end
    
    return space
end

# Exportar tipos y funciones principales
export SemanticRepresentation, Semantic3DSpace,
       add_concept!, remove_concept!, query, semantic_search,
       create_concept_from_text, merge_representations,
       find_related_concepts, process_query_in_brain!,
       save_semantic_space, load_semantic_space,
       visualize_semantic_space

end # module SemanticSpace