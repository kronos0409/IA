# architecture/HippocampalMemory.jl
# Implementa un sistema de memoria inspirado en el hipocampo

module Hippocampal_Memory

using LinearAlgebra
using Statistics
using Random
using DataStructures
using UUIDs
# Importaciones de otros módulos de RNTA
using ..BrainSpace
using ..TensorNeuron
using ..Connections
using ..TensorOperations
using ..SpatialAttention

"""
    MemoryPattern

Representa un patrón almacenado en la memoria hipocampal.
"""
struct MemoryPattern
    # Identificador único
    id::UUID
    
    # Representación tensorial del patrón
    tensor::Array{Float32,3}
    
    # Contexto asociado (opcional)
    context::Union{Array{Float32,3}, Nothing}
    
    # Etiqueta o metadatos del patrón
    label::String
    
    # Fuerza del patrón (0.0-1.0)
    strength::Float32
    
    # Tiempo de creación (para decaimiento)
    creation_time::Float64
    
    # Contador de accesos
    access_count::Int
    
    # Último tiempo de acceso
    last_access_time::Float64
    
    # Metadatos adicionales
    metadata::Dict{Symbol, Any}
end

"""
Constructor para MemoryPattern
"""
function MemoryPattern(
    tensor::Array{T,3},
    label::String="";
    context::Union{Array{S,3}, Nothing}=nothing,
    strength::Float32=1.0f0,
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}()
) where {T <: AbstractFloat, S <: AbstractFloat}
    current_time = time()
    
    return MemoryPattern(
        uuid4(),
        convert(Array{Float32,3}, tensor),
        context,
        label,
        strength,
        current_time,
        1,
        current_time,
        metadata
    )
end

"""
    MemoryConfig

Configuración para el sistema de memoria hipocampal.
"""
struct MemoryConfig
    # Dimensiones del espacio de memoria
    dimensions::NTuple{3,Int}
    
    # Capacidad máxima de patrones
    max_capacity::Int
    
    # Factor de decaimiento temporal
    decay_factor::Float32
    
    # Umbral de similitud para considerarse el mismo patrón
    similarity_threshold::Float32
    
    # Estrategia de consolidación de memoria
    consolidation_strategy::Symbol
    
    # Modo de recuperación
    retrieval_mode::Symbol
    
    # Peso del contexto en la recuperación
    context_weight::Float32
    
    # Peso para recencia de acceso
    recency_weight::Float32
    
    # Peso para frecuencia de acceso
    frequency_weight::Float32
end

# Constructor con valores por defecto
function MemoryConfig(;
    dimensions::NTuple{3,Int}=(10, 10, 10),
    max_capacity::Int=1000,
    decay_factor::Float32=0.95f0,
    similarity_threshold::Float32=0.8f0,
    consolidation_strategy::Symbol=:strength_based,
    retrieval_mode::Symbol=:similarity,
    context_weight::Float32=0.3f0,
    recency_weight::Float32=0.4f0,
    frequency_weight::Float32=0.3f0
)
    return MemoryConfig(
        dimensions,
        max_capacity,
        decay_factor,
        similarity_threshold,
        consolidation_strategy,
        retrieval_mode,
        context_weight,
        recency_weight,
        frequency_weight
    )
end

"""
    HippocampalMemory

Sistema de memoria inspirado en el hipocampo para almacenamiento y recuperación de patrones.
"""
mutable struct HippocampalMemory
    # Identificador único
    id::UUID
    
    # Configuración del sistema
    config::MemoryConfig
    
    # Patrones almacenados
    patterns::Vector{MemoryPattern}
    
    # Índice de similitud para búsqueda rápida
    similarity_index::Dict{UInt64, Vector{UUID}}
    
    # Contexto actual
    current_context::Union{Array{Float32,3}, Nothing}
    
    # Caché de patrones recientes
    recent_cache::OrderedDict{UUID, Float32}
    
    # Contador de operaciones
    operation_count::Int
    
    # Último estado de salida
    last_output::Union{Array{Float32,3}, Nothing}
    
    # Metadatos adicionales
    metadata::Dict{Symbol, Any}
end

"""
Constructor para HippocampalMemory
"""
function HippocampalMemory(
    config::MemoryConfig;
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    return HippocampalMemory(
        uuid4(),
        config,
        Vector{MemoryPattern}(),
        Dict{UInt64, Vector{UUID}}(),
        nothing,
        OrderedDict{UUID, Float32}(),
        0,
        nothing,
        metadata
    )
end

"""
    store_pattern!(memory, pattern_tensor, label=""; context=nothing, metadata=Dict{Symbol,Any}())

Almacena un nuevo patrón en la memoria hipocampal.
"""
function store_pattern!(
    memory::HippocampalMemory,
    pattern_tensor::Array{T,3},
    label::String="";
    context::Union{Array{S,3}, Nothing}=nothing,
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}()
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(pattern_tensor) != memory.config.dimensions
        pattern_tensor = tensor_interpolation(pattern_tensor, memory.config.dimensions)
    end
    
    if !isnothing(context) && size(context) != memory.config.dimensions
        context = tensor_interpolation(context, memory.config.dimensions)
    end
    
    # Si no hay contexto explícito, usar el contexto actual
    if isnothing(context) && !isnothing(memory.current_context)
        context = memory.current_context
    end
    
    # Verificar si ya existe un patrón similar
    similar_pattern_id = find_similar_pattern(memory, pattern_tensor)
    
    if !isnothing(similar_pattern_id)
        # Actualizar patrón existente
        update_existing_pattern!(memory, similar_pattern_id, pattern_tensor, context)
        return similar_pattern_id
    end
    
    # Crear nuevo patrón
    pattern = MemoryPattern(
        pattern_tensor,
        label,
        context=context,
        metadata=metadata
    )
    
    # Verificar capacidad
    if length(memory.patterns) >= memory.config.max_capacity
        # Liberar espacio según estrategia de consolidación
        consolidate_memory!(memory)
    end
    
    # Almacenar patrón
    push!(memory.patterns, pattern)
    
    # Actualizar índice de similitud
    update_similarity_index!(memory, pattern)
    
    # Incrementar contador de operaciones
    memory.operation_count += 1
    
    return pattern.id
end

"""
    update_existing_pattern!(memory, pattern_id, new_tensor, new_context)

Actualiza un patrón existente con nueva información.
"""
function update_existing_pattern!(
    memory::HippocampalMemory,
    pattern_id::UUID,
    new_tensor::Array{T,3},
    new_context::Union{Array{S,3}, Nothing}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Buscar patrón
    idx = findfirst(p -> p.id == pattern_id, memory.patterns)
    
    if isnothing(idx)
        return nothing
    end
    
    # Extraer patrón
    pattern = memory.patterns[idx]
    
    # Crear patrón actualizado
    updated_pattern = MemoryPattern(
        # Combinar tensor antiguo y nuevo
        0.7f0 * pattern.tensor + 0.3f0 * new_tensor,
        pattern.label,
        # Actualizar contexto si se proporciona
        context=isnothing(new_context) ? pattern.context : 
                0.7f0 * pattern.context + 0.3f0 * new_context,
        # Incrementar fuerza
        strength=min(1.0f0, pattern.strength + 0.1f0),
        # Mantener metadatos
        metadata=pattern.metadata
    )
    
    # Mantener ID y contadores del patrón original
    updated_pattern = MemoryPattern(
        pattern.id,  # Mantener ID original
        updated_pattern.tensor,
        updated_pattern.context,
        updated_pattern.label,
        updated_pattern.strength,
        pattern.creation_time,  # Mantener tiempo de creación
        pattern.access_count + 1,  # Incrementar contador de accesos
        time(),  # Actualizar último acceso
        updated_pattern.metadata
    )
    
    # Actualizar en colección
    memory.patterns[idx] = updated_pattern
    
    # Actualizar caché de recientes
    memory.recent_cache[pattern.id] = 1.0f0
    
    # Incrementar contador de operaciones
    memory.operation_count += 1
    
    return pattern.id
end

"""
    find_similar_pattern(memory, query_tensor)

Busca un patrón similar al tensor de consulta.
"""
function find_similar_pattern(
    memory::HippocampalMemory,
    query_tensor::Array{T,3}
) where T <: AbstractFloat
    # Si no hay patrones, devolver nada
    if isempty(memory.patterns)
        return nothing
    end
    
    # Buscar patrón más similar
    max_similarity = 0.0f0
    most_similar_id = nothing
    
    for pattern in memory.patterns
        similarity = tensor_similarity(pattern.tensor, query_tensor)
        
        if similarity > max_similarity
            max_similarity = similarity
            most_similar_id = pattern.id
        end
    end
    
    # Verificar umbral de similitud
    if max_similarity >= memory.config.similarity_threshold
        return most_similar_id
    else
        return nothing
    end
end

"""
    tensor_similarity(tensor1, tensor2)

Calcula la similitud entre dos tensores.
"""
function tensor_similarity(
    tensor1::Array{T,3},
    tensor2::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(tensor1) != size(tensor2)
        tensor2 = tensor_interpolation(tensor2, size(tensor1))
    end
    
    # Aplanar tensores
    flat1 = vec(tensor1)
    flat2 = vec(tensor2)
    
    # Calcular similitud coseno
    dot_product = dot(flat1, flat2)
    norm1 = norm(flat1)
    norm2 = norm(flat2)
    
    # Evitar división por cero
    if norm1 < 1e-8 || norm2 < 1e-8
        return 0.0f0
    end
    
    return dot_product / (norm1 * norm2)
end

"""
    update_similarity_index!(memory, pattern)

Actualiza el índice de similitud con un nuevo patrón.
"""
function update_similarity_index!(
    memory::HippocampalMemory,
    pattern::MemoryPattern
)
    # Calcular características para indexación
    features = extract_pattern_features(pattern.tensor)
    
    # Para cada característica, añadir patrón al índice
    for feature in features
        feature_hash = hash(feature)
        
        if !haskey(memory.similarity_index, feature_hash)
            memory.similarity_index[feature_hash] = UUID[]
        end
        
        push!(memory.similarity_index[feature_hash], pattern.id)
    end
    
    return memory.similarity_index
end

"""
    extract_pattern_features(tensor)

Extrae características para indexación de un tensor.
"""
function extract_pattern_features(tensor::Array{T,3}) where T <: AbstractFloat
    # Esta es una implementación simplificada
    # En la práctica, se usarían características más sofisticadas
    
    dim_x, dim_y, dim_z = size(tensor)
    features = []
    
    # Características de bajo nivel: valores medios por región
    regions_x = 2
    regions_y = 2
    regions_z = 2
    
    for x in 1:regions_x
        for y in 1:regions_y
            for z in 1:regions_z
                x_start = div((x-1) * dim_x, regions_x) + 1
                y_start = div((y-1) * dim_y, regions_y) + 1
                z_start = div((z-1) * dim_z, regions_z) + 1
                
                x_end = div(x * dim_x, regions_x)
                y_end = div(y * dim_y, regions_y)
                z_end = div(z * dim_z, regions_z)
                
                # Extraer región
                region = tensor[x_start:x_end, y_start:y_end, z_start:z_end]
                
                # Calcular valor medio
                mean_value = mean(region)
                
                # Discretizar para indexación
                discretized = round(mean_value * 10) / 10
                
                # Crear característica
                feature = (x, y, z, discretized)
                push!(features, feature)
            end
        end
    end
    
    return features
end

"""
    retrieve_pattern(memory, query_tensor; context=nothing, top_k=1)

Recupera los patrones más similares a la consulta.
"""
function retrieve_pattern(
    memory::HippocampalMemory,
    query_tensor::Array{T,3};
    context::Union{Array{S,3}, Nothing}=nothing,
    top_k::Int=1
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(query_tensor) != memory.config.dimensions
        query_tensor = tensor_interpolation(query_tensor, memory.config.dimensions)
    end
    
    if !isnothing(context) && size(context) != memory.config.dimensions
        context = tensor_interpolation(context, memory.config.dimensions)
    end
    
    # Si no hay contexto explícito, usar el contexto actual
    if isnothing(context) && !isnothing(memory.current_context)
        context = memory.current_context
    end
    
    # Si no hay patrones, devolver nada
    if isempty(memory.patterns)
        return []
    end
    
    # Calcular similitud con cada patrón
    similarities = []
    
    for pattern in memory.patterns
        # Similitud de tensor
        tensor_sim = tensor_similarity(pattern.tensor, query_tensor)
        
        # Similitud de contexto
        context_sim = 0.0f0
        if !isnothing(context) && !isnothing(pattern.context)
            context_sim = tensor_similarity(pattern.context, context)
        end
        
        # Calcular puntuación combinada
        if memory.config.retrieval_mode == :similarity
            # Combinar similitud de tensor y contexto
            score = (1.0f0 - memory.config.context_weight) * tensor_sim +
                    memory.config.context_weight * context_sim
        elseif memory.config.retrieval_mode == :adaptive
            # Combinar tensor, contexto, recencia y frecuencia
            recency_score = 0.0f0
            frequency_score = 0.0f0
            
            # Calcular puntuación de recencia
            time_diff = time() - pattern.last_access_time
            recency_score = exp(-time_diff / 3600)  # Decaimiento exponencial por hora
            
            # Calcular puntuación de frecuencia
            frequency_score = min(1.0f0, pattern.access_count / 20.0f0)  # Normalizar a 0-1
            
            # Combinar puntuaciones
            score = (1.0f0 - memory.config.context_weight - memory.config.recency_weight - memory.config.frequency_weight) * tensor_sim +
                    memory.config.context_weight * context_sim +
                    memory.config.recency_weight * recency_score +
                    memory.config.frequency_weight * frequency_score
        else
            # Por defecto, usar solo similitud de tensor
            score = tensor_sim
        end
        
        # Aplicar factor de fuerza del patrón
        score *= pattern.strength
        
        push!(similarities, (pattern, score))
    end
    
    # Ordenar por similitud descendente
    sort!(similarities, by=x -> x[2], rev=true)
    
    # Limitar a top_k resultados
    results = similarities[1:min(top_k, length(similarities))]
    
    # Actualizar estadísticas de los patrones recuperados
    for (pattern, _) in results
        update_pattern_stats!(memory, pattern.id)
    end
    
    # Si hay resultados, actualizar último output
    if !isempty(results)
        memory.last_output = results[1][1].tensor
    end
    
    # Incrementar contador de operaciones
    memory.operation_count += 1
    
    return results
end

"""
    update_pattern_stats!(memory, pattern_id)

Actualiza estadísticas de un patrón tras su acceso.
"""
function update_pattern_stats!(
    memory::HippocampalMemory,
    pattern_id::UUID
)
    # Buscar patrón
    idx = findfirst(p -> p.id == pattern_id, memory.patterns)
    
    if isnothing(idx)
        return
    end
    
    # Extraer patrón
    pattern = memory.patterns[idx]
    
    # Crear patrón actualizado con estadísticas incrementadas
    updated_pattern = MemoryPattern(
        pattern.id,
        pattern.tensor,
        pattern.context,
        pattern.label,
        pattern.strength,
        pattern.creation_time,
        pattern.access_count + 1,  # Incrementar contador
        time(),  # Actualizar último acceso
        pattern.metadata
    )
    
    # Actualizar en colección
    memory.patterns[idx] = updated_pattern
    
    # Actualizar caché de recientes
    memory.recent_cache[pattern.id] = 1.0f0
    
    # Si la caché excede cierto tamaño, eliminar entradas antiguas
    if length(memory.recent_cache) > 20
        # Eliminar las entradas más antiguas
        for _ in 1:5
            if !isempty(memory.recent_cache)
                popfirst!(memory.recent_cache)
            end
        end
    end
    
    return updated_pattern
end

"""
    consolidate_memory!(memory)

Consolida la memoria eliminando patrones menos importantes.
"""
function consolidate_memory!(
    memory::HippocampalMemory
)
    # No hacer nada si no se alcanza la capacidad
    if length(memory.patterns) < memory.config.max_capacity
        return
    end
    
    # Estrategia de consolidación
    if memory.config.consolidation_strategy == :strength_based
        # Ordenar patrones por fuerza (los más débiles primero)
        sort!(memory.patterns, by=p -> p.strength)
        
        # Eliminar los patrones más débiles (10% del total)
        num_to_remove = max(1, div(length(memory.patterns), 10))
        
        # Guardar IDs para eliminar del índice
        removed_ids = [p.id for p in memory.patterns[1:num_to_remove]]
        
        # Eliminar patrones
        deleteat!(memory.patterns, 1:num_to_remove)
        
        # Actualizar índice
        clean_similarity_index!(memory, removed_ids)
        
    elseif memory.config.consolidation_strategy == :recency_based
        # Ordenar patrones por tiempo de último acceso (los más antiguos primero)
        sort!(memory.patterns, by=p -> p.last_access_time)
        
        # Eliminar los patrones menos recientes (10% del total)
        num_to_remove = max(1, div(length(memory.patterns), 10))
        
        # Guardar IDs para eliminar del índice
        removed_ids = [p.id for p in memory.patterns[1:num_to_remove]]
        
        # Eliminar patrones
        deleteat!(memory.patterns, 1:num_to_remove)
        
        # Actualizar índice
        clean_similarity_index!(memory, removed_ids)
        
    elseif memory.config.consolidation_strategy == :combined
        # Calcular puntuación combinada para cada patrón
        pattern_scores = []
        
        current_time = time()
        for pattern in memory.patterns
            # Componente de fuerza
            strength_score = pattern.strength
            
            # Componente de recencia
            time_diff = current_time - pattern.last_access_time
            recency_score = exp(-time_diff / 86400)  # Decaimiento exponencial por día
            
            # Componente de frecuencia
            frequency_score = min(1.0f0, pattern.access_count / 100.0f0)
            
            # Puntuación combinada
            score = 0.4f0 * strength_score + 0.4f0 * recency_score + 0.2f0 * frequency_score
            
            push!(pattern_scores, (pattern, score))
        end
        
        # Ordenar por puntuación (los menos importantes primero)
        sort!(pattern_scores, by=x -> x[2])
        
        # Eliminar los patrones menos importantes (10% del total)
        num_to_remove = max(1, div(length(pattern_scores), 10))
        
        # Guardar IDs para eliminar del índice
        removed_ids = [p[1].id for p in pattern_scores[1:num_to_remove]]
        
        # Filtrar patrones a mantener
        keep_patterns = [p[1] for p in pattern_scores[num_to_remove+1:end]]
        memory.patterns = keep_patterns
        
        # Actualizar índice
        clean_similarity_index!(memory, removed_ids)
    end
    
    return memory
end

"""
    clean_similarity_index!(memory, removed_ids)

Limpia el índice de similitud eliminando patrones.
"""
function clean_similarity_index!(
    memory::HippocampalMemory,
    removed_ids::Vector{UUID}
)
    # Para cada entrada en el índice
    for (feature_hash, pattern_ids) in memory.similarity_index
        # Filtrar IDs eliminados
        filtered_ids = filter(id -> !(id in removed_ids), pattern_ids)
        
        if isempty(filtered_ids)
            # Eliminar entrada si queda vacía
            delete!(memory.similarity_index, feature_hash)
        else
            # Actualizar con lista filtrada
            memory.similarity_index[feature_hash] = filtered_ids
        end
    end
    
    return memory.similarity_index
end

"""
    set_context!(memory, context_tensor)

Establece el contexto actual para la memoria.
"""
function set_context!(
    memory::HippocampalMemory,
    context_tensor::Array{T,3}
) where T <: AbstractFloat
    # Asegurar dimensiones compatibles
    if size(context_tensor) != memory.config.dimensions
        context_tensor = tensor_interpolation(context_tensor, memory.config.dimensions)
    end
    
    # Actualizar contexto
    memory.current_context = context_tensor
    
    return memory
end

"""
    apply_memory_decay!(memory)

Aplica decaimiento temporal a los patrones de memoria.
"""
function apply_memory_decay!(memory::HippocampalMemory)
    # No hacer nada si no hay patrones
    if isempty(memory.patterns)
        return
    end
    
    # Calcular factor de decaimiento basado en tiempo desde la última operación
    current_time = time()
    
    # Actualizar fuerza de cada patrón
    for (i, pattern) in enumerate(memory.patterns)
        # Calcular tiempo desde la creación
        time_diff = current_time - pattern.creation_time
        
        # Calcular factor de decaimiento
        decay = memory.config.decay_factor^(time_diff / 86400)  # Decaimiento por día
        
        # Aplicar decaimiento a la fuerza
        new_strength = pattern.strength * decay
        
        # Actualizar patrón
        memory.patterns[i] = MemoryPattern(
            pattern.id,
            pattern.tensor,
            pattern.context,
            pattern.label,
            new_strength,
            pattern.creation_time,
            pattern.access_count,
            pattern.last_access_time,
            pattern.metadata
        )
    end
    
    return memory
end

"""
    complete_pattern(memory, partial_pattern; threshold=0.6)

Completa un patrón parcial usando la memoria.
"""
function complete_pattern(
    memory::HippocampalMemory,
    partial_pattern::Array{T,3};
    threshold::Float32=0.6f0
) where T <: AbstractFloat
    # Asegurar dimensiones compatibles
    if size(partial_pattern) != memory.config.dimensions
        partial_pattern = tensor_interpolation(partial_pattern, memory.config.dimensions)
    end
    
    # Encontrar patrones similares
    matches = retrieve_pattern(memory, partial_pattern, top_k=5)
    
    if isempty(matches)
        return partial_pattern
    end
    
    # Crear máscara para valores significativos en el patrón parcial
    mask = abs.(partial_pattern) .> threshold
    
    # Combinar patrones recuperados
    completed_pattern = copy(partial_pattern)
    
    # Mantener valores originales donde la máscara es verdadera
    for (pattern, similarity) in matches
        # Combinar solo si la similitud es suficiente
        if similarity > threshold
            # Completar partes faltantes
            completed_pattern[.!mask] = pattern.tensor[.!mask]
            break
        end
    end
    
    return completed_pattern
end

"""
    associate_patterns(memory, pattern1, pattern2)

Crea una asociación entre dos patrones.
"""
function associate_patterns(
    memory::HippocampalMemory,
    pattern1_id::UUID,
    pattern2_id::UUID
)
    # Buscar patrones
    idx1 = findfirst(p -> p.id == pattern1_id, memory.patterns)
    idx2 = findfirst(p -> p.id == pattern2_id, memory.patterns)
    
    if isnothing(idx1) || isnothing(idx2)
        return nothing
    end
    
    pattern1 = memory.patterns[idx1]
    pattern2 = memory.patterns[idx2]
    
    # Crear asociación bidireccional actualizando los contextos
    
    # Actualizar contexto de pattern1 con tensor de pattern2
    new_context1 = isnothing(pattern1.context) ? 
                 pattern2.tensor : 
                 0.7f0 * pattern1.context + 0.3f0 * pattern2.tensor
    
    # Actualizar contexto de pattern2 con tensor de pattern1
    new_context2 = isnothing(pattern2.context) ? 
                 pattern1.tensor : 
                 0.7f0 * pattern2.context + 0.3f0 * pattern1.tensor
    
    # Actualizar patrones
    memory.patterns[idx1] = MemoryPattern(
        pattern1.id,
        pattern1.tensor,
        new_context1,
        pattern1.label,
        pattern1.strength,
        pattern1.creation_time,
        pattern1.access_count,
        pattern1.last_access_time,
        pattern1.metadata
    )
    
    memory.patterns[idx2] = MemoryPattern(
        pattern2.id,
        pattern2.tensor,
        new_context2,
        pattern2.label,
        pattern2.strength,
        pattern2.creation_time,
        pattern2.access_count,
        pattern2.last_access_time,
        pattern2.metadata
    )
    
    # Incrementar contador de operaciones
    memory.operation_count += 1
    
    return true
end

"""
    summarize_memory(memory)

Genera un resumen estadístico de la memoria.
"""
function summarize_memory(memory::HippocampalMemory)
    # Calcular estadísticas básicas
    num_patterns = length(memory.patterns)
    
    # Calcular fuerza media
    mean_strength = isempty(memory.patterns) ? 0.0f0 : 
                   mean([p.strength for p in memory.patterns])
    
    # Calcular antigüedad media
    current_time = time()
    mean_age = isempty(memory.patterns) ? 0.0f0 : 
              mean([current_time - p.creation_time for p in memory.patterns]) / 86400  # En días
    
    # Calcular accesos medios
    mean_accesses = isempty(memory.patterns) ? 0.0f0 : 
                   mean([p.access_count for p in memory.patterns])
    
    # Construir resumen
    summary = Dict{Symbol, Any}(
        :capacity => memory.config.max_capacity,
        :used => num_patterns,
        :usage_percent => 100.0f0 * num_patterns / memory.config.max_capacity,
        :mean_strength => mean_strength,
        :mean_age_days => mean_age,
        :mean_accesses => mean_accesses,
        :operation_count => memory.operation_count,
        :retrieval_mode => memory.config.retrieval_mode,
        :consolidation_strategy => memory.config.consolidation_strategy
    )
    
    return summary
end

# Exportar tipos y funciones principales
export MemoryPattern, MemoryConfig, HippocampalMemory,
       store_pattern!, retrieve_pattern, set_context!,
       complete_pattern, associate_patterns, apply_memory_decay!,
       summarize_memory

end # module HippocampalMemory