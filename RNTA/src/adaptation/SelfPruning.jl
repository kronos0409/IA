# adaptation/SelfPruning.jl
# Implementa mecanismos de auto-optimización y poda de conexiones

module SelfPruning

using Statistics
using LinearAlgebra
using Random
using UUIDs
# Importaciones de otros módulos de RNTA
using ..BrainSpace
using ..TensorNeuron
using ..Connections
"""
    PruningParameters

Parámetros que controlan el proceso de auto-poda y optimización.
"""
struct PruningParameters
    # Umbral de actividad mínima para mantener conexiones
    activity_threshold::Float32
    
    # Probabilidad base de poda para conexiones inactivas
    base_pruning_probability::Float32
    
    # Factor que modula la poda según la edad de la conexión
    age_factor::Float32
    
    # Umbral de edad (en segundos) por encima del cual se protegen las conexiones
    age_protection_threshold::Float64
    
    # Factor de protección para conexiones importantes
    importance_protection_factor::Float32
    
    # Equilibrio excitatorio-inhibitorio objetivo (proporción)
    ei_balance_target::Float32
    
    # Peso para mantener el equilibrio excitatorio-inhibitorio
    ei_balance_weight::Float32
    
    # Estrategia de poda
    pruning_strategy::Symbol
end

# Constructor con valores por defecto
function PruningParameters(;
    activity_threshold::Float32=0.1f0,
    base_pruning_probability::Float32=0.5f0,
    age_factor::Float32=0.2f0,
    age_protection_threshold::Float64=300.0,  # 5 minutos
    importance_protection_factor::Float32=2.0f0,
    ei_balance_target::Float32=0.8f0,  # 80% excitatorias, 20% inhibitorias
    ei_balance_weight::Float32=0.3f0,
    pruning_strategy::Symbol=:adaptive
)
    return PruningParameters(
        activity_threshold,
        base_pruning_probability,
        age_factor,
        age_protection_threshold,
        importance_protection_factor,
        ei_balance_target,
        ei_balance_weight,
        pruning_strategy
    )
end

"""
    self_prune!(brain, params=PruningParameters())

Ejecuta el proceso de auto-poda en el espacio cerebral completo.
"""
function self_prune!(
    brain::Brain_Space,
    params::PruningParameters=PruningParameters()
)
    # Estadísticas iniciales
    initial_connections = length(brain.connections)
    
    # Analizar conexiones antes de la poda
    initial_analysis = analyze_connections(brain)
    
    # Seleccionar estrategia de poda
    if params.pruning_strategy == :activity
        # Poda basada únicamente en actividad
        pruned_indices = activity_based_pruning(brain.connections, params)
    elseif params.pruning_strategy == :importance
        # Poda basada en importancia de conexiones
        pruned_indices = importance_based_pruning(brain, params)
    elseif params.pruning_strategy == :balanced
        # Poda que mantiene balance excitatorio-inhibitorio
        pruned_indices = balanced_pruning(brain, params)
    else
        # Estrategia adaptativa (por defecto)
        pruned_indices = adaptive_pruning(brain, params)
    end
    
    # Ordenar índices en orden descendente para eliminar sin afectar otros índices
    sort!(pruned_indices, rev=true)
    
    # Eliminar conexiones
    for idx in pruned_indices
        if idx <= length(brain.connections)  # Verificación de seguridad
            deleteat!(brain.connections, idx)
        end
    end
    
    # Estadísticas finales
    final_connections = length(brain.connections)
    num_pruned = initial_connections - final_connections
    
    # Analizar conexiones después de la poda
    final_analysis = analyze_connections(brain)
    
    # Construir informe de poda
    pruning_report = Dict{Symbol, Any}(
        :initial_connections => initial_connections,
        :final_connections => final_connections,
        :connections_pruned => num_pruned,
        :pruning_ratio => num_pruned / max(1, initial_connections),
        :initial_analysis => initial_analysis,
        :final_analysis => final_analysis
    )
    
    return pruning_report
end

"""
    activity_based_pruning(connections, params)

Realiza poda basada únicamente en niveles de actividad.
"""
function activity_based_pruning(
    connections::Vector{Tensor_Connection},
    params::PruningParameters
)
    pruned_indices = Int[]
    
    # Para cada conexión, evaluar si debe ser podada
    for (idx, connection) in enumerate(connections)
        # Calcular actividad media reciente
        recent_activity = mean(connection.activity_history)
        
        # Verificar si la conexión debe ser podada
        if recent_activity < params.activity_threshold
            # Probabilidad de poda basada en edad de la conexión
            age = time() - connection.creation_time
            age_protection = min(1.0f0, Float32(age / params.age_protection_threshold))
            
            # Reducir probabilidad de poda para conexiones más antiguas
            prune_probability = params.base_pruning_probability * (1.0f0 - params.age_factor * age_protection)
            
            # Decidir si podar
            if rand() < prune_probability
                push!(pruned_indices, idx)
            end
        end
    end
    
    return pruned_indices
end

"""
    importance_based_pruning(brain, params)

Realiza poda basada en importancia relativa de conexiones.
"""
function importance_based_pruning(
    brain::Brain_Space,
    params::PruningParameters
)
    # Calcular importancia de cada conexión
    importance_scores = calculate_connection_importance(brain)
    
    # Determinar umbral de importancia para poda
    # Podamos conexiones en el percentil inferior
    importance_threshold = quantile(importance_scores, 0.2)
    
    pruned_indices = Int[]
    
    # Para cada conexión, evaluar si debe ser podada
    for (idx, connection) in enumerate(brain.connections)
        # Obtener importancia
        importance = importance_scores[idx]
        
        # Verificar si es candidata para poda
        if importance < importance_threshold
            # Calcular actividad reciente
            recent_activity = mean(connection.activity_history)
            
            # Factor combinado de protección
            protection_factor = (importance / importance_threshold) * params.importance_protection_factor
            
            # Probabilidad ajustada de poda
            prune_probability = params.base_pruning_probability / protection_factor
            
            # Reducir probabilidad si tiene actividad reciente
            if recent_activity > 0
                prune_probability *= (params.activity_threshold / max(params.activity_threshold, recent_activity))
            end
            
            # Decidir si podar
            if rand() < prune_probability
                push!(pruned_indices, idx)
            end
        end
    end
    
    return pruned_indices
end

"""
    balanced_pruning(brain, params)

Realiza poda manteniendo un balance entre conexiones excitatorias e inhibitorias.
"""
function balanced_pruning(
    brain::Brain_Space,
    params::PruningParameters
)
    # Calcular proporciones actuales
    excitatory_count = count(c -> c.connection_type == :excitatory, brain.connections)
    inhibitory_count = count(c -> c.connection_type == :inhibitory, brain.connections)
    total_count = excitatory_count + inhibitory_count
    
    # Calcular ratio actual
    current_ratio = excitatory_count / max(1, total_count)
    
    # Determinar tipo a podar preferentemente
    prune_excitatory = current_ratio > params.ei_balance_target
    
    pruned_indices = Int[]
    
    # Para cada conexión, evaluar si debe ser podada
    for (idx, connection) in enumerate(brain.connections)
        # Calcular actividad media reciente
        recent_activity = mean(connection.activity_history)
        
        # Verificar si la conexión es candidata para poda
        if recent_activity < params.activity_threshold
            # Probabilidad base de poda
            prune_probability = params.base_pruning_probability
            
            # Ajustar según desequilibrio
            if (connection.connection_type == :excitatory && prune_excitatory) ||
               (connection.connection_type == :inhibitory && !prune_excitatory)
                # Aumentar probabilidad para el tipo que está en exceso
                balance_factor = abs(current_ratio - params.ei_balance_target) * 
                                 params.ei_balance_weight
                prune_probability *= (1.0f0 + balance_factor)
            else
                # Reducir probabilidad para el tipo que está en déficit
                balance_factor = abs(current_ratio - params.ei_balance_target) * 
                                 params.ei_balance_weight
                prune_probability *= (1.0f0 - balance_factor)
            end
            
            # Ajustar por edad
            age = time() - connection.creation_time
            age_protection = min(1.0f0, Float32(age / params.age_protection_threshold))
            prune_probability *= (1.0f0 - params.age_factor * age_protection)
            
            # Decidir si podar
            if rand() < prune_probability
                push!(pruned_indices, idx)
            end
        end
    end
    
    return pruned_indices
end

"""
    adaptive_pruning(brain, params)

Realiza poda adaptativa combinando múltiples estrategias.
"""
function adaptive_pruning(
    brain::Brain_Space,
    params::PruningParameters
)
    # Análisis de conectividad global
    connectivity_analysis = analyze_connections(brain)
    
    # Calcular importancia de cada conexión
    importance_scores = calculate_connection_importance(brain)
    
    # Proporciones actuales
    excitatory_count = connectivity_analysis[:excitatory_count]
    inhibitory_count = connectivity_analysis[:inhibitory_count]
    total_count = excitatory_count + inhibitory_count
    
    # Calcular ratio actual
    current_ratio = excitatory_count / max(1, total_count)
    
    # Determinar tipo a podar preferentemente
    prune_excitatory = current_ratio > params.ei_balance_target
    
    # Análisis de densidad de conexiones
    density = connectivity_analysis[:density]
    
    # Ajustar umbral de actividad según densidad
    # Cerebros más densos tienen umbrales más altos
    adaptive_threshold = params.activity_threshold * (1.0f0 + density)
    
    pruned_indices = Int[]
    
    # Para cada conexión, evaluar si debe ser podada
    for (idx, connection) in enumerate(brain.connections)
        # Calcular actividad media reciente
        recent_activity = mean(connection.activity_history)
        
        # Solo considerar conexiones con baja actividad
        if recent_activity < adaptive_threshold
            # Probabilidad base de poda
            prune_probability = params.base_pruning_probability
            
            # Ajustar según importancia relativa
            importance = importance_scores[idx]
            importance_factor = 1.0f0 - min(1.0f0, importance * params.importance_protection_factor)
            prune_probability *= importance_factor
            
            # Ajustar según balance E/I
            if (connection.connection_type == :excitatory && prune_excitatory) ||
               (connection.connection_type == :inhibitory && !prune_excitatory)
                # Aumentar probabilidad para el tipo que está en exceso
                balance_factor = abs(current_ratio - params.ei_balance_target) * 
                                 params.ei_balance_weight
                prune_probability *= (1.0f0 + balance_factor)
            else
                # Reducir probabilidad para el tipo que está en déficit
                balance_factor = abs(current_ratio - params.ei_balance_target) * 
                                 params.ei_balance_weight
                prune_probability *= (1.0f0 - balance_factor)
            end
            
            # Ajustar por edad
            age = time() - connection.creation_time
            age_protection = min(1.0f0, Float32(age / params.age_protection_threshold))
            prune_probability *= (1.0f0 - params.age_factor * age_protection)
            
            # Decidir si podar
            if rand() < prune_probability
                push!(pruned_indices, idx)
            end
        end
    end
    
    return pruned_indices
end

"""
    calculate_connection_importance(brain)

Calcula la importancia relativa de cada conexión en el cerebro.
"""
function calculate_connection_importance(brain::Brain_Space)
    # Vector de puntuaciones de importancia
    importance_scores = Float32[]
    
    # Para cada conexión, calcular su importancia
    for connection in brain.connections
        # Factores que contribuyen a importancia
        
        # 1. Fuerza de conexión
        strength_factor = connection.strength
        
        # 2. Actividad histórica
        activity_factor = mean(connection.activity_history)
        
        # 3. Centralidad de neuronas conectadas
        source_centrality = calculate_neuron_centrality(brain, connection.source_id)
        target_centrality = calculate_neuron_centrality(brain, connection.target_id)
        centrality_factor = (source_centrality + target_centrality) / 2
        
        # 4. Especialización de neuronas conectadas
        source_spec = get_neuron_specialization(brain, connection.source_id)
        target_spec = get_neuron_specialization(brain, connection.target_id)
        specialization_factor = (source_spec + target_spec) / 2
        
        # Combinar factores en puntuación final
        importance = 0.3f0 * strength_factor + 
                     0.3f0 * activity_factor + 
                     0.2f0 * centrality_factor + 
                     0.2f0 * specialization_factor
        
        push!(importance_scores, importance)
    end
    
    return importance_scores
end

"""
    calculate_neuron_centrality(brain, neuron_id)

Calcula la centralidad de una neurona basada en sus conexiones.
"""
function calculate_neuron_centrality(brain::Brain_Space, neuron_id::UUID)
    # Contar conexiones entrantes y salientes
    in_connections = count(c -> c.target_id == neuron_id, brain.connections)
    out_connections = count(c -> c.source_id == neuron_id, brain.connections)
    
    # Normalizar por número total de conexiones
    total_connections = length(brain.connections)
    if total_connections == 0
        return 0.0f0
    end
    
    # Centralidad como proporción de conexiones que involucran esta neurona
    centrality = (in_connections + out_connections) / (2 * total_connections)
    
    return centrality
end

"""
    get_neuron_specialization(brain, neuron_id)

Obtiene el nivel de especialización de una neurona.
"""
function get_neuron_specialization(brain::Brain_Space, neuron_id::UUID)
    # Encontrar neurona por ID
    for (_, neuron) in brain.neurons
        if neuron.id == neuron_id
            return neuron.specialization
        end
    end
    
    # Si no se encuentra, devolver valor bajo
    return 0.1f0
end

"""
    analyze_connections(brain)

Analiza la estructura de conexiones del cerebro.
"""
function analyze_connections(brain::Brain_Space)
    # Estadísticas básicas
    total_connections = length(brain.connections)
    excitatory_count = count(c -> c.connection_type == :excitatory, brain.connections)
    inhibitory_count = count(c -> c.connection_type == :inhibitory, brain.connections)
    
    # Calcular actividades medias
    all_activities = Float32[]
    excitatory_activities = Float32[]
    inhibitory_activities = Float32[]
    
    for connection in brain.connections
        activity = mean(connection.activity_history)
        push!(all_activities, activity)
        
        if connection.connection_type == :excitatory
            push!(excitatory_activities, activity)
        else
            push!(inhibitory_activities, activity)
        end
    end
    
    # Calcular densidad de conexiones
    total_possible = length(brain.neurons) * (length(brain.neurons) - 1)
    density = total_possible > 0 ? total_connections / total_possible : 0.0
    
    # Construir análisis
    analysis = Dict{Symbol, Any}(
        :total_connections => total_connections,
        :excitatory_count => excitatory_count,
        :inhibitory_count => inhibitory_count,
        :ei_ratio => excitatory_count / max(1, total_connections),
        :mean_activity => isempty(all_activities) ? 0.0f0 : mean(all_activities),
        :excitatory_activity => isempty(excitatory_activities) ? 0.0f0 : mean(excitatory_activities),
        :inhibitory_activity => isempty(inhibitory_activities) ? 0.0f0 : mean(inhibitory_activities),
        :density => density
    )
    
    return analysis
end

"""
    optimize_connection_strengths!(brain)

Optimiza las fuerzas de conexión basado en su actividad.
"""
function optimize_connection_strengths!(brain::Brain_Space)
    # Contador de conexiones modificadas
    modified_count = 0
    
    # Para cada conexión
    for connection in brain.connections
        # Calcular actividad media reciente
        recent_activity = mean(connection.activity_history)
        
        # Umbral adaptativo basado en nivel de actividad global
        all_activities = [mean(c.activity_history) for c in brain.connections]
        mean_activity = mean(all_activities)
        std_activity = std(all_activities)
        
        # Umbral = media + desviación
        threshold = mean_activity + 0.5f0 * std_activity
        
        # Si la actividad es alta, reforzar conexión
        if recent_activity > threshold
            # Factor de refuerzo
            strengthening_factor = 1.0f0 + 0.1f0 * (recent_activity - threshold) / std_activity
            
            # Limitar factor máximo
            strengthening_factor = min(1.2f0, strengthening_factor)
            
            # Aplicar refuerzo
            connection.strength *= strengthening_factor
            connection.weight .*= strengthening_factor
            
            modified_count += 1
            
        # Si la actividad es baja pero no tanto como para podar, debilitar
        elseif recent_activity < threshold - 0.5f0 * std_activity && recent_activity > 0.05f0
            # Factor de debilitamiento
            weakening_factor = 1.0f0 - 0.05f0 * (threshold - recent_activity) / std_activity
            
            # Limitar factor mínimo
            weakening_factor = max(0.9f0, weakening_factor)
            
            # Aplicar debilitamiento
            connection.strength *= weakening_factor
            connection.weight .*= weakening_factor
            
            modified_count += 1
        end
    end
    
    return modified_count
end

"""
    redistribute_connection_resources!(brain, pruning_report)

Redistribuye recursos de conexiones podadas para fortalecer conexiones importantes.
"""
function redistribute_connection_resources!(
    brain::Brain_Space,
    pruning_report::Dict{Symbol, Any}
)
    # Si no se podó nada, no hay nada que redistribuir
    num_pruned = pruning_report[:connections_pruned]
    if num_pruned == 0
        return 0
    end
    
    # Calcular importancia de conexiones restantes
    importance_scores = calculate_connection_importance(brain)
    
    # Determinar las conexiones más importantes (top 30%)
    if isempty(importance_scores)
        return 0
    end
    
    sorted_indices = sortperm(importance_scores, rev=true)
    num_to_strengthen = max(1, round(Int, 0.3 * length(importance_scores)))
    strengthen_indices = sorted_indices[1:min(end, num_to_strengthen)]
    
    # Factor de fortalecimiento base (dependiente de cuánto se podó)
    pruning_ratio = pruning_report[:pruning_ratio]
    strengthen_factor = 1.0f0 + 0.2f0 * pruning_ratio
    
    # Fortalecer conexiones importantes
    for idx in strengthen_indices
        connection = brain.connections[idx]
        
        # Aplicar fortalecimiento
        connection.strength *= strengthen_factor
        connection.weight .*= strengthen_factor
    end
    
    return length(strengthen_indices)
end

# Exportar tipos y funciones principales
export PruningParameters,
       self_prune!, analyze_connections,
       optimize_connection_strengths!,
       redistribute_connection_resources!

end # module SelfPruning