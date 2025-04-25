"""
Types.jl - Definiciones de tipos básicos para RNTA
==================================================

Contiene definiciones de estructuras de datos fundamentales utilizadas en el sistema RNTA.
"""
module Types

using Dates

"""
    PlasticityParameters

Parámetros para controlar la plasticidad neuronal.
"""
mutable struct PlasticityParameters
    # Tipo de regla de plasticidad a utilizar
    rule_type::Symbol
    
    # Tasa de aprendizaje base para plasticidad
    learning_rate::Float32
    
    # Factor de estabilidad para evitar cambios bruscos
    stability_factor::Float32
    
    # Factor homeostático para mantener niveles de actividad estables
    homeostatic_factor::Float32
    
    # Nivel de sensibilidad al contexto
    context_sensitivity::Float32
    
    # Factor de decaimiento de plasticidad con la especialización
    specialization_decay::Float32
    
    # Historial temporal que considerar para plasticidad
    temporal_window::Int
    
    # Parámetros específicos para cada regla
    rule_params::Dict{Symbol, Any}
end

# Constructor con valores por defecto
function PlasticityParameters(;
    rule_type::Symbol = :hebbian,
    learning_rate::Float32 = 0.01f0,
    stability_factor::Float32 = 0.1f0,
    homeostatic_factor::Float32 = 0.2f0,
    context_sensitivity::Float32 = 0.3f0,
    specialization_decay::Float32 = 0.5f0,
    temporal_window::Int = 3,
    rule_params::Dict{Symbol, Any} = Dict{Symbol, Any}()
)
    return PlasticityParameters(
        rule_type,
        learning_rate,
        stability_factor,
        homeostatic_factor,
        context_sensitivity,
        specialization_decay,
        temporal_window,
        rule_params
    )
end

"""
    AdaptivityParameters

Parámetros para controlar mecanismos de adaptación del cerebro.
"""
mutable struct AdaptivityParameters
    # Umbral de actividad para expansión
    expansion_threshold::Float32
    
    # Umbral para poda de conexiones
    pruning_threshold::Float32
    
    # Umbral para especialización de neuronas
    specialization_threshold::Float32
    
    # Máximo de neuronas permitidas
    max_neurons::Int
    
    # Máximo de conexiones por neurona
    max_connections_per_neuron::Int
    
    # Período entre ciclos de adaptación (en pasos de simulación)
    adaptation_cycle::Int
    
    # Proporción máxima de neuronas que pueden expandirse en un ciclo
    max_expansion_ratio::Float32
    
    # Proporción máxima de conexiones que pueden podarse en un ciclo
    max_pruning_ratio::Float32
    
    # Parámetros adicionales para mecanismos adaptativos
    extra_params::Dict{Symbol, Any}
end

# Constructor con valores por defecto
function AdaptivityParameters(;
    expansion_threshold::Float32 = 0.8f0,
    pruning_threshold::Float32 = 0.1f0,
    specialization_threshold::Float32 = 0.6f0,
    max_neurons::Int = 10000,
    max_connections_per_neuron::Int = 200,
    adaptation_cycle::Int = 100,
    max_expansion_ratio::Float32 = 0.05f0,
    max_pruning_ratio::Float32 = 0.2f0,
    extra_params::Dict{Symbol, Any} = Dict{Symbol, Any}()
)
    return AdaptivityParameters(
        expansion_threshold,
        pruning_threshold,
        specialization_threshold,
        max_neurons,
        max_connections_per_neuron,
        adaptation_cycle,
        max_expansion_ratio,
        max_pruning_ratio,
        extra_params
    )
end

"""
    NeuronMetadata

Metadatos asociados con una neurona tensorial.
"""
struct NeuronMetadata
    # Identificador único
    id::String
    
    # Etiqueta descriptiva (opcional)
    label::String
    
    # Fecha y hora de creación
    creation_time::DateTime
    
    # Coordenadas en el espacio cerebral
    position::NTuple{3, Float32}
    
    # Tipo funcional de la neurona
    functional_type::Symbol
    
    # Nivel de especialización
    specialization::Float32
    
    # Estadísticas de activación
    activation_stats::Dict{Symbol, Any}
    
    # Datos adicionales específicos de la aplicación
    extra_data::Dict{Symbol, Any}
end

# Constructor con valores por defecto
function NeuronMetadata(;
    id::String = string(Base.uuid4()),
    label::String = "",
    creation_time::DateTime = now(),
    position::NTuple{3, Float32} = (0.0f0, 0.0f0, 0.0f0),
    functional_type::Symbol = :generic,
    specialization::Float32 = 0.0f0,
    activation_stats::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    extra_data::Dict{Symbol, Any} = Dict{Symbol, Any}()
)
    return NeuronMetadata(
        id,
        label,
        creation_time,
        position,
        functional_type,
        specialization,
        activation_stats,
        extra_data
    )
end

"""
    TrainingParameters

Parámetros para el entrenamiento del espacio cerebral.
"""
mutable struct TrainingParameters
    # Tasa de aprendizaje base
    learning_rate::Float32
    
    # Tipo de optimizador a utilizar
    optimizer_type::Symbol
    
    # Tamaño del lote (batch)
    batch_size::Int
    
    # Función de pérdida a utilizar
    loss_function::Symbol
    
    # Número máximo de épocas
    max_epochs::Int
    
    # Umbral de convergencia para detener el entrenamiento
    convergence_threshold::Float32
    
    # Valor para recorte de gradientes
    gradient_clip::Float32
    
    # Parámetros específicos del optimizador
    optimizer_params::Dict{Symbol, Any}
    
    # Parámetros adicionales de entrenamiento
    extra_params::Dict{Symbol, Any}
end

# Constructor con valores por defecto
function TrainingParameters(;
    learning_rate::Float32 = 0.001f0,
    optimizer_type::Symbol = :adam,
    batch_size::Int = 32,
    loss_function::Symbol = :mse,
    max_epochs::Int = 100,
    convergence_threshold::Float32 = 0.001f0,
    gradient_clip::Float32 = 5.0f0,
    optimizer_params::Dict{Symbol, Any} = Dict{Symbol, Any}(:beta1 => 0.9f0, :beta2 => 0.999f0, :epsilon => 1f-8),
    extra_params::Dict{Symbol, Any} = Dict{Symbol, Any}()
)
    return TrainingParameters(
        learning_rate,
        optimizer_type,
        batch_size,
        loss_function,
        max_epochs,
        convergence_threshold,
        gradient_clip,
        optimizer_params,
        extra_params
    )
end

# Exportar tipos
export PlasticityParameters, AdaptivityParameters, NeuronMetadata, TrainingParameters

end # module Types