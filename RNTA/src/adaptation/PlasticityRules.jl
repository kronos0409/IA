"""
PlasticityRules.jl - Implementación de reglas de plasticidad para RNTA
======================================================================

Este módulo implementa las reglas de plasticidad neuronal que permiten
a las neuronas tensoriales adaptarse y aprender de sus entradas.
"""
module PlasticityRules

using Statistics
using LinearAlgebra
using Random
using Dates

# Tipos de neuronas y conexiones (serán importados del módulo principal en la versión final)
# Por ahora los definiremos aquí para que el archivo sea independiente
struct BrainSpace
    dimensions::NTuple{3,Int}
    neurons::Dict{String,Any}
    connections::Dict{String,Any}
    global_state::Array{Float32,3}
    attention_map::Array{Float32,3}
    activity_map::Array{Float32,3}
    history::Vector{Dict{Symbol,Any}}
    creation_time::DateTime
    metadata::Dict{Symbol,Any}
end

struct TensorNeuron
    id::String
    position::NTuple{3,Float32}
    receptive_field::Any
    output_field::Any
    transformation_kernel::Array{Float32,3}
    bias_tensor::Array{Float32,3}
    state::Array{Float32,3}
    activation_history::Vector{Array{Float32,3}}
    functional_type::Symbol
    specialization::Float32
    creation_time::DateTime
    metadata::Dict{Symbol,Any}
    plasticity::Any  # Aquí irá PlasticityParameters
end

struct TensorConnection
    id::String
    source_id::String
    target_id::String
    weight::Array{Float32,3}
    strength::Float32
    activity_history::Vector{Float32}
    connection_type::Symbol
    creation_time::DateTime
    metadata::Dict{Symbol,Any}
end

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
    PlasticityRuleType

Tipos de reglas de plasticidad disponibles.
"""
@enum PlasticityRuleType begin
    HebbianRule
    BCMRule
    STDPRule
    HomeostaticRule
    ContextualRule
    MixedRule
end

"""
    PlasticityContext

Contexto para aplicación de reglas de plasticidad.
"""
struct PlasticityContext
    # Tasa de aprendizaje base
    learning_rate::Float32
    
    # Tiempo actual de simulación
    current_time::Float64
    
    # Tipo de regla a aplicar
    rule_type::PlasticityRuleType
    
    # Parámetros específicos de la regla
    rule_params::Dict{Symbol, Any}
    
    # Tensor de modulación (opcional)
    modulation_tensor::Union{Array{Float32,3}, Nothing}
    
    # Factor de penalización para pesos grandes (regularización)
    weight_decay::Float32
    
    # Flag para normalización de pesos
    normalize_weights::Bool
end

# Constructor con valores por defecto
function PlasticityContext(;
    learning_rate::Float32=0.01f0,
    current_time::Float64=time(),
    rule_type::PlasticityRuleType=HebbianRule,
    rule_params::Dict{Symbol, Any}=Dict{Symbol, Any}(),
    modulation_tensor::Union{Array{Float32,3}, Nothing}=nothing,
    weight_decay::Float32=0.0001f0,
    normalize_weights::Bool=true
)
    return PlasticityContext(
        learning_rate,
        current_time,
        rule_type,
        rule_params,
        modulation_tensor,
        weight_decay,
        normalize_weights
    )
end

"""
    tensor_interpolation(tensor, output_size)

Redimensiona un tensor al tamaño especificado mediante interpolación.
Función auxiliar simplificada para este módulo.
"""
function tensor_interpolation(tensor::Array{T,3}, output_size::NTuple{3,Int}) where T <: AbstractFloat
    # Implementación muy básica - en la versión completa esto estaría en otro módulo
    result = zeros(T, output_size)
    
    # Factores de escala
    scale_x = size(tensor, 1) / output_size[1]
    scale_y = size(tensor, 2) / output_size[2]
    scale_z = size(tensor, 3) / output_size[3]
    
    # Interpolación básica
    for i in 1:output_size[1], j in 1:output_size[2], k in 1:output_size[3]
        # Coordenadas en el tensor original
        x = min(size(tensor, 1), max(1, round(Int, i * scale_x)))
        y = min(size(tensor, 2), max(1, round(Int, j * scale_y)))
        z = min(size(tensor, 3), max(1, round(Int, k * scale_z)))
        
        # Copiar valor
        result[i, j, k] = tensor[x, y, z]
    end
    
    return result
end

"""
    apply_plasticity!(neuron, pre_activation, post_activation, context)

Aplica la regla de plasticidad especificada en el contexto a una neurona.
"""
function apply_plasticity!(
    neuron::TensorNeuron,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Seleccionar regla de plasticidad
    if context.rule_type == HebbianRule
        apply_hebbian_plasticity!(neuron, pre_activation, post_activation, context)
    elseif context.rule_type == BCMRule
        apply_bcm_plasticity!(neuron, pre_activation, post_activation, context)
    elseif context.rule_type == STDPRule
        apply_stdp_plasticity!(neuron, pre_activation, post_activation, context)
    elseif context.rule_type == HomeostaticRule
        apply_homeostatic_plasticity!(neuron, pre_activation, post_activation, context)
    elseif context.rule_type == ContextualRule
        apply_contextual_plasticity!(neuron, pre_activation, post_activation, context)
    else # MixedRule
        apply_mixed_plasticity!(neuron, pre_activation, post_activation, context)
    end
    
    # Aplicar decaimiento de pesos (regularización)
    if context.weight_decay > 0
        neuron.transformation_kernel .*= (1.0f0 - context.weight_decay)
    end
    
    # Normalizar pesos si es necesario
    if context.normalize_weights
        normalize_kernel!(neuron.transformation_kernel)
    end
    
    return neuron
end

"""
    apply_plasticity!(connection, pre_activation, post_activation, context)

Aplica la regla de plasticidad especificada en el contexto a una conexión.
"""
function apply_plasticity!(
    connection::TensorConnection,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Seleccionar regla de plasticidad
    if context.rule_type == HebbianRule
        apply_hebbian_plasticity!(connection, pre_activation, post_activation, context)
    elseif context.rule_type == BCMRule
        apply_bcm_plasticity!(connection, pre_activation, post_activation, context)
    elseif context.rule_type == STDPRule
        apply_stdp_plasticity!(connection, pre_activation, post_activation, context)
    elseif context.rule_type == HomeostaticRule
        apply_homeostatic_plasticity!(connection, pre_activation, post_activation, context)
    elseif context.rule_type == ContextualRule
        apply_contextual_plasticity!(connection, pre_activation, post_activation, context)
    else # MixedRule
        apply_mixed_plasticity!(connection, pre_activation, post_activation, context)
    end
    
    # Aplicar decaimiento de pesos (regularización)
    if context.weight_decay > 0
        connection.weight .*= (1.0f0 - context.weight_decay)
    end
    
    # Normalizar pesos si es necesario
    if context.normalize_weights
        normalize_kernel!(connection.weight)
    end
    
    # Actualizar fuerza de conexión
    connection.strength = mean(abs.(connection.weight))
    
    return connection
end

"""
    apply_hebbian_plasticity!(neuron, pre_activation, post_activation, context)

Aplica plasticidad Hebbiana: "Las neuronas que se disparan juntas, se conectan juntas".
"""
function apply_hebbian_plasticity!(
    neuron::TensorNeuron,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que pre_activation tiene las mismas dimensiones que el kernel
    if size(pre_activation) != size(neuron.transformation_kernel)
        pre_activation = tensor_interpolation(pre_activation, size(neuron.transformation_kernel))
    end
    
    # Asegurar que post_activation tiene las mismas dimensiones que el kernel
    if size(post_activation) != size(neuron.transformation_kernel)
        post_activation = tensor_interpolation(post_activation, size(neuron.transformation_kernel))
    end
    
    # Parámetros específicos de la regla Hebbiana
    hebbian_coef = get(context.rule_params, :hebbian_coefficient, 1.0f0)
    anti_hebbian_coef = get(context.rule_params, :anti_hebbian_coefficient, 0.5f0)
    threshold = get(context.rule_params, :threshold, 0.1f0)
    
    # Factor de modulación (opcional)
    modulation = 1.0f0
    if !isnothing(context.modulation_tensor)
        if size(context.modulation_tensor) == size(neuron.transformation_kernel)
            modulation = context.modulation_tensor
        else
            modulation = tensor_interpolation(
                context.modulation_tensor, 
                size(neuron.transformation_kernel)
            )
        end
    end
    
    # Tasa de aprendizaje efectiva
    effective_lr = context.learning_rate * (1.0f0 - neuron.specialization * neuron.plasticity.specialization_decay)
    
    # Calcular cambio Hebbiano
    for i in CartesianIndices(neuron.transformation_kernel)
        # Producto Hebbiano
        pre = pre_activation[i]
        post = post_activation[i]
        
        # Cambio de peso
        if pre * post > threshold
            # Potenciación (correlación positiva)
            delta_w = hebbian_coef * pre * post
        elseif pre * post < -threshold
            # Depresión (correlación negativa)
            delta_w = anti_hebbian_coef * pre * post
        else
            # Sin cambio (correlación débil)
            delta_w = 0.0f0
        end
        
        # Aplicar modulación
        if isa(modulation, Array)
            delta_w *= modulation[i]
        else
            delta_w *= modulation
        end
        
        # Actualizar kernel
        neuron.transformation_kernel[i] += effective_lr * delta_w
    end
    
    return neuron
end

"""
    apply_hebbian_plasticity!(connection, pre_activation, post_activation, context)

Aplica plasticidad Hebbiana a una conexión entre neuronas.
"""
function apply_hebbian_plasticity!(
    connection::TensorConnection,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que pre_activation tiene las mismas dimensiones que el peso
    if size(pre_activation) != size(connection.weight)
        pre_activation = tensor_interpolation(pre_activation, size(connection.weight))
    end
    
    # Asegurar que post_activation tiene las mismas dimensiones que el peso
    if size(post_activation) != size(connection.weight)
        post_activation = tensor_interpolation(post_activation, size(connection.weight))
    end
    
    # Parámetros específicos de la regla Hebbiana
    hebbian_coef = get(context.rule_params, :hebbian_coefficient, 1.0f0)
    anti_hebbian_coef = get(context.rule_params, :anti_hebbian_coefficient, 0.5f0)
    threshold = get(context.rule_params, :threshold, 0.1f0)
    
    # Factor de modulación (opcional)
    modulation = 1.0f0
    if !isnothing(context.modulation_tensor)
        if size(context.modulation_tensor) == size(connection.weight)
            modulation = context.modulation_tensor
        else
            modulation = tensor_interpolation(
                context.modulation_tensor, 
                size(connection.weight)
            )
        end
    end
    
    # Calcular cambio Hebbiano
    for i in CartesianIndices(connection.weight)
        # Producto Hebbiano
        pre = pre_activation[i]
        post = post_activation[i]
        
        # Cambio de peso
        if pre * post > threshold
            # Potenciación (correlación positiva)
            delta_w = hebbian_coef * pre * post
        elseif pre * post < -threshold
            # Depresión (correlación negativa)
            delta_w = anti_hebbian_coef * pre * post
        else
            # Sin cambio (correlación débil)
            delta_w = 0.0f0
        end
        
        # Aplicar modulación
        if isa(modulation, Array)
            delta_w *= modulation[i]
        else
            delta_w *= modulation
        end
        
        # Inversor para conexiones inhibitorias
        if connection.connection_type == :inhibitory
            delta_w *= -1.0f0
        end
        
        # Actualizar peso
        connection.weight[i] += context.learning_rate * delta_w
    end
    
    return connection
end

"""
    apply_bcm_plasticity!(neuron, pre_activation, post_activation, context)

Aplica plasticidad BCM (Bienenstock-Cooper-Munro).
"""
function apply_bcm_plasticity!(
    neuron::TensorNeuron,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que pre_activation tiene las mismas dimensiones que el kernel
    if size(pre_activation) != size(neuron.transformation_kernel)
        pre_activation = tensor_interpolation(pre_activation, size(neuron.transformation_kernel))
    end
    
    # Asegurar que post_activation tiene las mismas dimensiones que el kernel
    if size(post_activation) != size(neuron.transformation_kernel)
        post_activation = tensor_interpolation(post_activation, size(neuron.transformation_kernel))
    end
    
    # Parámetros específicos de BCM
    theta_scale = get(context.rule_params, :theta_scale, 1.0f0)
    learning_rate_scale = get(context.rule_params, :learning_rate_scale, 1.0f0)
    
    # Calcular theta deslizante (umbral adaptativo)
    # En BCM, theta es proporcional al cuadrado de la actividad postsináptica promediada
    theta = theta_scale * mean(post_activation.^2)
    
    # Tasa de aprendizaje efectiva
    effective_lr = context.learning_rate * learning_rate_scale * 
                   (1.0f0 - neuron.specialization * neuron.plasticity.specialization_decay)
    
    # Calcular cambio BCM
    for i in CartesianIndices(neuron.transformation_kernel)
        pre = pre_activation[i]
        post = post_activation[i]
        
        # Cambio de peso según regla BCM
        # post * (post - theta) * pre
        delta_w = post * (post - theta) * pre
        
        # Actualizar kernel
        neuron.transformation_kernel[i] += effective_lr * delta_w
    end
    
    return neuron
end

"""
    apply_bcm_plasticity!(connection, pre_activation, post_activation, context)

Aplica plasticidad BCM a una conexión entre neuronas.
"""
function apply_bcm_plasticity!(
    connection::TensorConnection,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que pre_activation tiene las mismas dimensiones que el peso
    if size(pre_activation) != size(connection.weight)
        pre_activation = tensor_interpolation(pre_activation, size(connection.weight))
    end
    
    # Asegurar que post_activation tiene las mismas dimensiones que el peso
    if size(post_activation) != size(connection.weight)
        post_activation = tensor_interpolation(post_activation, size(connection.weight))
    end
    
    # Parámetros específicos de BCM
    theta_scale = get(context.rule_params, :theta_scale, 1.0f0)
    learning_rate_scale = get(context.rule_params, :learning_rate_scale, 1.0f0)
    
    # Calcular theta deslizante (umbral adaptativo)
    theta = theta_scale * mean(post_activation.^2)
    
    # Calcular cambio BCM
    for i in CartesianIndices(connection.weight)
        pre = pre_activation[i]
        post = post_activation[i]
        
        # Cambio de peso según regla BCM
        delta_w = post * (post - theta) * pre
        
        # Inversor para conexiones inhibitorias
        if connection.connection_type == :inhibitory
            delta_w *= -1.0f0
        end
        
        # Actualizar peso
        connection.weight[i] += context.learning_rate * learning_rate_scale * delta_w
    end
    
    return connection
end

"""
    apply_stdp_plasticity!(neuron, pre_activation, post_activation, context)

Aplica plasticidad STDP (Spike-Timing-Dependent Plasticity).
"""
function apply_stdp_plasticity!(
    neuron::TensorNeuron,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que pre_activation tiene las mismas dimensiones que el kernel
    if size(pre_activation) != size(neuron.transformation_kernel)
        pre_activation = tensor_interpolation(pre_activation, size(neuron.transformation_kernel))
    end
    
    # Asegurar que post_activation tiene las mismas dimensiones que el kernel
    if size(post_activation) != size(neuron.transformation_kernel)
        post_activation = tensor_interpolation(post_activation, size(neuron.transformation_kernel))
    end
    
    # Parámetros específicos de STDP
    a_plus = get(context.rule_params, :a_plus, 1.0f0)
    a_minus = get(context.rule_params, :a_minus, 0.8f0)
    tau_plus = get(context.rule_params, :tau_plus, 20.0f0)
    tau_minus = get(context.rule_params, :tau_minus, 20.0f0)
    
    # Para STDP necesitamos historial temporal
    # En esta implementación simplificada, usamos el historial del estado neuronal
    
    # Obtener activaciones previas
    if length(neuron.activation_history) < 2
        # No hay suficiente historial para STDP
        return neuron
    end
    
    prev_post_activation = neuron.activation_history[end-1]
    if size(prev_post_activation) != size(post_activation)
        prev_post_activation = tensor_interpolation(prev_post_activation, size(post_activation))
    end
    
    # Tasa de aprendizaje efectiva
    effective_lr = context.learning_rate * (1.0f0 - neuron.specialization * neuron.plasticity.specialization_decay)
    
    # Calcular cambio STDP
    for i in CartesianIndices(neuron.transformation_kernel)
        pre = pre_activation[i]
        post = post_activation[i]
        prev_post = prev_post_activation[i]
        
        # STDP simplificado
        # Si post > prev_post, asumimos que post spikeó después de pre (potenciación)
        # Si post < prev_post, asumimos que post spikeó antes que pre (depresión)
        if post > prev_post
            # Potenciación (pre → post)
            delta_w = a_plus * pre * post
        else
            # Depresión (post → pre)
            delta_w = -a_minus * pre * post
        end
        
        # Actualizar kernel
        neuron.transformation_kernel[i] += effective_lr * delta_w
    end
    
    return neuron
end

"""
    apply_stdp_plasticity!(connection, pre_activation, post_activation, context)

Aplica plasticidad STDP a una conexión entre neuronas.
"""
function apply_stdp_plasticity!(
    connection::TensorConnection,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que pre_activation tiene las mismas dimensiones que el peso
    if size(pre_activation) != size(connection.weight)
        pre_activation = tensor_interpolation(pre_activation, size(connection.weight))
    end
    
    # Asegurar que post_activation tiene las mismas dimensiones que el peso
    if size(post_activation) != size(connection.weight)
        post_activation = tensor_interpolation(post_activation, size(connection.weight))
    end
    
    # Parámetros específicos de STDP
    a_plus = get(context.rule_params, :a_plus, 1.0f0)
    a_minus = get(context.rule_params, :a_minus, 0.8f0)
    
    # Para un STDP simplificado, necesitamos al menos una actividad previa
    if length(connection.activity_history) < 2
        # No hay suficiente historial
        return connection
    end
    
    # Obtener actividad previa
    prev_activity = connection.activity_history[end-1]
    curr_activity = mean(connection.activity_history)
    
    # Calcular cambio STDP para cada elemento del tensor
    for i in CartesianIndices(connection.weight)
        pre = pre_activation[i]
        post = post_activation[i]
        
        # STDP simplificado basado en la diferencia de actividad
        if curr_activity > prev_activity
            # Potenciación (pre → post)
            delta_w = a_plus * pre * post
        else
            # Depresión (post → pre)
            delta_w = -a_minus * pre * post
        end
        
        # Inversor para conexiones inhibitorias
        if connection.connection_type == :inhibitory
            delta_w *= -1.0f0
        end
        
        # Actualizar peso
        connection.weight[i] += context.learning_rate * delta_w
    end
    
    return connection
end

"""
    apply_homeostatic_plasticity!(neuron, pre_activation, post_activation, context)

Aplica plasticidad homeostática para mantener niveles de actividad estables.
"""
function apply_homeostatic_plasticity!(
    neuron::TensorNeuron,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Parámetros homeostáticos
    target_activity = get(context.rule_params, :target_activity, 0.1f0)
    homeostatic_scale = get(context.rule_params, :homeostatic_scale, 0.1f0)
    
    # Calcular actividad media actual
    current_activity = mean(abs.(post_activation))
    
    # Calcular factor de escala homeostático
    # Si la actividad es mayor que el objetivo, reducir los pesos
    # Si la actividad es menor que el objetivo, aumentar los pesos
    scale_factor = target_activity / max(current_activity, 1e-6)
    
    # Limitar cambios extremos
    scale_factor = min(1.0f0 + homeostatic_scale, max(1.0f0 - homeostatic_scale, scale_factor))
    
    # Aplicar cambio homeostático
    neuron.transformation_kernel .*= scale_factor
    
    return neuron
end

"""
    apply_homeostatic_plasticity!(connection, pre_activation, post_activation, context)

Aplica plasticidad homeostática a una conexión.
"""
function apply_homeostatic_plasticity!(
    connection::TensorConnection,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Parámetros homeostáticos
    target_activity = get(context.rule_params, :target_activity, 0.1f0)
    homeostatic_scale = get(context.rule_params, :homeostatic_scale, 0.1f0)
    
    # Calcular actividad media actual
    current_activity = mean(connection.activity_history)
    
    # Calcular factor de escala homeostático
    scale_factor = target_activity / max(current_activity, 1e-6)
    
    # Limitar cambios extremos
    scale_factor = min(1.0f0 + homeostatic_scale, max(1.0f0 - homeostatic_scale, scale_factor))
    
    # Aplicar cambio homeostático
    connection.weight .*= scale_factor
    
    return connection
end

"""
    apply_contextual_plasticity!(neuron, pre_activation, post_activation, context)

Aplica plasticidad contextual sensible al entorno.
"""
function apply_contextual_plasticity!(
    neuron::TensorNeuron,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Para plasticidad contextual necesitamos tensor de modulación
    if isnothing(context.modulation_tensor)
        # Sin modulación, usar Hebbian por defecto
        return apply_hebbian_plasticity!(neuron, pre_activation, post_activation, context)
    end
    
    # Parámetros contextuales
    context_strength = get(context.rule_params, :context_strength, 1.0f0)
    
    # Asegurar que modulation_tensor tiene las dimensiones correctas
    modulation = context.modulation_tensor
    if size(modulation) != size(neuron.transformation_kernel)
        modulation = tensor_interpolation(modulation, size(neuron.transformation_kernel))
    end
    
    # Calcular cambio contextual (combinación de Hebbian y modulación)
    for i in CartesianIndices(neuron.transformation_kernel)
        pre = pre_activation[i]
        post = post_activation[i]
        mod = modulation[i]
        
        # Cambio básico Hebbiano
        base_delta = pre * post
        
        # Modulación contextual
        context_delta = base_delta * mod * context_strength
        
        # Aplicar cambio
        neuron.transformation_kernel[i] += context.learning_rate * context_delta
    end
    
    return neuron
end


"""
    apply_mixed_plasticity!(neuron, pre_activation, post_activation, context)

Aplica una combinación ponderada de diferentes reglas de plasticidad.
"""
function apply_mixed_plasticity!(
    neuron::TensorNeuron,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Pesos para cada regla
    hebbian_weight = get(context.rule_params, :hebbian_weight, 0.4f0)
    bcm_weight = get(context.rule_params, :bcm_weight, 0.2f0)
    stdp_weight = get(context.rule_params, :stdp_weight, 0.2f0)
    homeostatic_weight = get(context.rule_params, :homeostatic_weight, 0.1f0)
    contextual_weight = get(context.rule_params, :contextual_weight, 0.1f0)
    
    # Copia del kernel original
    original_kernel = copy(neuron.transformation_kernel)
    
    # Aplicar cada regla por separado y combinar resultados
    apply_hebbian_plasticity!(neuron, pre_activation, post_activation, context)
    hebbian_kernel = copy(neuron.transformation_kernel)
    
    # Restaurar kernel original
    neuron.transformation_kernel = copy(original_kernel)
    
    apply_bcm_plasticity!(neuron, pre_activation, post_activation, context)
    bcm_kernel = copy(neuron.transformation_kernel)
    
    # Restaurar kernel original
    neuron.transformation_kernel = copy(original_kernel)
    
    apply_stdp_plasticity!(neuron, pre_activation, post_activation, context)
    stdp_kernel = copy(neuron.transformation_kernel)
    
    # Restaurar kernel original
    neuron.transformation_kernel = copy(original_kernel)
    
    apply_homeostatic_plasticity!(neuron, pre_activation, post_activation, context)
    homeostatic_kernel = copy(neuron.transformation_kernel)
    
    # Restaurar kernel original
    neuron.transformation_kernel = copy(original_kernel)
    
    apply_contextual_plasticity!(neuron, pre_activation, post_activation, context)
    contextual_kernel = copy(neuron.transformation_kernel)
    
    # Combinar resultados usando pesos
    neuron.transformation_kernel = 
        original_kernel * (1.0f0 - (hebbian_weight + bcm_weight + stdp_weight + homeostatic_weight + contextual_weight)) +
        hebbian_kernel * hebbian_weight +
        bcm_kernel * bcm_weight +
        stdp_kernel * stdp_weight +
        homeostatic_kernel * homeostatic_weight +
        contextual_kernel * contextual_weight
    
    return neuron
end

"""
    apply_mixed_plasticity!(connection, pre_activation, post_activation, context)

Aplica una combinación ponderada de diferentes reglas de plasticidad a una conexión.
"""
function apply_mixed_plasticity!(
    connection::TensorConnection,
    pre_activation::Array{T,3},
    post_activation::Array{S,3},
    context::PlasticityContext
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Pesos para cada regla
    hebbian_weight = get(context.rule_params, :hebbian_weight, 0.4f0)
    bcm_weight = get(context.rule_params, :bcm_weight, 0.2f0)
    stdp_weight = get(context.rule_params, :stdp_weight, 0.2f0)
    homeostatic_weight = get(context.rule_params, :homeostatic_weight, 0.1f0)
    contextual_weight = get(context.rule_params, :contextual_weight, 0.1f0)
    
    # Copia del peso original
    original_weight = copy(connection.weight)
    
    # Aplicar cada regla por separado y combinar resultados
    apply_hebbian_plasticity!(connection, pre_activation, post_activation, context)
    hebbian_weight_tensor = copy(connection.weight)
    
    # Restaurar peso original
    connection.weight = copy(original_weight)
    
    apply_bcm_plasticity!(connection, pre_activation, post_activation, context)
    bcm_weight_tensor = copy(connection.weight)
    
    # Restaurar peso original
    connection.weight = copy(original_weight)
    
    apply_stdp_plasticity!(connection, pre_activation, post_activation, context)
    stdp_weight_tensor = copy(connection.weight)
    
    # Restaurar peso original
    connection.weight = copy(original_weight)
    
    apply_homeostatic_plasticity!(connection, pre_activation, post_activation, context)
    homeostatic_weight_tensor = copy(connection.weight)
    
    # Restaurar peso original
    connection.weight = copy(original_weight)
    
    apply_contextual_plasticity!(connection, pre_activation, post_activation, context)
    contextual_weight_tensor = copy(connection.weight)
    
    # Combinar resultados usando pesos
    connection.weight = 
        original_weight * (1.0f0 - (hebbian_weight + bcm_weight + stdp_weight + homeostatic_weight + contextual_weight)) +
        hebbian_weight_tensor * hebbian_weight +
        bcm_weight_tensor * bcm_weight +
        stdp_weight_tensor * stdp_weight +
        homeostatic_weight_tensor * homeostatic_weight +
        contextual_weight_tensor * contextual_weight
    
    return connection
end

"""
    normalize_kernel!(kernel)

Normaliza un kernel para evitar crecimiento descontrolado de pesos.
"""
function normalize_kernel!(kernel::Array{T,3}) where T <: AbstractFloat
    # Calcular norma del kernel
    norm_val = norm(vec(kernel))
    
    # Evitar división por cero
    if norm_val > 1e-8
        # Normalizar kernel manteniendo su dirección pero limitando magnitud
        if norm_val > 5.0f0
            # Solo normalizar si la norma es grande
            kernel ./= (norm_val / 5.0f0)
        end
    end
    
    return kernel
end

"""
    create_modulation_tensor(brain, type=:attention)

Crea un tensor de modulación para plasticidad contextual.
"""
function create_modulation_tensor(
    brain::BrainSpace,
    type::Symbol=:attention
)
    # Inicializar tensor de modulación
    modulation_tensor = ones(Float32, brain.dimensions)
    
    if type == :attention
        # Usar mapa de atención como modulación
        modulation_tensor = copy(brain.attention_map)
        
    elseif type == :error
        # Crear modulación basada en error (si hay un objetivo disponible)
        if hasfield(typeof(brain), :target_state) && !isnothing(brain.target_state)
            # Calcular error elemento a elemento
            error_tensor = abs.(brain.global_state - brain.target_state)
            
            # Normalizar error
            max_error = maximum(error_tensor)
            if max_error > 0
                error_tensor ./= max_error
            end
            
            # Usar error como modulación (mayor error = mayor plasticidad)
            modulation_tensor = 1.0f0 .+ 2.0f0 .* error_tensor
        end
        
    elseif type == :novelty
        # Crear modulación basada en novedad
        if length(brain.history) > 1
            # Calcular cambio respecto al estado anterior
            prev_state = brain.history[end-1]
            if size(prev_state) != brain.dimensions
                prev_state = tensor_interpolation(prev_state, brain.dimensions)
            end
            
            # Diferencia absoluta con estado anterior
            novelty_tensor = abs.(brain.global_state - prev_state)
            
            # Normalizar novedad
            max_novelty = maximum(novelty_tensor)
            if max_novelty > 0
                novelty_tensor ./= max_novelty
            end
            
            # Usar novedad como modulación (mayor novedad = mayor plasticidad)
            modulation_tensor = 1.0f0 .+ 2.0f0 .* novelty_tensor
        end
        
    elseif type == :gradient
        # Crear modulación basada en gradientes espaciales
        gradients = spatial_gradients(brain.global_state)
        gradient_magnitude = zeros(Float32, brain.dimensions)
        
        # Calcular magnitud del gradiente para cada posición
        for x in 2:brain.dimensions[1]-1
            for y in 2:brain.dimensions[2]-1
                for z in 2:brain.dimensions[3]-1
                    # Gradientes en cada dirección
                    gx = (brain.global_state[x+1, y, z] - brain.global_state[x-1, y, z]) / 2
                    gy = (brain.global_state[x, y+1, z] - brain.global_state[x, y-1, z]) / 2
                    gz = (brain.global_state[x, y, z+1] - brain.global_state[x, y, z-1]) / 2
                    
                    # Magnitud del gradiente
                    gradient_magnitude[x, y, z] = sqrt(gx^2 + gy^2 + gz^2)
                end
            end
        end
        
        # Normalizar magnitud del gradiente
        max_magnitude = maximum(gradient_magnitude)
        if max_magnitude > 0
            gradient_magnitude ./= max_magnitude
        end
        
        # Usar gradiente como modulación (mayor gradiente = mayor plasticidad)
        modulation_tensor = 1.0f0 .+ gradient_magnitude
    end
    
    return modulation_tensor
end

"""
    apply_brain_plasticity!(brain, input_tensor, output_tensor, rule_type=HebbianRule)

Aplica plasticidad a todas las neuronas y conexiones en el cerebro.
"""
function apply_brain_plasticity!(
    brain::BrainSpace,
    input_tensor::Array{T,3},
    output_tensor::Array{S,3},
    rule_type::PlasticityRuleType=HebbianRule
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Crear contexto de plasticidad
    modulation_tensor = create_modulation_tensor(brain, :attention)
    
    context = PlasticityContext(
        learning_rate=0.01f0,
        current_time=time(),
        rule_type=rule_type,
        rule_params=Dict{Symbol,Any}(),
        modulation_tensor=modulation_tensor,
        weight_decay=0.0001f0,
        normalize_weights=true
    )
    
    # Contador de elementos actualizados
    updated_neurons = 0
    updated_connections = 0
    
    # Aplicar plasticidad a cada neurona
    for (_, neuron) in brain.neurons
        # Obtener activaciones de entrada y salida para esta neurona
        neuron_input = extract_neuron_input(brain, neuron, input_tensor)
        neuron_output = neuron.state
        
        # Aplicar plasticidad
        apply_plasticity!(neuron, neuron_input, neuron_output, context)
        updated_neurons += 1
    end
    
    # Aplicar plasticidad a cada conexión
    for (_, connection) in brain.connections
        # Encontrar neuronas origen y destino
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
        
        if !isnothing(source_neuron) && !isnothing(target_neuron)
            # Obtener activaciones
            source_activation = source_neuron.state
            target_activation = target_neuron.state
            
            # Aplicar plasticidad
            apply_plasticity!(connection, source_activation, target_activation, context)
            updated_connections += 1
        end
    end
    
    return updated_neurons, updated_connections
end

"""
    extract_neuron_input(brain, neuron, input_tensor)

Extrae la entrada específica para una neurona desde el tensor de entrada global.
"""
function extract_neuron_input(
    brain::BrainSpace,
    neuron::TensorNeuron,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Implementación simplificada - en la versión real, debería usar el campo receptivo
    # Para esta versión usamos el tensor completo
    return input_tensor
end

"""
    spatial_gradients(tensor)

Calcula los gradientes espaciales de un tensor 3D.
"""
function spatial_gradients(tensor::Array{T,3}) where T <: AbstractFloat
    dim_x, dim_y, dim_z = size(tensor)
    
    # Inicializar gradientes
    grad_x = zeros(T, dim_x-2, dim_y-2, dim_z-2)
    grad_y = zeros(T, dim_x-2, dim_y-2, dim_z-2)
    grad_z = zeros(T, dim_x-2, dim_y-2, dim_z-2)
    
    # Calcular gradientes usando diferencias centrales
    for x in 2:dim_x-1
        for y in 2:dim_y-1
            for z in 2:dim_z-1
                grad_x[x-1, y-1, z-1] = (tensor[x+1, y, z] - tensor[x-1, y, z]) / 2
                grad_y[x-1, y-1, z-1] = (tensor[x, y+1, z] - tensor[x, y-1, z]) / 2
                grad_z[x-1, y-1, z-1] = (tensor[x, y, z+1] - tensor[x, y, z-1]) / 2
            end
        end
    end
    
    return (x=grad_x, y=grad_y, z=grad_z)
end

# Función de conveniencia para convertir un símbolo a tipo de regla
function symbol_to_rule_type(symbol::Symbol)
    if symbol == :hebbian
        return HebbianRule
    elseif symbol == :bcm
        return BCMRule
    elseif symbol == :stdp
        return STDPRule
    elseif symbol == :homeostatic
        return HomeostaticRule
    elseif symbol == :contextual
        return ContextualRule
    elseif symbol == :mixed
        return MixedRule
    else
        return HebbianRule  # Por defecto
    end
end

# Exportar tipos y funciones principales
export PlasticityParameters, PlasticityRuleType, HebbianRule, BCMRule, STDPRule, 
       HomeostaticRule, ContextualRule, MixedRule, PlasticityContext,
       apply_plasticity!, apply_brain_plasticity!, create_modulation_tensor, 
       normalize_kernel!, symbol_to_rule_type
end