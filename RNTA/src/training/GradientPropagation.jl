# training/GradientPropagation.jl
# Implementa propagación de gradientes en el espacio tensorial 3D

module GradientPropagation

using LinearAlgebra
using Statistics
using Random
using UUIDs
# Importaciones de otros módulos de RNTA
using ..TensorNeuron
using ..Connections
using ..BrainSpace
using ..TensorOperations

"""
    GradientConfig

Configuración para propagación de gradientes.
"""
struct GradientConfig
    # Factor de propagación hacia atrás
    backprop_factor::Float32
    
    # Factor de propagación lateral
    lateral_factor::Float32
    
    # Método de actualización de gradientes
    update_method::Symbol
    
    # Factor de momentum para actualizaciones
    momentum::Float32
    
    # Umbral para recorte de gradientes
    gradient_clip::Float32
    
    # Usar gradientes tensoriales completos vs. escalares
    use_tensor_gradients::Bool
    
    # Usar regularización para evitar overfitting
    regularization_lambda::Float32
    
    # Tipo de regularización (L1, L2)
    regularization_type::Symbol
end

# Constructor con valores por defecto
function GradientConfig(;
    backprop_factor::Float32=1.0f0,
    lateral_factor::Float32=0.3f0,
    update_method::Symbol=:adam,
    momentum::Float32=0.9f0,
    gradient_clip::Float32=5.0f0,
    use_tensor_gradients::Bool=true,
    regularization_lambda::Float32=0.0001f0,
    regularization_type::Symbol=:L2
)
    return GradientConfig(
        backprop_factor,
        lateral_factor,
        update_method,
        momentum,
        gradient_clip,
        use_tensor_gradients,
        regularization_lambda,
        regularization_type
    )
end

"""
    AdamState

Estado del optimizador Adam para una neurona o conexión.
"""
mutable struct AdamState
    # Momento de primer orden
    m::Union{Array{Float32,3}, Float32}
    
    # Momento de segundo orden
    v::Union{Array{Float32,3}, Float32}
    
    # Contador de iteraciones
    t::Int
end

"""
    initialize_adam_state(tensor_size)

Inicializa el estado de Adam para un tensor del tamaño dado.
"""
function initialize_adam_state(tensor_size::NTuple{3,Int})
    m = zeros(Float32, tensor_size)
    v = zeros(Float32, tensor_size)
    return AdamState(m, v, 0)
end

"""
    initialize_adam_state(scalar=true)

Inicializa el estado de Adam para un valor escalar.
"""
function initialize_adam_state(scalar::Bool=true)
    return AdamState(0.0f0, 0.0f0, 0)
end

"""
    compute_gradients(brain, loss_gradient, config)

Computa gradientes para todas las neuronas y conexiones en el cerebro.
"""
function compute_gradients(
    brain::Brain_Space,
    loss_gradient::Array{T,3},
    config::GradientConfig=GradientConfig()
) where T <: AbstractFloat
    # Verificar que el gradiente tiene dimensiones compatibles
    if size(loss_gradient) != brain.dimensions
        loss_gradient = tensor_interpolation(loss_gradient, brain.dimensions)
    end
    
    # Gradiente global para el cerebro
    brain_gradient = copy(loss_gradient)
    
    # Diccionarios para almacenar gradientes
    neuron_gradients = Dict{UUID, Array{Float32,3}}()
    connection_gradients = Dict{UUID, Array{Float32,3}}()
    
    # Para cada neurona, calcular gradiente
    for (pos, neuron) in brain.neurons
        # Extraer gradiente local para esta neurona
        local_gradient = extract_local_gradient(brain_gradient, pos, config)
        
        # Calcular gradiente para esta neurona
        neuron_gradient = compute_neuron_gradient(neuron, local_gradient, config)
        
        # Almacenar gradiente
        neuron_gradients[neuron.id] = neuron_gradient
    end
    
    # Para cada conexión, calcular gradiente
    for connection in brain.connections
        # Calcular gradiente para esta conexión
        connection_gradient = compute_connection_gradient(
            connection, 
            brain, 
            neuron_gradients, 
            config
        )
        
        # Almacenar gradiente
        connection_gradients[connection.id] = connection_gradient
    end
    
    return neuron_gradients, connection_gradients
end

"""
    extract_local_gradient(global_gradient, position, config)

Extrae gradiente local para una neurona en la posición dada.
"""
function extract_local_gradient(
    global_gradient::Array{T,3},
    position::NTuple{3,Int},
    config::GradientConfig
) where T <: AbstractFloat
    # Dimensiones del gradiente global
    dim_x, dim_y, dim_z = size(global_gradient)
    
    # Radio para extracción local
    radius = 1
    
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
    compute_neuron_gradient(neuron, local_gradient, config)

Computa el gradiente para una neurona específica.
"""
function compute_neuron_gradient(
    neuron::Tensor_Neuron,
    local_gradient::Array{T,3},
    config::GradientConfig
) where T <: AbstractFloat
    # Si no usamos gradientes tensoriales, reducir a escalar
    if !config.use_tensor_gradients
        return mean(local_gradient)
    end
    
    # Asegurar dimensiones compatibles
    if size(local_gradient) != size(neuron.transformation_kernel)
        local_gradient = tensor_interpolation(local_gradient, size(neuron.transformation_kernel))
    end
    
    # Aplicar regularización
    if config.regularization_lambda > 0
        if config.regularization_type == :L1
            # Regularización L1 (Lasso)
            reg_gradient = config.regularization_lambda * sign.(neuron.transformation_kernel)
        else
            # Regularización L2 (Ridge) - por defecto
            reg_gradient = config.regularization_lambda * neuron.transformation_kernel
        end
        
        # Combinar con gradiente de pérdida
        gradient = local_gradient + reg_gradient
    else
        gradient = local_gradient
    end
    
    # Recortar gradientes extremos
    if config.gradient_clip > 0
        gradient = clamp.(gradient, -config.gradient_clip, config.gradient_clip)
    end
    
    return gradient
end

"""
    compute_connection_gradient(connection, brain, neuron_gradients, config)

Computa el gradiente para una conexión específica.
"""
function compute_connection_gradient(
    connection::Tensor_Connection,
    brain::Brain_Space,
    neuron_gradients::Dict{UUID, Array{T,3}},
    config::GradientConfig
) where T <: AbstractFloat
    # Encontrar gradiente de neurona destino
    if !haskey(neuron_gradients, connection.target_id)
        # Si no hay gradiente para destino, devolver ceros
        return zeros(Float32, size(connection.weight))
    end
    
    target_gradient = neuron_gradients[connection.target_id]
    
    # Encontrar neurona origen
    source_neuron = nothing
    for (_, neuron) in brain.neurons
        if neuron.id == connection.source_id
            source_neuron = neuron
            break
        end
    end
    
    if isnothing(source_neuron)
        # Si no se encuentra neurona origen, devolver ceros
        return zeros(Float32, size(connection.weight))
    end
    
    # Obtener activación de origen
    source_activation = source_neuron.state
    
    # Si no usamos gradientes tensoriales, calcular escalar
    if !config.use_tensor_gradients
        mean_target_gradient = mean(target_gradient)
        mean_source_activation = mean(source_activation)
        
        # Gradiente escalar
        scalar_gradient = mean_target_gradient * mean_source_activation
        
        # Expandir a tensor
        return fill(scalar_gradient, size(connection.weight))
    end
    
    # Calcular gradiente tensorial
    # El gradiente de la conexión es el producto externo del gradiente del destino
    # y la activación de la fuente
    
    # Asegurar dimensiones compatibles
    if size(target_gradient) != size(connection.weight)
        target_gradient = tensor_interpolation(target_gradient, size(connection.weight))
    end
    
    if size(source_activation) != size(connection.weight)
        source_activation = tensor_interpolation(source_activation, size(connection.weight))
    end
    
    # Calcular producto elemento a elemento
    gradient = target_gradient .* source_activation
    
    # Aplicar regularización
    if config.regularization_lambda > 0
        if config.regularization_type == :L1
            # Regularización L1 (Lasso)
            reg_gradient = config.regularization_lambda * sign.(connection.weight)
        else
            # Regularización L2 (Ridge) - por defecto
            reg_gradient = config.regularization_lambda * connection.weight
        end
        
        # Combinar con gradiente calculado
        gradient .+= reg_gradient
    end
    
    # Recortar gradientes extremos
    if config.gradient_clip > 0
        gradient = clamp.(gradient, -config.gradient_clip, config.gradient_clip)
    end
    
    # Para conexiones inhibitorias, invertir signo
    if connection.connection_type == :inhibitory
        gradient .*= -1
    end
    
    return gradient
end

"""
    apply_gradients!(brain, neuron_gradients, connection_gradients, learning_rate, config)

Aplica los gradientes calculados a las neuronas y conexiones.
"""
function apply_gradients!(
    brain::Brain_Space,
    neuron_gradients::Dict{UUID, Array{T,3}},
    connection_gradients::Dict{UUID, Array{S,3}},
    learning_rate::Float32,
    config::GradientConfig=GradientConfig()
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Diccionarios para almacenar estados de optimizadores
    if !hasfield(typeof(brain), :optimizer_states)
        # Inicializar si no existe
        brain.optimizer_states = Dict{UUID, Any}()
    end
    
    # Contador de elementos actualizados
    updated_neurons = 0
    updated_connections = 0
    
    # Para cada neurona, aplicar gradiente
    for (_, neuron) in brain.neurons
        if haskey(neuron_gradients, neuron.id)
            gradient = neuron_gradients[neuron.id]
            
            # Obtener o inicializar estado del optimizador
            if !haskey(brain.optimizer_states, neuron.id)
                if config.use_tensor_gradients
                    brain.optimizer_states[neuron.id] = initialize_adam_state(size(neuron.transformation_kernel))
                else
                    brain.optimizer_states[neuron.id] = initialize_adam_state(true)
                end
            end
            
            optimizer_state = brain.optimizer_states[neuron.id]
            
            # Aplicar actualización según método configurado
            if config.update_method == :sgd
                apply_sgd_update!(neuron, gradient, learning_rate, config)
            elseif config.update_method == :momentum
                apply_momentum_update!(neuron, gradient, learning_rate, optimizer_state, config)
            else
                # Adam por defecto
                apply_adam_update!(neuron, gradient, learning_rate, optimizer_state, config)
            end
            
            updated_neurons += 1
        end
    end
    
    # Para cada conexión, aplicar gradiente
    for connection in brain.connections
        if haskey(connection_gradients, connection.id)
            gradient = connection_gradients[connection.id]
            
            # Obtener o inicializar estado del optimizador
            if !haskey(brain.optimizer_states, connection.id)
                if config.use_tensor_gradients
                    brain.optimizer_states[connection.id] = initialize_adam_state(size(connection.weight))
                else
                    brain.optimizer_states[connection.id] = initialize_adam_state(true)
                end
            end
            
            optimizer_state = brain.optimizer_states[connection.id]
            
            # Aplicar actualización según método configurado
            if config.update_method == :sgd
                apply_sgd_update!(connection, gradient, learning_rate, config)
            elseif config.update_method == :momentum
                apply_momentum_update!(connection, gradient, learning_rate, optimizer_state, config)
            else
                # Adam por defecto
                apply_adam_update!(connection, gradient, learning_rate, optimizer_state, config)
            end
            
            # Actualizar fuerza de conexión
            connection.strength = mean(abs.(connection.weight))
            
            updated_connections += 1
        end
    end
    
    return updated_neurons, updated_connections
end

"""
    apply_sgd_update!(neuron, gradient, learning_rate, config)

Aplica actualización SGD (descenso de gradiente estocástico) a una neurona.
"""
function apply_sgd_update!(
    neuron::Tensor_Neuron,
    gradient::Union{Array{T,3}, T},
    learning_rate::Float32,
    config::GradientConfig
) where T <: AbstractFloat
    # Calcular tasa de aprendizaje efectiva
    effective_lr = learning_rate * (1.0f0 - neuron.specialization * neuron.plasticity.specialization_decay)
    
    # Aplicar actualización
    if isa(gradient, Array)
        neuron.transformation_kernel .-= effective_lr .* gradient
    else
        # Gradient es escalar, aplicar a todo el kernel
        neuron.transformation_kernel .-= effective_lr * gradient
    end
    
    return neuron
end

"""
    apply_sgd_update!(connection, gradient, learning_rate, config)

Aplica actualización SGD a una conexión.
"""
function apply_sgd_update!(
    connection::Tensor_Connection,
    gradient::Union{Array{T,3}, T},
    learning_rate::Float32,
    config::GradientConfig
) where T <: AbstractFloat
    # Aplicar actualización
    if isa(gradient, Array)
        connection.weight .-= learning_rate .* gradient
    else
        # Gradient es escalar, aplicar a todo el tensor
        connection.weight .-= learning_rate * gradient
    end
    
    return connection
end

"""
    apply_momentum_update!(neuron, gradient, learning_rate, optimizer_state, config)

Aplica actualización con momentum a una neurona.
"""
function apply_momentum_update!(
    neuron::Tensor_Neuron,
    gradient::Union{Array{T,3}, T},
    learning_rate::Float32,
    optimizer_state::AdamState,
    config::GradientConfig
) where T <: AbstractFloat
    # Calcular tasa de aprendizaje efectiva
    effective_lr = learning_rate * (1.0f0 - neuron.specialization * neuron.plasticity.specialization_decay)
    
    # Actualizar momentum
    if isa(gradient, Array) && isa(optimizer_state.m, Array)
        optimizer_state.m = config.momentum .* optimizer_state.m .+ 
                          (1.0f0 - config.momentum) .* gradient
    else
        # Versión escalar
        optimizer_state.m = config.momentum * optimizer_state.m + 
                          (1.0f0 - config.momentum) * gradient
    end
    
    # Aplicar actualización
    if isa(optimizer_state.m, Array)
        neuron.transformation_kernel .-= effective_lr .* optimizer_state.m
    else
        neuron.transformation_kernel .-= effective_lr * optimizer_state.m
    end
    
    return neuron
end

"""
    apply_momentum_update!(connection, gradient, learning_rate, optimizer_state, config)

Aplica actualización con momentum a una conexión.
"""
function apply_momentum_update!(
    connection::Tensor_Connection,
    gradient::Union{Array{T,3}, T},
    learning_rate::Float32,
    optimizer_state::AdamState,
    config::GradientConfig
) where T <: AbstractFloat
    # Actualizar momentum
    if isa(gradient, Array) && isa(optimizer_state.m, Array)
        optimizer_state.m = config.momentum .* optimizer_state.m .+ 
                          (1.0f0 - config.momentum) .* gradient
    else
        # Versión escalar
        optimizer_state.m = config.momentum * optimizer_state.m + 
                          (1.0f0 - config.momentum) * gradient
    end
    
    # Aplicar actualización
    if isa(optimizer_state.m, Array)
        connection.weight .-= learning_rate .* optimizer_state.m
    else
        connection.weight .-= learning_rate * optimizer_state.m
    end
    
    return connection
end

"""
    apply_adam_update!(neuron, gradient, learning_rate, optimizer_state, config)

Aplica actualización Adam a una neurona.
"""
function apply_adam_update!(
    neuron::Tensor_Neuron,
    gradient::Union{Array{T,3}, T},
    learning_rate::Float32,
    optimizer_state::AdamState,
    config::GradientConfig
) where T <: AbstractFloat
    # Parámetros de Adam
    β1 = 0.9f0
    β2 = 0.999f0
    ϵ = 1e-8f0
    
    # Calcular tasa de aprendizaje efectiva
    effective_lr = learning_rate * (1.0f0 - neuron.specialization * neuron.plasticity.specialization_decay)
    
    # Incrementar contador de iteraciones
    optimizer_state.t += 1
    t = optimizer_state.t
    
    # Para implementación tensorial
    if isa(gradient, Array) && isa(optimizer_state.m, Array)
        # Actualizar momentos
        optimizer_state.m = β1 .* optimizer_state.m .+ (1.0f0 - β1) .* gradient
        optimizer_state.v = β2 .* optimizer_state.v .+ (1.0f0 - β2) .* gradient.^2
        
        # Corregir sesgo
        m_hat = optimizer_state.m ./ (1.0f0 - β1^t)
        v_hat = optimizer_state.v ./ (1.0f0 - β2^t)
        
        # Calcular actualización
        update = m_hat ./ (sqrt.(v_hat) .+ ϵ)
        
        # Aplicar actualización
        neuron.transformation_kernel .-= effective_lr .* update
        
    else
        # Implementación escalar
        # Actualizar momentos
        optimizer_state.m = β1 * optimizer_state.m + (1.0f0 - β1) * gradient
        optimizer_state.v = β2 * optimizer_state.v + (1.0f0 - β2) * gradient^2
        
        # Corregir sesgo
        m_hat = optimizer_state.m / (1.0f0 - β1^t)
        v_hat = optimizer_state.v / (1.0f0 - β2^t)
        
        # Calcular actualización
        update = m_hat / (sqrt(v_hat) + ϵ)
        
        # Aplicar actualización
        neuron.transformation_kernel .-= effective_lr * update
    end
    
    return neuron
end

"""
    apply_adam_update!(connection, gradient, learning_rate, optimizer_state, config)

Aplica actualización Adam a una conexión.
"""
function apply_adam_update!(
    connection::Tensor_Connection,
    gradient::Union{Array{T,3}, T},
    learning_rate::Float32,
    optimizer_state::AdamState,
    config::GradientConfig
) where T <: AbstractFloat
    # Parámetros de Adam
    β1 = 0.9f0
    β2 = 0.999f0
    ϵ = 1e-8f0
    
    # Incrementar contador de iteraciones
    optimizer_state.t += 1
    t = optimizer_state.t
    
    # Para implementación tensorial
    if isa(gradient, Array) && isa(optimizer_state.m, Array)
        # Actualizar momentos
        optimizer_state.m = β1 .* optimizer_state.m .+ (1.0f0 - β1) .* gradient
        optimizer_state.v = β2 .* optimizer_state.v .+ (1.0f0 - β2) .* gradient.^2
        
        # Corregir sesgo
        m_hat = optimizer_state.m ./ (1.0f0 - β1^t)
        v_hat = optimizer_state.v ./ (1.0f0 - β2^t)
        
        # Calcular actualización
        update = m_hat ./ (sqrt.(v_hat) .+ ϵ)
        
        # Aplicar actualización
        connection.weight .-= learning_rate .* update
        
    else
        # Implementación escalar
        # Actualizar momentos
        optimizer_state.m = β1 * optimizer_state.m + (1.0f0 - β1) * gradient
        optimizer_state.v = β2 * optimizer_state.v + (1.0f0 - β2) * gradient^2
        
        # Corregir sesgo
        m_hat = optimizer_state.m / (1.0f0 - β1^t)
        v_hat = optimizer_state.v / (1.0f0 - β2^t)
        
        # Calcular actualización
        update = m_hat / (sqrt(v_hat) + ϵ)
        
        # Aplicar actualización
        connection.weight .-= learning_rate * update
    end
    
    return connection
end

"""
    backpropagate_gradients!(brain, loss_gradient, learning_rate, config)

Propaga gradientes a través del cerebro y aplica actualizaciones.
"""
function backpropagate_gradients!(
    brain::Brain_Space,
    loss_gradient::Array{T,3},
    learning_rate::Float32,
    config::GradientConfig=GradientConfig()
) where T <: AbstractFloat
    # Calcular gradientes
    neuron_gradients, connection_gradients = compute_gradients(brain, loss_gradient, config)
    
    # Aplicar gradientes
    updated_neurons, updated_connections = apply_gradients!(
        brain, 
        neuron_gradients, 
        connection_gradients, 
        learning_rate, 
        config
    )
    
    return updated_neurons, updated_connections
end

"""
    process_batch!(brain, input_batch, target_batch, loss_function, learning_rate, config)

Procesa un batch de entrenamiento completo.
"""
function process_batch!(
    brain::Brain_Space,
    input_batch::Vector{Array{T,3}},
    target_batch::Vector{Array{S,3}},
    loss_function,
    learning_rate::Float32,
    config::GradientConfig=GradientConfig()
) where {T <: AbstractFloat, S <: AbstractFloat}
    batch_size = length(input_batch)
    if batch_size == 0
        return 0.0f0
    end
    
    total_loss = 0.0f0
    
    # Procesar cada muestra del batch
    for i in 1:batch_size
        # Obtener entrada y objetivo
        input = input_batch[i]
        target = target_batch[i]
        
        # Propagar hacia adelante
        output = process(brain, input)
        
        # Calcular pérdida y gradiente
        loss, loss_gradient = calculate_loss(loss_function, output, target)
        
        # Acumular pérdida
        total_loss += loss
        
        # Propagar gradientes hacia atrás
        backpropagate_gradients!(brain, loss_gradient, learning_rate, config)
    end
    
    # Devolver pérdida media del batch
    return total_loss / batch_size
end

"""
    calculate_loss(loss_function, prediction, target)

Calcula pérdida y gradiente dado un par predicción-objetivo.
"""
function calculate_loss(
    loss_function,
    prediction::Array{T,3},
    target::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Calcular pérdida y gradiente
    loss, gradient = calculate_loss(loss_function, prediction, target)
    
    return loss, gradient
end

# Funciones auxiliares para diferentes tipos de pérdida

"""
    mse_loss(prediction, target)

Calcula pérdida de error cuadrático medio y su gradiente.
"""
function mse_loss(
    prediction::Array{T,3},
    target::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(prediction) != size(target)
        target = tensor_interpolation(target, size(prediction))
    end
    
    # Calcular error
    diff = prediction - target
    
    # Calcular pérdida MSE
    loss = mean(diff.^2)
    
    # Gradiente de MSE: 2 * (prediction - target) / n
    gradient = 2.0f0 * diff / length(diff)
    
    return loss, gradient
end

"""
    mae_loss(prediction, target)

Calcula pérdida de error absoluto medio y su gradiente.
"""
function mae_loss(
    prediction::Array{T,3},
    target::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(prediction) != size(target)
        target = tensor_interpolation(target, size(prediction))
    end
    
    # Calcular error
    diff = prediction - target
    
    # Calcular pérdida MAE
    loss = mean(abs.(diff))
    
    # Gradiente de MAE: sign(prediction - target) / n
    gradient = sign.(diff) / length(diff)
    
    return loss, gradient
end

"""
    huber_loss(prediction, target, delta=1.0)

Calcula pérdida Huber (combinación de MSE y MAE) y su gradiente.
"""
function huber_loss(
    prediction::Array{T,3},
    target::Array{S,3},
    delta::Float32=1.0f0
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar dimensiones compatibles
    if size(prediction) != size(target)
        target = tensor_interpolation(target, size(prediction))
    end
    
    # Calcular error
    diff = prediction - target
    
    # Calcular pérdida Huber
    abs_diff = abs.(diff)
    quadratic_mask = abs_diff .<= delta
    linear_mask = .!quadratic_mask
    
    # Parte cuadrática
    loss_quadratic = 0.5f0 * diff[quadratic_mask].^2
    
    # Parte lineal
    loss_linear = delta * (abs_diff[linear_mask] .- 0.5f0 * delta)
    
    # Combinar
    loss = (sum(loss_quadratic) + sum(loss_linear)) / length(diff)
    
    # Gradiente
    gradient = similar(diff)
    gradient[quadratic_mask] = diff[quadratic_mask] / length(diff)
    gradient[linear_mask] = delta * sign.(diff[linear_mask]) / length(diff)
    
    return loss, gradient
end

# Exportar tipos y funciones principales
export GradientConfig, AdamState,
       compute_gradients, apply_gradients!,
       backpropagate_gradients!, process_batch!,
       mse_loss, mae_loss, huber_loss

end # module GradientPropagation