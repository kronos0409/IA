# training/MultidimensionalLoss.jl
# Implementa funciones de pérdida para tensores 3D
module dimensionalLoss

export MultidimensionalLoss,
       calculate_loss,
       calculate_base_loss,
       calculate_coherence_loss,
       spatial_gradients,
       calculate_regularization,
       apply_focus,
       default_loss
"""
    MultidimensionalLoss

Función de pérdida que opera sobre tensores 3D completos.
"""
struct MultidimensionalLoss
    # Tipo de pérdida base
    base_type::Symbol
    
    # Peso para término de coherencia espacial
    coherence_weight::Float32
    
    # Peso para término de regularización
    regularization_weight::Float32
    
    # Factor de enfoque en regiones de alto error
    focus_factor::Float32
    
    # Umbral para considerar un error como significativo
    error_threshold::Float32
end

"""
Constructor principal para MultidimensionalLoss
"""
function MultidimensionalLoss(;
    base_type::Symbol=:mse,
    coherence_weight::Float32=0.2f0,
    regularization_weight::Float32=0.01f0,
    focus_factor::Float32=2.0f0,
    error_threshold::Float32=0.5f0
)
    return MultidimensionalLoss(
        base_type,
        coherence_weight,
        regularization_weight,
        focus_factor,
        error_threshold
    )
end

"""
    calculate_loss(loss_function, prediction, target)

Calcula la pérdida y su gradiente entre predicción y objetivo.
"""
function calculate_loss(
    loss_function::MultidimensionalLoss,
    prediction::Array{T,3},
    target::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que las dimensiones coincidan
    if size(prediction) != size(target)
        target = tensor_interpolation(convert(Array{Float32,3}, target), size(prediction))
    end
    
    # Calcular pérdida base
    base_loss, base_gradient = calculate_base_loss(
        loss_function.base_type,
        prediction,
        target
    )
    
    # Calcular pérdida de coherencia espacial
    if loss_function.coherence_weight > 0
        coherence_loss, coherence_gradient = calculate_coherence_loss(
            prediction,
            target
        )
        
        # Combinar con pérdida base
        total_loss = base_loss + loss_function.coherence_weight * coherence_loss
        total_gradient = base_gradient + loss_function.coherence_weight * coherence_gradient
    else
        total_loss = base_loss
        total_gradient = base_gradient
    end
    
    # Aplicar regularización si es necesario
    if loss_function.regularization_weight > 0
        reg_loss, reg_gradient = calculate_regularization(prediction)
        
        total_loss += loss_function.regularization_weight * reg_loss
        total_gradient += loss_function.regularization_weight * reg_gradient
    end
    
    # Aplicar enfoque en regiones de alto error
    if loss_function.focus_factor > 1.0f0
        total_gradient = apply_focus(
            total_gradient,
            prediction,
            target,
            loss_function.focus_factor,
            loss_function.error_threshold
        )
    end
    
    return total_loss, total_gradient
end

"""
    calculate_base_loss(type, prediction, target)

Calcula la pérdida base según el tipo especificado.
"""
function calculate_base_loss(
    type::Symbol,
    prediction::Array{T,3},
    target::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    if type == :mse
        # Error cuadrático medio
        diff = prediction - target
        loss = mean(diff.^2)
        gradient = 2.0f0 * diff / length(diff)
        
    elseif type == :mae
        # Error absoluto medio
        diff = prediction - target
        loss = mean(abs.(diff))
        gradient = sign.(diff) / length(diff)
        
    elseif type == :huber
        # Pérdida Huber (combinación de MSE y MAE)
        delta = 1.0f0
        diff = prediction - target
        
        # Calcular pérdida y gradiente por elemento
        mask_quadratic = abs.(diff) .<= delta
        
        loss = zeros(Float32, size(diff))
        gradient = zeros(Float32, size(diff))
        
        # Región cuadrática
        loss[mask_quadratic] = 0.5f0 * diff[mask_quadratic].^2
        gradient[mask_quadratic] = diff[mask_quadratic]
        
        # Región lineal
        mask_linear = .!mask_quadratic
        loss[mask_linear] = delta * (abs.(diff[mask_linear]) - 0.5f0 * delta)
        gradient[mask_linear] = delta * sign.(diff[mask_linear])
        
        # Promediar
        loss = mean(loss)
        gradient = gradient / length(gradient)
        
    else
        # Por defecto, usar MSE
        diff = prediction - target
        loss = mean(diff.^2)
        gradient = 2.0f0 * diff / length(diff)
    end
    
    return loss, gradient
end

"""
    calculate_coherence_loss(prediction, target)

Calcula la pérdida de coherencia espacial basada en gradientes locales.
"""
function calculate_coherence_loss(
    prediction::Array{T,3},
    target::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Calcular gradientes espaciales
    pred_gradients = spatial_gradients(prediction)
    target_gradients = spatial_gradients(target)
    
    # Calcular diferencia entre gradientes
    grad_diff_x = pred_gradients.x - target_gradients.x
    grad_diff_y = pred_gradients.y - target_gradients.y
    grad_diff_z = pred_gradients.z - target_gradients.z
    
    # Calcular pérdida como MSE de diferencias de gradientes
    loss = mean(grad_diff_x.^2 + grad_diff_y.^2 + grad_diff_z.^2)
    
    # Calcular gradiente de la pérdida (simplificado)
    # En la implementación completa, esto se calcularía adecuadamente con diferenciación automática
    # Aquí usamos una aproximación
    gradient = zeros(Float32, size(prediction))
    
    dim_x, dim_y, dim_z = size(prediction)
    
    # Propagar gradiente a través de operadores de diferencia
    for x in 2:dim_x-1
        for y in 2:dim_y-1
            for z in 2:dim_z-1
                # Contribución del gradiente X
                gradient[x+1, y, z] += grad_diff_x[x, y, z] / length(grad_diff_x)
                gradient[x-1, y, z] -= grad_diff_x[x, y, z] / length(grad_diff_x)
                
                # Contribución del gradiente Y
                gradient[x, y+1, z] += grad_diff_y[x, y, z] / length(grad_diff_y)
                gradient[x, y-1, z] -= grad_diff_y[x, y, z] / length(grad_diff_y)
                
                # Contribución del gradiente Z
                gradient[x, y, z+1] += grad_diff_z[x, y, z] / length(grad_diff_z)
                gradient[x, y, z-1] -= grad_diff_z[x, y, z] / length(grad_diff_z)
            end
        end
    end
    
    return loss, gradient
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

"""
    calculate_regularization(tensor)

Calcula un término de regularización para prevenir valores extremos.
"""
function calculate_regularization(tensor::Array{T,3}) where T <: AbstractFloat
    # Regularización L2 simple
    loss = mean(tensor.^2)
    gradient = 2.0f0 * tensor / length(tensor)
    
    return loss, gradient
end

"""
    apply_focus(gradient, prediction, target, focus_factor, threshold)

Amplifica el gradiente en regiones con error por encima del umbral.
"""
function apply_focus(
    gradient::Array{T,3},
    prediction::Array{S,3},
    target::Array{U,3},
    focus_factor::Float32,
    threshold::Float32
) where {T <: AbstractFloat, S <: AbstractFloat, U <: AbstractFloat}
    # Calcular error absoluto
    error = abs.(prediction - target)
    
    # Determinar regiones con error significativo
    significant_error = error .> threshold
    
    # Aplicar factor de enfoque
    focused_gradient = copy(gradient)
    focused_gradient[significant_error] .*= focus_factor
    
    return focused_gradient
end

"""
    default_loss()

Crea una función de pérdida multidimensional con parámetros por defecto.
"""
function default_loss()
    return MultidimensionalLoss()
end
end