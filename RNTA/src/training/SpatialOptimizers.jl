# training/SpatialOptimizers.jl
# Implementa optimizadores especializados para espacios tensoriales 3D
module SpatialOptimizers

using UUIDs
using LinearAlgebra
using Statistics
using ..BrainSpace
using ..TensorNeuron
using ..TensorOperations
export SpatialOptimizerState,
       SpatialOptimizer,
       optimization_step!,
       update_spatial_modulator!,
       default_optimizer
"""
    SpatialOptimizerState

Estado interno para el optimizador espacial adaptativo.
"""
mutable struct SpatialOptimizerState
    # Momentos tensoriales de primer orden
    m::Dict{UUID, Array{Float32,3}}
    
    # Momentos tensoriales de segundo orden
    v::Dict{UUID, Array{Float32,3}}
    
    # Contadores de pasos para corrección de sesgo
    t::Int
    
    # Mapas de importancia informacional para cada neurona
    importance::Dict{UUID, Array{Float32,3}}
    
    # Factores de modulación espacial
    spatial_modulators::Dict{UUID, Array{Float32,3}}
end

"""
    SpatialOptimizer

Optimizador adaptativo con conciencia espacial para redes tensoriales.
"""
struct SpatialOptimizer
    # Tasa de aprendizaje base
    alpha::Float32
    
    # Factores de decaimiento para momentos
    beta1::Float32
    beta2::Float32
    
    # Factor de modulación espacial
    lambda::Float32
    
    # Factor de sensibilidad a la importancia
    gamma::Float32
    
    # Amplitud de oscilación de la tasa de aprendizaje
    delta::Float32
    
    # Período de oscilación (en pasos)
    period::Int
    
    # Factor de estabilidad numérica
    epsilon::Float32
    
    # Estado del optimizador
    state::SpatialOptimizerState
end

"""
Constructor principal para SpatialOptimizer
"""
function SpatialOptimizer(
    brain::Brain_Space;
    alpha::Float32=0.001f0,
    beta1::Float32=0.9f0,
    beta2::Float32=0.999f0,
    lambda::Float32=0.2f0,
    gamma::Float32=1.0f0,
    delta::Float32=0.1f0,
    period::Int=1000,
    epsilon::Float32=1f-8
)
    # Inicializar estado
    m = Dict{UUID, Array{Float32,3}}()
    v = Dict{UUID, Array{Float32,3}}()
    importance = Dict{UUID, Array{Float32,3}}()
    spatial_modulators = Dict{UUID, Array{Float32,3}}()
    
    # Inicializar para cada neurona
    for (_, neuron) in brain.neurons
        # Obtener forma del kernel de transformación
        kernel_shape = size(neuron.transformation_kernel)
        
        # Inicializar momentos a cero
        m[neuron.id] = zeros(Float32, kernel_shape)
        v[neuron.id] = zeros(Float32, kernel_shape)
        
        # Inicializar importancia y moduladores a uno
        importance[neuron.id] = ones(Float32, kernel_shape)
        spatial_modulators[neuron.id] = ones(Float32, kernel_shape)
    end
    
    # Inicializar estado
    state = SpatialOptimizerState(m, v, 0, importance, spatial_modulators)
    
    return SpatialOptimizer(
        alpha,
        beta1,
        beta2,
        lambda,
        gamma,
        delta,
        period,
        epsilon,
        state
    )
end

"""
    optimization_step!(neuron, gradient, optimizer)

Aplica un paso de optimización a una neurona usando el optimizador espacial.
"""
function optimization_step!(
    neuron::Tensor_Neuron,
    gradient::Array{T,3},
    optimizer::SpatialOptimizer
) where T <: AbstractFloat
    # Aumentar contador de pasos
    optimizer.state.t += 1
    t = optimizer.state.t
    
    # Calcular fase temporal para oscilación de tasa de aprendizaje
    phi_t = 2.0f0 * π * (t % optimizer.period) / optimizer.period
    
    # Actualizar momento de primer orden
    optimizer.state.m[neuron.id] = optimizer.beta1 * optimizer.state.m[neuron.id] +
                                 (1.0f0 - optimizer.beta1) * gradient
    
    # Actualizar momento de segundo orden
    optimizer.state.v[neuron.id] = optimizer.beta2 * optimizer.state.v[neuron.id] +
                                 (1.0f0 - optimizer.beta2) * (gradient .^ 2)
    
    # Corregir sesgo
    m_hat = optimizer.state.m[neuron.id] / (1.0f0 - optimizer.beta1^t)
    v_hat = optimizer.state.v[neuron.id] / (1.0f0 - optimizer.beta2^t)
    
    # Actualizar factor de importancia informacional
    avg_grad_norm = mean(abs.(gradient))
    if avg_grad_norm > 0
        for i in CartesianIndices(gradient)
            rel_importance = abs(gradient[i]) / (avg_grad_norm + optimizer.epsilon)
            optimizer.state.importance[neuron.id][i] = tanh(optimizer.gamma * rel_importance)
        end
    end
    
    # Actualizar modulador espacial
    update_spatial_modulator!(optimizer.state.spatial_modulators[neuron.id], gradient, optimizer.lambda)
    
    # Calcular tasa de aprendizaje adaptativa con oscilación
    alpha_t = optimizer.alpha * (1.0f0 + optimizer.delta * cos(phi_t))
    
    # Calcular actualización final
    update = similar(gradient)
    
    for i in CartesianIndices(gradient)
        # Tasa de aprendizaje específica por posición
        alpha_pos = alpha_t * optimizer.state.importance[neuron.id][i]
        
        # Actualización con modulación espacial
        update[i] = alpha_pos * m_hat[i] / 
                    (optimizer.state.spatial_modulators[neuron.id][i] * sqrt(v_hat[i]) + optimizer.epsilon)
    end
    
    # Aplicar actualización
    neuron.transformation_kernel .-= update
    
    return neuron
end

"""
    update_spatial_modulator!(modulator, gradient, lambda)

Actualiza el modulador espacial basado en gradientes locales.
"""
function update_spatial_modulator!(
    modulator::Array{Float32,3},
    gradient::Array{T,3},
    lambda::Float32
) where T <: AbstractFloat
    dim_x, dim_y, dim_z = size(gradient)
    
    # Para cada posición, calcular diferencia con vecinos
    for x in 2:dim_x-1
        for y in 2:dim_y-1
            for z in 2:dim_z-1
                # Gradiente en posición actual
                current_grad = gradient[x, y, z]
                
                # Calcular diferencia cuadrática con vecinos
                diff_sum = 0.0f0
                
                for dx in -1:1
                    for dy in -1:1
                        for dz in -1:1
                            # Omitir posición actual
                            if dx == 0 && dy == 0 && dz == 0
                                continue
                            end
                            
                            # Gradiente del vecino
                            neighbor_grad = gradient[x+dx, y+dy, z+dz]
                            
                            # Diferencia cuadrática
                            diff_sum += (neighbor_grad - current_grad)^2
                        end
                    end
                end
                
                # Calcular modulador espacial
                modulator[x, y, z] = 1.0f0 + lambda * sqrt(diff_sum)
            end
        end
    end
    
    # Tratar bordes - copiar valores adyacentes
    for x in 1:dim_x
        for y in 1:dim_y
            for z in 1:dim_z
                if x == 1 || x == dim_x || y == 1 || y == dim_y || z == 1 || z == dim_z
                    # Calcular posición válida más cercana
                    nx = max(2, min(x, dim_x-1))
                    ny = max(2, min(y, dim_y-1))
                    nz = max(2, min(z, dim_z-1))
                    
                    # Copiar valor
                    modulator[x, y, z] = modulator[nx, ny, nz]
                end
            end
        end
    end
    
    return modulator
end

"""
    default_optimizer(brain)

Crea un optimizador con parámetros por defecto para el espacio cerebral.
"""
function default_optimizer(brain::Brain_Space)
    return SpatialOptimizer(brain)
end
end