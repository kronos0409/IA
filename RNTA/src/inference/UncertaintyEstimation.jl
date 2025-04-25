# inference/UncertaintyEstimation.jl
# Implementa estimación de incertidumbre para el sistema RNTA

module UncertaintyEstimation

using LinearAlgebra
using Statistics
using Random
using Distributions

# Importaciones de otros módulos de RNTA
using ..BrainSpace
using ..TensorOperations
using ..InternalDialogue
using ..ReasoningPathways

"""
    UncertaintyMetrics

Métricas de incertidumbre calculadas para una inferencia.
"""
struct UncertaintyMetrics
    # Incertidumbre aleatoria (variabilidad inherente)
    aleatoric::Float32
    
    # Incertidumbre epistémica (falta de conocimiento)
    epistemic::Float32
    
    # Incertidumbre distribucional (desajuste con distribuciones conocidas)
    distributional::Float32
    
    # Incertidumbre estructural (relacionada con la arquitectura)
    structural::Float32
    
    # Incertidumbre combinada (métrica global)
    combined::Float32
    
    # Descomposición espacial de la incertidumbre (tensor)
    uncertainty_map::Array{Float32,3}
    
    # Detalles adicionales
    details::Dict{Symbol, Any}
end

"""
Constructor para UncertaintyMetrics
"""
function UncertaintyMetrics(
    aleatoric::Float32,
    epistemic::Float32,
    distributional::Float32,
    structural::Float32,
    uncertainty_map::Array{T,3};
    details::Dict{Symbol, Any}=Dict{Symbol, Any}()
) where T <: AbstractFloat
    # Calcular incertidumbre combinada
    combined = sqrt(aleatoric^2 + epistemic^2 + distributional^2 + structural^2)
    
    return UncertaintyMetrics(
        aleatoric,
        epistemic,
        distributional,
        structural,
        combined,
        convert(Array{Float32,3}, uncertainty_map),
        details
    )
end

"""
    UncertaintyEstimator

Estimador de incertidumbre para el sistema RNTA.
"""
mutable struct UncertaintyEstimator
    # Cerebro base
    brain::Brain_Space
    
    # Número de muestras para Monte Carlo
    num_samples::Int
    
    # Nivel de ruido para perturbaciones
    noise_level::Float32
    
    # Historial de estimaciones
    estimation_history::Vector{UncertaintyMetrics}
    
    # Tensor de referencia para incertidumbre previa
    prior_uncertainty::Union{Array{Float32,3}, Nothing}
    
    # Configuración del estimador
    config::Dict{Symbol, Any}
end

"""
Constructor para UncertaintyEstimator
"""
function UncertaintyEstimator(
    brain::Brain_Space;
    num_samples::Int=10,
    noise_level::Float32=0.05f0,
    config::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    # Inicializar estimador
    return UncertaintyEstimator(
        brain,
        num_samples,
        noise_level,
        Vector{UncertaintyMetrics}(),
        nothing,
        config
    )
end

"""
    estimate_uncertainty(estimator, input_tensor)

Estima la incertidumbre para un tensor de entrada.
"""
function estimate_uncertainty(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Verificar que las dimensiones coinciden con el cerebro
    if size(input_tensor) != estimator.brain.dimensions
        input_tensor = tensor_interpolation(input_tensor, estimator.brain.dimensions)
    end
    
    # Estimar cada tipo de incertidumbre
    aleatoric = estimate_aleatoric_uncertainty(estimator, input_tensor)
    epistemic = estimate_epistemic_uncertainty(estimator, input_tensor)
    distributional = estimate_distributional_uncertainty(estimator, input_tensor)
    structural = estimate_structural_uncertainty(estimator, input_tensor)
    
    # Calcular mapa espacial de incertidumbre
    uncertainty_map = calculate_uncertainty_map(
        estimator, 
        input_tensor,
        aleatoric,
        epistemic,
        distributional,
        structural
    )
    
    # Crear métricas
    metrics = UncertaintyMetrics(
        aleatoric,
        epistemic,
        distributional,
        structural,
        uncertainty_map
    )
    
    # Guardar en historial
    push!(estimator.estimation_history, metrics)
    
    # Actualizar incertidumbre previa
    estimator.prior_uncertainty = copy(uncertainty_map)
    
    return metrics
end

"""
    estimate_aleatoric_uncertainty(estimator, input_tensor)

Estima la incertidumbre aleatoria mediante múltiples pases con ruido.
"""
function estimate_aleatoric_uncertainty(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Guardar tensor original para referencia
    original_output = process(estimator.brain, input_tensor)
    
    # Realizar múltiples pases con ruido
    noisy_outputs = []
    
    for _ in 1:estimator.num_samples
        # Añadir ruido al tensor de entrada
        noise = randn(Float32, size(input_tensor)) * estimator.noise_level * mean(abs.(input_tensor))
        noisy_input = input_tensor + noise
        
        # Procesar con el cerebro
        noisy_output = process(estimator.brain, noisy_input)
        
        push!(noisy_outputs, noisy_output)
    end
    
    # Calcular varianza media a través de todas las muestras
    sample_variance = zeros(Float32, size(original_output))
    
    for output in noisy_outputs
        sample_variance .+= (output .- original_output).^2
    end
    
    sample_variance ./= estimator.num_samples
    
    # Calcular métrica de incertidumbre aleatoria
    aleatoric_uncertainty = mean(sqrt.(sample_variance))
    
    return aleatoric_uncertainty
end

"""
    estimate_epistemic_uncertainty(estimator, input_tensor)

Estima la incertidumbre epistémica mediante diálogo interno.
"""
function estimate_epistemic_uncertainty(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Usar diálogo interno para obtener múltiples opiniones
    result, stats = internal_dialogue(
        estimator.brain,
        input_tensor,
        max_steps=5,
        num_agents=3
    )
    
    # Extraer convergencia como indicador inverso de incertidumbre epistémica
    convergence = stats[:convergence_level]
    
    # Convertir convergencia a incertidumbre (mayor convergencia = menor incertidumbre)
    epistemic_uncertainty = 1.0f0 - convergence
    
    return epistemic_uncertainty
end

"""
    estimate_distributional_uncertainty(estimator, input_tensor)

Estima la incertidumbre distribucional mediante comparación con datos conocidos.
"""
function estimate_distributional_uncertainty(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Para estimar incertidumbre distribucional, comparamos la salida con
    # distribuciones conocidas o esperadas
    
    # Procesar entrada
    output = process(estimator.brain, input_tensor)
    
    # En una implementación completa, compararíamos con distribuciones de referencia
    # Por ahora, usar una métrica simplificada basada en normalidad
    
    # Calcular estadísticas de la salida
    flat_output = vec(output)
    output_mean = mean(flat_output)
    output_std = std(flat_output)
    
    # Verificar normalidad mediante test Jarque-Bera simplificado
    n = length(flat_output)
    
    # Calcular asimetría
    skewness = sum((flat_output .- output_mean).^3) / (n * output_std^3)
    
    # Calcular curtosis
    kurtosis = sum((flat_output .- output_mean).^4) / (n * output_std^4) - 3.0f0
    
    # Estadístico JB
    jb = n / 6 * (skewness^2 + kurtosis^2 / 4)
    
    # Normalizar a [0,1]
    distributional_uncertainty = min(1.0f0, jb / 10.0f0)
    
    return distributional_uncertainty
end

"""
    estimate_structural_uncertainty(estimator, input_tensor)

Estima la incertidumbre estructural mediante análisis de la arquitectura.
"""
function estimate_structural_uncertainty(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # La incertidumbre estructural se relaciona con limitaciones de la arquitectura
    
    # Un enfoque es medir la activación de regiones funcionales especializadas
    functional_regions = estimator.brain.functional_regions
    
    if isempty(functional_regions)
        # Si no hay regiones funcionales definidas, usar enfoque alternativo
        return estimate_structural_uncertainty_fallback(estimator, input_tensor)
    end
    
    # Obtener activación por región funcional
    region_activations = Dict{Symbol, Float32}()
    
    for (region_type, positions) in functional_regions
        # Calcular activación media en esta región
        activations = Float32[]
        
        for pos in positions
            # Solo considerar posiciones válidas
            if all(1 .<= pos .<= size(estimator.brain.global_state))
                push!(activations, abs(estimator.brain.global_state[pos...]))
            end
        end
        
        region_activations[region_type] = isempty(activations) ? 0.0f0 : mean(activations)
    end
    
    # Calcular varianza de activación entre regiones
    activation_values = collect(values(region_activations))
    
    if isempty(activation_values)
        return 0.5f0  # Valor neutro si no hay datos
    end
    
    # Normalizar activaciones
    if maximum(activation_values) > 0
        activation_values ./= maximum(activation_values)
    end
    
    # Calcular coeficiente de variación como medida de especialización
    cv = std(activation_values) / (mean(activation_values) + 1e-8f0)
    
    # Convertir a incertidumbre (mayor variación = menor incertidumbre estructural)
    structural_uncertainty = max(0.0f0, 1.0f0 - cv)
    
    return structural_uncertainty
end

"""
    estimate_structural_uncertainty_fallback(estimator, input_tensor)

Método alternativo para estimar incertidumbre estructural cuando no hay regiones funcionales.
"""
function estimate_structural_uncertainty_fallback(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Procesar entrada
    output = process(estimator.brain, input_tensor)
    
    # Calcular activación por cada dimensión
    dim_x, dim_y, dim_z = size(output)
    
    # Activación media por dimensión x
    x_activations = [mean(abs.(output[i,:,:])) for i in 1:dim_x]
    
    # Activación media por dimensión y
    y_activations = [mean(abs.(output[:,j,:])) for j in 1:dim_y]
    
    # Activación media por dimensión z
    z_activations = [mean(abs.(output[:,:,k])) for k in 1:dim_z]
    
    # Calcular coeficientes de variación
    cv_x = std(x_activations) / (mean(x_activations) + 1e-8f0)
    cv_y = std(y_activations) / (mean(y_activations) + 1e-8f0)
    cv_z = std(z_activations) / (mean(z_activations) + 1e-8f0)
    
    # Promediar coeficientes
    avg_cv = (cv_x + cv_y + cv_z) / 3
    
    # Convertir a incertidumbre
    structural_uncertainty = max(0.0f0, 1.0f0 - avg_cv)
    
    return structural_uncertainty
end

"""
    calculate_uncertainty_map(estimator, input_tensor, aleatoric, epistemic, distributional, structural)

Calcula un mapa espacial de incertidumbre.
"""
function calculate_uncertainty_map(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3},
    aleatoric::Float32,
    epistemic::Float32,
    distributional::Float32,
    structural::Float32
) where T <: AbstractFloat
    # Inicializar mapa de incertidumbre
    uncertainty_map = zeros(Float32, size(input_tensor))
    
    # Realizar múltiples pases con ruido
    sample_variance = zeros(Float32, size(input_tensor))
    
    for i in 1:estimator.num_samples
        # Añadir ruido al tensor de entrada
        noise = randn(Float32, size(input_tensor)) * estimator.noise_level * mean(abs.(input_tensor))
        noisy_input = input_tensor + noise
        
        # Procesar con el cerebro
        noisy_output = process(estimator.brain, noisy_input)
        
        # Acumular varianza
        if i > 1
            sample_variance .+= (noisy_output .- output).^2
        end
        
        # Guardar último output
        output = noisy_output
    end
    
    # Normalizar varianza
    sample_variance ./= max(1, estimator.num_samples - 1)
    
    # Convertir varianza a incertidumbre local
    local_uncertainty = sqrt.(sample_variance)
    
    # Normalizar
    max_uncertainty = maximum(local_uncertainty)
    if max_uncertainty > 0
        local_uncertainty ./= max_uncertainty
    end
    
    # Incorporar componentes globales
    for i in CartesianIndices(uncertainty_map)
        # Combinar incertidumbre local con componentes globales
        local_factor = local_uncertainty[i]
        
        # Ponderar componentes
        weighted_aleatoric = 0.7f0 * local_factor + 0.3f0 * aleatoric
        weighted_epistemic = epistemic  # Componente global
        weighted_distributional = distributional  # Componente global
        weighted_structural = structural  # Componente global
        
        # Combinar componentes
        uncertainty_map[i] = sqrt(
            weighted_aleatoric^2 + 
            weighted_epistemic^2 + 
            weighted_distributional^2 + 
            weighted_structural^2
        )
    end
    
    # Incorporar incertidumbre previa si está disponible
    if !isnothing(estimator.prior_uncertainty)
        if size(estimator.prior_uncertainty) == size(uncertainty_map)
            # Factor de decaimiento para incertidumbre previa
            decay_factor = 0.7f0
            
            # Combinar con incertidumbre actual
            uncertainty_map = (1.0f0 - decay_factor) * uncertainty_map + 
                              decay_factor * estimator.prior_uncertainty
        end
    end
    
    # Normalizar mapa final
    uncertainty_map ./= maximum(uncertainty_map)
    
    return uncertainty_map
end

"""
    calculate_uncertainty_metrics(estimator, outputs)

Calcula métricas de incertidumbre para un conjunto de salidas.
"""
function calculate_uncertainty_metrics(
    estimator::UncertaintyEstimator,
    outputs::Vector{Array{T,3}}
) where T <: AbstractFloat
    # Verificar que hay suficientes muestras
    if length(outputs) < 2
        error("Se necesitan al menos 2 muestras para calcular incertidumbre")
    end
    
    # Calcular media
    mean_output = zeros(Float32, size(outputs[1]))
    
    for output in outputs
        mean_output .+= output
    end
    
    mean_output ./= length(outputs)
    
    # Calcular varianza
    variance = zeros(Float32, size(outputs[1]))
    
    for output in outputs
        variance .+= (output .- mean_output).^2
    end
    
    variance ./= length(outputs)
    
    # Calcular métricas estadísticas
    metrics = Dict{Symbol, Any}(
        :mean => mean(mean_output),
        :std => sqrt(mean(variance)),
        :entropy => calculate_entropy(mean_output),
        :max_variance => maximum(variance),
        :total_variance => sum(variance)
    )
    
    return metrics
end

"""
    calculate_entropy(tensor)

Calcula la entropía de un tensor, una medida de incertidumbre.
"""
function calculate_entropy(tensor::Array{T,3}) where T <: AbstractFloat
    # Normalizar tensor a distribución de probabilidad
    flat = vec(tensor)
    
    # Transformar a valores positivos
    shifted = flat .- minimum(flat)
    
    # Evitar división por cero
    if sum(shifted) < 1e-8
        return 0.0f0
    end
    
    # Normalizar a distribución de probabilidad
    prob_dist = shifted ./ sum(shifted)
    
    # Calcular entropía
    entropy = 0.0f0
    
    for p in prob_dist
        if p > 1e-8
            entropy -= p * log2(p)
        end
    end
    
    return entropy
end

"""
    estimate_uncertainty_from_pathway(estimator, pathway)

Estima la incertidumbre a partir de una trayectoria de razonamiento.
"""
function estimate_uncertainty_from_pathway(
    estimator::UncertaintyEstimator,
    pathway::ReasoningPathway
)
    # Obtener resultado final de la trayectoria
    result = get_pathway_result(pathway)
    
    if isnothing(result)
        error("La trayectoria no tiene resultado")
    end
    
    # Usar confianza de la trayectoria como indicador inverso de incertidumbre
    epistemic_uncertainty = 1.0f0 - pathway.confidence
    
    # Estimar otros componentes
    aleatoric = 0.2f0  # Valor por defecto
    distributional = 0.3f0  # Valor por defecto
    structural = 0.2f0  # Valor por defecto
    
    # Intentar estimar componentes de la estructura de la trayectoria
    try
        # Analizar variabilidad entre nodos de la trayectoria
        node_tensors = []
        
        for (_, node) in pathway.nodes
            push!(node_tensors, node.tensor)
        end
        
        # Si hay suficientes nodos, calcular métricas
        if length(node_tensors) >= 2
            metrics = calculate_uncertainty_metrics(estimator, node_tensors)
            
            # Extraer componentes de las métricas
            aleatoric = min(1.0f0, metrics[:std] * 2.0f0)
            distributional = min(1.0f0, metrics[:entropy] / 4.0f0)
        end
    catch e
        # Si hay error, mantener valores por defecto
        @warn "Error al calcular componentes de incertidumbre: $e"
    end
    
    # Calcular mapa de incertidumbre
    uncertainty_map = zeros(Float32, size(result))
    
    # Llenar mapa con incertidumbre
    for i in CartesianIndices(uncertainty_map)
        uncertainty_map[i] = sqrt(
            aleatoric^2 + 
            epistemic_uncertainty^2 + 
            distributional^2 + 
            structural^2
        )
    end
    
    # Crear métricas
    metrics = UncertaintyMetrics(
        aleatoric,
        epistemic_uncertainty,
        distributional,
        structural,
        uncertainty_map,
        details=Dict{Symbol, Any}(
            :pathway_id => string(pathway.id),
            :nodes_count => length(pathway.nodes),
            :edges_count => length(pathway.edges)
        )
    )
    
    # Guardar en historial
    push!(estimator.estimation_history, metrics)
    
    # Actualizar incertidumbre previa
    estimator.prior_uncertainty = copy(uncertainty_map)
    
    return metrics
end

"""
    calibrate_uncertainty(estimator, ground_truth, predictions, uncertainties)

Calibra el estimador de incertidumbre usando datos de referencia.
"""
function calibrate_uncertainty(
    estimator::UncertaintyEstimator,
    ground_truth::Vector{Array{T,3}},
    predictions::Vector{Array{S,3}},
    uncertainties::Vector{UncertaintyMetrics}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Verificar que hay suficientes muestras
    if length(ground_truth) < 10 || 
       length(predictions) != length(ground_truth) || 
       length(uncertainties) != length(ground_truth)
        error("Se necesitan al menos 10 muestras con predicciones y métricas para calibrar")
    end
    
    # Calcular errores
    errors = []
    
    for i in 1:length(ground_truth)
        # Asegurar dimensiones compatibles
        if size(predictions[i]) != size(ground_truth[i])
            predictions[i] = tensor_interpolation(predictions[i], size(ground_truth[i]))
        end
        
        # Calcular error
        error = mean(abs.(predictions[i] - ground_truth[i]))
        push!(errors, error)
    end
    
    # Extraer componentes de incertidumbre
    aleatoric_values = [u.aleatoric for u in uncertainties]
    epistemic_values = [u.epistemic for u in uncertainties]
    distributional_values = [u.distributional for u in uncertainties]
    structural_values = [u.structural for u in uncertainties]
    combined_values = [u.combined for u in uncertainties]
    
    # Calcular correlaciones
    correlations = Dict{Symbol, Float32}(
        :aleatoric => correlation(errors, aleatoric_values),
        :epistemic => correlation(errors, epistemic_values),
        :distributional => correlation(errors, distributional_values),
        :structural => correlation(errors, structural_values),
        :combined => correlation(errors, combined_values)
    )
    
    # Calcular factores de calibración
    calibration_factors = Dict{Symbol, Float32}(
        :aleatoric => mean(errors) / mean(aleatoric_values),
        :epistemic => mean(errors) / mean(epistemic_values),
        :distributional => mean(errors) / mean(distributional_values),
        :structural => mean(errors) / mean(structural_values),
        :combined => mean(errors) / mean(combined_values)
    )
    
    # Actualizar configuración del estimador
    estimator.config[:calibration_factors] = calibration_factors
    estimator.config[:correlations] = correlations
    estimator.config[:calibrated] = true
    
    # Devolver métricas de calibración
    return Dict{Symbol, Any}(
        :correlations => correlations,
        :calibration_factors => calibration_factors,
        :mean_error => mean(errors),
        :error_std => std(errors)
    )
end

"""
    correlation(x, y)

Calcula la correlación entre dos vectores.
"""
function correlation(x::Vector{T}, y::Vector{S}) where {T <: Real, S <: Real}
    if length(x) != length(y) || isempty(x)
        return 0.0f0
    end
    
    # Normalizar a media cero
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    
    # Calcular correlación
    numerator = sum(x_centered .* y_centered)
    denominator = sqrt(sum(x_centered.^2) * sum(y_centered.^2))
    
    # Evitar división por cero
    if denominator < 1e-8
        return 0.0f0
    end
    
    return numerator / denominator
end

"""
    apply_calibration(estimator, metrics)

Aplica factores de calibración a métricas de incertidumbre.
"""
function apply_calibration(
    estimator::UncertaintyEstimator,
    metrics::UncertaintyMetrics
)
    # Verificar si hay factores de calibración
    if !haskey(estimator.config, :calibration_factors) || !estimator.config[:calibrated]
        return metrics
    end
    
    # Obtener factores
    factors = estimator.config[:calibration_factors]
    
    # Aplicar factores a componentes
    aleatoric = metrics.aleatoric * factors[:aleatoric]
    epistemic = metrics.epistemic * factors[:epistemic]
    distributional = metrics.distributional * factors[:distributional]
    structural = metrics.structural * factors[:structural]
    
    # Recalcular mapa de incertidumbre
    uncertainty_map = metrics.uncertainty_map * factors[:combined]
    
    # Crear métricas calibradas
    calibrated = UncertaintyMetrics(
        aleatoric,
        epistemic,
        distributional,
        structural,
        uncertainty_map,
        details=merge(
            metrics.details,
            Dict{Symbol, Any}(:calibrated => true)
        )
    )
    
    return calibrated
end

"""
    visualize_uncertainty(metrics; options...)

Genera una visualización de la incertidumbre.
"""
function visualize_uncertainty(
    metrics::UncertaintyMetrics;
    projection::Symbol=:max,
    colormap=:viridis
)
    # Esta función sería implementada con alguna biblioteca de visualización
    # En esta implementación, devolvemos datos para visualización
    
    # Proyectar mapa de incertidumbre 3D a 2D
    projection_x = dropdims(maximum(metrics.uncertainty_map, dims=1), dims=1)
    projection_y = dropdims(maximum(metrics.uncertainty_map, dims=2), dims=2)
    projection_z = dropdims(maximum(metrics.uncertainty_map, dims=3), dims=3)
    
    # Preparar datos para visualización
    visualization_data = Dict{Symbol, Any}(
        :components => Dict{Symbol, Float32}(
            :aleatoric => metrics.aleatoric,
            :epistemic => metrics.epistemic,
            :distributional => metrics.distributional,
            :structural => metrics.structural,
            :combined => metrics.combined
        ),
        :projections => Dict{Symbol, Array{Float32,2}}(
            :x => projection_x,
            :y => projection_y,
            :z => projection_z
        ),
        :details => metrics.details
    )
    
    return visualization_data
end

"""
    uncertainty_threshold(metrics, threshold=0.7)

Determina si la incertidumbre supera un umbral crítico.
"""
function uncertainty_threshold(
    metrics::UncertaintyMetrics,
    threshold::Float32=0.7f0
)
    # Verificar si la incertidumbre combinada supera el umbral
    high_uncertainty = metrics.combined > threshold
    
    # Identificar componentes críticos
    critical_components = Symbol[]
    
    if metrics.aleatoric > threshold
        push!(critical_components, :aleatoric)
    end
    
    if metrics.epistemic > threshold
        push!(critical_components, :epistemic)
    end
    
    if metrics.distributional > threshold
        push!(critical_components, :distributional)
    end
    
    if metrics.structural > threshold
        push!(critical_components, :structural)
    end
    
    # Calcular fracción del volumen con alta incertidumbre
    high_volume_fraction = count(metrics.uncertainty_map .> threshold) / length(metrics.uncertainty_map)
    
    # Preparar resultado
    result = Dict{Symbol, Any}(
        :high_uncertainty => high_uncertainty,
        :critical_components => critical_components,
        :high_volume_fraction => high_volume_fraction,
        :threshold => threshold,
        :combined => metrics.combined,
        :recommendation => high_uncertainty ? :reject : :accept
    )
    
    return result
end

"""
    monte_carlo_uncertainty(estimator, input_tensor, num_samples=20)

Estima incertidumbre con mayor precisión usando muestreo Monte Carlo.
"""
function monte_carlo_uncertainty(
    estimator::UncertaintyEstimator,
    input_tensor::Array{T,3},
    num_samples::Int=20
) where T <: AbstractFloat
    # Guardar configuración original
    original_num_samples = estimator.num_samples
    
    # Actualizar configuración
    estimator.num_samples = num_samples
    
    # Estimar incertidumbre
    metrics = estimate_uncertainty(estimator, input_tensor)
    
    # Restaurar configuración
    estimator.num_samples = original_num_samples
    
    return metrics
end

# Exportar tipos y funciones principales
export UncertaintyMetrics, UncertaintyEstimator,
       estimate_uncertainty, estimate_uncertainty_from_pathway,
       calibrate_uncertainty, apply_calibration,
       visualize_uncertainty, uncertainty_threshold,
       monte_carlo_uncertainty

end # module UncertaintyEstimation