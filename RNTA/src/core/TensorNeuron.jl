# core/TensorNeuron.jl
# Define la unidad neuronal fundamental basada en tensores

"""
    TensorNeuron

Unidad computacional básica del sistema RNTA que opera con tensores 3D
en lugar de escalares.
"""
# Añadir al inicio del archivo:
module TensorNeuron

using DataStructures # Para CircularBuffer
using UUIDs         # Para uuid4()
using ..SpatialField # Asumiendo que SpatialField es un módulo hermano
using ..PlasticityRules: PlasticityParameters # Importar PlasticityParameters

export Tensor_Neuron, process_input, update_weights!, adapt_receptive_field!, clone, should_expand, update_functional_type!, apply_hebbian_learning!
mutable struct Tensor_Neuron
    # Identificador único
    id::UUID
    
    # Posición en el espacio 3D
    position::NTuple{3,Int}
    
    # Campo receptivo - define la región del espacio de entrada que esta neurona observa
    receptive_field::Spatial_Field
    
    # Estado interno - tensor de activación actual
    state::Array{Float32,3}
    
    # Kernel de transformación - determina cómo esta neurona transforma sus entradas
    transformation_kernel::Array{Float32,3}
    
    # Propiedades de plasticidad
    plasticity::PlasticityParameters
    
    # Historia de activación reciente (para mecanismos de aprendizaje)
    activation_history::CircularBuffer{Array{Float32,3}}
    
    # Nivel de especialización (0.0-1.0) - aumenta con el entrenamiento
    specialization::Float32
    
    # Tipo funcional - se especializa con el tiempo
    functional_type::Symbol
end

"""
Constructor principal para TensorNeuron
"""
function Tensor_Neuron(
    position::NTuple{3,Int}, 
    receptive_field_size::NTuple{3,Int};
    state_size::NTuple{3,Int}=(3,3,3),
    kernel_size::NTuple{3,Int}=(3,3,3),
    init_scale::Float32=0.1f0,
    history_length::Int=10,
    plasticity::PlasticityParameters=PlasticityParameters()
)
    id = uuid4()
    
    # Crear campo receptivo
    receptive_field = SpatialField(position, receptive_field_size)
    
    # Inicializar estado interno
    state = zeros(Float32, state_size)
    
    # Inicializar kernel de transformación con pequeños valores aleatorios
    transformation_kernel = randn(Float32, kernel_size) * init_scale
    
    # Inicializar buffer circular para historial de activación
    activation_history = CircularBuffer{Array{Float32,3}}(history_length)
    for _ in 1:history_length
        push!(activation_history, zeros(Float32, state_size))
    end
    
    # Inicializar con baja especialización
    specialization = 0.0f0
    
    # Inicialmente, todas las neuronas son de tipo general
    functional_type = :general
    
    return TensorNeuron(
        id, 
        position, 
        receptive_field, 
        state, 
        transformation_kernel,
        plasticity,
        activation_history,
        specialization,
        functional_type
    )
end

"""
    process_input(neuron, input_tensor)

Procesa un tensor de entrada y actualiza el estado interno de la neurona.
"""
function process_input(neuron::Tensor_Neuron, input_tensor::Array{T,3}) where T <: AbstractFloat
    # Extraer la región relevante del tensor de entrada según el campo receptivo
    local_field = extract_tensor(input_tensor, neuron.receptive_field)
    
    # Aplicar transformación tensorial (convolución adaptativa)
    transformed = tensor_convolution(local_field, neuron.transformation_kernel)
    
    # Crear parámetros de activación basados en el tipo funcional
    activation_params = ActivationParameters(
        activation_type = neuron.functional_type,
        history = [copy(state) for state in neuron.activation_history]
    )
    
    # Aplicar función de activación volumétrica
    activated = volumetric_activation(transformed, 
                                     type=neuron.functional_type, 
                                     parameters=activation_params)
    
    # Actualizar estado interno mediante integración con estado anterior
    new_state = update_neuronal_state(neuron.state, activated, neuron.plasticity)
    
    # Guardar nuevo estado
    neuron.state = copy(new_state)
    
    # Actualizar historial de activación
    push!(neuron.activation_history, copy(new_state))
    
    return new_state
end

"""
    update_weights!(neuron, gradient, learning_rate)

Actualiza los pesos (kernel de transformación) de la neurona según el gradiente.
"""
function update_weights!(neuron::Tensor_Neuron, gradient::Array{T,3}, learning_rate::Float32) where T <: AbstractFloat
    # Aplicar reglas de plasticidad para modular el aprendizaje
    # Las neuronas más especializadas aprenden más lentamente
    effective_lr = learning_rate * (1.0f0 - neuron.specialization * neuron.plasticity.specialization_decay)
    
    # Actualizar kernel de transformación
    neuron.transformation_kernel .-= effective_lr .* gradient
    
    # Incrementar ligeramente la especialización con cada actualización
    neuron.specialization = min(1.0f0, neuron.specialization + 0.0001f0)
    
    # Posiblemente actualizar el tipo funcional basado en patrones de activación
    update_functional_type!(neuron)
    
    return neuron
end

"""
    adapt_receptive_field!(neuron, input_sensitivities)

Adapta el campo receptivo de la neurona basándose en la sensibilidad a diferentes
regiones de entrada.
"""
function adapt_receptive_field!(neuron::Tensor_Neuron, input_sensitivities::Array{Float32,3})
    # Encontrar región de máxima sensibilidad
    max_val, max_idx = findmax(input_sensitivities)
    
    # Convertir índice a coordenadas
    max_coords = Tuple(max_idx)
    
    # Calcular dirección desde el centro del campo receptivo hacia la máxima sensibilidad
    direction = (
        max_coords[1] - neuron.receptive_field.center[1],
        max_coords[2] - neuron.receptive_field.center[2],
        max_coords[3] - neuron.receptive_field.center[3]
    )
    
    # Normalizar dirección
    dir_length = sqrt(direction[1]^2 + direction[2]^2 + direction[3]^2)
    if dir_length > 0
        normalized_dir = (
            direction[1] / dir_length,
            direction[2] / dir_length,
            direction[3] / dir_length
        )
        
        # Ajustar campo receptivo
        neuron.receptive_field = shift_field!(
            neuron.receptive_field, 
            normalized_dir, 
            neuron.plasticity.adaptation_rate
        )
    end
    
    return neuron
end

"""
    clone(neuron)

Crea una copia profunda de la neurona para propósitos de deliberación interna.
"""
function clone(neuron::Tensor_Neuron)
    return TensorNeuron(
        neuron.id,
        neuron.position,
        clone(neuron.receptive_field),
        copy(neuron.state),
        copy(neuron.transformation_kernel),
        neuron.plasticity,  # Structs son inmutables, no necesitan copiarse
        deepcopy(neuron.activation_history),
        neuron.specialization,
        neuron.functional_type
    )
end

"""
    should_expand(neuron)

Determina si esta neurona debería expandirse (generar neuronas hijas)
basado en su actividad y saturación.
"""
function should_expand(neuron::Tensor_Neuron)
    # Calcular nivel de actividad media reciente
    recent_activity = mean([sum(abs.(state)) / length(state) for state in neuron.activation_history])
    
    # Calcular saturación (qué porcentaje de las unidades están cerca de su max/min)
    saturation = calculate_saturation(neuron.state)
    
    # Una neurona debería expandirse si está muy activa y saturada
    return recent_activity > neuron.plasticity.expansion_activity_threshold && 
           saturation > neuron.plasticity.expansion_saturation_threshold
end

"""
    update_functional_type!(neuron)

Actualiza el tipo funcional de la neurona basándose en sus patrones de activación.
"""
function update_functional_type!(neuron::Tensor_Neuron)
    # Solo actualizar si la neurona no está muy especializada ya
    if neuron.specialization < 0.5f0
        # Analizar patrones de activación recientes
        patterns = analyze_activation_patterns(neuron.activation_history)
        
        # Determinar tipo funcional basado en estos patrones
        if patterns.temporal_sensitivity > 0.7f0
            neuron.functional_type = :temporal
        elseif patterns.spatial_sensitivity > 0.7f0
            neuron.functional_type = :spatial
        elseif patterns.feature_sensitivity > 0.7f0
            neuron.functional_type = :feature
        elseif patterns.context_sensitivity > 0.7f0
            neuron.functional_type = :contextual
        else
            neuron.functional_type = :general
        end
    end
    
    return neuron
end

"""
    apply_hebbian_learning!(neuron, pre_synaptic, post_synaptic)

Aplica aprendizaje Hebbiano: "Las neuronas que se disparan juntas, se conectan"
"""
function apply_hebbian_learning!(
    neuron::Tensor_Neuron, 
    pre_synaptic::Array{T,3}, 
    post_synaptic::Array{S,3},
    learning_rate::Float32=0.01f0
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que pre_synaptic tiene las mismas dimensiones que el kernel
    if size(pre_synaptic) != size(neuron.transformation_kernel)
        pre_synaptic = tensor_interpolation(pre_synaptic, size(neuron.transformation_kernel))
    end
    
    # Si post_synaptic es un escalar, convertirlo a tensor
    if isa(post_synaptic, Number)
        post_value = convert(Float32, post_synaptic)
        post_synaptic = ones(Float32, size(neuron.transformation_kernel)) .* post_value
    end
    
    # Asegurar que post_synaptic tiene las mismas dimensiones que el kernel
    if size(post_synaptic) != size(neuron.transformation_kernel)
        post_synaptic = tensor_interpolation(post_synaptic, size(neuron.transformation_kernel))
    end
    
    # Calcular correlación Hebbiana
    hebbian_update = pre_synaptic .* post_synaptic
    
    # Aplicar reglas de plasticidad específicas de Hebbian
    for i in CartesianIndices(hebbian_update)
        if hebbian_update[i] > neuron.plasticity.hebbian_threshold
            # Potenciación (correlación positiva fuerte)
            neuron.transformation_kernel[i] += learning_rate * hebbian_update[i]
        elseif hebbian_update[i] < neuron.plasticity.anti_hebbian_threshold
            # Depresión (correlación negativa fuerte)
            neuron.transformation_kernel[i] -= learning_rate * abs(hebbian_update[i])
        end
    end
    
    # Normalizar kernel para evitar crecimiento descontrolado
    max_val = maximum(abs.(neuron.transformation_kernel))
    if max_val > 1.0f0
        neuron.transformation_kernel ./= max_val
    end
    
    return neuron
end

# Funciones auxiliares internas

"""
    update_neuronal_state(current_state, activation, plasticity)

Actualiza el estado de la neurona combinando el estado actual con la nueva activación.
"""
function update_neuronal_state(
    current_state::Array{T,3}, 
    activation::Array{S,3}, 
    plasticity::PlasticityParameters
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Si las dimensiones no coinciden, redimensionar activación
    if size(current_state) != size(activation)
        activation = tensor_interpolation(activation, size(current_state))
    end
    
    # Factor de persistencia basado en plasticidad
    persistence = 1.0f0 - plasticity.base_rate
    
    # Combinar estado actual (con decaimiento) y nueva activación
    new_state = persistence .* current_state .+ plasticity.base_rate .* activation
    
    return new_state
end

"""
    calculate_saturation(state)

Calcula el nivel de saturación del estado neuronal.
"""
function calculate_saturation(state::Array{T,3}) where T <: AbstractFloat
    # Consideramos saturados los valores > 0.9 o < -0.9
    saturation_threshold = 0.9f0
    
    saturated_count = count(x -> abs(x) > saturation_threshold, state)
    total_count = length(state)
    
    return saturated_count / total_count
end

"""
    analyze_activation_patterns(history)

Analiza patrones de activación para determinar el tipo funcional.
"""
function analyze_activation_patterns(history)
    # Extraer información de los patrones de activación para determinar la especialización
    history_array = collect(history)
    
    # Calcular diferencias temporales para detectar sensibilidad temporal
    temporal_diffs = Float32[]
    for i in 2:length(history_array)
        push!(temporal_diffs, sum(abs.(history_array[i] - history_array[i-1])) / length(history_array[i]))
    end
    temporal_sensitivity = mean(temporal_diffs)
    
    # Calcular gradientes espaciales para detectar sensibilidad espacial
    spatial_sensitivity = 0.0f0
    if length(history_array) > 0
        last_state = history_array[end]
        if ndims(last_state) == 3 && all(size(last_state) .> 2)
            spatial_gradients = Float32[]
            for x in 2:(size(last_state, 1)-1)
                for y in 2:(size(last_state, 2)-1)
                    for z in 2:(size(last_state, 3)-1)
                        # Calcular gradientes en cada dirección
                        grad_x = (last_state[x+1, y, z] - last_state[x-1, y, z]) / 2
                        grad_y = (last_state[x, y+1, z] - last_state[x, y-1, z]) / 2
                        grad_z = (last_state[x, y, z+1] - last_state[x, y, z-1]) / 2
                        
                        # Magnitud del gradiente
                        grad_magnitude = sqrt(grad_x^2 + grad_y^2 + grad_z^2)
                        push!(spatial_gradients, grad_magnitude)
                    end
                end
            end
            spatial_sensitivity = mean(spatial_gradients)
        end
    end
    
    # Calcular sensibilidad a características específicas
    feature_sensitivity = 0.0f0
    if length(history_array) > 0
        last_state = history_array[end]
        # Calcular varianza como indicador de selectividad
        feature_sensitivity = std(last_state) / (mean(abs.(last_state)) + 1e-5f0)
    end
    
    # Calcular sensibilidad contextual
    context_sensitivity = 0.0f0
    if length(history_array) > 1
        # Correlación entre estados consecutivos como indicador de sensibilidad contextual
        correlations = Float32[]
        for i in 2:length(history_array)
            # Correlación normalizada entre estados
            norm_corr = sum(history_array[i] .* history_array[i-1]) / 
                       (norm(history_array[i]) * norm(history_array[i-1]) + 1e-5f0)
            push!(correlations, norm_corr)
        end
        context_sensitivity = mean(correlations)
    end
    
    # Normalizar sensibilidades para que sumen aproximadamente 1
    total = temporal_sensitivity + spatial_sensitivity + feature_sensitivity + context_sensitivity
    if total > 0
        temporal_sensitivity /= total
        spatial_sensitivity /= total
        feature_sensitivity /= total
        context_sensitivity /= total
    end
    
    # Devolver como una tupla nombrada
    return (
        temporal_sensitivity = temporal_sensitivity,
        spatial_sensitivity = spatial_sensitivity,
        feature_sensitivity = feature_sensitivity,
        context_sensitivity = context_sensitivity
    )
end
end