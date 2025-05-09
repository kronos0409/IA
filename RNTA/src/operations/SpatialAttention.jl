# operations/SpatialAttention.jl
# Implementa mecanismos de atención volumétrica para el espacio cerebral

module SpatialAttention

using LinearAlgebra
using Statistics

# Importaciones de otros módulos de RNTA
using ..TensorOperations

"""
    SpatialAttentionMap

Representa un mapa de atención que modula la actividad en el espacio 3D.
"""
struct SpatialAttentionMap
    # Tensor de activación de atención
    attention_field::Array{Float32,3}
    
    # Factor de enfoque (mayor valor = atención más concentrada)
    focus_factor::Float32
    
    # Radio de atención efectivo
    effective_radius::Float32
    
    # Centro actual de atención
    focus_center::NTuple{3,Float32}
    
    # Tipo de decaimiento de atención
    decay_type::Symbol
end

"""
Constructor principal para SpatialAttentionMap
"""
function SpatialAttentionMap(
    dimensions::NTuple{3,Int};
    focus_factor::Float32=2.0f0,
    effective_radius::Float32=5.0f0,
    decay_type::Symbol=:gaussian
)
    # Inicializar campo de atención uniforme
    attention_field = ones(Float32, dimensions)
    
    # Establecer enfoque en el centro por defecto
    focus_center = (
        dimensions[1] / 2,
        dimensions[2] / 2, 
        dimensions[3] / 2
    )
    
    return SpatialAttentionMap(
        attention_field,
        focus_factor,
        effective_radius,
        focus_center,
        decay_type
    )
end

"""
    shift_attention!(attention_map, new_center)

Desplaza el centro de atención a una nueva posición.
"""
function shift_attention!(
    attention_map::SpatialAttentionMap, 
    new_center::NTuple{3,Float32}
)
    # Actualizar centro de atención
    attention_map.focus_center = new_center
    
    # Recalcular campo de atención basado en nuevo centro
    recalculate_attention_field!(attention_map)
    
    return attention_map
end

"""
    adjust_focus!(attention_map, new_focus_factor)

Ajusta el factor de enfoque (concentración) de la atención.
"""
function adjust_focus!(
    attention_map::SpatialAttentionMap, 
    new_focus_factor::Float32
)
    # Actualizar factor de enfoque
    attention_map.focus_factor = max(1.0f0, new_focus_factor)
    
    # Recalcular campo de atención
    recalculate_attention_field!(attention_map)
    
    return attention_map
end

"""
    adjust_radius!(attention_map, new_radius)

Ajusta el radio efectivo de atención.
"""
function adjust_radius!(
    attention_map::SpatialAttentionMap, 
    new_radius::Float32
)
    # Actualizar radio
    attention_map.effective_radius = max(1.0f0, new_radius)
    
    # Recalcular campo de atención
    recalculate_attention_field!(attention_map)
    
    return attention_map
end

"""
    apply_attention(tensor, attention_map)

Aplica el mapa de atención a un tensor de entrada.
"""
function apply_attention(
    tensor::Array{T,3}, 
    attention_map::SpatialAttentionMap
) where T <: AbstractFloat
    # Asegurar que las dimensiones coincidan
    if size(tensor) != size(attention_map.attention_field)
        # Redimensionar mapa de atención si es necesario
        resized_attention = tensor_interpolation(
            attention_map.attention_field,
            size(tensor)
        )
        
        # Aplicar atención
        return tensor .* resized_attention
    else
        # Aplicar atención directamente
        return tensor .* attention_map.attention_field
    end
end

"""
    recalculate_attention_field!(attention_map)

Recalcula el campo de atención basado en los parámetros actuales.
"""
function recalculate_attention_field!(attention_map::SpatialAttentionMap)
    dimensions = size(attention_map.attention_field)
    
    # Reiniciar campo de atención
    attention_field = zeros(Float32, dimensions)
    
    # Calcular distribución de atención basada en la distancia al centro
    for x in 1:dimensions[1]
        for y in 1:dimensions[2]
            for z in 1:dimensions[3]
                # Distancia al centro de atención
                dx = x - attention_map.focus_center[1]
                dy = y - attention_map.focus_center[2]
                dz = z - attention_map.focus_center[3]
                
                distance = sqrt(dx^2 + dy^2 + dz^2)
                
                # Calcular valor de atención según tipo de decaimiento
                if attention_map.decay_type == :gaussian
                    # Decaimiento gaussiano
                    sigma = attention_map.effective_radius / attention_map.focus_factor
                    attention_field[x, y, z] = exp(-(distance^2) / (2 * sigma^2))
                    
                elseif attention_map.decay_type == :exponential
                    # Decaimiento exponencial
                    scale = attention_map.focus_factor / attention_map.effective_radius
                    attention_field[x, y, z] = exp(-distance * scale)
                    
                elseif attention_map.decay_type == :linear
                    # Decaimiento lineal
                    max_distance = attention_map.effective_radius * attention_map.focus_factor
                    attention_field[x, y, z] = max(0.0f0, 1.0f0 - distance / max_distance)
                    
                else
                    # Por defecto, usar decaimiento gaussiano
                    sigma = attention_map.effective_radius / attention_map.focus_factor
                    attention_field[x, y, z] = exp(-(distance^2) / (2 * sigma^2))
                end
            end
        end
    end
    
    # Normalizar campo de atención
    max_attention = maximum(attention_field)
    if max_attention > 0
        attention_field ./= max_attention
    else
        attention_field .= 1.0f0
    end
    
    # Asegurar un nivel mínimo de atención en todo el campo
    min_attention = 0.1f0
    attention_field .= max.(attention_field, min_attention)
    
    # Actualizar campo de atención
    attention_map.attention_field = attention_field
    
    return attention_map
end

"""
    create_attention_from_activity(activity_tensor; options...)

Crea un mapa de atención basado en un tensor de actividad.
"""
function create_attention_from_activity(
    activity_tensor::Array{T,3};
    threshold::Float32=0.5f0,
    radius::Float32=3.0f0,
    focus_factor::Float32=2.0f0,
    decay_type::Symbol=:gaussian
) where T <: AbstractFloat
    # Crear mapa de atención con dimensiones del tensor de actividad
    attention_map = SpatialAttentionMap(
        size(activity_tensor),
        focus_factor=focus_factor,
        effective_radius=radius,
        decay_type=decay_type
    )
    
    # Encontrar punto de máxima actividad
    max_val, max_idx = findmax(abs.(activity_tensor))
    
    # Solo enfocar la atención si la actividad supera el umbral
    if max_val > threshold
        # Convertir índice multidimensional a coordenadas
        max_x, max_y, max_z = Tuple(max_idx)
        
        # Enfocar la atención en el punto de máxima actividad
        shift_attention!(attention_map, (Float32(max_x), Float32(max_y), Float32(max_z)))
    end
    
    return attention_map
end

"""
    multi_focus_attention(activity_tensor, num_foci; options...)

Crea un mapa de atención con múltiples focos basado en puntos de actividad.
"""
function multi_focus_attention(
    activity_tensor::Array{T,3},
    num_foci::Int=3;
    threshold::Float32=0.3f0,
    radius::Float32=3.0f0,
    focus_factor::Float32=1.5f0
) where T <: AbstractFloat
    dimensions = size(activity_tensor)
    
    # Inicializar campo de atención
    attention_field = zeros(Float32, dimensions)
    
    # Encontrar puntos de actividad significativa
    significant_positions = Tuple{Int,Int,Int}[]
    significant_values = Float32[]
    
    # Umbral adaptativo
    adaptive_threshold = max(
        threshold,
        mean(abs.(activity_tensor)) + 1.5f0 * std(abs.(activity_tensor))
    )
    
    # Encontrar todos los puntos que superan el umbral
    for idx in CartesianIndices(activity_tensor)
        if abs(activity_tensor[idx]) > adaptive_threshold
            push!(significant_positions, Tuple(idx))
            push!(significant_values, abs(activity_tensor[idx]))
        end
    end
    
    # Si hay suficientes puntos significativos
    if length(significant_positions) > 0
        # Ordenar por valor de actividad
        sorted_indices = sortperm(significant_values, rev=true)
        
        # Limitar al número de focos especificado
        num_actual_foci = min(num_foci, length(significant_positions))
        
        # Crear mapa de atención combinando múltiples focos
        for i in 1:num_actual_foci
            pos = significant_positions[sorted_indices[i]]
            val = significant_values[sorted_indices[i]]
            
            # Crear foco individual
            focus_map = SpatialAttentionMap(
                dimensions,
                focus_factor=focus_factor,
                effective_radius=radius
            )
            
            # Enfocar en posición
            shift_attention!(focus_map, (Float32(pos[1]), Float32(pos[2]), Float32(pos[3])))
            
            # Ponderar por valor de actividad normalizado
            weight = val / significant_values[sorted_indices[1]]
            
            # Añadir a mapa combinado
            attention_field .+= focus_map.attention_field .* weight
        end
        
        # Normalizar campo de atención combinado
        max_attention = maximum(attention_field)
        if max_attention > 0
            attention_field ./= max_attention
        end
    else
        # Si no hay puntos significativos, usar atención uniforme
        attention_field .= 1.0f0
    end
    
    # Crear mapa de atención final
    attention_map = SpatialAttentionMap(
        attention_field,
        focus_factor,
        radius,
        (Float32(dimensions[1]/2), Float32(dimensions[2]/2), Float32(dimensions[3]/2)),
        :gaussian
    )
    
    return attention_map
end

"""
    attention_guided_convolution(input, kernel, attention_map; options...)

Realiza una convolución tensorial guiada por atención.
"""
function attention_guided_convolution(
    input::Array{T,3}, 
    kernel::Array{S,3},
    attention_map::SpatialAttentionMap;
    stride::NTuple{3,Int}=(1,1,1), 
    padding::Int=0
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Aplicar atención a la entrada
    attended_input = apply_attention(input, attention_map)
    
    # Realizar convolución normal sobre entrada atendida
    return tensor_convolution(attended_input, kernel, stride=stride, padding=padding)
end

# Exportar tipos y funciones principales
export SpatialAttentionMap,
       shift_attention!, adjust_focus!, adjust_radius!,
       apply_attention, recalculate_attention_field!,
       create_attention_from_activity, multi_focus_attention,
       attention_guided_convolution

end # module SpatialAttention