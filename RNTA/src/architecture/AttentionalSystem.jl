# architecture/AttentionalSystem.jl
# Implementa un sistema de atención adaptativo para el espacio cerebral

module Attentional_System

using LinearAlgebra
using Statistics
using Random
using SparseArrays

# Importaciones de otros módulos de RNTA
using ..BrainSpace
using ..TensorNeuron
using ..Connections
using ..TensorOperations
using ..SpatialAttention
using ..VolumetricActivations

"""
    AttentionalConfig

Configuración para el sistema atencional.
"""
struct AttentionalConfig
    # Dimensiones del espacio atencional
    dimensions::NTuple{3,Int}
    
    # Número de focos de atención simultáneos
    num_foci::Int
    
    # Radio base para cada foco de atención
    base_radius::Float32
    
    # Factor de intensidad del foco
    focus_factor::Float32
    
    # Velocidad de desplazamiento del foco
    shift_speed::Float32
    
    # Tipo de decaimiento para la atención
    decay_type::Symbol
    
    # Factor de persistencia temporal
    temporal_persistence::Float32
    
    # Umbral de saliencia para capturar atención
    salience_threshold::Float32
    
    # Factor de inhibición para atención distribuida
    inhibition_factor::Float32
end

# Constructor con valores por defecto
function AttentionalConfig(;
    dimensions::NTuple{3,Int}=(10, 10, 10),
    num_foci::Int=3,
    base_radius::Float32=3.0f0,
    focus_factor::Float32=2.0f0,
    shift_speed::Float32=0.5f0,
    decay_type::Symbol=:gaussian,
    temporal_persistence::Float32=0.8f0,
    salience_threshold::Float32=0.7f0,
    inhibition_factor::Float32=0.3f0
)
    return AttentionalConfig(
        dimensions,
        num_foci,
        base_radius,
        focus_factor,
        shift_speed,
        decay_type,
        temporal_persistence,
        salience_threshold,
        inhibition_factor
    )
end

"""
    AttentionalFocus

Representa un foco individual de atención.
"""
mutable struct AttentionalFocus
    # Identificador único
    id::Int
    
    # Posición actual (puede ser fraccional para movimiento suave)
    position::NTuple{3,Float32}
    
    # Radio de atención
    radius::Float32
    
    # Intensidad del foco (fuerza de atención)
    intensity::Float32
    
    # Factor de expansión de radio (para zoom in/out)
    expansion_rate::Float32
    
    # Velocidad de movimiento actual
    velocity::NTuple{3,Float32}
    
    # Tiempo de creación del foco
    creation_time::Float64
    
    # Tiempo de vida en segundos (0 = permanente)
    lifetime::Float32
    
    # Tipo de foco (bottom-up, top-down, etc.)
    focus_type::Symbol
end

"""
Constructor para AttentionalFocus
"""
function AttentionalFocus(
    id::Int,
    position::NTuple{3,Float32};
    radius::Float32=3.0f0,
    intensity::Float32=1.0f0,
    focus_type::Symbol=:bottom_up,
    lifetime::Float32=0.0f0
)
    return AttentionalFocus(
        id,
        position,
        radius,
        intensity,
        0.0f0,                          # expansion_rate
        (0.0f0, 0.0f0, 0.0f0),          # velocity
        time(),                          # creation_time
        lifetime,                        # lifetime
        focus_type                       # focus_type
    )
end

"""
    AttentionalSystem

Sistema adaptativo de atención para dirigir el procesamiento del cerebro.
"""
mutable struct AttentionalSystem
    # Configuración del sistema
    config::AttentionalConfig
    
    # Mapa de atención global
    global_map::SpatialAttentionMap
    
    # Focos de atención activos
    active_foci::Vector{AttentionalFocus}
    
    # Mapa de inhibición de retorno
    inhibition_map::Array{Float32,3}
    
    # Historial de mapas de atención
    map_history::Vector{SpatialAttentionMap}
    
    # Contador de focos creados (para IDs)
    focus_counter::Int
    
    # Cerebro al que está conectado
    brain::Union{Brain_Space, Nothing}
    
    # Tensor de control top-down (para atención voluntaria)
    top_down_control::Array{Float32,3}
    
    # Metadatos del sistema
    metadata::Dict{Symbol, Any}
end

"""
Constructor para AttentionalSystem
"""
function AttentionalSystem(
    config::AttentionalConfig;
    brain::Union{Brain_Space, Nothing}=nothing
)
    # Crear mapa de atención inicial
    global_map = SpatialAttentionMap(
        config.dimensions,
        focus_factor=config.focus_factor,
        decay_type=config.decay_type
    )
    
    # Inicializar mapa de inhibición
    inhibition_map = zeros(Float32, config.dimensions)
    
    # Inicializar control top-down
    top_down_control = zeros(Float32, config.dimensions)
    
    return AttentionalSystem(
        config,
        global_map,
        Vector{AttentionalFocus}(),  # active_foci
        inhibition_map,
        Vector{SpatialAttentionMap}(),  # map_history
        0,                              # focus_counter
        brain,
        top_down_control,
        Dict{Symbol, Any}()             # metadata
    )
end

"""
    update_attention!(system, input_tensor)

Actualiza el estado atencional basado en un tensor de entrada.
"""
function update_attention!(
    system::AttentionalSystem,
    input_tensor::Array{T,3}
) where T <: AbstractFloat
    # Asegurar que las dimensiones coincidan
    if size(input_tensor) != system.config.dimensions
        input_tensor = tensor_interpolation(input_tensor, system.config.dimensions)
    end
    
    # Guardar mapa anterior en historial
    push!(system.map_history, system.global_map)
    if length(system.map_history) > 10
        popfirst!(system.map_history)
    end
    
    # Actualizar focos de atención existentes
    update_foci!(system)
    
    # Detectar regiones salientes en la entrada
    salience_map = compute_salience_map(input_tensor, system.inhibition_map)
    
    # Crear nuevos focos basados en saliencia
    create_bottom_up_foci!(system, salience_map)
    
    # Combinar con control top-down
    if sum(abs.(system.top_down_control)) > 0
        apply_top_down_control!(system)
    end
    
    # Recalcular mapa de atención global
    recalculate_global_map!(system)
    
    return system.global_map
end

"""
    update_foci!(system)

Actualiza los focos de atención existentes.
"""
function update_foci!(system::AttentionalSystem)
    # Tiempo actual
    current_time = time()
    
    # Focos a eliminar
    remove_indices = Int[]
    
    for (i, focus) in enumerate(system.active_foci)
        # Verificar tiempo de vida
        if focus.lifetime > 0
            if current_time - focus.creation_time > focus.lifetime
                # Expiró, marcar para eliminación
                push!(remove_indices, i)
                continue
            end
        end
        
        # Actualizar posición basado en velocidad
        new_position = (
            focus.position[1] + focus.velocity[1] * system.config.shift_speed,
            focus.position[2] + focus.velocity[2] * system.config.shift_speed,
            focus.position[3] + focus.velocity[3] * system.config.shift_speed
        )
        
        # Verificar límites
        dimensions = system.config.dimensions
        new_position = (
            clamp(new_position[1], 1.0f0, Float32(dimensions[1])),
            clamp(new_position[2], 1.0f0, Float32(dimensions[2])),
            clamp(new_position[3], 1.0f0, Float32(dimensions[3]))
        )
        
        # Actualizar posición
        focus.position = new_position
        
        # Actualizar radio basado en tasa de expansión
        focus.radius += focus.expansion_rate
        focus.radius = max(1.0f0, min(focus.radius, Float32(minimum(dimensions) / 2)))
        
        # Actualizar intensidad basada en tiempo de vida
        if focus.lifetime > 0
            remaining_life = 1.0f0 - Float32((current_time - focus.creation_time) / focus.lifetime)
            focus.intensity *= max(0.5f0, remaining_life)
        end
    end
    
    # Eliminar focos expirados
    if !isempty(remove_indices)
        deleteat!(system.active_foci, sort(remove_indices, rev=true))
    end
end

"""
    compute_salience_map(input_tensor, inhibition_map)

Calcula un mapa de saliencia a partir del tensor de entrada y el mapa de inhibición.
"""
function compute_salience_map(
    input_tensor::Array{T,3},
    inhibition_map::Array{Float32,3}
) where T <: AbstractFloat
    dimensions = size(input_tensor)
    
    # Normalizar el tensor de entrada
    normalized_input = normalize_tensor(input_tensor)
    
    # Calcular gradientes locales
    gradient_map = compute_gradient_magnitude(normalized_input)
    
    # Calcular mapa de contraste local
    contrast_map = compute_local_contrast(normalized_input)
    
    # Detectar puntos de alta actividad
    activity_map = normalized_input .^ 2
    
    # Calcular variación temporal (si hay historia disponible)
    # Por ahora usamos un mapa vacío
    temporal_map = zeros(Float32, dimensions)
    
    # Combinación ponderada de las características
    salience_map = 0.4f0 .* gradient_map .+
                  0.3f0 .* contrast_map .+
                  0.3f0 .* activity_map .+
                  0.2f0 .* temporal_map
    
    # Aplicar inhibición de retorno
    salience_map = salience_map .* (1.0f0 .- inhibition_map)
    
    # Normalizar el mapa de saliencia final
    return normalize_tensor(salience_map)
end

"""
    normalize_tensor(tensor)

Normaliza un tensor al rango [0,1].
"""
function normalize_tensor(tensor::Array{T,3}) where T <: AbstractFloat
    min_val = minimum(tensor)
    max_val = maximum(tensor)
    
    if max_val > min_val
        return Float32.((tensor .- min_val) ./ (max_val - min_val))
    else
        return zeros(Float32, size(tensor))
    end
end

"""
    compute_gradient_magnitude(tensor)

Calcula la magnitud del gradiente en cada punto del tensor.
"""
function compute_gradient_magnitude(tensor::Array{T,3}) where T <: AbstractFloat
    dims = size(tensor)
    result = zeros(Float32, dims)
    
    for x in 2:(dims[1]-1), y in 2:(dims[2]-1), z in 2:(dims[3]-1)
        # Gradientes en las tres direcciones
        dx = (tensor[x+1, y, z] - tensor[x-1, y, z]) / 2
        dy = (tensor[x, y+1, z] - tensor[x, y-1, z]) / 2
        dz = (tensor[x, y, z+1] - tensor[x, y, z-1]) / 2
        
        # Magnitud del gradiente
        result[x, y, z] = sqrt(dx^2 + dy^2 + dz^2)
    end
    
    return result
end

"""
    compute_local_contrast(tensor)

Calcula el contraste local para cada punto del tensor.
"""
function compute_local_contrast(tensor::Array{T,3}) where T <: AbstractFloat
    dims = size(tensor)
    result = zeros(Float32, dims)
    
    # Tamaño de la ventana para calcular estadísticas locales
    window_size = 3
    radius = div(window_size, 2)
    
    for x in (radius+1):(dims[1]-radius),
        y in (radius+1):(dims[2]-radius),
        z in (radius+1):(dims[3]-radius)
        
        # Extraer ventana local
        window = tensor[
            (x-radius):(x+radius),
            (y-radius):(y+radius),
            (z-radius):(z+radius)
        ]
        
        # Calcular estadísticas locales
        local_mean = mean(window)
        local_std = std(window)
        
        # Contraste como desviación del punto central respecto a la media local
        # normalizado por la desviación estándar local
        central_value = tensor[x, y, z]
        
        if local_std > 0
            result[x, y, z] = abs(central_value - local_mean) / local_std
        else
            result[x, y, z] = 0.0f0
        end
    end
    
    return result
end

"""
    create_bottom_up_foci!(system, salience_map)

Crea nuevos focos de atención bottom-up basados en el mapa de saliencia.
"""
function create_bottom_up_foci!(
    system::AttentionalSystem,
    salience_map::Array{Float32,3}
)
    # Determinar cuántos focos adicionales podemos crear
    max_new_foci = system.config.num_foci - length(system.active_foci)
    
    if max_new_foci <= 0
        return  # Ya tenemos el máximo de focos
    end
    
    # Encontrar los puntos más salientes
    threshold = system.config.salience_threshold
    candidates = []
    
    # Buscar puntos que superen el umbral
    dims = size(salience_map)
    for x in 2:(dims[1]-1), y in 2:(dims[2]-1), z in 2:(dims[3]-1)
        if salience_map[x, y, z] > threshold
            # Verificar si es un máximo local
            is_local_max = true
            for dx in -1:1, dy in -1:1, dz in -1:1
                nx, ny, nz = x+dx, y+dy, z+dz
                if 1 <= nx <= dims[1] && 1 <= ny <= dims[2] && 1 <= nz <= dims[3]
                    if salience_map[nx, ny, nz] > salience_map[x, y, z]
                        is_local_max = false
                        break
                    end
                end
            end
            
            if is_local_max
                push!(candidates, (x, y, z, salience_map[x, y, z]))
            end
        end
    end
    
    # Ordenar candidatos por saliencia
    sort!(candidates, by=c -> c[4], rev=true)
    
    # Crear nuevos focos (limitado por max_new_foci)
    num_to_create = min(max_new_foci, length(candidates))
    
    for i in 1:num_to_create
        x, y, z, salience = candidates[i]
        
        # Incrementar contador de focos
        system.focus_counter += 1
        
        # Calcular intensidad basada en saliencia
        intensity = 0.5f0 + 0.5f0 * salience
        
        # Calcular radio basado en la configuración
        radius = system.config.base_radius * (0.8f0 + 0.4f0 * salience)
        
        # Crear nuevo foco
        new_focus = AttentionalFocus(
            system.focus_counter,
            (Float32(x), Float32(y), Float32(z)),
            radius=radius,
            intensity=intensity,
            focus_type=:bottom_up,
            lifetime=5.0f0  # Tiempo de vida estándar para focos bottom-up
        )
        
        push!(system.active_foci, new_focus)
        
        # Actualizar mapa de inhibición para evitar focos muy cercanos
        update_inhibition_map!(system, x, y, z, radius)
    end
end

"""
    update_inhibition_map!(system, x, y, z, radius)

Actualiza el mapa de inhibición después de crear un nuevo foco.
"""
function update_inhibition_map!(
    system::AttentionalSystem,
    x::Int,
    y::Int,
    z::Int,
    radius::Float32
)
    dims = system.config.dimensions
    inhibition_factor = system.config.inhibition_factor
    
    # Radio de inhibición (ligeramente mayor que el radio del foco)
    inhibition_radius = ceil(Int, radius * 1.5)
    
    # Actualizar región de inhibición
    for ix in max(1, x-inhibition_radius):min(dims[1], x+inhibition_radius),
        iy in max(1, y-inhibition_radius):min(dims[2], y+inhibition_radius),
        iz in max(1, z-inhibition_radius):min(dims[3], z+inhibition_radius)
        
        # Calcular distancia al centro
        dist = sqrt(Float32((ix-x)^2 + (iy-y)^2 + (iz-z)^2))
        
        if dist <= inhibition_radius
            # Aplicar inhibición con decaimiento gaussiano
            inhibition_value = inhibition_factor * exp(-0.5f0 * (dist/radius)^2)
            
            # Asegurar que no exceda 1.0
            system.inhibition_map[ix, iy, iz] = min(
                1.0f0,
                system.inhibition_map[ix, iy, iz] + inhibition_value
            )
        end
    end
end

"""
    apply_top_down_control!(system)

Aplica control top-down al sistema de atención.
"""
function apply_top_down_control!(system::AttentionalSystem)
    # Normalizar tensor de control top-down
    normalized_control = normalize_tensor(system.top_down_control)
    
    # Buscar regiones de alta activación en el control top-down
    dims = system.config.dimensions
    candidates = []
    
    # Umbral para control top-down
    threshold = 0.7f0
    
    for x in 2:(dims[1]-1), y in 2:(dims[2]-1), z in 2:(dims[3]-1)
        if normalized_control[x, y, z] > threshold
            # Verificar si es un máximo local
            is_local_max = true
            for dx in -1:1, dy in -1:1, dz in -1:1
                nx, ny, nz = x+dx, y+dy, z+dz
                if 1 <= nx <= dims[1] && 1 <= ny <= dims[2] && 1 <= nz <= dims[3]
                    if normalized_control[nx, ny, nz] > normalized_control[x, y, z]
                        is_local_max = false
                        break
                    end
                end
            end
            
            if is_local_max
                push!(candidates, (x, y, z, normalized_control[x, y, z]))
            end
        end
    end
    
    # Ordenar candidatos por valor de control
    sort!(candidates, by=c -> c[4], rev=true)
    
    # Determinar cuántos focos top-down podemos crear
    max_top_down = max(1, div(system.config.num_foci, 2))
    num_to_create = min(max_top_down, length(candidates))
    
    for i in 1:num_to_create
        x, y, z, control_value = candidates[i]
        
        # Incrementar contador de focos
        system.focus_counter += 1
        
        # Calcular intensidad y radio
        intensity = 0.7f0 + 0.3f0 * control_value
        radius = system.config.base_radius * 1.2f0
        
        # Crear foco top-down
        new_focus = AttentionalFocus(
            system.focus_counter,
            (Float32(x), Float32(y), Float32(z)),
            radius=radius,
            intensity=intensity,
            focus_type=:top_down,
            lifetime=10.0f0  # Tiempo de vida más largo para focos top-down
        )
        
        # Darle más influencia a los focos top-down
        # Aumentar prioridad eliminando un foco bottom-up si es necesario
        if length(system.active_foci) >= system.config.num_foci
            # Encontrar un foco bottom-up para reemplazar
            for (j, focus) in enumerate(system.active_foci)
                if focus.focus_type == :bottom_up
                    deleteat!(system.active_foci, j)
                    break
                end
            end
        end
        
        push!(system.active_foci, new_focus)
    end
    
    # Reiniciar el control top-down después de usarlo
    fill!(system.top_down_control, 0.0f0)
end

"""
    recalculate_global_map!(system)

Recalcula el mapa de atención global basado en los focos activos.
"""
function recalculate_global_map!(system::AttentionalSystem)
    # Crear nuevo mapa de atención
    dims = system.config.dimensions
    attention_values = zeros(Float32, dims)
    
    # Aplicar cada foco al mapa
    for focus in system.active_foci
        apply_focus_to_map!(attention_values, focus, dims)
    end
    
    # Aplicar persistencia temporal
    persistence = system.config.temporal_persistence
    if !isempty(system.map_history)
        previous_map = system.map_history[end]
        previous_values = previous_map.values
        
        # Combinación ponderada con mapa anterior
        attention_values = (1.0f0 - persistence) .* attention_values .+ 
                           persistence .* previous_values
    end
    
    # Normalizar mapa final
    normalized_map = normalize_tensor(attention_values)
    
    # Actualizar mapa global
    system.global_map = SpatialAttentionMap(
        dims,
        values=normalized_map,
        focus_factor=system.config.focus_factor,
        decay_type=system.config.decay_type
    )
end

"""
    apply_focus_to_map!(attention_map, focus, dimensions)

Aplica un foco de atención al mapa de atención.
"""
function apply_focus_to_map!(
    attention_map::Array{Float32,3},
    focus::AttentionalFocus,
    dimensions::NTuple{3,Int}
)
    # Extraer parámetros del foco
    cx, cy, cz = focus.position
    radius = focus.radius
    intensity = focus.intensity
    
    # Calcular límites de la región afectada
    x_min = max(1, floor(Int, cx - 2*radius))
    x_max = min(dimensions[1], ceil(Int, cx + 2*radius))
    y_min = max(1, floor(Int, cy - 2*radius))
    y_max = min(dimensions[2], ceil(Int, cy + 2*radius))
    z_min = max(1, floor(Int, cz - 2*radius))
    z_max = min(dimensions[3], ceil(Int, cz + 2*radius))
    
    # Aplicar distribución gaussiana
    for x in x_min:x_max, y in y_min:y_max, z in z_min:z_max
        # Calcular distancia al centro del foco
        dx = Float32(x) - cx
        dy = Float32(y) - cy
        dz = Float32(z) - cz
        dist_sq = dx^2 + dy^2 + dz^2
        
        # Calcular atenuación gaussiana
        attenuation = exp(-0.5f0 * dist_sq / (radius^2))
        
        # Aplicar efecto del foco
        attention_value = intensity * attenuation
        
        # Actualizar mapa usando máximo para evitar interferencia entre focos
        attention_map[x, y, z] = max(attention_map[x, y, z], attention_value)
    end
end

"""
    set_top_down_attention!(system, control_tensor)

Establece el control top-down para dirigir la atención voluntariamente.
"""
function set_top_down_attention!(
    system::AttentionalSystem,
    control_tensor::Array{T,3}
) where T <: AbstractFloat
    # Verificar dimensiones
    if size(control_tensor) != system.config.dimensions
        control_tensor = tensor_interpolation(control_tensor, system.config.dimensions)
    end
    
    # Actualizar tensor de control
    system.top_down_control = Float32.(control_tensor)
end

"""
    create_focus!(system, position; kwargs...)

Crea explícitamente un nuevo foco de atención en una posición específica.
"""
function create_focus!(
    system::AttentionalSystem,
    position::NTuple{3,Float32};
    radius::Float32=system.config.base_radius,
    intensity::Float32=1.0f0,
    focus_type::Symbol=:explicit,
    lifetime::Float32=0.0f0,
    velocity::NTuple{3,Float32}=(0.0f0, 0.0f0, 0.0f0)
)
    # Incrementar contador de focos
    system.focus_counter += 1
    
    # Crear nuevo foco
    new_focus = AttentionalFocus(
        system.focus_counter,
        position,
        radius=radius,
        intensity=intensity,
        focus_type=focus_type,
        lifetime=lifetime
    )
    
    # Establecer velocidad
    new_focus.velocity = velocity
    
    # Verificar si excedemos el número máximo de focos
    if length(system.active_foci) >= system.config.num_foci
        # Eliminar el foco más antiguo
        popfirst!(system.active_foci)
    end
    
    push!(system.active_foci, new_focus)
    
    # Actualizar mapa de inhibición
    update_inhibition_map!(
        system,
        round(Int, position[1]),
        round(Int, position[2]),
        round(Int, position[3]),
        radius
    )
    
    return new_focus
end

"""
    decay_inhibition!(system, decay_rate=0.1)

Decae el mapa de inhibición con el tiempo.
"""
function decay_inhibition!(
    system::AttentionalSystem,
    decay_rate::Float32=0.1f0
)
    # Aplicar decaimiento exponencial al mapa de inhibición
    system.inhibition_map .*= (1.0f0 - decay_rate)
end

"""
    get_attention_at(system, position)

Obtiene el valor de atención en una posición específica.
"""
function get_attention_at(
    system::AttentionalSystem,
    position::NTuple{3,Float32}
)
    # Convertir posición a índices enteros
    x = round(Int, position[1])
    y = round(Int, position[2])
    z = round(Int, position[3])
    
    # Verificar límites
    dims = system.config.dimensions
    if 1 <= x <= dims[1] && 1 <= y <= dims[2] && 1 <= z <= dims[3]
        return system.global_map.values[x, y, z]
    else
        return 0.0f0
    end
end

"""
    get_most_attended_regions(system, n=3)

Obtiene las n regiones más atendidas.
"""
function get_most_attended_regions(
    system::AttentionalSystem,
    n::Int=3
)
    # Obtener mapa de valores
    values = system.global_map.values
    dims = size(values)
    
    # Encontrar máximos locales
    maxima = []
    
    for x in 2:(dims[1]-1), y in 2:(dims[2]-1), z in 2:(dims[3]-1)
        value = values[x, y, z]
        
        if value > 0.3f0  # Umbral mínimo
            # Verificar si es un máximo local
            is_local_max = true
            for dx in -1:1, dy in -1:1, dz in -1:1
                nx, ny, nz = x+dx, y+dy, z+dz
                if 1 <= nx <= dims[1] && 1 <= ny <= dims[2] && 1 <= nz <= dims[3]
                    if values[nx, ny, nz] > value
                        is_local_max = false
                        break
                    end
                end
            end
            
            if is_local_max
                push!(maxima, (x, y, z, value))
            end
        end
    end
    
    # Ordenar por valor de atención
    sort!(maxima, by=m -> m[4], rev=true)
    
    # Tomar los n primeros (o menos si no hay suficientes)
    top_n = maxima[1:min(n, length(maxima))]
    
    # Convertir a tuplas (posición, valor)
    return [(Float32.(t[1:3]), t[4]) for t in top_n]
end

"""
    apply_attention!(system, target_tensor)

Aplica el mapa de atención actual a un tensor objetivo.
"""
function apply_attention!(
    system::AttentionalSystem,
    target_tensor::Array{T,3}
) where T <: AbstractFloat
    # Asegurar que las dimensiones coincidan
    if size(target_tensor) != system.config.dimensions
        target_tensor = tensor_interpolation(target_tensor, system.config.dimensions)
    end
    
    # Aplicar mapa de atención
    result = apply_attention(target_tensor, system.global_map)
    
    return result
end

"""
    focus_on_region!(system, center, radius)

Dirige la atención explícitamente a una región específica.
"""
function focus_on_region!(
    system::AttentionalSystem,
    center::NTuple{3,Float32},
    radius::Float32=system.config.base_radius;
    intensity::Float32=1.0f0,
    lifetime::Float32=5.0f0
)
    # Eliminar todos los focos de tipo :explicit anteriores
    filter!(f -> f.focus_type != :explicit, system.active_foci)
    
    # Crear nuevo foco explícito
    create_focus!(
        system,
        center,
        radius=radius,
        intensity=intensity,
        focus_type=:explicit,
        lifetime=lifetime
    )
    
    # Recalcular mapa de atención global
    recalculate_global_map!(system)
    
    return system.global_map
end

"""
    merge_attention_maps(system, other_system; weight=0.5)

Combina el mapa de atención de este sistema con otro.
"""
function merge_attention_maps(
    system::AttentionalSystem,
    other_system::AttentionalSystem;
    weight::Float32=0.5f0
)
    # Verificar que las dimensiones sean compatibles
    if system.config.dimensions != other_system.config.dimensions
        error("Los sistemas de atención tienen dimensiones incompatibles")
    end
    
    # Combinar mapas
    combined_values = (1.0f0 - weight) .* system.global_map.values .+
                      weight .* other_system.global_map.values
    
    # Normalizar
    combined_values = normalize_tensor(combined_values)
    
    # Crear nuevo mapa
    return SpatialAttentionMap(
        system.config.dimensions,
        values=combined_values,
        focus_factor=system.config.focus_factor,
        decay_type=system.config.decay_type
    )
end

"""
    track_moving_stimulus!(system, position, velocity, lifetime=10.0)

Crea un foco que sigue un estímulo en movimiento.
"""
function track_moving_stimulus!(
    system::AttentionalSystem,
    position::NTuple{3,Float32},
    velocity::NTuple{3,Float32},
    lifetime::Float32=10.0f0
)
    # Crear foco con velocidad
    create_focus!(
        system,
        position,
        radius=system.config.base_radius * 1.2f0,
        intensity=1.0f0,
        focus_type=:tracking,
        lifetime=lifetime,
        velocity=velocity
    )
    
    # Recalcular mapa global
    recalculate_global_map!(system)
    
    return system.global_map
end

"""
    create_attentional_mask(system, threshold=0.5)

Crea una máscara binaria basada en el mapa de atención.
"""
function create_attentional_mask(
    system::AttentionalSystem,
    threshold::Float32=0.5f0
)
    # Crear máscara binaria
    mask = system.global_map.values .>= threshold
    
    return BitArray(mask)
end

"""
    get_attention_statistics(system)

Obtiene estadísticas generales sobre el estado actual de atención.
"""
function get_attention_statistics(system::AttentionalSystem)
    values = system.global_map.values
    
    stats = Dict{Symbol, Any}()
    
    # Estadísticas básicas
    stats[:mean] = mean(values)
    stats[:max] = maximum(values)
    stats[:min] = minimum(values)
    stats[:std] = std(values)
    
    # Calcular cobertura (porcentaje del espacio con atención significativa)
    stats[:coverage] = sum(values .> 0.3f0) / length(values)
    
    # Número de focos activos
    stats[:num_foci] = length(system.active_foci)
    
    # Calcular "centro de masa" de la atención
    total_mass = sum(values)
    if total_mass > 0
        dims = size(values)
        x_coords = [i for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
        y_coords = [j for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
        z_coords = [k for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
        
        center_x = sum(x_coords .* values) / total_mass
        center_y = sum(y_coords .* values) / total_mass
        center_z = sum(z_coords .* values) / total_mass
        
        stats[:center_of_attention] = (center_x, center_y, center_z)
    else
        stats[:center_of_attention] = nothing
    end
    
    return stats
end

"""
    reset!(system)

Reinicia el sistema de atención a su estado inicial.
"""
function reset!(system::AttentionalSystem)
    # Limpiar focos activos
    empty!(system.active_foci)
    
    # Reiniciar contador de focos
    system.focus_counter = 0
    
    # Reiniciar mapa de inhibición
    fill!(system.inhibition_map, 0.0f0)
    
    # Reiniciar control top-down
    fill!(system.top_down_control, 0.0f0)
    
    # Reiniciar mapa global
    system.global_map = SpatialAttentionMap(
        system.config.dimensions,
        focus_factor=system.config.focus_factor,
        decay_type=system.config.decay_type
    )
    
    # Limpiar historial
    empty!(system.map_history)
    
    return system
end

# Exportación de funciones principales
export AttentionalConfig, AttentionalFocus, AttentionalSystem
export update_attention!, set_top_down_attention!, focus_on_region!
export get_most_attended_regions, apply_attention!, create_attentional_mask
export get_attention_statistics, reset!

end # module