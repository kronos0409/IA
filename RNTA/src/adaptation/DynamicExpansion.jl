# adaptation/DynamicExpansion.jl
# Implementa mecanismos para expansión dinámica del espacio cerebral
module DynamicExpansion

# Importaciones necesarias
using ..BrainSpace
# Otras importaciones...

# Funciones actuales...

export identify_expansion_regions, update_region_map!, find_high_value_regions,
       expand_region!, should_expand_space, calculate_saturation

"""
    identify_expansion_regions(brain)

Identifica regiones del espacio cerebral que deberían expandirse.
"""
function identify_expansion_regions(brain::Brain_Space)
    # Mapa de actividad para identificar regiones de alta densidad
    activity_map = zeros(Float32, brain.dimensions)
    saturation_map = zeros(Float32, brain.dimensions)
    
    # Calcular actividad y saturación para cada neurona
    for (pos, neuron) in brain.neurons
        # Calcular actividad media reciente
        recent_activity = mean([sum(abs.(state)) / length(state) for state in neuron.activation_history])
        
        # Calcular saturación
        saturation = calculate_saturation(neuron.state)
        
        # Actualizar mapas en la posición de la neurona
        activity_map[pos...] = recent_activity
        saturation_map[pos...] = saturation
        
        # Propagar a vecinos con atenuación
        update_region_map!(activity_map, pos, recent_activity, radius=2)
        update_region_map!(saturation_map, pos, saturation, radius=2)
    end
    
    # Combinar mapas para encontrar regiones candidatas
    combined_map = activity_map .* saturation_map
    
    # Encontrar clusters de alta actividad combinada
    threshold = maximum(combined_map) * 0.7f0
    regions = find_high_value_regions(combined_map, threshold)
    
    return regions
end

"""
    update_region_map!(map, position, value; radius=2)

Actualiza una región del mapa alrededor de una posición.
"""
function update_region_map!(
    map::Array{Float32,3},
    position::NTuple{3,Int},
    value::Float32;
    radius::Int=2
)
    dim_x, dim_y, dim_z = size(map)
    
    # Iterar sobre región
    for dx in -radius:radius
        for dy in -radius:radius
            for dz in -radius:radius
                # Calcular posición
                x = position[1] + dx
                y = position[2] + dy
                z = position[3] + dz
                
                # Verificar límites
                if 1 <= x <= dim_x && 1 <= y <= dim_y && 1 <= z <= dim_z
                    # Distancia al centro
                    distance = sqrt(dx^2 + dy^2 + dz^2)
                    
                    # Factor de atenuación por distancia
                    attenuation = exp(-distance)
                    
                    # Actualizar mapa de actividad
                    map[x, y, z] = max(map[x, y, z], value * attenuation)
                end
            end
        end
    end
end

"""
    find_high_value_regions(map, threshold)

Encuentra regiones con valores por encima del umbral.
"""
function find_high_value_regions(
    map::Array{Float32,3},
    threshold::Float32
)
    # Crear mapa binario
    binary_map = map .> threshold
    
    # Encontrar componentes conectados
    regions = NTuple{3,UnitRange{Int}}[]
    
    # Recorrer el mapa en busca de regiones activas
    dim_x, dim_y, dim_z = size(map)
    visited = falses(size(binary_map))
    
    for x in 1:dim_x
        for y in 1:dim_y
            for z in 1:dim_z
                if binary_map[x, y, z] && !visited[x, y, z]
                    # Encontrar región conectada
                    region_bounds = find_connected_region(binary_map, visited, (x, y, z))
                    push!(regions, region_bounds)
                end
            end
        end
    end
    
    # Si no se encontraron regiones, expandir la región con mayor valor
    if isempty(regions)
        max_val, max_idx = findmax(map)
        x, y, z = Tuple(max_idx)
        
        # Crear región pequeña alrededor del máximo
        radius = 2
        x_range = max(1, x-radius):min(dim_x, x+radius)
        y_range = max(1, y-radius):min(dim_y, y+radius)
        z_range = max(1, z-radius):min(dim_z, z+radius)
        
        push!(regions, (x_range, y_range, z_range))
    end
    
    return regions
end

"""
    expand_region!(brain, region)

Expande una región específica del espacio cerebral.
"""
function expand_region!(brain::Brain_Space, region::NTuple{3,UnitRange{Int}})
    # Calcular dimensiones actuales de la región
    region_dim_x = length(region[1])
    region_dim_y = length(region[2])
    region_dim_z = length(region[3])
    
    # Calcular nuevas dimensiones expandidas
    new_dim_x = ceil(Int, region_dim_x * brain.config.expansion_factor)
    new_dim_y = ceil(Int, region_dim_y * brain.config.expansion_factor)
    new_dim_z = ceil(Int, region_dim_z * brain.config.expansion_factor)
    
    # Asegurar que las nuevas dimensiones no excedan las dimensiones del cerebro
    new_dim_x = min(new_dim_x, brain.dimensions[1])
    new_dim_y = min(new_dim_y, brain.dimensions[2])
    new_dim_z = min(new_dim_z, brain.dimensions[3])
    
    # Calcular el centro de la región
    center_x = div(region[1].start + region[1].stop, 2)
    center_y = div(region[2].start + region[2].stop, 2)
    center_z = div(region[3].start + region[3].stop, 2)
    
    # Calcular nueva región expandida (centrada en el mismo punto)
    half_x = div(new_dim_x, 2)
    half_y = div(new_dim_y, 2)
    half_z = div(new_dim_z, 2)
    
    expanded_region = (
        max(1, center_x - half_x):min(brain.dimensions[1], center_x + half_x),
        max(1, center_y - half_y):min(brain.dimensions[2], center_y + half_y),
        max(1, center_z - half_z):min(brain.dimensions[3], center_z + half_z)
    )
    
    # Contar neuronas actuales en la región
    current_neurons = 0
    for (pos, _) in brain.neurons
        if pos[1] in expanded_region[1] && pos[2] in expanded_region[2] && pos[3] in expanded_region[3]
            current_neurons += 1
        end
    end
    
    # Calcular cuántas neuronas nuevas añadir
    expanded_volume = length(expanded_region[1]) * length(expanded_region[2]) * length(expanded_region[3])
    target_neurons = round(Int, expanded_volume * brain.config.initial_density)
    new_neurons_to_add = max(0, target_neurons - current_neurons)
    
    # Añadir nuevas neuronas
    new_positions = Set{NTuple{3,Int}}()
    
    # Intentar añadir el número objetivo de neuronas
    attempts = 0
    max_attempts = new_neurons_to_add * 10  # Límite para evitar bucles infinitos
    
    while length(new_positions) < new_neurons_to_add && attempts < max_attempts
        # Generar posición aleatoria dentro de la región expandida
        x = rand(expanded_region[1])
        y = rand(expanded_region[2])
        z = rand(expanded_region[3])
        
        pos = (x, y, z)
        
        # Verificar que la posición no esté ya ocupada
        if !haskey(brain.neurons, pos)
            push!(new_positions, pos)
        end
        
        attempts += 1
    end
    
    # Crear nuevas neuronas en las posiciones seleccionadas
    for pos in new_positions
        # Tamaño de campo receptivo
        receptive_field_size = (3, 3, 3)
        
        # Crear neurona
        neuron = TensorNeuron(
            pos, 
            receptive_field_size,
            init_scale=brain.config.init_scale
        )
        
        # Añadir a diccionario
        brain.neurons[pos] = neuron
    end
    
    # Establecer conexiones para las nuevas neuronas
    establish_new_connections!(brain)
    
    # Registrar evento de expansión
    push!(brain.expansion_history, ExpansionEvent(
        time(), 
        expanded_region, 
        length(new_positions)
    ))
    
    return expanded_region
end

"""
    should_expand_space(brain)

Determina si el espacio cerebral debería expandirse.
"""
function should_expand_space(brain::Brain_Space)
    # Contar neuronas candidatas para expansión
    expansion_candidates = 0
    
    for (_, neuron) in brain.neurons
        if should_expand(neuron)
            expansion_candidates += 1
        end
    end
    
    # Calcular porcentaje
    percentage = expansion_candidates / max(1, length(brain.neurons))
    
    # Expandir si más del 20% de neuronas están saturadas
    return percentage > 0.2
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
end # module DynamicExpansion