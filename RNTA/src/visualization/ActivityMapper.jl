module ActivityMapper

using LinearAlgebra
using Statistics
using Colors
using Makie
using GLMakie
using ..BrainSpace, ..SpatialField, ..TensorNeuron

export create_activity_map, temporal_activity_sequence, activity_difference_map,
       region_activity_profile, generate_activation_heatmap, track_neuron_activity,
       create_activity_comparison, export_activity_data, activation_threshold_map

"""
    create_activity_map(brain_space::BrainSpace; 
                        resolution=(100, 100, 100),
                        normalize::Bool=true)

Crea un mapa volumétrico de la actividad neuronal actual en el espacio cerebral.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral a mapear
- `resolution=(100, 100, 100)`: Resolución del volumen 3D (x, y, z)
- `normalize::Bool=true`: Si se normaliza la actividad al rango [0,1]

# Retorna
- `activity_volume`: Array 3D con la actividad mapeada
"""
function create_activity_map(brain_space::Brain_Space; 
                            resolution=(100, 100, 100),
                            normalize::Bool=true)
    # Inicializar volumen de actividad
    activity_volume = zeros(Float32, resolution)
    
    # Obtener dimensiones del espacio cerebral
    x_range, y_range, z_range = get_brain_space_dimensions(brain_space)
    
    # Factores de escala para mapear coordenadas a índices del volumen
    x_scale = (resolution[1] - 1) / (x_range[2] - x_range[1])
    y_scale = (resolution[2] - 1) / (y_range[2] - y_range[1])
    z_scale = (resolution[3] - 1) / (z_range[2] - z_range[1])
    
    # Mapear cada neurona al volumen
    for neuron in brain_space.neurons
        # Convertir posición a índices del volumen
        x_idx = round(Int, (neuron.position.x - x_range[1]) * x_scale) + 1
        y_idx = round(Int, (neuron.position.y - y_range[1]) * y_scale) + 1
        z_idx = round(Int, (neuron.position.z - z_range[1]) * z_scale) + 1
        
        # Asegurarse de que los índices están dentro de los límites
        if 1 <= x_idx <= resolution[1] && 
           1 <= y_idx <= resolution[2] && 
           1 <= z_idx <= resolution[3]
            
            # Acumular actividad (usar máximo si hay múltiples neuronas en un voxel)
            activity_volume[x_idx, y_idx, z_idx] = 
                max(activity_volume[x_idx, y_idx, z_idx], neuron.activation)
        end
    end
    
    # Normalizar si es necesario
    if normalize
        min_val, max_val = extrema(activity_volume)
        if min_val != max_val  # Evitar división por cero
            activity_volume = (activity_volume .- min_val) ./ (max_val - min_val)
        end
    end
    
    return activity_volume
end

"""
    temporal_activity_sequence(brain_space::BrainSpace, num_steps::Int; 
                              step_function, delay::Float64=0.1)

Genera una secuencia temporal de mapas de actividad durante varios pasos de simulación.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral a mapear
- `num_steps::Int`: Número de pasos temporales a simular
- `step_function`: Función que avanza la simulación un paso temporal
- `delay::Float64=0.1`: Retraso entre pasos (en segundos)

# Retorna
- `activity_sequence`: Vector de mapas de actividad 3D
"""
function temporal_activity_sequence(brain_space::Brain_Space, num_steps::Int; 
                                   step_function, delay::Float64=0.1)
    # Vector para almacenar la secuencia de actividad
    activity_sequence = []
    
    # Crear mapa inicial
    initial_map = create_activity_map(brain_space)
    push!(activity_sequence, copy(initial_map))
    
    # Simular pasos temporales
    for i in 1:num_steps
        # Avanzar la simulación
        step_function(brain_space)
        
        # Capturar mapa de actividad actual
        current_map = create_activity_map(brain_space)
        push!(activity_sequence, copy(current_map))
        
        # Opcional: esperar entre pasos para simulaciones en tiempo real
        if delay > 0
            sleep(delay)
        end
    end
    
    return activity_sequence
end

"""
    activity_difference_map(map1, map2; threshold::Float64=0.1)

Calcula un mapa de diferencia entre dos mapas de actividad para visualizar cambios.

# Argumentos
- `map1`: Primer mapa de actividad (Array 3D)
- `map2`: Segundo mapa de actividad (Array 3D)
- `threshold::Float64=0.1`: Umbral mínimo para considerar un cambio significativo

# Retorna
- `diff_map`: Mapa de diferencias con valores positivos/negativos
"""
function activity_difference_map(map1, map2; threshold::Float64=0.1)
    # Verificar que los mapas tienen las mismas dimensiones
    if size(map1) != size(map2)
        error("Los mapas de actividad deben tener las mismas dimensiones")
    end
    
    # Calcular diferencia
    diff_map = map2 - map1
    
    # Aplicar umbral para eliminar ruido
    diff_map[abs.(diff_map) .< threshold] .= 0.0
    
    return diff_map
end

"""
    region_activity_profile(brain_space::BrainSpace, region_bounds)

Genera un perfil de actividad para una región específica del espacio cerebral.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral completo
- `region_bounds`: Tupla ((x_min, x_max), (y_min, y_max), (z_min, z_max)) con los límites

# Retorna
- `profile`: Diccionario con estadísticas de actividad para la región
"""
function region_activity_profile(brain_space::Brain_Space, region_bounds)
    # Filtrar neuronas en la región
    region_neurons = filter_neurons_in_region(brain_space, region_bounds)
    
    # Si no hay neuronas en la región, devolver perfil vacío
    if isempty(region_neurons)
        return Dict(
            "num_neurons" => 0,
            "mean_activity" => 0.0,
            "max_activity" => 0.0,
            "active_ratio" => 0.0
        )
    end
    
    # Obtener activaciones
    activities = [neuron.activation for neuron in region_neurons]
    
    # Calcular estadísticas
    mean_act = mean(activities)
    max_act = maximum(activities)
    active_ratio = count(act -> act > 0.1, activities) / length(activities)
    
    # Calcular distribución de actividad
    hist_data = fit(Histogram, activities, 0:0.1:1)
    
    # Crear perfil
    profile = Dict(
        "num_neurons" => length(region_neurons),
        "mean_activity" => mean_act,
        "max_activity" => max_act,
        "active_ratio" => active_ratio,
        "activity_histogram" => hist_data,
        "activity_std" => std(activities),
        "activity_median" => median(activities)
    )
    
    return profile
end

"""
    generate_activation_heatmap(activity_map; slice_dimension=3, slice_index=nothing)

Genera un mapa de calor 2D a partir de un mapa de actividad 3D, tomando un corte específico.

# Argumentos
- `activity_map`: Mapa de actividad 3D
- `slice_dimension=3`: Dimensión a lo largo de la cual hacer el corte (1=x, 2=y, 3=z)
- `slice_index=nothing`: Índice del corte. Si es `nothing`, se usa el índice medio

# Retorna
- `fig`: La figura que contiene el mapa de calor
- `heatmap_data`: Los datos 2D del mapa de calor
"""
function generate_activation_heatmap(activity_map; slice_dimension=3, slice_index=nothing)
    # Obtener dimensiones
    dims = size(activity_map)
    
    # Si no se especifica índice, usar el centro
    if isnothing(slice_index)
        slice_index = div(dims[slice_dimension], 2)
    end
    
    # Extraer slice 2D según la dimensión
    if slice_dimension == 1
        heatmap_data = activity_map[slice_index, :, :]
    elseif slice_dimension == 2
        heatmap_data = activity_map[:, slice_index, :]
    else  # slice_dimension == 3
        heatmap_data = activity_map[:, :, slice_index]
    end
    
    # Crear figura
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1])
    
    # Crear mapa de calor
    hm = heatmap!(ax, heatmap_data, colormap=:inferno)
    
    # Añadir barra de color
    Colorbar(fig[1, 2], hm)
    
    # Establecer etiquetas según la dimensión
    dim_labels = [("Y", "Z"), ("X", "Z"), ("X", "Y")]
    ax.xlabel, ax.ylabel = dim_labels[slice_dimension]
    
    # Título con información del corte
    dim_names = ["X", "Y", "Z"]
    ax.title = "Actividad en corte $(dim_names[slice_dimension])=$(slice_index)"
    
    return fig, heatmap_data
end

"""
    track_neuron_activity(brain_space::BrainSpace, neuron_ids, num_steps::Int; 
                         step_function)

Registra la actividad de neuronas específicas a lo largo de varios pasos temporales.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral a monitorear
- `neuron_ids`: Vector de IDs de neuronas a rastrear
- `num_steps::Int`: Número de pasos temporales
- `step_function`: Función que avanza la simulación un paso

# Retorna
- `tracked_data`: Matriz donde cada fila contiene la actividad de una neurona a lo largo del tiempo
"""
function track_neuron_activity(brain_space::Brain_Space, neuron_ids, num_steps::Int; 
                              step_function)
    # Inicializar matriz para almacenar actividad
    # Filas: neuronas, Columnas: pasos temporales
    tracked_data = zeros(Float32, length(neuron_ids), num_steps + 1)
    
    # Registrar estado inicial
    for (i, id) in enumerate(neuron_ids)
        tracked_data[i, 1] = get_neuron_activation(brain_space, id)
    end
    
    # Simular y registrar actividad para cada paso
    for step in 1:num_steps
        # Avanzar simulación
        step_function(brain_space)
        
        # Registrar activación de cada neurona
        for (i, id) in enumerate(neuron_ids)
            tracked_data[i, step + 1] = get_neuron_activation(brain_space, id)
        end
    end
    
    return tracked_data
end

"""
    create_activity_comparison(brain_space1::BrainSpace, brain_space2::BrainSpace; 
                             resolution=(80, 80, 80))

Compara la actividad entre dos espacios cerebrales diferentes, útil para comparar
estados antes/después o diferentes modelos.

# Argumentos
- `brain_space1::BrainSpace`: Primer espacio cerebral
- `brain_space2::BrainSpace`: Segundo espacio cerebral
- `resolution=(80, 80, 80)`: Resolución para los mapas de actividad

# Retorna
- `comparison_results`: Diccionario con datos de comparación
"""
function create_activity_comparison(brain_space1::Brain_Space, brain_space2::Brain_Space; 
                                  resolution=(80, 80, 80))
    # Generar mapas de actividad para ambos espacios
    map1 = create_activity_map(brain_space1, resolution=resolution)
    map2 = create_activity_map(brain_space2, resolution=resolution)
    
    # Calcular mapa de diferencias
    diff_map = activity_difference_map(map1, map2)
    
    # Calcular estadísticas de comparación
    mean_diff = mean(abs.(diff_map))
    max_diff = maximum(abs.(diff_map))
    
    # Identificar regiones con mayores diferencias
    significant_threshold = 0.3  # Umbral para diferencias significativas
    significant_diffs = count(x -> abs(x) > significant_threshold, diff_map)
    significant_ratio = significant_diffs / prod(resolution)
    
    # Crear resultados de comparación
    comparison_results = Dict(
        "map1" => map1,
        "map2" => map2,
        "diff_map" => diff_map,
        "mean_difference" => mean_diff,
        "max_difference" => max_diff,
        "significant_differences" => significant_diffs,
        "significant_ratio" => significant_ratio
    )
    
    return comparison_results
end

"""
    export_activity_data(activity_map, filename::String; format=:jld2)

Exporta datos de actividad a un archivo para análisis posterior.

# Argumentos
- `activity_map`: Mapa de actividad 3D o secuencia temporal
- `filename::String`: Nombre del archivo donde guardar los datos
- `format=:jld2`: Formato de archivo (:jld2, :csv, :npz)
"""
function export_activity_data(activity_map, filename::String; format=:jld2)
    if format == :jld2
        # Guardar en formato Julia Data
        save_jld(filename, Dict("activity_map" => activity_map))
    elseif format == :csv
        # Para CSV, aplanar el mapa 3D o guardar múltiples archivos
        export_to_csv(activity_map, filename)
    elseif format == :npz
        # Formato compatible con NumPy de Python
        save_npz(filename, Dict("activity_map" => activity_map))
    else
        error("Formato no soportado: $format")
    end
    
    println("Datos de actividad exportados a: $filename")
end

"""
    activation_threshold_map(activity_map, threshold::Float64; 
                            binary::Bool=false)

Aplica un umbral al mapa de actividad para filtrar valores bajos.

# Argumentos
- `activity_map`: Mapa de actividad 3D
- `threshold::Float64`: Umbral de activación (valores menores se reducen)
- `binary::Bool=false`: Si es true, convierte a mapa binario (0 o 1)

# Retorna
- `thresholded_map`: Mapa de actividad con umbral aplicado
"""
function activation_threshold_map(activity_map, threshold::Float64; 
                                binary::Bool=false)
    # Crear copia del mapa para no modificar el original
    thresholded_map = copy(activity_map)
    
    if binary
        # Convertir a mapa binario (0 o 1)
        thresholded_map = Float32.(thresholded_map .>= threshold)
    else
        # Aplicar umbral suave (mantener valores, pero reducir los bajos)
        below_threshold = thresholded_map .< threshold
        thresholded_map[below_threshold] .*= (thresholded_map[below_threshold] ./ threshold)
    end
    
    return thresholded_map
end

# Funciones auxiliares internas

"""Obtiene las dimensiones del espacio cerebral"""
function get_brain_space_dimensions(brain_space::Brain_Space)
    # Obtener posiciones de todas las neuronas
    positions = [(n.position.x, n.position.y, n.position.z) for n in brain_space.neurons]
    
    # Calcular límites para cada dimensión
    x_vals = [p[1] for p in positions]
    y_vals = [p[2] for p in positions]
    z_vals = [p[3] for p in positions]
    
    x_range = extrema(x_vals)
    y_range = extrema(y_vals)
    z_range = extrema(z_vals)
    
    return x_range, y_range, z_range
end

"""Filtra neuronas dentro de una región específica"""
function filter_neurons_in_region(brain_space, region_bounds)
    (x_bounds, y_bounds, z_bounds) = region_bounds
    
    # Filtrar neuronas dentro de los límites
    return [neuron for neuron in brain_space.neurons 
            if x_bounds[1] <= neuron.position.x <= x_bounds[2] &&
               y_bounds[1] <= neuron.position.y <= y_bounds[2] &&
               z_bounds[1] <= neuron.position.z <= z_bounds[2]]
end

"""Obtiene la activación de una neurona por su ID"""
function get_neuron_activation(brain_space::Brain_Space, neuron_id)
    # Buscar neurona por ID
    for neuron in brain_space.neurons
        if neuron.id == neuron_id
            return neuron.activation
        end
    end
    
    # Si no se encuentra, devolver 0
    return 0.0
end

"""Guarda datos en formato JLD"""
function save_jld(filename, data_dict)
    # En una implementación real, se usaría JLD2.jl
    # Por simplicidad, aquí solo simulamos la función
    println("Guardando datos en formato JLD: $filename")
end

"""Exporta datos a formato CSV"""
function export_to_csv(activity_map, filename)
    # Para 3D, guardar múltiples archivos 2D (uno por slice)
    if ndims(activity_map) == 3
        for z in 1:size(activity_map, 3)
            slice_filename = replace(filename, ".csv" => "_slice$(z).csv")
            slice_data = activity_map[:, :, z]
            
            # Guardar slice como CSV
            open(slice_filename, "w") do io
                for i in 1:size(slice_data, 1)
                    line = join(slice_data[i, :], ",")
                    println(io, line)
                end
            end
        end
    else
        # Para datos 2D, guardar directamente
        open(filename, "w") do io
            for i in 1:size(activity_map, 1)
                line = join(activity_map[i, :], ",")
                println(io, line)
            end
        end
    end
end

"""Guarda datos en formato NPZ (NumPy)"""
function save_npz(filename, data_dict)
    # En una implementación real, se usaría un paquete para compatibilidad con NumPy
    # Por simplicidad, aquí solo simulamos la función
    println("Guardando datos en formato NPZ: $filename")
end

end # module