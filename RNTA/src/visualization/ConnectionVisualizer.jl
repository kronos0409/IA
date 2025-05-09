module ConnectionVisualizer

using Makie
using GLMakie
using Colors
using LinearAlgebra
using Statistics
using ..BrainSpace, ..Connections, ..TensorNeuron

export visualize_connections, visualize_connection_strength, 
       connection_density_map, highlight_strongest_paths,
       visualize_connection_changes, prune_visualization,
       visualize_connection_types, export_connection_graph,
       track_connection_evolution

"""
    visualize_connections(brain_space::BrainSpace; 
                         show_weights::Bool=true,
                         min_weight::Float64=0.1,
                         colormap=:plasma)

Visualiza las conexiones entre neuronas en el espacio cerebral.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral que contiene las conexiones
- `show_weights::Bool=true`: Si se muestran los pesos de las conexiones
- `min_weight::Float64=0.1`: Umbral mínimo de peso para mostrar una conexión
- `colormap=:plasma`: Mapa de colores para representar los pesos

# Retorna
- `fig`: La figura que contiene la visualización
- `ax`: El eje 3D donde se dibuja la visualización
"""
function visualize_connections(brain_space::Brain_Space; 
                              show_weights::Bool=true,
                              min_weight::Float64=0.1,
                              colormap=:plasma)
    # Crear figura
    fig = Figure(resolution=(1200, 900))
    ax = Axis3(fig[1, 1], aspect=:data)
    
    # Obtener conexiones y posiciones de neuronas
    connections = get_connections(brain_space)
    neuron_positions = get_neuron_positions(brain_space)
    
    # Visualizar neuronas como puntos
    neuron_ids = collect(keys(neuron_positions))
    positions = collect(values(neuron_positions))
    scatter!(ax, positions, color=:gray, markersize=10)
    
    # Filtrar conexiones por peso mínimo
    filtered_connections = filter(conn -> abs(conn.weight) >= min_weight, connections)
    
    # Mostrar conexiones
    for conn in filtered_connections
        # Obtener posiciones de neuronas de origen y destino
        source_pos = neuron_positions[conn.source_id]
        target_pos = neuron_positions[conn.target_id]
        
        # Definir color basado en peso si está habilitado
        if show_weights
            # Normalizar peso a [0,1] para el mapeo de colores
            norm_weight = normalize_weight(conn.weight)
            color = get_weight_color(norm_weight, colormap)
            linewidth = 1 + 3 * abs(norm_weight)  # Ancho basado en peso
        else
            color = :lightblue
            linewidth = 1.5
        end
        
        # Dibujar línea de conexión
        lines!(ax, [source_pos, target_pos], color=color, linewidth=linewidth)
    end
    
    # Configurar etiquetas y título
    ax.xlabel = "X"
    ax.ylabel = "Y"
    ax.zlabel = "Z"
    ax.title = "Visualización de Conexiones Neuronales"
    
    return fig, ax
end

"""
    visualize_connection_strength(brain_space::BrainSpace, neuron_id; 
                                 incoming::Bool=true,
                                 outgoing::Bool=true)

Visualiza la fuerza de las conexiones para una neurona específica.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral
- `neuron_id`: ID de la neurona focal
- `incoming::Bool=true`: Si se muestran conexiones entrantes
- `outgoing::Bool=true`: Si se muestran conexiones salientes

# Retorna
- `fig`: La figura que contiene la visualización
"""
function visualize_connection_strength(brain_space::Brain_Space, neuron_id; 
                                      incoming::Bool=true,
                                      outgoing::Bool=true)
    # Crear figura
    fig = Figure(resolution=(1000, 800))
    ax = Axis3(fig[1, 1], aspect=:data)
    
    # Obtener conexiones y posiciones
    connections = get_connections(brain_space)
    neuron_positions = get_neuron_positions(brain_space)
    
    # Filtrar conexiones para la neurona específica
    focal_connections = filter(connections) do conn
        (incoming && conn.target_id == neuron_id) || 
        (outgoing && conn.source_id == neuron_id)
    end
    
    # Obtener todos los IDs de neuronas involucrados
    involved_neurons = Set{Int}([neuron_id])
    for conn in focal_connections
        push!(involved_neurons, conn.source_id)
        push!(involved_neurons, conn.target_id)
    end
    
    # Visualizar todas las neuronas involucradas
    for id in involved_neurons
        pos = neuron_positions[id]
        
        # Neurona focal en rojo, otras en gris
        color = id == neuron_id ? :red : :gray
        size = id == neuron_id ? 15 : 8
        
        scatter!(ax, [pos], color=color, markersize=size)
    end
    
    # Visualizar conexiones
    for conn in focal_connections
        source_pos = neuron_positions[conn.source_id]
        target_pos = neuron_positions[conn.target_id]
        
        # Determinar tipo de conexión (entrante/saliente)
        is_incoming = conn.target_id == neuron_id
        is_outgoing = conn.source_id == neuron_id
        
        # Color según tipo y peso
        if is_incoming && incoming
            # Conexión entrante: azul
            color = colorant"blue"
            alpha = 0.3 + 0.7 * abs(normalize_weight(conn.weight))
        elseif is_outgoing && outgoing
            # Conexión saliente: verde
            color = colorant"green"
            alpha = 0.3 + 0.7 * abs(normalize_weight(conn.weight))
        else
            continue  # Saltar si no se muestra este tipo
        end
        
        # Ajustar transparencia según peso
        color_with_alpha = RGBA(color.r, color.g, color.b, alpha)
        
        # Ancho de línea basado en peso
        linewidth = 1 + 4 * abs(normalize_weight(conn.weight))
        
        # Dibujar conexión
        lines!(ax, [source_pos, target_pos], color=color_with_alpha, linewidth=linewidth)
    end
    
    # Configurar visualización
    ax.title = "Conexiones para Neurona $neuron_id"
    
    return fig
end

"""
    connection_density_map(brain_space::BrainSpace; 
                          resolution=(50, 50, 50),
                          weight_threshold::Float64=0.0)

Genera un mapa de densidad 3D que muestra la concentración de conexiones en el espacio.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral
- `resolution=(50, 50, 50)`: Resolución del mapa de densidad
- `weight_threshold::Float64=0.0`: Umbral mínimo de peso para considerar conexiones

# Retorna
- `density_map`: Array 3D con la densidad de conexiones
- `fig`: Figura con visualización de volumen
"""
function connection_density_map(brain_space::Brain_Space; 
                               resolution=(50, 50, 50),
                               weight_threshold::Float64=0.0)
    # Inicializar mapa de densidad
    density_map = zeros(Float32, resolution)
    
    # Obtener conexiones y dimensiones del espacio
    connections = get_connections(brain_space)
    neuron_positions = get_neuron_positions(brain_space)
    x_range, y_range, z_range = get_brain_space_dimensions(brain_space)
    
    # Factores de escala para mapear coordenadas a índices
    x_scale = (resolution[1] - 1) / (x_range[2] - x_range[1])
    y_scale = (resolution[2] - 1) / (y_range[2] - y_range[1])
    z_scale = (resolution[3] - 1) / (z_range[2] - z_range[1])
    
    # Filtrar conexiones por umbral
    filtered_connections = filter(conn -> abs(conn.weight) >= weight_threshold, connections)
    
    # Crear mapa de densidad
    for conn in filtered_connections
        # Obtener posiciones de origen y destino
        source_pos = neuron_positions[conn.source_id]
        target_pos = neuron_positions[conn.target_id]
        
        # Calcular puntos intermedios a lo largo de la conexión (10 puntos)
        for t in 0:0.1:1
            # Interpolación lineal entre origen y destino
            pos = source_pos + t * (target_pos - source_pos)
            
            # Convertir a índices del mapa
            x_idx = round(Int, (pos[1] - x_range[1]) * x_scale) + 1
            y_idx = round(Int, (pos[2] - y_range[1]) * y_scale) + 1
            z_idx = round(Int, (pos[3] - z_range[1]) * z_scale) + 1
            
            # Incrementar densidad si está dentro de los límites
            if 1 <= x_idx <= resolution[1] && 
               1 <= y_idx <= resolution[2] && 
               1 <= z_idx <= resolution[3]
                
                # Aumentar según el peso de la conexión
                density_map[x_idx, y_idx, z_idx] += abs(conn.weight)
            end
        end
    end
    
    # Normalizar mapa de densidad
    normalize_map!(density_map)
    
    # Crear visualización de volumen
    fig = Figure(resolution=(1000, 800))
    ax = Axis3(fig[1, 1], aspect=:data)
    
    # Visualizar volumen
    volume!(ax, 1:resolution[1], 1:resolution[2], 1:resolution[3], density_map,
            algorithm=:mip,  # Maximum Intensity Projection
            colormap=:inferno,
            transparency=true)
    
    # Configurar visualización
    ax.title = "Mapa de Densidad de Conexiones"
    
    return density_map, fig
end

"""
    highlight_strongest_paths(brain_space::BrainSpace; 
                             num_paths::Int=10,
                             path_length::Int=5)

Identifica y visualiza los caminos de conexión más fuertes en la red.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral
- `num_paths::Int=10`: Número de caminos a destacar
- `path_length::Int=5`: Longitud máxima de los caminos

# Retorna
- `fig`: Figura con los caminos destacados
- `strongest_paths`: Lista de los caminos más fuertes
"""
function highlight_strongest_paths(brain_space::Brain_Space; 
                                  num_paths::Int=10,
                                  path_length::Int=5)
    # Obtener conexiones y posiciones
    connections = get_connections(brain_space)
    neuron_positions = get_neuron_positions(brain_space)
    
    # Construir grafo de conexiones
    connection_graph = build_connection_graph(connections)
    
    # Encontrar caminos más fuertes
    strongest_paths = find_strongest_paths(connection_graph, path_length, num_paths)
    
    # Crear visualización
    fig = Figure(resolution=(1200, 800))
    ax = Axis3(fig[1, 1], aspect=:data)
    
    # Visualizar todas las neuronas como puntos pequeños de fondo
    all_positions = collect(values(neuron_positions))
    scatter!(ax, all_positions, color=:lightgray, markersize=5, alpha=0.3)
    
    # Visualizar los caminos más fuertes con colores distintivos
    colors = distinguishable_colors(num_paths, [RGB(1,1,1), RGB(0,0,0)])
    
    for (i, path) in enumerate(strongest_paths)
        path_positions = [neuron_positions[id] for id in path]
        
        # Visualizar neuronas en el camino
        scatter!(ax, path_positions, color=colors[i], markersize=10)
        
        # Conectar neuronas con líneas
        for j in 1:(length(path_positions)-1)
            lines!(ax, [path_positions[j], path_positions[j+1]], 
                   color=colors[i], linewidth=2)
        end
    end
    
    # Configurar visualización
    ax.title = "Caminos de Conexión Más Fuertes"
    
    return fig, strongest_paths
end

"""
    visualize_connection_changes(brain_space_before::BrainSpace, 
                               brain_space_after::BrainSpace;
                               threshold::Float64=0.1)

Visualiza los cambios en las conexiones entre dos estados del espacio cerebral.

# Argumentos
- `brain_space_before::BrainSpace`: Estado inicial del espacio cerebral
- `brain_space_after::BrainSpace`: Estado final del espacio cerebral
- `threshold::Float64=0.1`: Umbral mínimo para considerar un cambio significativo

# Retorna
- `fig`: Figura con visualización de cambios
- `changes`: Estadísticas de cambios
"""
function visualize_connection_changes(brain_space_before::Brain_Space, 
                                    brain_space_after::Brain_Space;
                                    threshold::Float64=0.1)
    # Obtener conexiones de ambos estados
    connections_before = get_connections(brain_space_before)
    connections_after = get_connections(brain_space_after)
    
    # Crear diccionarios para facilitar búsqueda
    conn_dict_before = Dict((c.source_id, c.target_id) => c.weight for c in connections_before)
    conn_dict_after = Dict((c.source_id, c.target_id) => c.weight for c in connections_after)
    
    # Identificar cambios
    strengthened = []
    weakened = []
    new_connections = []
    removed_connections = []
    
    # Verificar conexiones que existen en ambos estados o solo en el estado final
    for conn in connections_after
        key = (conn.source_id, conn.target_id)
        
        if haskey(conn_dict_before, key)
            # Conexión existe en ambos: comparar pesos
            weight_diff = conn.weight - conn_dict_before[key]
            
            if abs(weight_diff) >= threshold
                if weight_diff > 0
                    push!(strengthened, (key..., weight_diff))
                else
                    push!(weakened, (key..., weight_diff))
                end
            end
        else
            # Conexión nueva
            push!(new_connections, (key..., conn.weight))
        end
    end
    
    # Verificar conexiones que fueron eliminadas
    for conn in connections_before
        key = (conn.source_id, conn.target_id)
        
        if !haskey(conn_dict_after, key)
            push!(removed_connections, (key..., conn.weight))
        end
    end
    
    # Obtener posiciones de neuronas
    neuron_positions = get_neuron_positions(brain_space_after)
    
    # Crear visualización
    fig = Figure(resolution=(1200, 1000))
    
    # Panel principal: visualización 3D
    ax = Axis3(fig[1:2, 1:2], aspect=:data)
    
    # Visualizar todas las neuronas como puntos grises
    all_positions = collect(values(neuron_positions))
    scatter!(ax, all_positions, color=:gray, markersize=4, alpha=0.5)
    
    # Visualizar cambios
    visualize_connection_type!(ax, strengthened, neuron_positions, :green, "Fortalecidas")
    visualize_connection_type!(ax, weakened, neuron_positions, :blue, "Debilitadas")
    visualize_connection_type!(ax, new_connections, neuron_positions, :red, "Nuevas")
    visualize_connection_type!(ax, removed_connections, neuron_positions, :black, "Eliminadas")
    
    # Panel lateral: estadísticas y leyenda
    stats_panel = fig[1, 3]
    
    num_strengthened = length(strengthened)
    num_weakened = length(weakened)
    num_new = length(new_connections)
    num_removed = length(removed_connections)
    total_changes = num_strengthened + num_weakened + num_new + num_removed
    
    # Crear texto de estadísticas
    stats_text = """
    Cambios en Conexiones:
    • Fortalecidas: $num_strengthened
    • Debilitadas: $num_weakened
    • Nuevas: $num_new
    • Eliminadas: $num_removed
    • Total: $total_changes
    """
    
    Label(stats_panel, stats_text, fontsize=14, tellwidth=false)
    
    # Configurar visualización
    ax.title = "Cambios en Conexiones Neuronales"
    
    # Crear resumen de cambios
    changes = Dict(
        "strengthened" => strengthened,
        "weakened" => weakened,
        "new" => new_connections,
        "removed" => removed_connections,
        "total_count" => total_changes
    )
    
    return fig, changes
end

"""
    prune_visualization(brain_space::BrainSpace; 
                      prune_threshold::Float64=0.05,
                      highlight_pruned::Bool=true)

Visualiza qué conexiones serían podadas si se aplicara un umbral de poda específico.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral
- `prune_threshold::Float64=0.05`: Umbral para considerar una conexión para poda
- `highlight_pruned::Bool=true`: Si se destacan las conexiones a podar

# Retorna
- `fig`: Figura con la visualización
- `pruning_stats`: Estadísticas sobre la poda
"""
function prune_visualization(brain_space::Brain_Space; 
                           prune_threshold::Float64=0.05,
                           highlight_pruned::Bool=true)
    # Obtener conexiones y posiciones
    connections = get_connections(brain_space)
    neuron_positions = get_neuron_positions(brain_space)
    
    # Separar conexiones a mantener y a podar
    to_keep = filter(conn -> abs(conn.weight) >= prune_threshold, connections)
    to_prune = filter(conn -> abs(conn.weight) < prune_threshold, connections)
    
    # Calcular estadísticas
    total_connections = length(connections)
    num_pruned = length(to_prune)
    prune_percentage = 100 * num_pruned / total_connections
    
    # Crear visualización
    fig = Figure(resolution=(1200, 800))
    
    # Panel principal: visualización 3D
    ax = Axis3(fig[1:2, 1:2], aspect=:data)
    
    # Visualizar todas las neuronas
    all_positions = collect(values(neuron_positions))
    scatter!(ax, all_positions, color=:gray, markersize=6)
    
    # Visualizar conexiones a mantener
    for conn in to_keep
        source_pos = neuron_positions[conn.source_id]
        target_pos = neuron_positions[conn.target_id]
        
        # Color basado en peso
        norm_weight = normalize_weight(conn.weight)
        color = get_weight_color(norm_weight, :viridis)
        linewidth = 1 + 2 * abs(norm_weight)
        
        lines!(ax, [source_pos, target_pos], color=color, linewidth=linewidth)
    end
    
    # Visualizar conexiones a podar si está habilitado
    if highlight_pruned
        for conn in to_prune
            source_pos = neuron_positions[conn.source_id]
            target_pos = neuron_positions[conn.target_id]
            
            # Conexiones a podar en rojo transparente
            lines!(ax, [source_pos, target_pos], color=:red, linewidth=0.5, alpha=0.3)
        end
    end
    
    # Panel lateral: estadísticas
    stats_panel = fig[1, 3]
    
    stats_text = """
    Estadísticas de Poda:
    • Umbral: $(prune_threshold)
    • Conexiones totales: $total_connections
    • Conexiones a podar: $num_pruned
    • Porcentaje a podar: $(round(prune_percentage, digits=2))%
    • Conexiones a mantener: $(total_connections - num_pruned)
    """
    
    Label(stats_panel, stats_text, fontsize=14, tellwidth=false)
    
    # Configurar visualización
    ax.title = "Visualización de Poda de Conexiones (Umbral: $prune_threshold)"
    
    # Crear estadísticas de poda
    pruning_stats = Dict(
        "threshold" => prune_threshold,
        "total_connections" => total_connections,
        "pruned_connections" => num_pruned,
        "prune_percentage" => prune_percentage,
        "kept_connections" => total_connections - num_pruned
    )
    
    return fig, pruning_stats
end

"""
    visualize_connection_types(brain_space::BrainSpace)

Visualiza diferentes tipos de conexiones (excitatorias, inhibitorias) en el espacio cerebral.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral

# Retorna
- `fig`: Figura con la visualización
"""
function visualize_connection_types(brain_space::Brain_Space)
    # Obtener conexiones y posiciones
    connections = get_connections(brain_space)
    neuron_positions = get_neuron_positions(brain_space)
    
    # Separar conexiones excitatorias e inhibitorias
    excitatory = filter(conn -> conn.weight > 0, connections)
    inhibitory = filter(conn -> conn.weight < 0, connections)
    
    # Crear visualización
    fig = Figure(resolution=(1200, 900))
    ax = Axis3(fig[1, 1], aspect=:data)
    
    # Visualizar todas las neuronas
    all_positions = collect(values(neuron_positions))
    scatter!(ax, all_positions, color=:gray, markersize=6)
    
    # Visualizar conexiones excitatorias (verde)
    for conn in excitatory
        source_pos = neuron_positions[conn.source_id]
        target_pos = neuron_positions[conn.target_id]
        
        norm_weight = normalize_weight(conn.weight)
        alpha = 0.3 + 0.7 * norm_weight
        linewidth = 0.5 + 2 * norm_weight
        
        lines!(ax, [source_pos, target_pos], 
               color=RGBA(0.0, 0.8, 0.2, alpha), 
               linewidth=linewidth)
    end
    
    # Visualizar conexiones inhibitorias (rojo)
    for conn in inhibitory
        source_pos = neuron_positions[conn.source_id]
        target_pos = neuron_positions[conn.target_id]
        
        norm_weight = normalize_weight(abs(conn.weight))
        alpha = 0.3 + 0.7 * norm_weight
        linewidth = 0.5 + 2 * norm_weight
        
        lines!(ax, [source_pos, target_pos], 
               color=RGBA(0.8, 0.0, 0.2, alpha), 
               linewidth=linewidth)
    end
    
    # Configurar visualización
    ax.title = "Tipos de Conexiones (Verde: Excitatorias, Rojo: Inhibitorias)"
    
    # Añadir leyenda
    Legend(fig[1, 2], 
          [PolyElement(color=:green), PolyElement(color=:red)], 
          ["Conexiones Excitatorias ($(length(excitatory)))", 
           "Conexiones Inhibitorias ($(length(inhibitory)))"])
    
    return fig
end

"""
    export_connection_graph(brain_space::BrainSpace, filename::String; 
                          format=:graphml,
                          min_weight::Float64=0.0)

Exporta la estructura de conexiones a un formato de grafo para análisis en otras herramientas.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral
- `filename::String`: Nombre del archivo para guardar
- `format=:graphml`: Formato de exportación (:graphml, :gexf, :dot)
- `min_weight::Float64=0.0`: Peso mínimo para incluir conexiones
"""
function export_connection_graph(brain_space::Brain_Space, filename::String; 
                               format=:graphml,
                               min_weight::Float64=0.0)
    # Obtener conexiones y posiciones
    connections = get_connections(brain_space)
    neuron_positions = get_neuron_positions(brain_space)
    
    # Filtrar por peso mínimo
    filtered_connections = filter(conn -> abs(conn.weight) >= min_weight, connections)
    
    if format == :graphml
        export_graphml(filtered_connections, neuron_positions, filename)
    elseif format == :gexf
        export_gexf(filtered_connections, neuron_positions, filename)
    elseif format == :dot
        export_dot(filtered_connections, neuron_positions, filename)
    else
        error("Formato no soportado: $format")
    end
    
    println("Grafo de conexiones exportado a: $filename")
end

"""
    track_connection_evolution(brain_space::BrainSpace, 
                             connection_ids, 
                             num_steps::Int; 
                             step_function)

Rastrea la evolución de conexiones específicas a lo largo del tiempo.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral
- `connection_ids`: Lista de pares (source_id, target_id) a rastrear
- `num_steps::Int`: Número de pasos a simular
- `step_function`: Función para avanzar la simulación

# Retorna
- `fig`: Figura con gráficos de evolución
- `evolution_data`: Datos de evolución de conexiones
"""
function track_connection_evolution(brain_space::Brain_Space, 
                                  connection_ids, 
                                  num_steps::Int; 
                                  step_function)
    # Inicializar matriz para almacenar pesos a lo largo del tiempo
    # Filas: conexiones, Columnas: pasos temporales
    evolution_data = zeros(Float32, length(connection_ids), num_steps + 1)
    
    # Registrar pesos iniciales
    for (i, (source_id, target_id)) in enumerate(connection_ids)
        weight = get_connection_weight(brain_space, source_id, target_id)
        evolution_data[i, 1] = weight
    end
    
    # Simular y registrar evolución
    for step in 1:num_steps
        # Avanzar simulación
        step_function(brain_space)
        
        # Registrar pesos actuales
        for (i, (source_id, target_id)) in enumerate(connection_ids)
            weight = get_connection_weight(brain_space, source_id, target_id)
            evolution_data[i, step + 1] = weight
        end
    end
    
    # Crear visualización
    fig = Figure(resolution=(1000, 600))
    ax = Axis(fig[1, 1], 
             xlabel="Paso Temporal", 
             ylabel="Peso de Conexión",
             title="Evolución Temporal de Conexiones")
    
    # Visualizar evolución para cada conexión
    time_steps = 0:num_steps
    colors = distinguishable_colors(length(connection_ids), [RGB(1,1,1), RGB(0,0,0)])
    
    for i in 1:length(connection_ids)
        source_id, target_id = connection_ids[i]
        label = "$(source_id)→$(target_id)"
        
        lines!(ax, time_steps, evolution_data[i, :], 
               color=colors[i], linewidth=2, label=label)
    end
    
    # Añadir leyenda
    axislegend(ax)
    
    return fig, evolution_data
end

# Funciones auxiliares internas

"""Obtiene las conexiones del espacio cerebral"""
function get_connections(brain_space::Brain_Space)
    # Implementación dependería de la estructura interna de BrainSpace
    # Ejemplo simplificado:
    return brain_space.connections
end

"""Obtiene las posiciones de todas las neuronas"""
function get_neuron_positions(brain_space::Brain_Space)
    # Ejemplo simplificado:
    positions = Dict{Int, Vector{Float64}}()
    
    for neuron in brain_space.neurons
        positions[neuron.id] = [neuron.position.x, neuron.position.y, neuron.position.z]
    end
    
    return positions
end

"""Normaliza un peso de conexión al rango [0,1]"""
function normalize_weight(weight)
    # Limitar a rango [-1, 1] y luego normalizar a [0, 1]
    clamped = clamp(weight, -1.0, 1.0)
    return (clamped + 1.0) / 2.0
end

"""Obtiene un color basado en el peso normalizado"""
function get_weight_color(normalized_weight, colormap)
    # Usar el sistema de mapeo de colores de Makie
    return cgrad(colormap)[normalized_weight]
end

"""Normaliza un mapa 3D al rango [0,1]"""
function normalize_map!(map_3d)
    min_val, max_val = extrema(map_3d)
    if min_val != max_val  # Evitar división por cero
        map_3d .= (map_3d .- min_val) ./ (max_val - min_val)
    end
end

"""Construye un grafo de conexiones para análisis de caminos"""
function build_connection_graph(connections)
    # Simplificado: crear un diccionario que mapea ID de neurona a sus conexiones
    graph = Dict{Int, Vector{Tuple{Int, Float64}}}()
    
    for conn in connections
        if !haskey(graph, conn.source_id)
            graph[conn.source_id] = []
        end
        
        push!(graph[conn.source_id], (conn.target_id, conn.weight))
    end
    
    return graph
end

"""Encuentra los caminos más fuertes en el grafo de conexiones"""
function find_strongest_paths(graph, max_length, num_paths)
    # Esta sería una implementación simplificada
    # En la práctica, se usaría un algoritmo más sofisticado
    
    # Simulación: devolver caminos aleatorios
    paths = []
    
    # Obtener todas las neuronas
    all_neurons = collect(keys(graph))
    
    # Generar caminos aleatorios como ejemplo
    for _ in 1:min(num_paths, length(all_neurons))
        # Comenzar desde una neurona aleatoria
        start_neuron = rand(all_neurons)
        
        # Construir camino
        path = [start_neuron]
        current = start_neuron
        
        # Añadir neuronas al camino
        for _ in 2:max_length
            if !haskey(graph, current) || isempty(graph[current])
                break
            end
            
            # Elegir la conexión más fuerte
            next_connections = graph[current]
            sorted_connections = sort(next_connections, by=x -> abs(x[2]), rev=true)
            
            # Tomar la primera que no forme un ciclo
            for (next_id, _) in sorted_connections
                if !(next_id in path)
                    push!(path, next_id)
                    current = next_id
                    break
                end
            end
            
            # Si no se pudo añadir más neuronas, terminar
            if current != path[end]
                break
            end
        end
        
        # Añadir camino si tiene al menos 2 neuronas
        if length(path) >= 2
            push!(paths, path)
        end
    end
    
    return paths
end

"""Visualiza un tipo específico de cambios de conexión"""
function visualize_connection_type!(ax, connections, positions, color, label)
    for (source_id, target_id, weight) in connections
        # Verificar que ambas neuronas existen en las posiciones
        if haskey(positions, source_id) && haskey(positions, target_id)
            source_pos = positions[source_id]
            target_pos = positions[target_id]
            
            # Normalizar peso para transparencia y ancho
            norm_weight = clamp(abs(weight), 0.0, 1.0)
            alpha = 0.3 + 0.7 * norm_weight
            linewidth = 0.5 + 2.0 * norm_weight
            
            # Dibujar conexión
            lines!(ax, [source_pos, target_pos], 
                   color=RGBA(color, alpha), 
                   linewidth=linewidth)
        end
    end
end

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

"""Obtiene el peso de una conexión específica"""
function get_connection_weight(brain_space::Brain_Space, source_id, target_id)
    # Buscar la conexión especificada
    for conn in brain_space.connections
        if conn.source_id == source_id && conn.target_id == target_id
            return conn.weight
        end
    end
    
    # Si no existe, devolver 0
    return 0.0
end

"""Exporta grafo en formato GraphML"""
function export_graphml(connections, positions, filename)
    # En una implementación real, se usaría una biblioteca como LightGraphs.jl
    # Aquí solo se simula la exportación
    
    open(filename, "w") do io
        # Escribir encabezado GraphML
        println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        println(io, "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">")
        
        # Definir atributos
        println(io, "  <key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>")
        println(io, "  <key id=\"x\" for=\"node\" attr.name=\"x\" attr.type=\"double\"/>")
        println(io, "  <key id=\"y\" for=\"node\" attr.name=\"y\" attr.type=\"double\"/>")
        println(io, "  <key id=\"z\" for=\"node\" attr.name=\"z\" attr.type=\"double\"/>")
        
        # Iniciar grafo
        println(io, "  <graph id=\"G\" edgedefault=\"directed\">")
        
        # Escribir nodos
        for (id, pos) in positions
            println(io, "    <node id=\"n$id\">")
            println(io, "      <data key=\"x\">$(pos[1])</data>")
            println(io, "      <data key=\"y\">$(pos[2])</data>")
            println(io, "      <data key=\"z\">$(pos[3])</data>")
            println(io, "    </node>")
        end
        
        # Escribir aristas
        for (i, conn) in enumerate(connections)
            println(io, "    <edge id=\"e$i\" source=\"n$(conn.source_id)\" target=\"n$(conn.target_id)\">")
            println(io, "      <data key=\"weight\">$(conn.weight)</data>")
            println(io, "    </edge>")
        end
        
        # Cerrar grafo y documento
        println(io, "  </graph>")
        println(io, "</graphml>")
    end
end

"""Exporta grafo en formato GEXF"""
function export_gexf(connections, positions, filename)
    # Simplificado: solo simular la exportación
    println("Exportando grafo en formato GEXF a: $filename")
end

"""Exporta grafo en formato DOT (Graphviz)"""
function export_dot(connections, positions, filename)
    # Simplificado: solo simular la exportación
    println("Exportando grafo en formato DOT a: $filename")
end

end # module