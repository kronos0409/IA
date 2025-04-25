module BrainSpaceVisualizer

using Makie
using GLMakie
using Colors
using LinearAlgebra
using ..BrainSpace, ..SpatialField, ..TensorNeuron

export visualize_brain_space, visualize_region, capture_activity_snapshot, 
       create_interactive_view, export_visualization, generate_activity_heatmap

"""
    visualize_brain_space(brain_space::BrainSpace; 
                          show_connections::Bool=true, 
                          activity_color_map=:viridis, 
                          background_color=:black)

Visualiza el espacio neuronal 3D completo con activaciones neuronales y opcionalmente conexiones.

# Argumentos
- `brain_space::BrainSpace`: El espacio neuronal a visualizar
- `show_connections::Bool=true`: Si se deben mostrar las conexiones entre neuronas
- `activity_color_map=:viridis`: Mapa de colores para representar la actividad neuronal
- `background_color=:black`: Color de fondo para la visualización

# Retorna
- `fig`: La figura que contiene la visualización
- `ax`: El eje 3D donde se dibuja la visualización
"""
function visualize_brain_space(brain_space::Brain_Space; 
                              show_connections::Bool=true, 
                              activity_color_map=:viridis, 
                              background_color=:black)
    fig = Figure(resolution=(1200, 900), backgroundcolor=background_color)
    ax = Axis3(fig[1, 1], 
              aspect=:data, 
              xlabel="X", ylabel="Y", zlabel="Z",
              title="Espacio Neuronal 3D")
    
    # Obtener datos de posición y activación de neuronas
    positions = get_neuron_positions(brain_space)
    activities = get_neuron_activities(brain_space)
    
    # Normalizar actividades para el mapeo de colores
    normalized_activities = normalize_values(activities)
    
    # Crear puntos para las neuronas con colores basados en su activación
    colors = [get_activity_color(act, activity_color_map) for act in normalized_activities]
    sizes = [10.0 + 20.0 * act for act in normalized_activities]
    
    # Visualizar neuronas
    scatter!(ax, positions, color=colors, markersize=sizes)
    
    # Visualizar conexiones si está habilitado
    if show_connections
        draw_connections!(ax, brain_space)
    end
    
    # Configurar iluminación y cámara
    setup_lighting!(ax)
    set_default_camera!(ax)
    
    return fig, ax
end

"""
    visualize_region(brain_space::BrainSpace, region_bounds;
                     show_connections::Bool=true, 
                     highlight_active::Bool=true)

Visualiza una región específica del espacio neuronal definida por límites 3D.

# Argumentos
- `brain_space::BrainSpace`: El espacio neuronal completo
- `region_bounds`: Tupla ((x_min, x_max), (y_min, y_max), (z_min, z_max)) con los límites de la región
- `show_connections::Bool=true`: Si se deben mostrar las conexiones
- `highlight_active::Bool=true`: Resaltar neuronas con alta actividad

# Retorna
- `fig`: La figura que contiene la visualización
- `ax`: El eje 3D donde se dibuja la visualización
"""
function visualize_region(brain_space::Brain_Space, region_bounds;
                         show_connections::Bool=true, 
                         highlight_active::Bool=true)
    (x_bounds, y_bounds, z_bounds) = region_bounds
    
    fig = Figure(resolution=(1000, 800))
    ax = Axis3(fig[1, 1], aspect=:data)
    
    # Filtrar neuronas dentro de la región
    region_neurons = filter_neurons_in_region(brain_space, region_bounds)
    
    # Visualizar neuronas en la región
    positions = get_neuron_positions(region_neurons)
    activities = get_neuron_activities(region_neurons)
    
    # Colores basados en actividad
    colors = get_region_color_mapping(activities, highlight_active)
    
    # Visualizar neuronas
    scatter!(ax, positions, color=colors, markersize=12)
    
    # Visualizar conexiones dentro de la región
    if show_connections
        draw_region_connections!(ax, brain_space, region_bounds)
    end
    
    # Dibujar caja que delimita la región
    draw_region_box!(ax, region_bounds)
    
    return fig, ax
end

"""
    capture_activity_snapshot(brain_space::BrainSpace, filename::String; 
                             resolution=(1920, 1080), format=:png)

Captura una imagen del estado actual de activación del espacio neuronal y la guarda en un archivo.

# Argumentos
- `brain_space::BrainSpace`: El espacio neuronal a capturar
- `filename::String`: Nombre del archivo donde guardar la imagen
- `resolution=(1920, 1080)`: Resolución de la imagen de salida
- `format=:png`: Formato de imagen (:png, :jpg, :svg, etc)
"""
function capture_activity_snapshot(brain_space::Brain_Space, filename::String; 
                                  resolution=(1920, 1080), format=:png)
    fig, ax = visualize_brain_space(brain_space, show_connections=false)
    
    # Configurar resolución
    fig.resolution = resolution
    
    # Optimizar vista
    optimize_camera_angle!(ax)
    
    # Guardar figura
    save(filename, fig, format=format)
    
    return fig
end

"""
    create_interactive_view(brain_space::BrainSpace)

Crea una visualización interactiva del espacio neuronal que permite rotar, hacer zoom y seleccionar regiones.

# Argumentos
- `brain_space::BrainSpace`: El espacio neuronal a visualizar interactivamente

# Retorna
- `fig`: La figura interactiva
"""
function create_interactive_view(brain_space::Brain_Space)
    fig = Figure(resolution=(1200, 900))
    
    # Crear panel principal con visualización 3D
    ax_main = Axis3(fig[1, 1:2], aspect=:data)
    
    # Crear paneles para métricas y controles
    ax_metrics = Axis(fig[2, 1], xlabel="Tiempo", ylabel="Actividad")
    control_panel = fig[2, 2]
    
    # Visualizar espacio neuronal inicial
    positions = get_neuron_positions(brain_space)
    activities = get_neuron_activities(brain_space)
    scatter_plot = scatter!(ax_main, positions, color=activities, colormap=:plasma)
    
    # Añadir controles interactivos
    add_interactive_controls!(fig, control_panel, ax_main, scatter_plot, brain_space)
    
    # Añadir gráfico de actividad global
    plot_global_activity!(ax_metrics, brain_space)
    
    # Habilitar selección y zoom de regiones
    setup_region_selection!(fig, ax_main, brain_space)
    
    return fig
end

"""
    export_visualization(visualization, filename::String; 
                        format=:html, include_interactivity::Bool=true)

Exporta una visualización a diferentes formatos para compartir o incluir en documentación.

# Argumentos
- `visualization`: La visualización a exportar (figura de Makie)
- `filename::String`: Nombre del archivo donde guardar la exportación
- `format=:html`: Formato de exportación (:html, :png, :svg, :pdf)
- `include_interactivity::Bool=true`: Si se debe mantener la interactividad (solo para HTML)
"""
function export_visualization(visualization, filename::String; 
                             format=:html, include_interactivity::Bool=true)
    if format == :html
        # Exportar como visualización interactiva HTML con JS
        html_content = convert_to_interactive_html(visualization, include_interactivity)
        open(filename, "w") do io
            write(io, html_content)
        end
    else
        # Exportar como imagen estática
        save(filename, visualization)
    end
end

"""
    generate_activity_heatmap(brain_space::BrainSpace, slice_dimension::Symbol, slice_index::Int)

Genera un mapa de calor 2D de la actividad neuronal en un corte específico del espacio 3D.

# Argumentos
- `brain_space::BrainSpace`: El espacio neuronal a visualizar
- `slice_dimension::Symbol`: Dimensión a lo largo de la cual hacer el corte (:x, :y, o :z)
- `slice_index::Int`: Índice del corte en la dimensión especificada

# Retorna
- `fig`: La figura que contiene el mapa de calor
"""
function generate_activity_heatmap(brain_space::Brain_Space, slice_dimension::Symbol, slice_index::Int)
    fig = Figure()
    ax = Axis(fig[1, 1])
    
    # Obtener datos de actividad para el corte especificado
    slice_data = extract_slice_data(brain_space, slice_dimension, slice_index)
    
    # Generar mapa de calor
    heatmap!(ax, slice_data, colormap=:inferno)
    
    # Añadir barra de color
    Colorbar(fig[1, 2], colormap=:inferno, limits=(0, 1))
    
    # Configurar título y etiquetas
    dimensions = Dict(:x => ("Y", "Z"), :y => ("X", "Z"), :z => ("X", "Y"))
    ax.title = "Actividad Neuronal: Corte $(slice_dimension)=$(slice_index)"
    ax.xlabel = dimensions[slice_dimension][1]
    ax.ylabel = dimensions[slice_dimension][2]
    
    return fig
end

# Funciones auxiliares internas

"""Obtiene las posiciones de las neuronas en el espacio 3D"""
function get_neuron_positions(brain_space::Brain_Space)
    # Implementación real dependería de la estructura interna de BrainSpace
    # Ejemplo simplificado:
    return [(neuron.position.x, neuron.position.y, neuron.position.z) 
            for neuron in brain_space.neurons]
end

"""Obtiene los niveles de activación de las neuronas"""
function get_neuron_activities(brain_space::Brain_Space)
    # Implementación simplificada:
    return [neuron.activation for neuron in brain_space.neurons]
end

"""Normaliza valores a un rango [0,1]"""
function normalize_values(values)
    min_val, max_val = extrema(values)
    if min_val == max_val
        return fill(0.5, length(values))
    end
    return (values .- min_val) ./ (max_val - min_val)
end

"""Obtiene un color basado en el nivel de activación"""
function get_activity_color(activity, colormap)
    # Usa el sistema de mapeo de colores de Makie
    return cgrad(colormap)[activity]
end

"""Configura la iluminación para visualización 3D"""
function setup_lighting!(ax)
    # Configurar iluminación para mejor visualización 3D
    scene = ax.scene
    scene.lights = [DirectionalLight(RGB(0.7, 0.7, 0.7), [0, 1, 1]),
                    AmbientLight(RGB(0.3, 0.3, 0.3))]
end

"""Dibuja conexiones entre neuronas"""
function draw_connections!(ax, brain_space)
    # Obtener conexiones desde el espacio neuronal
    connections = get_neuron_connections(brain_space)
    
    for (source, target, strength) in connections
        # Obtener posiciones de neuronas
        source_pos = get_neuron_position(brain_space, source)
        target_pos = get_neuron_position(brain_space, target)
        
        # Dibujar línea con color y ancho basados en la fuerza de conexión
        color = get_connection_color(strength)
        width = 0.5 + 2.0 * abs(strength)
        
        # Dibujar conexión
        lines!(ax, [source_pos, target_pos], color=color, linewidth=width)
    end
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

"""Dibuja una caja para delimitar una región del espacio"""
function draw_region_box!(ax, region_bounds)
    (x_bounds, y_bounds, z_bounds) = region_bounds
    
    # Definir los vértices de la caja
    vertices = [
        (x_bounds[1], y_bounds[1], z_bounds[1]),
        (x_bounds[2], y_bounds[1], z_bounds[1]),
        (x_bounds[2], y_bounds[2], z_bounds[1]),
        (x_bounds[1], y_bounds[2], z_bounds[1]),
        (x_bounds[1], y_bounds[1], z_bounds[2]),
        (x_bounds[2], y_bounds[1], z_bounds[2]),
        (x_bounds[2], y_bounds[2], z_bounds[2]),
        (x_bounds[1], y_bounds[2], z_bounds[2])
    ]
    
    # Definir las aristas
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 1), # Base inferior
        (5, 6), (6, 7), (7, 8), (8, 5), # Base superior
        (1, 5), (2, 6), (3, 7), (4, 8)  # Conexiones verticales
    ]
    
    # Dibujar las aristas
    for (i, j) in edges
        lines!(ax, [vertices[i], vertices[j]], color=:white, linestyle=:dash)
    end
end

"""Optimiza el ángulo de la cámara para mejor visualización"""
function optimize_camera_angle!(ax)
    # Implementación básica: establecer una vista predeterminada óptima
    update_cam!(ax.scene, campos=[10, 10, 10], lookat=[0, 0, 0])
end

"""Extrae datos de un corte 2D del espacio 3D"""
function extract_slice_data(brain_space, dimension, index)
    # Esta función extraería datos de un corte específico para el mapa de calor
    # Implementación dependería de cómo se almacenan los datos en BrainSpace
    
    # Dimensión de la cuadrícula 2D (simplificada)
    grid_size = (100, 100)
    
    # Matriz para el mapa de calor 
    slice_data = zeros(grid_size)
    
    # Obtener posiciones y activaciones de neuronas
    positions = get_neuron_positions(brain_space)
    activities = get_neuron_activities(brain_space)
    
    # Mapear neuronas al corte 2D según la dimensión
    dim_index = dimension == :x ? 1 : (dimension == :y ? 2 : 3)
    
    for (pos, act) in zip(positions, activities)
        # Verificar si la neurona está en el corte
        if round(Int, pos[dim_index]) == index
            # Determinar índices 2D según la dimensión del corte
            i, j = if dim_index == 1
                round.(Int, (pos[2], pos[3])) .+ (grid_size .÷ 2)
            elseif dim_index == 2
                round.(Int, (pos[1], pos[3])) .+ (grid_size .÷ 2)
            else
                round.(Int, (pos[1], pos[2])) .+ (grid_size .÷ 2)
            end
            
            # Asegurarse de que los índices están dentro de límites
            if 1 <= i <= grid_size[1] && 1 <= j <= grid_size[2]
                # Acumular actividad (promediar si hay múltiples neuronas)
                slice_data[i, j] = max(slice_data[i, j], act)
            end
        end
    end
    
    return slice_data
end

"""Añade controles interactivos a una visualización"""
function add_interactive_controls!(fig, control_panel, ax_main, scatter_plot, brain_space)
    # Implementar controles como sliders, botones, etc.
    # Por ejemplo, un slider para umbral de actividad
    threshold_slider = Slider(control_panel[1, 1], range=0:0.01:1, startvalue=0.2)
    Label(control_panel[1, 2], "Umbral de actividad")
    
    # Botón para alternar visualización de conexiones
    show_connections = Button(control_panel[2, 1:2], label="Mostrar conexiones")
    
    # Función para actualizar visualización según controles
    on(threshold_slider.value) do threshold
        # Actualizar visualización según umbral
        update_threshold_visualization!(ax_main, scatter_plot, brain_space, threshold)
    end
    
    # Manejar clic en botón
    on(show_connections.clicks) do n
        if n % 2 == 1  # Alternar estado
            show_connections.label = "Ocultar conexiones"
            draw_connections!(ax_main, brain_space)
        else
            show_connections.label = "Mostrar conexiones"
            clear_connections!(ax_main)
        end
    end
end

"""Actualiza visualización según umbral de actividad"""
function update_threshold_visualization!(ax, scatter_plot, brain_space, threshold)
    # Obtener datos
    activities = get_neuron_activities(brain_space)
    
    # Crear nueva máscara de visibilidad basada en el umbral
    visibility = activities .>= threshold
    
    # Actualizar puntos visibles
    scatter_plot.visible = visibility
end

"""Limpia las conexiones del eje"""
function clear_connections!(ax)
    # Eliminar todas las líneas (conexiones) del eje
    for child in ax.scene.children
        if child isa Lines
            child.visible = false
        end
    end
end

"""Configura la selección interactiva de regiones"""
function setup_region_selection!(fig, ax, brain_space)
    # Esta función implementaría la selección de regiones con el ratón
    # y actualizaría la visualización para mostrar solo esa región
    # Requiere implementación específica según las capacidades de Makie
end

"""Grafica la actividad global a lo largo del tiempo"""
function plot_global_activity!(ax_metrics, brain_space)
    # Esta función graficaría la actividad neuronal agregada
    # a lo largo del tiempo en un eje 2D separado
    
    # Por simplicidad, usamos datos de ejemplo
    times = 0:0.1:10
    activity = sin.(times) .* exp.(-times ./ 5) .+ 0.5
    
    lines!(ax_metrics, times, activity, color=:blue)
    ax_metrics.title = "Actividad neuronal global"
end

"""Convierte una visualización a HTML interactivo"""
function convert_to_interactive_html(visualization, include_interactivity)
    # Esta función convertiría una visualización de Makie a un formato HTML
    # con JavaScript para interactividad si está habilitada
    
    # Implementación simplificada:
    html_header = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Visualización del Espacio Neuronal</title>
        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    """
    
    if include_interactivity
        html_header *= """
        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
        <style>
            body { margin: 0; overflow: hidden; }
            canvas { display: block; }
        </style>
        """
    end
    
    html_header *= """
    </head>
    <body>
    """
    
    # Aquí iría la conversión real de la visualización a código JavaScript/Three.js
    
    html_footer = """
    </body>
    </html>
    """
    
    # Simplificado: en una implementación real, se generaría JS para reproducir la visualización
    return html_header * "<div>Visualización exportada</div>" * html_footer
end

end # module