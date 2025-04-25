module TrainingMonitor

using Makie
using GLMakie
using Colors
using Statistics
using Dates
using ..BrainSpace, ..SpatialField

export create_training_dashboard, track_loss_metrics, visualize_training_progress,
       region_specialization_tracking, track_gradient_flow, save_training_checkpoint,
       learning_rate_analysis, export_training_report, training_animation,
       compare_training_runs

"""
    create_training_dashboard(; initial_metrics=Dict())

Crea un panel de control interactivo para monitorizar el entrenamiento en tiempo real.

# Argumentos
- `initial_metrics=Dict()`: Métricas iniciales para mostrar en el dashboard

# Retorna
- `dashboard`: Objeto dashboard para actualización en tiempo real
- `update_function`: Función para actualizar el dashboard con nuevas métricas
"""
function create_training_dashboard(; initial_metrics=Dict())
    # Crear figura principal
    fig = Figure(resolution=(1400, 900), backgroundcolor=:white)
    
    # Panel de pérdida
    loss_panel = fig[1, 1]
    loss_ax = Axis(loss_panel, 
                  xlabel="Iteración", 
                  ylabel="Pérdida",
                  title="Evolución de Pérdida")
    
    # Panel de precisión u otras métricas
    metrics_panel = fig[1, 2]
    metrics_ax = Axis(metrics_panel,
                     xlabel="Iteración",
                     ylabel="Valor",
                     title="Métricas de Evaluación")
    
    # Panel de actividad neuronal
    activity_panel = fig[2, 1]
    activity_ax = Axis(activity_panel,
                      xlabel="Región", 
                      ylabel="Actividad Media",
                      title="Actividad por Región")
    
    # Panel de distribución de pesos
    weights_panel = fig[2, 2]
    weights_ax = Axis(weights_panel,
                     xlabel="Valor de Peso",
                     ylabel="Frecuencia",
                     title="Distribución de Pesos")
    
    # Inicializar con datos vacíos
    iterations = Observable(Int[])
    loss_values = Observable(Float64[])
    accuracy_values = Observable(Float64[])
    
    # Líneas para métricas principales
    loss_line = lines!(loss_ax, iterations, loss_values, color=:red, linewidth=2)
    
    # Si hay métricas iniciales, actualizar plots
    if haskey(initial_metrics, "loss") && !isempty(initial_metrics["loss"])
        iterations[] = 1:length(initial_metrics["loss"])
        loss_values[] = initial_metrics["loss"]
    end
    
    if haskey(initial_metrics, "accuracy") && !isempty(initial_metrics["accuracy"])
        accuracy_values[] = initial_metrics["accuracy"]
        accuracy_line = lines!(metrics_ax, iterations, accuracy_values, color=:blue, linewidth=2)
        
        # Añadir leyenda
        Legend(metrics_panel[1, 1, TopRight()], [accuracy_line], ["Precisión"])
    end
    
    # Configurar límites y otras propiedades visuales
    loss_ax.xgridvisible = false
    metrics_ax.xgridvisible = false
    
    # Status bar en la parte inferior
    status_bar = fig[3, 1:2]
    status_text = Observable("Inicializando monitoreo de entrenamiento...")
    Label(status_bar, status_text, tellwidth=false)
    
    # Función para actualizar el dashboard
    function update_dashboard(new_metrics)
        # Actualizar iteraciones
        if haskey(new_metrics, "iteration")
            push!(iterations[], new_metrics["iteration"])
            iterations[] = iterations[]
        end
        
        # Actualizar pérdida
        if haskey(new_metrics, "loss")
            push!(loss_values[], new_metrics["loss"])
            loss_values[] = loss_values[]
            
            # Actualizar límites del eje y si es necesario
            autolimits!(loss_ax)
        end
        
        # Actualizar precisión
        if haskey(new_metrics, "accuracy")
            push!(accuracy_values[], new_metrics["accuracy"])
            accuracy_values[] = accuracy_values[]
        end
        
        # Actualizar texto de estado
        if haskey(new_metrics, "status")
            status_text[] = new_metrics["status"]
        else
            status_text[] = "Iteración: $(length(iterations[])) | " *
                           "Última pérdida: $(round(loss_values[][end], digits=5))"
        end
        
        # Actualizar otros paneles específicos
        if haskey(new_metrics, "weights_histogram")
            empty!(weights_ax)
            hist!(weights_ax, new_metrics["weights_histogram"], bins=50, color=:purple)
        end
        
        if haskey(new_metrics, "region_activity")
            empty!(activity_ax)
            barplot!(activity_ax, 1:length(new_metrics["region_activity"]), 
                    new_metrics["region_activity"], color=:orange)
        end
    end
    
    # Crear y retornar el dashboard
    dashboard = Dict(
        "figure" => fig,
        "axes" => Dict(
            "loss" => loss_ax,
            "metrics" => metrics_ax,
            "activity" => activity_ax,
            "weights" => weights_ax
        ),
        "data" => Dict(
            "iterations" => iterations,
            "loss" => loss_values,
            "accuracy" => accuracy_values
        )
    )
    
    return dashboard, update_dashboard
end

"""
    track_loss_metrics(loss_history, validation_metrics=nothing; 
                      window_size=10, save_path=nothing)

Visualiza y analiza el historial de pérdida y otras métricas durante el entrenamiento.

# Argumentos
- `loss_history`: Vector con valores de pérdida por iteración
- `validation_metrics=nothing`: Diccionario con métricas de validación
- `window_size=10`: Tamaño de ventana para promedio móvil
- `save_path=nothing`: Ruta para guardar la visualización

# Retorna
- `fig`: Figura con la visualización
- `analysis_results`: Diccionario con estadísticas y análisis
"""
function track_loss_metrics(loss_history, validation_metrics=nothing; 
                           window_size=10, save_path=nothing)
    # Crear figura
    fig = Figure(resolution=(1200, 800))
    
    # Panel principal: evolución de pérdida
    loss_panel = fig[1, 1]
    loss_ax = Axis(loss_panel, 
                  xlabel="Iteración", 
                  ylabel="Pérdida",
                  title="Evolución de Pérdida durante Entrenamiento")
    
    # Calcular iteraciones
    iterations = 1:length(loss_history)
    
    # Trazar pérdida de entrenamiento
    lines!(loss_ax, iterations, loss_history, 
           color=:blue, linewidth=2, label="Pérdida de entrenamiento")
    
    # Calcular y trazar promedio móvil
    if length(loss_history) >= window_size
        moving_avg = moving_average(loss_history, window_size)
        moving_avg_iterations = window_size:length(loss_history)
        
        lines!(loss_ax, moving_avg_iterations, moving_avg, 
               color=:red, linewidth=2, label="Promedio móvil (n=$window_size)")
    end
    
    # Añadir pérdida de validación si está disponible
    if validation_metrics !== nothing && haskey(validation_metrics, "loss")
        val_loss = validation_metrics["loss"]
        val_iterations = validation_metrics["iterations"]
        
        scatter!(loss_ax, val_iterations, val_loss, 
                color=:green, markersize=8, label="Pérdida de validación")
    end
    
    # Añadir leyenda
    axislegend(loss_ax, position=:topright)
    
    # Panel para análisis de pérdida
    analysis_panel = fig[1, 2]
    analysis_ax = Axis(analysis_panel,
                      title="Análisis de Tendencia de Pérdida")
    
    # Análisis de tendencia: primera y segunda derivada
    if length(loss_history) > 2
        # Calcular primera derivada (tasa de cambio)
        first_derivative = diff(loss_history)
        
        # Segunda derivada (aceleración/desaceleración)
        second_derivative = diff(first_derivative)
        
        # Trazar primera derivada
        lines!(analysis_ax, 2:length(loss_history), first_derivative, 
               color=:purple, linewidth=2, label="Primera derivada")
        
        # Trazar segunda derivada
        lines!(analysis_ax, 3:length(loss_history), second_derivative, 
               color=:orange, linewidth=2, label="Segunda derivada")
        
        # Línea de referencia en y=0
        hlines!(analysis_ax, 0, color=:black, linestyle=:dash)
        
        axislegend(analysis_ax)
    end
    
    # Panel para métricas adicionales
    if validation_metrics !== nothing && length(validation_metrics) > 2
        metrics_panel = fig[2, 1:2]
        metrics_ax = Axis(metrics_panel,
                         xlabel="Iteración",
                         ylabel="Valor",
                         title="Métricas de Evaluación")
        
        # Visualizar todas las métricas excepto loss e iterations
        colors = [:green, :purple, :orange, :brown, :pink]
        color_idx = 1
        plotted_metrics = []
        
        for (key, values) in validation_metrics
            if key ∉ ["loss", "iterations"] && !isempty(values)
                metric_line = lines!(metrics_ax, validation_metrics["iterations"], values,
                                    color=colors[color_idx], linewidth=2, label=key)
                push!(plotted_metrics, metric_line)
                color_idx = (color_idx % length(colors)) + 1
            end
        end
        
        # Añadir leyenda si hay métricas adicionales
        if !isempty(plotted_metrics)
            axislegend(metrics_ax)
        end
    end
    
    # Calcular estadísticas
    analysis_results = Dict(
        "min_loss" => minimum(loss_history),
        "min_loss_iteration" => argmin(loss_history),
        "final_loss" => loss_history[end],
        "loss_reduction" => loss_history[1] - loss_history[end],
        "loss_reduction_percent" => 100 * (loss_history[1] - loss_history[end]) / loss_history[1]
    )
    
    # Guardar imagen si se proporciona ruta
    if save_path !== nothing
        save(save_path, fig)
    end
    
    return fig, analysis_results
end

"""
    visualize_training_progress(brain_space::BrainSpace, metrics_history; 
                              region_focus=nothing,
                              time_points=5)

Visualiza el progreso del entrenamiento con múltiples perspectivas del espacio cerebral.

# Argumentos
- `brain_space::BrainSpace`: Estado actual del espacio cerebral
- `metrics_history`: Historial de métricas durante el entrenamiento
- `region_focus=nothing`: Región específica para enfoque detallado
- `time_points=5`: Número de puntos temporales a visualizar

# Retorna
- `fig`: Figura con visualización de progreso
"""
function visualize_training_progress(brain_space::Brain_Space, metrics_history; 
                                   region_focus=nothing,
                                   time_points=5)
    # Crear figura
    fig = Figure(resolution=(1400, 1000))
    
    # Panel principal: métricas a lo largo del tiempo
    metrics_panel = fig[1, 1:2]
    metrics_ax = Axis(metrics_panel,
                     xlabel="Iteración",
                     ylabel="Valor",
                     title="Evolución de Métricas durante Entrenamiento")
    
    # Visualizar métricas principales
    iterations = 1:length(metrics_history["loss"])
    
    lines!(metrics_ax, iterations, metrics_history["loss"], 
           color=:red, linewidth=2, label="Pérdida")
    
    if haskey(metrics_history, "accuracy")
        lines!(metrics_ax, iterations, metrics_history["accuracy"], 
               color=:blue, linewidth=2, label="Precisión")
    end
    
    # Añadir otras métricas disponibles
    metrics_colors = Dict(
        "val_loss" => :darkred,
        "val_accuracy" => :darkblue,
        "learning_rate" => :green,
        "gradient_norm" => :purple
    )
    
    for (metric, color) in metrics_colors
        if haskey(metrics_history, metric) && !isempty(metrics_history[metric])
            lines!(metrics_ax, iterations, metrics_history[metric], 
                   color=color, linewidth=2, label=metric)
        end
    end
    
    axislegend(metrics_ax, position=:outertopright)
    
    # Panel para distribución de activaciones en diferentes puntos temporales
    if time_points > 1
        time_indices = round.(Int, range(1, length(iterations), length=time_points))
        
        # Generar y mostrar histogramas de activación
        activation_panel = fig[2, 1]
        activation_ax = Axis(activation_panel,
                            xlabel="Activación",
                            ylabel="Frecuencia",
                            title="Evolución de Distribución de Activaciones")
        
        colors = cgrad(:viridis, time_points)
        
        for (i, time_idx) in enumerate(time_indices)
            if haskey(metrics_history, "activation_histograms") && 
               length(metrics_history["activation_histograms"]) >= time_idx
                
                hist_data = metrics_history["activation_histograms"][time_idx]
                label = "Iteración $(iterations[time_idx])"
                
                density!(activation_ax, hist_data, color=(colors[i], 0.7), label=label)
            end
        end
        
        axislegend(activation_ax)
    end
    
    # Panel para visualización de pesos sinápticos a lo largo del tiempo
    weights_panel = fig[2, 2]
    weights_ax = Axis(weights_panel,
                     xlabel="Peso",
                     ylabel="Frecuencia",
                     title="Evolución de Distribución de Pesos")
    
    # Similar a activaciones, mostrar evolución de pesos
    if haskey(metrics_history, "weight_histograms")
        time_indices = round.(Int, range(1, length(iterations), length=time_points))
        colors = cgrad(:plasma, time_points)
        
        for (i, time_idx) in enumerate(time_indices)
            if length(metrics_history["weight_histograms"]) >= time_idx
                weight_data = metrics_history["weight_histograms"][time_idx]
                label = "Iteración $(iterations[time_idx])"
                
                density!(weights_ax, weight_data, color=(colors[i], 0.7), label=label)
            end
        end
        
        axislegend(weights_ax)
    end
    
    # Si hay un enfoque en una región específica, mostrar detalles
    if region_focus !== nothing
        region_panel = fig[3, 1:2]
        region_ax = Axis3(region_panel,
                         title="Evolución de Región: $region_focus")
        
        # Aquí se implementaría la visualización específica de la región
        # Dependiendo de cómo se estructuren los datos de región en RNTA
    end
    
    return fig
end

"""
    region_specialization_tracking(brain_space::BrainSpace, 
                                 training_history;
                                 num_regions=5)

Rastrea y visualiza cómo diferentes regiones del espacio cerebral se especializan
durante el entrenamiento.

# Argumentos
- `brain_space::BrainSpace`: Estado actual del espacio cerebral
- `training_history`: Historial del proceso de entrenamiento
- `num_regions=5`: Número de regiones a mostrar

# Retorna
- `fig`: Figura con visualización de especialización
- `region_data`: Datos de especialización por región
"""
function region_specialization_tracking(brain_space::Brain_Space, 
                                      training_history;
                                      num_regions=5)
    # Crear figura
    fig = Figure(resolution=(1200, 900))
    
    # Panel para evolución de especialización
    spec_panel = fig[1, 1:2]
    spec_ax = Axis(spec_panel,
                  xlabel="Iteración",
                  ylabel="Índice de Especialización",
                  title="Evolución de Especialización por Región")
    
    # Obtener datos de especialización de regiones a lo largo del tiempo
    if haskey(training_history, "region_specialization")
        specialization_data = training_history["region_specialization"]
        iterations = training_history["iterations"]
        
        # Determinar las regiones más especializadas al final
        final_specialization = specialization_data[end]
        top_regions = partialsortperm(final_specialization, 1:min(num_regions, length(final_specialization)), rev=true)
        
        # Colores para las diferentes regiones
        colors = distinguishable_colors(length(top_regions), [RGB(1,1,1), RGB(0,0,0)])
        
        # Trazar evolución para las regiones principales
        region_lines = []
        
        for (i, region_idx) in enumerate(top_regions)
            # Extraer datos de especialización para esta región
            region_spec = [spec[region_idx] for spec in specialization_data]
            
            # Trazar línea
            line = lines!(spec_ax, iterations, region_spec, 
                         color=colors[i], linewidth=2, 
                         label="Región $region_idx")
            
            push!(region_lines, line)
        end
        
        # Añadir leyenda
        Legend(spec_panel[1, 1, TopRight()], region_lines, ["Región $i" for i in top_regions])
    end
    
    # Panel para visualización de patrones de activación en las regiones
    activation_panel = fig[2, 1]
    activation_ax = Axis(activation_panel,
                        xlabel="Región",
                        ylabel="Activación Media",
                        title="Activación por Región (Estado Actual)")
    
    # Crear gráfico de barras para activación actual por región
    current_activations = get_region_activations(brain_space, num_regions)
    barplot!(activation_ax, 1:length(current_activations), current_activations, color=:orange)
    
    # Panel para visualización 3D de regiones especializadas
    region_panel = fig[2, 2]
    region_ax = Axis3(region_panel,
                     title="Regiones Especializadas (Vista 3D)")
    
    # Visualizar espacio cerebral con regiones coloreadas por especialización
    visualize_specialized_regions!(region_ax, brain_space, num_regions)
    
    # Recopilar datos de especialización por región
    region_data = analyze_region_specialization(brain_space, training_history)
    
    return fig, region_data
end

"""
    track_gradient_flow(brain_space::BrainSpace, gradient_history; 
                      layers_to_track=nothing)

Visualiza el flujo de gradientes a través de la red durante el entrenamiento.

# Argumentos
- `brain_space::BrainSpace`: Estado actual del espacio cerebral
- `gradient_history`: Historial de gradientes durante entrenamiento
- `layers_to_track=nothing`: Capas específicas a rastrear (todas si es nothing)

# Retorna
- `fig`: Figura con visualización de flujo de gradientes
"""
function track_gradient_flow(brain_space::Brain_Space, gradient_history; 
                           layers_to_track=nothing)
    # Crear figura
    fig = Figure(resolution=(1200, 800))
    
    # Panel para magnitud de gradiente por capa
    grad_panel = fig[1, 1:2]
    grad_ax = Axis(grad_panel,
                  xlabel="Iteración",
                  ylabel="Magnitud de Gradiente (log)",
                  title="Flujo de Gradientes por Capa",
                  yscale=log10)
    
    # Obtener iteraciones
    iterations = gradient_history["iterations"]
    
    # Determinar capas a visualizar
    all_layers = collect(keys(gradient_history))
    layers = filter(l -> l !== "iterations", all_layers)
    
    if layers_to_track !== nothing
        layers = filter(l -> l in layers_to_track, layers)
    end
    
    # Colores para diferentes capas
    colors = distinguishable_colors(length(layers), [RGB(1,1,1), RGB(0,0,0)])
    
    # Trazar magnitud de gradiente para cada capa
    layer_lines = []
    
    for (i, layer) in enumerate(layers)
        if haskey(gradient_history, layer)
            # Extraer magnitudes de gradiente
            magnitudes = gradient_history[layer]
            
            # Trazar línea
            line = lines!(grad_ax, iterations, magnitudes, 
                         color=colors[i], linewidth=2)
            
            push!(layer_lines, line)
        end
    end
    
    # Añadir leyenda
    Legend(grad_panel[1, 1, TopRight()], layer_lines, layers)
    
    # Panel para análisis de problemas de desvanecimiento/explosión
    analysis_panel = fig[2, 1]
    analysis_ax = Axis(analysis_panel,
                      xlabel="Capa",
                      ylabel="Gradiente Relativo",
                      title="Análisis de Desvanecimiento/Explosión de Gradiente")
    
    # Analizar último estado de gradientes
    if !isempty(iterations)
        last_iter = iterations[end]
        
        # Obtener gradientes relativos para la última iteración
        relative_gradients = []
        
        for layer in layers
            if haskey(gradient_history, layer)
                push!(relative_gradients, gradient_history[layer][end])
            end
        end
        
        # Normalizar respecto al gradiente máximo
        if !isempty(relative_gradients) && maximum(relative_gradients) > 0
            relative_gradients = relative_gradients ./ maximum(relative_gradients)
        end
        
        # Visualizar gradientes relativos
        barplot!(analysis_ax, 1:length(relative_gradients), relative_gradients, 
                color=:purple)
        
        # Añadir etiquetas de capa
        analysis_ax.xticks = (1:length(layers), layers)
        analysis_ax.xticklabelrotation = π/4
    end
    
    # Panel para histograma de distribución de gradientes
    hist_panel = fig[2, 2]
    hist_ax = Axis(hist_panel,
                  xlabel="Magnitud de Gradiente",
                  ylabel="Frecuencia",
                  title="Distribución de Gradientes (Última Iteración)")
    
    # Recopilar todos los valores de gradiente de la última iteración
    if haskey(gradient_history, "gradient_values") && !isempty(gradient_history["gradient_values"])
        last_gradients = gradient_history["gradient_values"][end]
        
        # Crear histograma
        hist!(hist_ax, last_gradients, bins=50, color=:blue, alpha=0.7)
        
        # Añadir líneas para estadísticas
        mean_val = mean(last_gradients)
        median_val = median(last_gradients)
        
        vlines!(hist_ax, mean_val, color=:red, linewidth=2, label="Media")
        vlines!(hist_ax, median_val, color=:green, linewidth=2, label="Mediana")
        
        axislegend(hist_ax)
    end
    
    return fig
end

"""
    save_training_checkpoint(brain_space::BrainSpace, 
                           metrics, 
                           checkpoint_dir::String;
                           include_visualization::Bool=true)

Guarda un punto de control del entrenamiento con métricas y estado del modelo.

# Argumentos
- `brain_space::BrainSpace`: Estado actual del espacio cerebral
- `metrics`: Métricas actuales del entrenamiento
- `checkpoint_dir::String`: Directorio donde guardar el checkpoint
- `include_visualization::Bool=true`: Si se incluye visualización del estado actual

# Retorna
- `checkpoint_path`: Ruta al archivo de checkpoint guardado
"""
function save_training_checkpoint(brain_space::Brain_Space, 
                                metrics, 
                                checkpoint_dir::String;
                                include_visualization::Bool=true)
    # Crear directorio si no existe
    if !isdir(checkpoint_dir)
        mkdir(checkpoint_dir)
    end
    
    # Generar nombre de archivo con timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    base_filename = "checkpoint_$(timestamp)"
    
    # Guardar estado del modelo
    model_path = joinpath(checkpoint_dir, "$(base_filename)_model.jld2")
    save_brain_space(brain_space, model_path)
    
    # Guardar métricas en formato JSON
    metrics_path = joinpath(checkpoint_dir, "$(base_filename)_metrics.json")
    save_metrics(metrics, metrics_path)
    
    # Crear y guardar visualización si está habilitado
    if include_visualization
        viz_path = joinpath(checkpoint_dir, "$(base_filename)_state.png")
        
        # Crear visualización del estado actual
        fig = Figure(resolution=(1200, 900))
        
        # Visualizar espacio cerebral (implementación depende de la estructura de BrainSpace)
        # Esta sería una versión simplificada
        ax = Axis3(fig[1, 1:2], aspect=:data, title="Estado del Modelo")
        visualize_brain_state!(ax, brain_space)
        
        # Añadir panel con métricas clave
        metrics_panel = fig[2, 1:2]
        
        # Extraer y mostrar métricas clave
        metrics_text = """
        Checkpoint: $(timestamp)
        Iteración: $(get(metrics, "iteration", "N/A"))
        Pérdida: $(get(metrics, "loss", "N/A"))
        Precisión: $(get(metrics, "accuracy", "N/A"))
        """
        
        Label(metrics_panel, metrics_text, fontsize=14, tellwidth=false)
        
        # Guardar visualización
        save(viz_path, fig)
    end
    
    # Ruta al archivo principal de checkpoint
    checkpoint_path = joinpath(checkpoint_dir, "$(base_filename)_model.jld2")
    
    return checkpoint_path
end

"""
    learning_rate_analysis(loss_history, lr_history; 
                         window_size=10)

Analiza la relación entre tasa de aprendizaje y pérdida para optimizar el entrenamiento.

# Argumentos
- `loss_history`: Vector con valores de pérdida
- `lr_history`: Vector con valores de tasa de aprendizaje
- `window_size=10`: Tamaño de ventana para suavizado

# Retorna
- `fig`: Figura con análisis de tasa de aprendizaje
- `optimal_lr`: Tasa de aprendizaje óptima estimada
"""
function learning_rate_analysis(loss_history, lr_history; 
                              window_size=10)
    # Verificar datos de entrada
    if length(loss_history) != length(lr_history)
        error("El historial de pérdida y tasa de aprendizaje deben tener la misma longitud")
    end
    
    # Crear figura
    fig = Figure(resolution=(1200, 800))
    
    # Panel para pérdida vs. tasa de aprendizaje
    lr_panel = fig[1, 1:2]
    lr_ax = Axis(lr_panel,
                xlabel="Tasa de Aprendizaje (log)",
                ylabel="Pérdida",
                title="Pérdida vs. Tasa de Aprendizaje",
                xscale=log10)
    
    # Graficar pérdida vs. tasa de aprendizaje
    scatter!(lr_ax, lr_history, loss_history, color=:blue, markersize=4, alpha=0.6)
    
    # Si hay suficientes puntos, calcular promedio móvil
    if length(loss_history) >= window_size
        # Ordenar por tasa de aprendizaje
        sorted_indices = sortperm(lr_history)
        sorted_lr = lr_history[sorted_indices]
        sorted_loss = loss_history[sorted_indices]
        
        # Calcular promedio móvil
        smoothed_loss = moving_average(sorted_loss, window_size)
        smoothed_lr = sorted_lr[window_size:end]
        
        # Trazar línea suavizada
        lines!(lr_ax, smoothed_lr, smoothed_loss, 
               color=:red, linewidth=2, label="Tendencia (ventana=$window_size)")
        
        axislegend(lr_ax)
    end
    
    # Panel para análisis de tasa de cambio
    derivative_panel = fig[2, 1]
    derivative_ax = Axis(derivative_panel,
                        xlabel="Tasa de Aprendizaje (log)",
                        ylabel="Tasa de Cambio de Pérdida",
                        title="Derivada de Pérdida respecto a LR",
                        xscale=log10)
    
    # Calcular y visualizar derivada (tasa de cambio)
    if length(loss_history) >= 3 && length(unique(lr_history)) >= 3
        # Ordenar por tasa de aprendizaje
        sorted_indices = sortperm(lr_history)
        sorted_lr = lr_history[sorted_indices]
        sorted_loss = loss_history[sorted_indices]
        
        # Calcular derivada aproximada
        d_loss = diff(sorted_loss) ./ diff(sorted_lr)
        d_lr = sorted_lr[1:end-1] .+ (diff(sorted_lr) ./ 2)  # Puntos medios para x
        
        # Visualizar derivada
        scatter!(derivative_ax, d_lr, d_loss, color=:purple, markersize=4)
        
        # Suavizar si hay suficientes puntos
        if length(d_loss) >= window_size
            # Reordenar por LR
            smooth_indices = sortperm(d_lr)
            smooth_d_lr = d_lr[smooth_indices]
            smooth_d_loss = d_loss[smooth_indices]
            
            # Calcular promedio móvil
            smoothed_d_loss = moving_average(smooth_d_loss, window_size)
            smoothed_d_lr = smooth_d_lr[window_size:end]
            
            # Trazar línea suavizada
            lines!(derivative_ax, smoothed_d_lr, smoothed_d_loss, 
                   color=:darkred, linewidth=2)
            
            # Línea de referencia en y=0
            hlines!(derivative_ax, 0, color=:black, linestyle=:dash)
        end
    end
    
    # Panel para recomendación
    recommendation_panel = fig[2, 2]
    
    # Estimar tasa de aprendizaje óptima
    optimal_lr = estimate_optimal_lr(loss_history, lr_history)
    
    # Mostrar recomendación
    optimal_text = """
    Análisis de Tasa de Aprendizaje:
    
    Mínima pérdida: $(minimum(loss_history))
    LR en mínima pérdida: $(lr_history[argmin(loss_history)])
    
    LR óptima estimada: $(optimal_lr)
    
    Recomendación:
    $(get_lr_recommendation(loss_history, lr_history, optimal_lr))
    """
    
    Label(recommendation_panel, optimal_text, tellwidth=false, fontsize=14)
    
    return fig, optimal_lr
end

"""
    export_training_report(brain_space::BrainSpace, 
                         training_history, 
                         output_path::String;
                         include_visualizations::Bool=true)

Genera y exporta un informe completo del proceso de entrenamiento.

# Argumentos
- `brain_space::BrainSpace`: Estado final del espacio cerebral
- `training_history`: Historial completo del entrenamiento
- `output_path::String`: Ruta donde guardar el informe
- `include_visualizations::Bool=true`: Si se incluyen visualizaciones

# Retorna
- `report_path`: Ruta al informe generado
"""
function export_training_report(brain_space::Brain_Space, 
                              training_history, 
                              output_path::String;
                              include_visualizations::Bool=true)
    # Crear directorio si no existe
    report_dir = dirname(output_path)
    if !isdir(report_dir)
        mkpath(report_dir)
    end
    
    # Generar informe en formato Markdown
    open(output_path, "w") do io
        # Encabezado
        write(io, "# Informe de Entrenamiento RNTA\n\n")
        write(io, "Fecha: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS\"))\n\n"))
        
        # Resumen general
        write(io,  Resumen de Entrenamiento\n\n")
        
        # Métricas clave
        if haskey(training_history, "loss")
            initial_loss = first(training_history["loss"])
            final_loss = last(training_history["loss"])
            best_loss = minimum(training_history["loss"])
            best_iter = argmin(training_history["loss"])
            
            write(io, "- **Pérdida inicial:** $(initial_loss)\n")
            write(io, "- **Pérdida final:** $(final_loss)\n")
            write(io, "- **Mejor pérdida:** $(best_loss) (iteración $(best_iter))\n")
            write(io, "- **Reducción de pérdida:** $(100*(initial_loss-final_loss)/initial_loss)%\n")
            write(io, "- **Iteraciones totales:** $(length(training_history["loss"]))\n")
        end
        
        if haskey(training_history, "accuracy")
            final_accuracy = last(training_history["accuracy"])
            best_accuracy = maximum(training_history["accuracy"])
            best_acc_iter = argmax(training_history["accuracy"])
            
            write(io, "- **Precisión final:** $(final_accuracy * 100)%\n")
            write(io, "- **Mejor precisión:** $(best_accuracy * 100)% (iteración $(best_acc_iter))\n")
        end
        
        # Estadísticas del modelo
        write(io, "\n## Estadísticas del Modelo\n\n")
        
        num_neurons = length(brain_space.neurons)
        num_connections = length(brain_space.connections)
        
        write(io, "- **Neuronas:** $(num_neurons)\n")
        write(io, "- **Conexiones:** $(num_connections)\n")
        write(io, "- **Densidad de conexiones:** $(num_connections / (num_neurons * (num_neurons-1)))\n")
        
        # Análisis de regiones
        write(io, "\n## Análisis de Regiones\n\n")
        
        # Identificar regiones más activas
        region_activities = get_region_activations(brain_space)
        top_regions = partialsortperm(region_activities, 1:min(5, length(region_activities)), rev=true)
        
        write(io, "### Regiones Más Activas\n\n")
        for (i, region_id) in enumerate(top_regions)
            write(io, "$(i). Región $(region_id): $(region_activities[region_id])\n")
        end
        
        # Análisis de especialización
        if haskey(training_history, "region_specialization") && 
           !isempty(training_history["region_specialization"])
            
            write(io, "\n### Evolución de Especialización\n\n")
            
            initial_spec = first(training_history["region_specialization"])
            final_spec = last(training_history["region_specialization"])
            
            # Regiones con mayor cambio
            spec_changes = final_spec .- initial_spec
            top_changed = partialsortperm(abs.(spec_changes), 1:min(5, length(spec_changes)), rev=true)
            
            write(io, "#### Regiones con Mayor Evolución\n\n")
            for (i, region_id) in enumerate(top_changed)
                change = spec_changes[region_id]
                direction = change > 0 ? "incremento" : "decremento"
                write(io, "$(i). Región $(region_id): $(abs(change)) ($(direction))\n")
            end
        end
        
        # Visualizaciones
        if include_visualizations
            write(io, "\n## Visualizaciones\n\n")
            
            # Guardar y enlazar visualizaciones
            viz_dir = joinpath(report_dir, "visualizations")
            if !isdir(viz_dir)
                mkdir(viz_dir)
            end
            
            # Pérdida y métricas
            if haskey(training_history, "loss")
                loss_fig, _ = track_loss_metrics(training_history["loss"])
                loss_path = joinpath(viz_dir, "loss_metrics.png")
                save(loss_path, loss_fig)
                
                rel_path = relpath(loss_path, report_dir)
                write(io, "### Evolución de Pérdida\n\n")
                write(io, "![Evolución de Pérdida]($(rel_path))\n\n")
            end
            
            # Actividad neuronal
            activity_fig = visualize_activity_summary(brain_space)
            activity_path = joinpath(viz_dir, "activity_summary.png")
            save(activity_path, activity_fig)
            
            rel_path = relpath(activity_path, report_dir)
            write(io, "### Resumen de Actividad Neuronal\n\n")
            write(io, "![Actividad Neuronal]($(rel_path))\n\n")
            
            # Otras visualizaciones específicas
        end
        
        # Conclusiones y recomendaciones
        write(io, "\n## Conclusiones y Recomendaciones\n\n")
        
        # Analizar tendencias y generar recomendaciones
        conclusions = generate_training_conclusions(training_history)
        
        for (category, items) in conclusions
            write(io, "### $(category)\n\n")
            for item in items
                write(io, "- $(item)\n")
            end
            write(io, "\n")
        end
    end
    
    return output_path
end

"""
    training_animation(training_snapshots; 
                      output_path::String="training_animation.mp4",
                      fps=10)

Crea una animación que muestra la evolución del espacio cerebral durante el entrenamiento.

# Argumentos
- `training_snapshots`: Vector de estados del espacio cerebral en diferentes puntos
- `output_path::String="training_animation.mp4"`: Ruta donde guardar la animación
- `fps=10`: Fotogramas por segundo para la animación

# Retorna
- `output_path`: Ruta a la animación generada
"""
function training_animation(training_snapshots; 
                           output_path::String="training_animation.mp4",
                           fps=10)
    # Crear figura base
    fig = Figure(resolution=(1200, 800))
    
    # Panel principal: visualización 3D del espacio cerebral
    brain_panel = fig[1:2, 1]
    brain_ax = Axis3(brain_panel, aspect=:data)
    
    # Panel secundario: métricas
    metrics_panel = fig[1, 2]
    metrics_ax = Axis(metrics_panel,
                     xlabel="Iteración",
                     ylabel="Valor",
                     title="Evolución de Métricas")
    
    # Panel para distribución de actividad
    activity_panel = fig[2, 2]
    activity_ax = Axis(activity_panel,
                      xlabel="Activación",
                      ylabel="Frecuencia",
                      title="Distribución de Activación")
    
    # Inicializar datos para animación
    iterations = Observable(Int[])
    loss_values = Observable(Float64[])
    
    # Línea de pérdida
    loss_line = lines!(metrics_ax, iterations, loss_values, color=:red, linewidth=2)
    
    # Crear animación
    record(fig, output_path, 1:length(training_snapshots); framerate=fps) do i
        # Actualizar espacio cerebral
        current_snapshot = training_snapshots[i]
        
        # Limpiar visualización anterior
        empty!(brain_ax)
        empty!(activity_ax)
        
        # Visualizar espacio cerebral actual
        visualize_brain_state!(brain_ax, current_snapshot)
        
        # Actualizar métricas
        push!(iterations[], i)
        push!(loss_values[], current_snapshot.metrics["loss"])
        
        iterations[] = iterations[]
        loss_values[] = loss_values[]
        
        # Visualizar distribución de activación
        activations = get_all_activations(current_snapshot)
        hist!(activity_ax, activations, bins=30, color=:blue)
        
        # Actualizar título con iteración actual
        brain_ax.title = "Espacio Cerebral - Iteración $(i) de $(length(training_snapshots))"
    end
    
    return output_path
end

"""
    compare_training_runs(run_results; 
                         metrics=["loss", "accuracy"],
                         normalize::Bool=true)

Compara resultados de múltiples ejecuciones de entrenamiento para análisis de rendimiento.

# Argumentos
- `run_results`: Diccionario con resultados de diferentes ejecuciones
- `metrics=["loss", "accuracy"]`: Métricas a comparar
- `normalize::Bool=true`: Si se normalizan las métricas para mejor comparación

# Retorna
- `fig`: Figura con comparación
- `comparison_stats`: Estadísticas de comparación
"""
function compare_training_runs(run_results; 
                              metrics=["loss", "accuracy"],
                              normalize::Bool=true)
    # Crear figura
    fig = Figure(resolution=(1400, 800))
    
    # Crear paneles para cada métrica
    num_metrics = length(metrics)
    panels = []
    
    for (i, metric) in enumerate(metrics)
        panel = fig[1, i]
        ax = Axis(panel,
                 xlabel="Iteración",
                 ylabel=metric,
                 title="Comparación de $metric")
        
        push!(panels, (panel, ax))
    end
    
    # Colores para diferentes ejecuciones
    run_names = collect(keys(run_results))
    colors = distinguishable_colors(length(run_names), [RGB(1,1,1), RGB(0,0,0)])
    
    # Trazar métricas para cada ejecución
    comparison_stats = Dict()
    
    for (i, run_name) in enumerate(run_names)
        run_data = run_results[run_name]
        
        comparison_stats[run_name] = Dict()
        
        for (j, metric) in enumerate(metrics)
            if haskey(run_data, metric)
                # Obtener datos de la métrica
                metric_data = run_data[metric]
                iterations = 1:length(metric_data)
                
                # Normalizar si es necesario
                if normalize
                    metric_data = normalize_metric(metric_data, metric)
                end
                
                # Trazar línea
                _, ax = panels[j]
                lines!(ax, iterations, metric_data, 
                       color=colors[i], linewidth=2, label=run_name)
                
                # Calcular estadísticas
                comparison_stats[run_name][metric] = Dict(
                    "final" => metric_data[end],
                    "best" => metric == "loss" ? minimum(metric_data) : maximum(metric_data),
                    "mean" => mean(metric_data),
                    "std" => std(metric_data)
                )
            end
        end
    end
    
    # Añadir leyendas
    for (_, ax) in panels
        axislegend(ax, position=:best)
    end
    
    # Panel de resumen de estadísticas
    summary_panel = fig[2, 1:num_metrics]
    
    # Crear tabla de resumen
    summary_text = "## Resumen Comparativo\n\n"
    
    for metric in metrics
        summary_text *= "### $metric\n\n"
        summary_text *= "| Ejecución | Final | Mejor | Media | Desv. Est. |\n"
        summary_text *= "|-----------|-------|-------|-------|------------|\n"
        
        for run_name in run_names
            if haskey(comparison_stats, run_name) && haskey(comparison_stats[run_name], metric)
                stats = comparison_stats[run_name][metric]
                
                final = round(stats["final"], digits=4)
                best = round(stats["best"], digits=4)
                mean_val = round(stats["mean"], digits=4)
                std_val = round(stats["std"], digits=4)
                
                summary_text *= "| $run_name | $final | $best | $mean_val | $std_val |\n"
            end
        end
        
        summary_text *= "\n"
    end
    
    Label(summary_panel, summary_text, tellwidth=false, fontsize=12)
    
    return fig, comparison_stats
end

# Funciones auxiliares internas

"""Calcula promedio móvil de un vector"""
function moving_average(values, window_size)
    n = length(values)
    result = Vector{Float64}(undef, n - window_size + 1)
    
    for i in 1:(n - window_size + 1)
        result[i] = mean(values[i:(i + window_size - 1)])
    end
    
    return result
end

"""Guarda un espacio cerebral en archivo"""
function save_brain_space(brain_space, filepath)
    # En una implementación real, se usaría JLD2 o similar
    # Aquí solo simulamos la función
    println("Guardando espacio cerebral en: $filepath")
end

"""Guarda métricas en formato JSON"""
function save_metrics(metrics, filepath)
    # En una implementación real, se usaría JSON.jl
    # Aquí solo simulamos la función
    println("Guardando métricas en: $filepath")
end

"""Visualiza el estado actual del espacio cerebral"""
function visualize_brain_state!(ax, brain_space)
    # Esta función dependería de la implementación específica de BrainSpace
    # Aquí se muestra una versión simplificada
    
    # Obtener posiciones de neuronas
    positions = [(n.position.x, n.position.y, n.position.z) for n in brain_space.neurons]
    
    # Obtener activaciones para colores
    activations = [n.activation for n in brain_space.neurons]
    
    # Normalizar activaciones
    if !isempty(activations)
        min_act, max_act = extrema(activations)
        if min_act != max_act
            normalized_act = (activations .- min_act) ./ (max_act - min_act)
        else
            normalized_act = fill(0.5, length(activations))
        end
        
        # Visualizar neuronas como puntos coloreados por activación
        scatter!(ax, positions, color=normalized_act, colormap=:inferno, 
                markersize=10)
    end
    
    # Opcionalmente, visualizar conexiones
    # (dependería de cómo se almacenan las conexiones en BrainSpace)
end

"""Obtiene activaciones de todas las regiones del espacio cerebral"""
function get_region_activations(brain_space, num_regions=nothing)
    # Esta función dependería de cómo se definen las regiones en BrainSpace
    # Aquí se muestra una versión simplificada que divide el espacio en regiones
    
    # Obtener todas las posiciones y activaciones
    positions = [(n.position.x, n.position.y, n.position.z) for n in brain_space.neurons]
    activations = [n.activation for n in brain_space.neurons]
    
    # Si no se especifica número de regiones, usar todas las disponibles
    if num_regions === nothing
        num_regions = haskey(brain_space, :regions) ? length(brain_space.regions) : 10
    end
    
    # Crear regiones artificiales si es necesario (para demostración)
    region_activations = zeros(Float64, num_regions)
    region_counts = zeros(Int, num_regions)
    
    # Asignar neuronas a regiones y acumular activación
    for (pos, act) in zip(positions, activations)
        # En una implementación real, se usaría la información real de región
        # Aquí simplemente asignamos según posición x (simplificado)
        x, y, z = pos
        
        # Determinar región (simplificado para demostración)
        region_idx = mod(floor(Int, abs(x * 10)) + 1, num_regions) + 1
        if region_idx > num_regions
            region_idx = num_regions
        end
        
        # Acumular activación
        region_activations[region_idx] += act
        region_counts[region_idx] += 1
    end
    
    # Calcular activación media por región
    for i in 1:num_regions
        if region_counts[i] > 0
            region_activations[i] /= region_counts[i]
        end
    end
    
    return region_activations
end

"""Visualiza regiones especializadas del espacio cerebral"""
function visualize_specialized_regions!(ax, brain_space, num_regions)
    # Esta función dependería de la implementación específica de BrainSpace
    # Aquí se muestra una versión simplificada
    
    # Obtener posiciones y activaciones de neuronas
    positions = [(n.position.x, n.position.y, n.position.z) for n in brain_space.neurons]
    activations = [n.activation for n in brain_space.neurons]
    
    # En una implementación real, se usaría información real de especialización
    # Aquí asignamos regiones artificialmente para demostración
    
    # Crear colores para las regiones
    region_colors = distinguishable_colors(num_regions, [RGB(1,1,1), RGB(0,0,0)])
    
    # Asignar colores según región (simplificado)
    colors = []
    
    for (i, pos) in enumerate(positions)
        x, y, z = pos
        
        # Determinar región simplificada para demostración
        region_idx = mod(floor(Int, abs(x * 10)), num_regions) + 1
        
        # Asignar color según región
        push!(colors, region_colors[region_idx])
    end
    
    # Visualizar neuronas coloreadas por región
    scatter!(ax, positions, color=colors, markersize=10)
end

"""Analiza la especialización de regiones del espacio cerebral"""
function analyze_region_specialization(brain_space, training_history)
    # En una implementación real, esto dependería de cómo se definen las regiones
    # Aquí se muestra una versión simplificada
    
    # Obtener activaciones actuales por región
    region_activations = get_region_activations(brain_space)
    
    # Calcular especialización (ejemplo simplificado)
    region_specialization = region_activations ./ sum(region_activations)
    
    # Crear datos de análisis
    region_data = Dict(
        "current_activations" => region_activations,
        "specialization" => region_specialization
    )
    
    # Añadir evolución temporal si está disponible
    if haskey(training_history, "region_specialization")
        region_data["evolution"] = training_history["region_specialization"]
    end
    
    return region_data
end

"""Estima la tasa de aprendizaje óptima basada en el historial"""
function estimate_optimal_lr(loss_history, lr_history)
    # Encontrar la tasa de aprendizaje con menor pérdida
    min_loss_idx = argmin(loss_history)
    min_loss_lr = lr_history[min_loss_idx]
    
    # En una implementación más sofisticada, se podría hacer un análisis de tendencia
    # Para esta versión simplificada, retornamos la LR con menor pérdida
    return min_loss_lr
end

"""Genera recomendaciones para la tasa de aprendizaje"""
function get_lr_recommendation(loss_history, lr_history, optimal_lr)
    # Analizar tendencia de pérdida
    is_converging = is_loss_converging(loss_history)
    
    if is_converging
        return "El entrenamiento está convergiendo bien. La tasa de aprendizaje actual parece adecuada."
    else
        # Verificar si estamos usando una LR cercana a la óptima
        current_lr = lr_history[end]
        lr_ratio = current_lr / optimal_lr
        
        if lr_ratio > 5
            return "La tasa de aprendizaje actual parece ser demasiado alta. Considere reducirla a un valor cercano a $(optimal_lr)."
        elseif lr_ratio < 0.2
            return "La tasa de aprendizaje actual parece ser demasiado baja. Considere aumentarla a un valor cercano a $(optimal_lr)."
        else
            return "La tasa de aprendizaje parece estar en un rango razonable, pero el entrenamiento no está convergiendo adecuadamente. Considere ajustar otros hiperparámetros o revisar la arquitectura del modelo."
        end
    end
end

"""Determina si la pérdida está convergiendo adecuadamente"""
function is_loss_converging(loss_history)
    # Esta es una verificación simplificada
    # Una implementación más sofisticada analizaría la tendencia con más detalle
    
    if length(loss_history) < 10
        return true  # No suficientes datos para analizar
    end
    
    # Verificar si la pérdida está disminuyendo en las últimas iteraciones
    last_losses = loss_history[end-9:end]
    
    # Calcular primera derivada (tasa de cambio)
    derivatives = diff(last_losses)
    
    # Si la mayoría de las derivadas son negativas, está convergiendo
    return count(d -> d < 0, derivatives) >= length(derivatives) / 2
end

"""Visualiza un resumen de la actividad neuronal"""
function visualize_activity_summary(brain_space)
    # Crear figura
    fig = Figure(resolution=(1200, 800))
    
    # Panel para distribución de activación
    dist_panel = fig[1, 1]
    dist_ax = Axis(dist_panel,
                  xlabel="Activación",
                  ylabel="Frecuencia",
                  title="Distribución de Activación Neuronal")
    
    # Obtener activaciones
    activations = [n.activation for n in brain_space.neurons]
    
    # Visualizar histograma de activación
    hist!(dist_ax, activations, bins=30, color=:blue, alpha=0.7)
    
    # Panel para activación por región
    region_panel = fig[1, 2]
    region_ax = Axis(region_panel,
                    xlabel="Región",
                    ylabel="Activación Media",
                    title="Activación por Región")
    
    # Obtener activaciones por región
    region_acts = get_region_activations(brain_space)
    
    # Visualizar activaciones por región
    barplot!(region_ax, 1:length(region_acts), region_acts, color=:orange)
    
    # Panel para visualización 3D de activación
    spatial_panel = fig[2, 1:2]
    spatial_ax = Axis3(spatial_panel,
                      title="Distribución Espacial de Activación")
    
    # Visualizar espacio cerebral coloreado por activación
    visualize_brain_state!(spatial_ax, brain_space)
    
    return fig
end

"""Obtiene todas las activaciones neuronales"""
function get_all_activations(brain_space)
    return [n.activation for n in brain_space.neurons]
end

"""Normaliza una métrica para mejor comparación"""
function normalize_metric(metric_data, metric_name)
    # Para pérdida, normalizar relativo al máximo
    if lowercase(metric_name) == "loss"
        max_val = maximum(metric_data)
        if max_val > 0
            return metric_data ./ max_val
        end
    end
    
    # Para otras métricas, mantener como están
    return metric_data
end

"""Genera conclusiones sobre el entrenamiento"""
function generate_training_conclusions(training_history)
    conclusions = Dict(
        "Observaciones" => String[],
        "Recomendaciones" => String[]
    )
    
    # Analizar pérdida
    if haskey(training_history, "loss")
        loss = training_history["loss"]
        
        # Verificar convergencia
        if length(loss) > 10
            last_losses = loss[end-9:end]
            
            if abs(last_losses[end] - last_losses[1]) < 0.01 * last_losses[1]
                push!(conclusions["Observaciones"], "La pérdida ha convergido en las últimas iteraciones.")
            elseif last_losses[end] > last_losses[1]
                push!(conclusions["Observaciones"], "La pérdida está aumentando en las últimas iteraciones, posible sobreajuste.")
                push!(conclusions["Recomendaciones"], "Considere aplicar regularización o detener el entrenamiento antes.")
            else
                push!(conclusions["Observaciones"], "La pérdida continúa disminuyendo, el entrenamiento podría beneficiarse de más iteraciones.")
            end
        end
    end
    
    # Analizar precisión si está disponible
    if haskey(training_history, "accuracy")
        accuracy = training_history["accuracy"]
        
        if length(accuracy) > 1
            if accuracy[end] > 0.95
                push!(conclusions["Observaciones"], "El modelo ha alcanzado una alta precisión (>95%).")
            elseif accuracy[end] < 0.7
                push!(conclusions["Observaciones"], "La precisión final es relativamente baja (<70%).")
                push!(conclusions["Recomendaciones"], "Considere revisar la arquitectura del modelo o aumentar la complejidad.")
            end
        end
    end
    
    # Conclusiones generales
    push!(conclusions["Recomendaciones"], "Guarde este modelo como punto de referencia para futuras comparaciones.")
    
    return conclusions
end

end # module