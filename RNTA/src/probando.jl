# test_linear_regression.jl
# Script para probar RNTA.jl con una regresión lineal simple f(x) = 2x + 1

using LinearAlgebra
using Statistics
using Random
using Plots

# Incluir RNTA.jl
include("RNTA.jl")
using .RNTA

# Configurar reproducibilidad
Random.seed!(42)

# Parámetros de prueba
num_samples = 100
input_range = -5:0.1:5
test_ratio = 0.2

println("=== Prueba de Regresión Lineal con RNTA ===")
println("Función objetivo: f(x) = 2x + 1")
println("Muestras: $num_samples")

# Generar datos lineales con algo de ruido
function generate_linear_data(num_samples, noise_level=0.1)
    x_values = collect(range(-5, 5, length=num_samples))
    y_true = 2 .* x_values .+ 1
    y_noisy = y_true + noise_level * randn(num_samples)
    
    return x_values, y_true, y_noisy
end

# Crear datos de entrenamiento y prueba
x_values, y_true, y_noisy = generate_linear_data(num_samples)
println("Datos generados con función lineal y ruido gaussiano")

# Dividir en conjuntos de entrenamiento y prueba
split_idx = Int(floor(num_samples * (1 - test_ratio)))
x_train, y_train = x_values[1:split_idx], y_noisy[1:split_idx]
x_test, y_test = x_values[split_idx+1:end], y_noisy[split_idx+1:end]

println("Datos divididos: $(length(x_train)) para entrenamiento, $(length(x_test)) para prueba")

# Convertir los datos a tensores 3D para RNTA
# Usando profundidad 1 para simular un tensor 2D
function create_tensor_data(x_values, y_values)
    input_tensors = []
    target_tensors = []
    
    for i in 1:length(x_values)
        # Crear tensor de entrada 3D (1×1×1)
        x_tensor = zeros(Float32, 1, 1, 1)
        x_tensor[1, 1, 1] = x_values[i]
        
        # Crear tensor de salida 3D (1×1×1)
        y_tensor = zeros(Float32, 1, 1, 1)
        y_tensor[1, 1, 1] = y_values[i]
        
        push!(input_tensors, x_tensor)
        push!(target_tensors, y_tensor)
    end
    
    return input_tensors, target_tensors
end

# Preparar datos para RNTA
train_input, train_target = create_tensor_data(x_train, y_train)
test_input, test_target = create_tensor_data(x_test, y_test)

println("Datos convertidos a formato tensorial para RNTA")

# Crear un espacio cerebral simple
try
    println("\n=== Inicializando RNTA ===")
    # Inicializar RNTA
    config = RNTA.initialize(use_cuda=false)
    
    # Crear un espacio cerebral para regresión simple
    # Usando dimensiones pequeñas (3x3x3) para esta tarea simple
    println("Creando espacio cerebral...")
    brain = RNTA.create_brain_space((3, 3, 3), config=config)
    
    # Mostrar información del cerebro
    println("Espacio cerebral creado")
    summary = RNTA.brain_summary(brain)
    println("Neuronas: $(summary[:neurons])")
    println("Conexiones: $(summary[:connections])")
    
    # Entrenar el modelo
    println("\n=== Entrenando modelo ===")
    training_params = Dict(
        :epochs => 50,
        :learning_rate => 0.01,
        :batch_size => 10
    )
    
    println("Parámetros de entrenamiento:")
    println("  Épocas: $(training_params[:epochs])")
    println("  Tasa de aprendizaje: $(training_params[:learning_rate])")
    println("  Tamaño de lote: $(training_params[:batch_size])")
    
    # Entrenar el modelo
    metrics = RNTA.train!(
        brain, 
        train_input, 
        train_target,
        epochs=training_params[:epochs],
        learning_rate=training_params[:learning_rate],
        batch_size=training_params[:batch_size],
        verbose=true
    )
    
    println("\n=== Evaluando modelo ===")
    # Evaluar el modelo
    predictions = []
    for input_tensor in test_input
        output = RNTA.process_input(brain, input_tensor)
        push!(predictions, output[1, 1, 1])  # Extraer valor escalar
    end
    
    # Calcular error cuadrático medio
    mse = mean((predictions .- y_test).^2)
    println("Error cuadrático medio en conjunto de prueba: $mse")
    
    # Visualizar resultados
    println("\n=== Generando visualizaciones ===")
    # Graficar datos y predicciones
    plot_data = plot(
        x_values, y_true, 
        label="Función real (2x+1)", 
        linewidth=2,
        title="Predicción con RNTA de f(x) = 2x+1",
        xlabel="x",
        ylabel="y"
    )
    
    scatter!(plot_data, x_train, y_train, label="Datos de entrenamiento", alpha=0.5)
    scatter!(plot_data, x_test, predictions, label="Predicciones", markersize=6, color=:red)
    
    # Guardar gráfico
    savefig(plot_data, "rnta_linear_regression.png")
    println("Gráfico guardado como 'rnta_linear_regression.png'")
    
    # Visualizar pérdida durante entrenamiento
    if !isempty(metrics[:loss])
        plot_loss = plot(
            1:length(metrics[:loss]), 
            metrics[:loss], 
            label="Pérdida", 
            linewidth=2,
            title="Curva de pérdida durante entrenamiento",
            xlabel="Época",
            ylabel="Pérdida"
        )
        savefig(plot_loss, "rnta_training_loss.png")
        println("Curva de pérdida guardada como 'rnta_training_loss.png'")
    end
    
    # Intentar visualizar el espacio cerebral
    try
        println("\n=== Visualizando el espacio cerebral ===")
        RNTA.visualize_brain(
            brain, 
            show_neurons=true, 
            show_connections=true, 
            show_activity=true,
            file_path="rnta_brain_visualization.png"
        )
        println("Visualización del cerebro guardada como 'rnta_brain_visualization.png'")
    catch e
        println("No se pudo generar la visualización del cerebro: $e")
    end
    
    println("\n=== Prueba completada con éxito ===")
    
catch e
    println("\n❌ Error durante la prueba: $e")
    println(stacktrace())
end