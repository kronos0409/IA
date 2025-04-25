module Serialization

using JLD2
using FileIO
using Dates
using Statistics

# Importaciones necesarias (ajustar según la estructura del proyecto)
# using CircularBuffers: CircularBuffer
using UUIDs: UUID
using ..BrainSpace
using ..TensorNeuron
using ..Connections
# using DataModels: Brain_Space, TensorNeuron, TensorConnection, SpatialField, ExpansionEvent
# using TensorialTokenizer: create_default_tokenizer, process_text

export save_brain,
       load_brain,
       save_checkpoint,
       to_tensor,
       from_tensor,
       brain_summary

"""
    save_brain(brain, filename)

Guarda un modelo RNTA en un archivo.
"""
function save_brain(brain::Brain_Space, filename::String)
    # Preparar datos para guardar
    data = Dict{String, Any}()
    
    # Guardar dimensiones
    data["dimensions"] = brain.dimensions
    
    # Guardar configuración
    data["config"] = brain.config
    
    # Guardar neuronas
    neurons_data = Dict{String, Any}()
    
    for (pos, neuron) in brain.neurons
        neuron_data = Dict{String, Any}()
        
        # Guardar propiedades de la neurona
        neuron_data["id"] = string(neuron.id)
        neuron_data["position"] = pos
        neuron_data["state"] = neuron.state
        neuron_data["transformation_kernel"] = neuron.transformation_kernel
        neuron_data["specialization"] = neuron.specialization
        neuron_data["functional_type"] = string(neuron.functional_type)
        
        # Guardar campo receptivo
        neuron_data["receptive_field"] = Dict{String, Any}(
            "center" => neuron.receptive_field.center,
            "size" => neuron.receptive_field.size,
            "bounds" => neuron.receptive_field.bounds
        )
        
        # Guardar historial de activación (limitado)
        # Convertimos el buffer circular a un array normal
        neuron_data["activation_history"] = [copy(state) for state in neuron.activation_history]
        
        # Guardar plasticidad
        neuron_data["plasticity"] = neuron.plasticity
        
        # Guardar en diccionario de neuronas
        neurons_data[string(neuron.id)] = neuron_data
    end
    
    data["neurons"] = neurons_data
    
    # Guardar conexiones
    connections_data = []
    
    for connection in brain.connections
        connection_data = Dict{String, Any}()
        
        # Guardar propiedades de la conexión
        connection_data["id"] = string(connection.id)
        connection_data["source_id"] = string(connection.source_id)
        connection_data["target_id"] = string(connection.target_id)
        connection_data["weight"] = connection.weight
        connection_data["strength"] = connection.strength
        connection_data["connection_type"] = string(connection.connection_type)
        connection_data["creation_time"] = connection.creation_time
        
        # Convertir historial de actividad
        connection_data["activity_history"] = collect(connection.activity_history)
        
        push!(connections_data, connection_data)
    end
    
    data["connections"] = connections_data
    
    # Guardar regiones funcionales
    functional_regions_data = Dict{String, Any}()
    
    for (type, positions) in brain.functional_regions
        functional_regions_data[string(type)] = positions
    end
    
    data["functional_regions"] = functional_regions_data
    
    # Guardar estado global
    data["global_state"] = brain.global_state
    
    # Guardar mapa de atención
    data["attention_map"] = brain.attention_map
    
    # Guardar historial de expansiones
    expansion_history_data = []
    
    for event in brain.expansion_history
        event_data = Dict{String, Any}(
            "timestamp" => event.timestamp,
            "region" => event.region,
            "new_neurons" => event.new_neurons
        )
        
        push!(expansion_history_data, event_data)
    end
    
    data["expansion_history"] = expansion_history_data
    
    # Guardar en archivo
    save(filename, data)
    
    return filename
end

"""
    load_brain(filename)

Carga un modelo RNTA desde un archivo.
"""
function load_brain(filename::String)
    # Cargar datos
    data = load(filename)
    
    # Reconstruir el cerebro
    dimensions = data["dimensions"]
    config = data["config"]
    
    # Crear cerebro vacío
    brain = BrainSpace(
        dimensions[1],
        dimensions[2],
        dimensions[3],
        config=config
    )
    
    # Limpiar neuronas y conexiones existentes
    empty!(brain.neurons)
    empty!(brain.connections)
    
    # Reconstruir neuronas
    neurons_data = data["neurons"]
    
    for (_, neuron_data) in neurons_data
        # Reconstruir UUID
        id = UUID(neuron_data["id"])
        
        # Reconstruir posición
        position = neuron_data["position"]
        
        # Reconstruir campo receptivo
        rf_data = neuron_data["receptive_field"]
        receptive_field = SpatialField(
            rf_data["center"],
            rf_data["size"],
            rf_data["bounds"]
        )
        
        # Reconstruir estado
        state = neuron_data["state"]
        
        # Reconstruir kernel de transformación
        transformation_kernel = neuron_data["transformation_kernel"]
        
        # Reconstruir plasticidad
        plasticity = neuron_data["plasticity"]
        
        # Reconstruir historial de activación
        activation_history_data = neuron_data["activation_history"]
        activation_history = CircularBuffer{Array{Float32,3}}(length(activation_history_data))
        
        for state in activation_history_data
            push!(activation_history, state)
        end
        
        # Reconstruir especialización
        specialization = neuron_data["specialization"]
        
        # Reconstruir tipo funcional
        functional_type = Symbol(neuron_data["functional_type"])
        
        # Crear neurona
        neuron = TensorNeuron(
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
        
        # Añadir a diccionario
        brain.neurons[position] = neuron
    end
    
    # Reconstruir conexiones
    connections_data = data["connections"]
    
    for connection_data in connections_data
        # Reconstruir ID
        id = UUID(connection_data["id"])
        
        # Reconstruir IDs de origen y destino
        source_id = UUID(connection_data["source_id"])
        target_id = UUID(connection_data["target_id"])
        
        # Reconstruir peso
        weight = connection_data["weight"]
        
        # Reconstruir fuerza
        strength = connection_data["strength"]
        
        # Reconstruir historial de actividad
        activity_history_data = connection_data["activity_history"]
        activity_history = CircularBuffer{Float32}(length(activity_history_data))
        
        for activity in activity_history_data
            push!(activity_history, activity)
        end
        
        # Reconstruir tiempo de creación
        creation_time = connection_data["creation_time"]
        
        # Reconstruir tipo de conexión
        connection_type = Symbol(connection_data["connection_type"])
        
        # Crear conexión
        connection = TensorConnection(
            id,
            source_id,
            target_id,
            weight,
            strength,
            activity_history,
            creation_time,
            connection_type
        )
        
        # Añadir a lista
        push!(brain.connections, connection)
    end
    
    # Reconstruir regiones funcionales
    functional_regions_data = data["functional_regions"]
    
    for (type_str, positions) in functional_regions_data
        brain.functional_regions[Symbol(type_str)] = positions
    end
    
    # Reconstruir estado global
    brain.global_state = data["global_state"]
    
    # Reconstruir mapa de atención
    brain.attention_map = data["attention_map"]
    
    # Reconstruir historial de expansiones
    expansion_history_data = data["expansion_history"]
    
    for event_data in expansion_history_data
        event = ExpansionEvent(
            event_data["timestamp"],
            event_data["region"],
            event_data["new_neurons"]
        )
        
        push!(brain.expansion_history, event)
    end
    
    return brain
end

"""
    save_checkpoint(brain, base_filename; max_checkpoints=5)

Guarda un checkpoint del modelo, manteniendo un número limitado de versiones.
"""
function save_checkpoint(
    brain::Brain_Space,
    base_filename::String;
    max_checkpoints::Int=5
)
    # Crear nombre de archivo con timestamp
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    filename = "$(base_filename)_checkpoint_$(timestamp).jld2"
    
    # Guardar checkpoint
    save_brain(brain, filename)
    
    # Eliminar checkpoints antiguos si exceden el límite
    checkpoints = filter(file -> startswith(basename(file), "$(basename(base_filename))_checkpoint_"),
                        readdir(dirname(base_filename), join=true))
    
    if length(checkpoints) > max_checkpoints
        # Ordenar por fecha (más antiguos primero)
        sort!(checkpoints, by=file -> stat(file).mtime)
        
        # Eliminar los más antiguos
        for i in 1:(length(checkpoints) - max_checkpoints)
            rm(checkpoints[i])
        end
    end
    
    return filename
end

"""
    to_tensor(data)

Convierte diferentes tipos de datos a representación tensorial 3D.
"""
function to_tensor(data)
    if isa(data, Array) && ndims(data) == 3
        # Ya es un tensor 3D
        return convert(Array{Float32,3}, data)
    elseif isa(data, Array) && ndims(data) == 2
        # Convertir matriz 2D a tensor 3D
        rows, cols = size(data)
        tensor = zeros(Float32, rows, cols, 1)
        tensor[:, :, 1] = data
        return tensor
    elseif isa(data, Array) && ndims(data) == 1
        # Convertir vector a tensor 3D
        len = length(data)
        tensor = zeros(Float32, len, 1, 1)
        tensor[:, 1, 1] = data
        return tensor
    elseif isa(data, String)
        # Convertir texto a tensor 3D usando tokenizador por defecto
        tokenizer = create_default_tokenizer()
        return process_text(tokenizer, data)
    else
        # Intentar convertir a Float32
        try
            value = convert(Float32, data)
            tensor = zeros(Float32, 1, 1, 1)
            tensor[1, 1, 1] = value
            return tensor
        catch
            error("No se puede convertir el tipo de datos $(typeof(data)) a tensor 3D")
        end
    end
end

"""
    from_tensor(tensor, output_type=:auto)

Convierte un tensor 3D de vuelta a un tipo de datos más simple si es posible.
"""
function from_tensor(tensor::Array{T,3}, output_type::Symbol=:auto) where T <: AbstractFloat
    if output_type == :tensor || output_type == :auto
        # Simplificar si es posible
        if size(tensor, 2) == 1 && size(tensor, 3) == 1
            # Es efectivamente un vector
            if output_type == :auto
                return tensor[:, 1, 1]
            end
        elseif size(tensor, 3) == 1
            # Es efectivamente una matriz
            if output_type == :auto
                return tensor[:, :, 1]
            end
        end
        
        # Devolver tensor original
        return tensor
    elseif output_type == :vector
        # Forzar conversión a vector
        if size(tensor, 2) == 1 && size(tensor, 3) == 1
            return tensor[:, 1, 1]
        else
            # Aplanar
            return reshape(tensor, :)
        end
    elseif output_type == :matrix
        # Forzar conversión a matriz
        if size(tensor, 3) == 1
            return tensor[:, :, 1]
        else
            # Proyectar a matriz (promediar en la tercera dimensión)
            return mean(tensor, dims=3)[:, :, 1]
        end
    elseif output_type == :scalar
        # Devolver valor escalar (promedio)
        return mean(tensor)
    else
        error("Tipo de salida no reconocido: $output_type")
    end
end

"""
    brain_summary(brain)

Genera un resumen estadístico del estado del cerebro.
"""
function brain_summary(brain::Brain_Space)
    # Contar neuronas por tipo
    neuron_counts = Dict{Symbol, Int}()
    
    for (_, neuron) in brain.neurons
        if !haskey(neuron_counts, neuron.functional_type)
            neuron_counts[neuron.functional_type] = 0
        end
        
        neuron_counts[neuron.functional_type] += 1
    end
    
    # Calcular nivel medio de especialización
    avg_specialization = mean([neuron.specialization for (_, neuron) in brain.neurons])
    
    # Contar conexiones por tipo
    excitatory_count = count(c -> c.connection_type == :excitatory, brain.connections)
    inhibitory_count = count(c -> c.connection_type == :inhibitory, brain.connections)
    
    # Calcular densidad de conexiones
    total_possible_connections = length(brain.neurons) * (length(brain.neurons) - 1)
    connection_density = total_possible_connections > 0 ? 
                         length(brain.connections) / total_possible_connections : 0
    
    # Calcular estadísticas de actividad global
    activity_stats = (
        mean = mean(abs.(brain.global_state)),
        max = maximum(abs.(brain.global_state)),
        std = std(brain.global_state)
    )
    
    # Generar resumen
    summary = Dict{String, Any}(
        "dimensions" => brain.dimensions,
        "num_neurons" => length(brain.neurons),
        "neuron_types" => neuron_counts,
        "avg_specialization" => avg_specialization,
        "connections" => Dict{String, Any}(
            "total" => length(brain.connections),
            "excitatory" => excitatory_count,
            "inhibitory" => inhibitory_count,
            "density" => connection_density
        ),
        "activity" => activity_stats,
        "expansions" => length(brain.expansion_history)
    )
    
    return summary
end
end