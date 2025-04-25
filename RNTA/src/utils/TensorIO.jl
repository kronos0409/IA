module TensorIO

using LinearAlgebra
using HDF5
using JSON
using FileIO
using JLD2
using ..BrainSpace, ..TensorNeuron, ..SpatialField

export save_tensor, load_tensor, export_model, import_model,
       tensor_to_hdf5, tensor_from_hdf5, metadata_to_json, metadata_from_json,
       batched_tensor_save, batched_tensor_load, tensor_checkpoint,
       model_versioning, model_conversion, validate_tensor_file

"""
    save_tensor(tensor, filename::String; format=:jld2, compression=true, metadata=nothing)

Guarda un tensor en un archivo con el formato especificado.

# Argumentos
- `tensor`: El tensor a guardar
- `filename::String`: Nombre del archivo donde guardar el tensor
- `format=:jld2`: Formato de archivo (:jld2, :hdf5, :csv)
- `compression=true`: Si se debe aplicar compresión al guardar
- `metadata=nothing`: Metadatos opcionales para incluir con el tensor

# Retorna
- `filepath`: Ruta donde se guardó el tensor
"""
function save_tensor(tensor, filename::String; format=:jld2, compression=true, metadata=nothing)
    # Verificar que el directorio existe, si no, crearlo
    dir = dirname(filename)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    # Seleccionar método de guardado según formato
    if format == :jld2
        filepath = save_tensor_jld2(tensor, filename, compression, metadata)
    elseif format == :hdf5
        filepath = save_tensor_hdf5(tensor, filename, compression, metadata)
    elseif format == :csv
        filepath = save_tensor_csv(tensor, filename, metadata)
    else
        error("Formato de archivo no soportado: $format")
    end
    
    return filepath
end

"""
    load_tensor(filename::String; format=nothing)

Carga un tensor desde un archivo.

# Argumentos
- `filename::String`: Nombre del archivo del tensor
- `format=nothing`: Formato de archivo (si es nothing, se infiere de la extensión)

# Retorna
- `tensor`: El tensor cargado
- `metadata`: Metadatos asociados al tensor, si existen
"""
function load_tensor(filename::String; format=nothing)
    # Verificar que el archivo existe
    if !isfile(filename)
        error("El archivo $filename no existe")
    end
    
    # Inferir formato si no se proporciona
    if format === nothing
        format = infer_format_from_extension(filename)
    end
    
    # Cargar según formato
    if format == :jld2
        return load_tensor_jld2(filename)
    elseif format == :hdf5
        return load_tensor_hdf5(filename)
    elseif format == :csv
        return load_tensor_csv(filename)
    else
        error("Formato de archivo no soportado: $format")
    end
end

"""
    export_model(brain_space::BrainSpace, filename::String; 
                include_weights=true, include_activations=false, format=:jld2)

Exporta un modelo RNTA completo a un archivo.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral a exportar
- `filename::String`: Nombre del archivo donde guardarlo
- `include_weights=true`: Si se incluyen los pesos de conexiones
- `include_activations=false`: Si se incluyen las activaciones actuales
- `format=:jld2`: Formato de exportación (:jld2, :hdf5)

# Retorna
- `filepath`: Ruta donde se guardó el modelo
"""
function export_model(brain_space::Brain_Space, filename::String; 
                     include_weights=true, include_activations=false, format=:jld2)
    # Preparar el modelo para exportación
    export_data = prepare_model_for_export(brain_space, include_weights, include_activations)
    
    # Añadir metadatos de exportación
    metadata = Dict(
        "export_date" => string(now()),
        "model_name" => brain_space.name,
        "rnta_version" => get_rnta_version(),
        "include_weights" => include_weights,
        "include_activations" => include_activations
    )
    
    # Guardar según formato
    if format == :jld2
        filepath = save_model_jld2(export_data, filename, metadata)
    elseif format == :hdf5
        filepath = save_model_hdf5(export_data, filename, metadata)
    else
        error("Formato de exportación no soportado: $format")
    end
    
    return filepath
end

"""
    import_model(filename::String; format=nothing)

Importa un modelo RNTA desde un archivo.

# Argumentos
- `filename::String`: Nombre del archivo del modelo
- `format=nothing`: Formato del archivo (si es nothing, se infiere de la extensión)

# Retorna
- `brain_space`: El espacio cerebral reconstruido
- `metadata`: Metadatos del modelo
"""
function import_model(filename::String; format=nothing)
    # Verificar que el archivo existe
    if !isfile(filename)
        error("El archivo $filename no existe")
    end
    
    # Inferir formato si no se proporciona
    if format === nothing
        format = infer_format_from_extension(filename)
    end
    
    # Cargar según formato
    if format == :jld2
        model_data, metadata = load_model_jld2(filename)
    elseif format == :hdf5
        model_data, metadata = load_model_hdf5(filename)
    else
        error("Formato de modelo no soportado: $format")
    end
    
    # Reconstruir el modelo
    brain_space = reconstruct_model_from_data(model_data)
    
    return brain_space, metadata
end

"""
    tensor_to_hdf5(tensor, filename::String; 
                 dataset_name="tensor", compression=true)

Guarda un tensor específicamente en formato HDF5 con opciones avanzadas.

# Argumentos
- `tensor`: El tensor a guardar
- `filename::String`: Nombre del archivo HDF5
- `dataset_name="tensor"`: Nombre del dataset dentro del archivo HDF5
- `compression=true`: Si se aplica compresión

# Retorna
- `filepath`: Ruta donde se guardó el tensor
"""
function tensor_to_hdf5(tensor, filename::String; 
                      dataset_name="tensor", compression=true)
    h5open(filename, "w") do file
        if compression
            write(file, dataset_name, tensor, compress=true)
        else
            write(file, dataset_name, tensor)
        end
        
        # Guardar dimensiones como atributo
        attrs(file[dataset_name])["dimensions"] = collect(size(tensor))
        attrs(file[dataset_name])["type"] = string(eltype(tensor))
    end
    
    return filename
end

"""
    tensor_from_hdf5(filename::String; dataset_name="tensor")

Carga un tensor desde un archivo HDF5 específico.

# Argumentos
- `filename::String`: Nombre del archivo HDF5
- `dataset_name="tensor"`: Nombre del dataset dentro del archivo HDF5

# Retorna
- `tensor`: El tensor cargado
- `dimensions`: Dimensiones originales del tensor
"""
function tensor_from_hdf5(filename::String; dataset_name="tensor")
    h5open(filename, "r") do file
        if !haskey(file, dataset_name)
            error("Dataset '$dataset_name' no encontrado en $filename")
        end
        
        tensor = read(file[dataset_name])
        
        # Leer atributos si existen
        dimensions = nothing
        if haskey(attrs(file[dataset_name]), "dimensions")
            dimensions = read(attrs(file[dataset_name])["dimensions"])
        end
        
        return tensor, dimensions
    end
end

"""
    metadata_to_json(metadata, filename::String)

Guarda metadatos en un archivo JSON separado.

# Argumentos
- `metadata`: Diccionario con metadatos
- `filename::String`: Nombre del archivo JSON

# Retorna
- `filepath`: Ruta donde se guardaron los metadatos
"""
function metadata_to_json(metadata, filename::String)
    open(filename, "w") do f
        JSON.print(f, metadata, 4)  # 4 espacios de indentación
    end
    
    return filename
end

"""
    metadata_from_json(filename::String)

Carga metadatos desde un archivo JSON.

# Argumentos
- `filename::String`: Nombre del archivo JSON

# Retorna
- `metadata`: Diccionario con los metadatos
"""
function metadata_from_json(filename::String)
    metadata = open(filename) do f
        JSON.parse(f)
    end
    
    return metadata
end

"""
    batched_tensor_save(tensors::Dict, base_filename::String; 
                      format=:jld2, compression=true, batch_size=10)

Guarda múltiples tensores en archivos por lotes para eficiencia.

# Argumentos
- `tensors::Dict`: Diccionario de tensores donde clave=nombre, valor=tensor
- `base_filename::String`: Nombre base para los archivos
- `format=:jld2`: Formato de archivo
- `compression=true`: Si se aplica compresión
- `batch_size=10`: Número de tensores por archivo

# Retorna
- `filepaths`: Lista de rutas a los archivos guardados
"""
function batched_tensor_save(tensors::Dict, base_filename::String; 
                           format=:jld2, compression=true, batch_size=10)
    # Verificar que hay tensores para guardar
    if isempty(tensors)
        return String[]
    end
    
    # Crear directorio si no existe
    dir = dirname(base_filename)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    # Dividir en lotes
    tensor_names = collect(keys(tensors))
    num_batches = ceil(Int, length(tensor_names) / batch_size)
    
    filepaths = String[]
    
    for batch_idx in 1:num_batches
        # Determinar nombres de tensores para este lote
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, length(tensor_names))
        batch_names = tensor_names[start_idx:end_idx]
        
        # Crear lote de tensores
        batch_tensors = Dict{String, Any}()
        for name in batch_names
            batch_tensors[name] = tensors[name]
        end
        
        # Crear nombre de archivo para este lote
        batch_filename = "$(base_filename)_batch$(batch_idx).$(string(format))"
        
        # Guardar el lote
        if format == :jld2
            filepath = save_batch_jld2(batch_tensors, batch_filename, compression)
        elseif format == :hdf5
            filepath = save_batch_hdf5(batch_tensors, batch_filename, compression)
        else
            error("Formato no soportado para guardado por lotes: $format")
        end
        
        push!(filepaths, filepath)
    end
    
    # Guardar índice de lotes
    index_data = Dict(
        "num_batches" => num_batches,
        "num_tensors" => length(tensor_names),
        "tensor_names" => tensor_names,
        "batch_files" => filepaths
    )
    
    index_filename = "$(base_filename)_index.json"
    metadata_to_json(index_data, index_filename)
    
    push!(filepaths, index_filename)
    
    return filepaths
end

"""
    batched_tensor_load(base_filename::String)

Carga múltiples tensores guardados por lotes.

# Argumentos
- `base_filename::String`: Nombre base usado para guardar los lotes

# Retorna
- `tensors`: Diccionario con todos los tensores cargados
"""
function batched_tensor_load(base_filename::String)
    # Cargar índice
    index_filename = "$(base_filename)_index.json"
    if !isfile(index_filename)
        error("Archivo de índice no encontrado: $index_filename")
    end
    
    index_data = metadata_from_json(index_filename)
    
    # Inicializar diccionario para todos los tensores
    tensors = Dict{String, Any}()
    
    # Cargar cada archivo de lote
    for batch_idx in 1:index_data["num_batches"]
        batch_filename = "$(base_filename)_batch$(batch_idx)"
        
        # Detectar extensión
        batch_file = ""
        for ext in [".jld2", ".h5", ".hdf5"]
            if isfile(batch_filename * ext)
                batch_file = batch_filename * ext
                break
            end
        end
        
        if isempty(batch_file)
            error("No se encontró el archivo de lote: $batch_filename con ninguna extensión")
        end
        
        # Inferir formato desde extensión
        format = infer_format_from_extension(batch_file)
        
        # Cargar lote
        batch_tensors = Dict{String, Any}()
        if format == :jld2
            batch_tensors = load_batch_jld2(batch_file)
        elseif format == :hdf5
            batch_tensors = load_batch_hdf5(batch_file)
        end
        
        # Agregar tensores al resultado final
        merge!(tensors, batch_tensors)
    end
    
    return tensors
end

"""
    tensor_checkpoint(tensor, checkpoint_dir::String, name::String; 
                    keep_last=5, metadata=nothing)

Guarda un punto de control de un tensor con historial limitado.

# Argumentos
- `tensor`: El tensor a guardar como checkpoint
- `checkpoint_dir::String`: Directorio para checkpoints
- `name::String`: Nombre base para el checkpoint
- `keep_last=5`: Número de checkpoints históricos a mantener
- `metadata=nothing`: Metadatos opcionales

# Retorna
- `checkpoint_path`: Ruta al checkpoint guardado
"""
function tensor_checkpoint(tensor, checkpoint_dir::String, name::String; 
                         keep_last=5, metadata=nothing)
    # Crear directorio si no existe
    if !isdir(checkpoint_dir)
        mkpath(checkpoint_dir)
    end
    
    # Añadir timestamp al nombre
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(checkpoint_dir, "$(name)_$(timestamp).jld2")
    
    # Guardar tensor
    save_tensor(tensor, filename, metadata=metadata)
    
    # Gestionar historial de checkpoints
    manage_checkpoint_history(checkpoint_dir, name, keep_last)
    
    return filename
end

"""
    model_versioning(brain_space::BrainSpace, version::String, repo_dir::String;
                   include_weights=true, include_activations=false,
                   description="")

Guarda una versión específica de un modelo RNTA en un repositorio versionado.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral a versionar
- `version::String`: Número o etiqueta de versión
- `repo_dir::String`: Directorio del repositorio de modelos
- `include_weights=true`: Si se incluyen los pesos
- `include_activations=false`: Si se incluyen las activaciones
- `description=""`: Descripción de la versión

# Retorna
- `version_info`: Información sobre la versión guardada
"""
function model_versioning(brain_space::Brain_Space, version::String, repo_dir::String;
                        include_weights=true, include_activations=false,
                        description="")
    # Crear directorio del repositorio si no existe
    if !isdir(repo_dir)
        mkpath(repo_dir)
    end
    
    # Crear directorio de versión
    version_dir = joinpath(repo_dir, "v$(version)")
    if !isdir(version_dir)
        mkdir(version_dir)
    end
    
    # Nombre del archivo de modelo
    model_filename = joinpath(version_dir, "model.jld2")
    
    # Exportar modelo
    filepath = export_model(
        brain_space, 
        model_filename, 
        include_weights=include_weights, 
        include_activations=include_activations
    )
    
    # Crear metadatos de versión
    version_info = Dict(
        "version" => version,
        "date" => string(now()),
        "description" => description,
        "include_weights" => include_weights,
        "include_activations" => include_activations,
        "model_name" => brain_space.name,
        "rnta_version" => get_rnta_version(),
        "neurons_count" => length(brain_space.neurons),
        "connections_count" => length(brain_space.connections)
    )
    
    # Guardar metadatos
    metadata_filename = joinpath(version_dir, "version_info.json")
    metadata_to_json(version_info, metadata_filename)
    
    # Actualizar índice de versiones
    update_version_index(repo_dir, version, version_info)
    
    return version_info
end

"""
    model_conversion(brain_space::BrainSpace, format::Symbol, output_file::String;
                    framework=:default, config=Dict())

Convierte un modelo RNTA a otro formato o framework.

# Argumentos
- `brain_space::BrainSpace`: El espacio cerebral a convertir
- `format::Symbol`: Formato de destino (:onnx, :tensorflow, :pytorch, etc.)
- `output_file::String`: Archivo de salida
- `framework=:default`: Framework específico de destino
- `config=Dict()`: Configuración adicional para la conversión

# Retorna
- `output_path`: Ruta al modelo convertido
"""
function model_conversion(brain_space::Brain_Space, format::Symbol, output_file::String;
                         framework=:default, config=Dict())
    # Verificar formato soportado
    supported_formats = [:onnx, :tensorflow, :pytorch, :caffe, :mxnet]
    if !(format in supported_formats)
        error("Formato de conversión no soportado: $format")
    end
    
    # Preparar modelo para conversión
    model_data = prepare_model_for_conversion(brain_space, format, framework)
    
    # Ejecutar conversión según formato
    if format == :onnx
        output_path = convert_to_onnx(model_data, output_file, config)
    elseif format == :tensorflow
        output_path = convert_to_tensorflow(model_data, output_file, config)
    elseif format == :pytorch
        output_path = convert_to_pytorch(model_data, output_file, config)
    elseif format == :caffe
        output_path = convert_to_caffe(model_data, output_file, config)
    elseif format == :mxnet
        output_path = convert_to_mxnet(model_data, output_file, config)
    end
    
    return output_path
end

"""
    validate_tensor_file(filename::String; format=nothing)

Valida un archivo de tensor, verificando su integridad y estructura.

# Argumentos
- `filename::String`: Archivo a validar
- `format=nothing`: Formato del archivo (si es nothing, se infiere)

# Retorna
- `is_valid`: Boolean indicando si el archivo es válido
- `validation_report`: Informe de validación con detalles
"""
function validate_tensor_file(filename::String; format=nothing)
    # Verificar que el archivo existe
    if !isfile(filename)
        return false, Dict("error" => "Archivo no encontrado: $filename")
    end
    
    # Inferir formato si no se proporciona
    if format === nothing
        format = infer_format_from_extension(filename)
    end
    
    # Inicializar reporte
    validation_report = Dict(
        "filename" => filename,
        "format" => format,
        "file_size" => filesize(filename),
        "is_valid" => false,
        "errors" => String[],
        "warnings" => String[]
    )
    
    # Validar según formato
    try
        if format == :jld2
            validate_jld2_file(filename, validation_report)
        elseif format == :hdf5
            validate_hdf5_file(filename, validation_report)
        elseif format == :csv
            validate_csv_file(filename, validation_report)
        else
            push!(validation_report["errors"], "Formato no soportado para validación: $format")
            return false, validation_report
        end
    catch e
        push!(validation_report["errors"], "Error durante validación: $(string(e))")
        return false, validation_report
    end
    
    # Determinar validez basada en errores
    validation_report["is_valid"] = isempty(validation_report["errors"])
    
    return validation_report["is_valid"], validation_report
end

# Funciones auxiliares internas

"""Guarda un tensor en formato JLD2"""
function save_tensor_jld2(tensor, filename, compression, metadata)
    # Asegurar extensión correcta
    if !endswith(filename, ".jld2")
        filename = filename * ".jld2"
    end
    
    # Guardar tensor y metadatos
    jldopen(filename, "w") do file
        file["tensor"] = tensor
        file["dimensions"] = collect(size(tensor))
        file["type"] = string(eltype(tensor))
        
        if metadata !== nothing
            file["metadata"] = metadata
        end
    end
    
    return filename
end

"""Carga un tensor desde formato JLD2"""
function load_tensor_jld2(filename)
    data = load(filename)
    
    tensor = data["tensor"]
    metadata = haskey(data, "metadata") ? data["metadata"] : nothing
    
    return tensor, metadata
end

"""Guarda un tensor en formato HDF5"""
function save_tensor_hdf5(tensor, filename, compression, metadata)
    # Asegurar extensión correcta
    if !endswith(filename, ".h5") && !endswith(filename, ".hdf5")
        filename = filename * ".h5"
    end
    
    h5open(filename, "w") do file
        if compression
            write(file, "tensor", tensor, compress=true)
        else
            write(file, "tensor", tensor)
        end
        
        # Guardar dimensiones y tipo como atributos
        attrs(file["tensor"])["dimensions"] = collect(size(tensor))
        attrs(file["tensor"])["type"] = string(eltype(tensor))
        
        # Guardar metadatos si existen
        if metadata !== nothing
            # Convertir a JSON para compatibilidad
            metadata_json = JSON.json(metadata)
            write(file, "metadata", metadata_json)
        end
    end
    
    return filename
end

"""Carga un tensor desde formato HDF5"""
function load_tensor_hdf5(filename)
    h5open(filename, "r") do file
        if !haskey(file, "tensor")
            error("Dataset 'tensor' no encontrado en $filename")
        end
        
        tensor = read(file["tensor"])
        
        # Cargar metadatos si existen
        metadata = nothing
        if haskey(file, "metadata")
            metadata_json = read(file["metadata"])
            metadata = JSON.parse(metadata_json)
        end
        
        return tensor, metadata
    end
end

"""Guarda un tensor en formato CSV (solo para tensores 2D)"""
function save_tensor_csv(tensor, filename, metadata)
    # Verificar que el tensor es 2D
    if ndims(tensor) != 2
        error("Solo tensores 2D pueden guardarse en formato CSV")
    end
    
    # Asegurar extensión correcta
    if !endswith(filename, ".csv")
        filename = filename * ".csv"
    end
    
    # Guardar tensor como CSV
    open(filename, "w") do f
        for i in 1:size(tensor, 1)
            line = join(tensor[i, :], ",")
            println(f, line)
        end
    end
    
    # Si hay metadatos, guardarlos en un archivo JSON separado
    if metadata !== nothing
        meta_filename = replace(filename, ".csv" => "_metadata.json")
        metadata_to_json(metadata, meta_filename)
    end
    
    return filename
end

"""Carga un tensor desde formato CSV"""
function load_tensor_csv(filename)
    # Leer archivo CSV
    lines = readlines(filename)
    
    # Determinar dimensiones
    if isempty(lines)
        return zeros(Float32, 0, 0), nothing
    end
    
    # Dividir primera línea por comas para determinar número de columnas
    row_values = split(lines[1], ",")
    ncols = length(row_values)
    nrows = length(lines)
    
    # Crear tensor
    tensor = zeros(Float32, nrows, ncols)
    
    # Llenar tensor con datos
    for (i, line) in enumerate(lines)
        values = split(line, ",")
        for (j, val) in enumerate(values)
            if j <= ncols  # Asegurar que no excedemos las dimensiones
                tensor[i, j] = parse(Float32, val)
            end
        end
    end
    
    # Verificar si hay metadatos
    meta_filename = replace(filename, ".csv" => "_metadata.json")
    metadata = nothing
    if isfile(meta_filename)
        metadata = metadata_from_json(meta_filename)
    end
    
    return tensor, metadata
end

"""Infiere el formato de archivo desde la extensión"""
function infer_format_from_extension(filename)
    if endswith(filename, ".jld2")
        return :jld2
    elseif endswith(filename, ".h5") || endswith(filename, ".hdf5")
        return :hdf5
    elseif endswith(filename, ".csv")
        return :csv
    else
        error("No se puede inferir formato desde extensión: $filename")
    end
end

"""Prepara el modelo para exportación"""
function prepare_model_for_export(brain_space, include_weights, include_activations)
    # Crear estructura de datos para exportación
    export_data = Dict(
        "name" => brain_space.name,
        "dimensions" => brain_space.dimensions,
        "neurons" => []
    )
    
    # Exportar neuronas
    for neuron in brain_space.neurons
        neuron_data = Dict(
            "id" => neuron.id,
            "position" => [neuron.position.x, neuron.position.y, neuron.position.z],
            "type" => string(neuron.type)
        )
        
        if include_activations
            neuron_data["activation"] = neuron.activation
        end
        
        push!(export_data["neurons"], neuron_data)
    end
    
    # Exportar conexiones si se incluyen pesos
    if include_weights
        export_data["connections"] = []
        
        for conn in brain_space.connections
            conn_data = Dict(
                "source_id" => conn.source_id,
                "target_id" => conn.target_id,
                "weight" => conn.weight,
                "type" => string(conn.type)
            )
            
            push!(export_data["connections"], conn_data)
        end
    end
    
    return export_data
end

"""Guarda el modelo en formato JLD2"""
function save_model_jld2(export_data, filename, metadata)
    # Asegurar extensión correcta
    if !endswith(filename, ".jld2")
        filename = filename * ".jld2"
    end
    
    # Guardar modelo y metadatos
    jldopen(filename, "w") do file
        file["model_data"] = export_data
        file["metadata"] = metadata
    end
    
    return filename
end

"""Carga un modelo desde formato JLD2"""
function load_model_jld2(filename)
    data = load(filename)
    
    if !haskey(data, "model_data")
        error("Datos de modelo no encontrados en $filename")
    end
    
    model_data = data["model_data"]
    metadata = haskey(data, "metadata") ? data["metadata"] : Dict()
    
    return model_data, metadata
end

"""Guarda un modelo en formato HDF5"""
function save_model_hdf5(export_data, filename, metadata)
    # Asegurar extensión correcta
    if !endswith(filename, ".h5") && !endswith(filename, ".hdf5")
        filename = filename * ".h5"
    end
    
    h5open(filename, "w") do file
        # Convertir a formato compatible con HDF5
        # Guardar información general
        write(file, "name", export_data["name"])
        write(file, "dimensions", export_data["dimensions"])
        
        # Guardar neuronas
        g_neurons = g_create(file, "neurons")
        for (i, neuron) in enumerate(export_data["neurons"])
            g_neuron = g_create(g_neurons, "neuron_$i")
            for (key, value) in neuron
                write(g_neuron, key, value)
            end
        end
        
        # Guardar conexiones si existen
        if haskey(export_data, "connections")
            g_connections = g_create(file, "connections")
            for (i, conn) in enumerate(export_data["connections"])
                g_conn = g_create(g_connections, "connection_$i")
                for (key, value) in conn
                    write(g_conn, key, value)
                end
            end
        end
        
        # Guardar metadatos
        g_metadata = g_create(file, "metadata")
        for (key, value) in metadata
            # Para valores complejos, convertir a JSON
            if isa(value, Dict) || isa(value, Vector)
                write(g_metadata, key, JSON.json(value))
            else
                write(g_metadata, key, string(value))
            end
        end
    end
    
    return filename
end

end