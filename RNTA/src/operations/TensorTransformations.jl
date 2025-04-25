# operations/TensorTransformations.jl
# Implementa transformaciones fundamentales para tensores 3D
module TensorOperations

export tensor_convolution, adaptive_pooling, tensor_interpolation, 
       spatial_attention_transform, zero_pad, distance_weights
"""
    tensor_convolution(input, kernel; stride=(1,1,1), padding=0)

Aplica una convolución 3D adaptativa al tensor de entrada.
"""
function tensor_convolution(
    input::Array{T,3}, 
    kernel::Array{S,3}; 
    stride::NTuple{3,Int}=(1,1,1), 
    padding::Int=0
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Obtener dimensiones
    in_dim_x, in_dim_y, in_dim_z = size(input)
    k_dim_x, k_dim_y, k_dim_z = size(kernel)
    
    # Calcular dimensiones de salida
    out_dim_x = div(in_dim_x - k_dim_x + 2*padding, stride[1]) + 1
    out_dim_y = div(in_dim_y - k_dim_y + 2*padding, stride[2]) + 1
    out_dim_z = div(in_dim_z - k_dim_z + 2*padding, stride[3]) + 1
    
    # Inicializar tensor de salida
    output = zeros(promote_type(T, S), out_dim_x, out_dim_y, out_dim_z)
    
    # Aplicar padding si es necesario
    if padding > 0
        padded_input = zero_pad(input, padding)
    else
        padded_input = input
    end
    
    # Realizar convolución
    for x in 1:out_dim_x
        for y in 1:out_dim_y
            for z in 1:out_dim_z
                # Calcular índices de inicio en el input
                start_x = (x - 1) * stride[1] + 1
                start_y = (y - 1) * stride[2] + 1
                start_z = (z - 1) * stride[3] + 1
                
                # Extraer región para esta operación
                region = padded_input[
                    start_x:start_x+k_dim_x-1,
                    start_y:start_y+k_dim_y-1,
                    start_z:start_z+k_dim_z-1
                ]
                
                # Aplicar kernel
                output[x, y, z] = sum(region .* kernel)
            end
        end
    end
    
    return output
end

"""
    adaptive_pooling(input, output_size; mode=:max)

Aplica pooling adaptativo para redimensionar el tensor a output_size.
"""
function adaptive_pooling(
    input::Array{T,3}, 
    output_size::NTuple{3,Int}; 
    mode::Symbol=:max
) where T <: AbstractFloat
    in_dim_x, in_dim_y, in_dim_z = size(input)
    out_dim_x, out_dim_y, out_dim_z = output_size
    
    # Calcular tamaños de ventana de pooling
    window_x = div(in_dim_x, out_dim_x)
    window_y = div(in_dim_y, out_dim_y)
    window_z = div(in_dim_z, out_dim_z)
    
    # Asegurar que los tamaños de ventana son al menos 1
    window_x = max(1, window_x)
    window_y = max(1, window_y)
    window_z = max(1, window_z)
    
    # Inicializar tensor de salida
    output = zeros(T, output_size)
    
    # Realizar pooling
    for x in 1:out_dim_x
        for y in 1:out_dim_y
            for z in 1:out_dim_z
                # Calcular índices de la ventana
                start_x = (x - 1) * window_x + 1
                start_y = (y - 1) * window_y + 1
                start_z = (z - 1) * window_z + 1
                
                # Calcular índices finales (asegurando que no exceda dimensiones)
                end_x = min(start_x + window_x - 1, in_dim_x)
                end_y = min(start_y + window_y - 1, in_dim_y)
                end_z = min(start_z + window_z - 1, in_dim_z)
                
                # Extraer región para esta operación de pooling
                region = input[start_x:end_x, start_y:end_y, start_z:end_z]
                
                # Aplicar operación de pooling según el modo
                if mode == :max
                    output[x, y, z] = maximum(region)
                elseif mode == :avg
                    output[x, y, z] = mean(region)
                elseif mode == :weighted_avg
                    # Pooling con peso por distancia al centro
                    weights = distance_weights(size(region))
                    output[x, y, z] = sum(region .* weights) / sum(weights)
                end
            end
        end
    end
    
    return output
end

"""
    tensor_interpolation(input, output_size; mode=:linear)

Redimensiona un tensor mediante interpolación.
"""
function tensor_interpolation(
    input::Array{T,3}, 
    output_size::NTuple{3,Int}; 
    mode::Symbol=:linear
) where T <: AbstractFloat
    in_dim_x, in_dim_y, in_dim_z = size(input)
    out_dim_x, out_dim_y, out_dim_z = output_size
    
    # Si las dimensiones son iguales, devolver copia
    if (in_dim_x, in_dim_y, in_dim_z) == output_size
        return copy(input)
    end
    
    # Inicializar tensor de salida
    output = zeros(T, output_size)
    
    # Calcular factores de escala
    scale_x = (in_dim_x - 1) / (out_dim_x - 1)
    scale_y = (in_dim_y - 1) / (out_dim_y - 1)
    scale_z = (in_dim_z - 1) / (out_dim_z - 1)
    
    # Manejar caso especial de una sola unidad en alguna dimensión
    if in_dim_x == 1
        scale_x = 0
    end
    if in_dim_y == 1
        scale_y = 0
    end
    if in_dim_z == 1
        scale_z = 0
    end
    
    for x in 1:out_dim_x
        for y in 1:out_dim_y
            for z in 1:out_dim_z
                # Calcular coordenadas correspondientes en el input
                input_x = 1 + (x - 1) * scale_x
                input_y = 1 + (y - 1) * scale_y
                input_z = 1 + (z - 1) * scale_z
                
                if mode == :nearest
                    # Interpolación por vecino más cercano
                    nx = round(Int, input_x)
                    ny = round(Int, input_y)
                    nz = round(Int, input_z)
                    
                    # Asegurar que estamos dentro de los límites
                    nx = max(1, min(nx, in_dim_x))
                    ny = max(1, min(ny, in_dim_y))
                    nz = max(1, min(nz, in_dim_z))
                    
                    output[x, y, z] = input[nx, ny, nz]
                    
                elseif mode == :linear
                    # Interpolación trilineal
                    # Índices de los vértices del cubo que rodea el punto
                    x0 = floor(Int, input_x)
                    y0 = floor(Int, input_y)
                    z0 = floor(Int, input_z)
                    
                    # Asegurar que los índices están dentro de los límites
                    x0 = max(1, min(x0, in_dim_x-1))
                    y0 = max(1, min(y0, in_dim_y-1))
                    z0 = max(1, min(z0, in_dim_z-1))
                    
                    x1 = min(x0 + 1, in_dim_x)
                    y1 = min(y0 + 1, in_dim_y)
                    z1 = min(z0 + 1, in_dim_z)
                    
                    # Pesos para la interpolación
                    wx = input_x - x0
                    wy = input_y - y0
                    wz = input_z - z0
                    
                    # Interpolación
                    c00 = input[x0, y0, z0] * (1 - wx) + input[x1, y0, z0] * wx
                    c01 = input[x0, y0, z1] * (1 - wx) + input[x1, y0, z1] * wx
                    c10 = input[x0, y1, z0] * (1 - wx) + input[x1, y1, z0] * wx
                    c11 = input[x0, y1, z1] * (1 - wx) + input[x1, y1, z1] * wx
                    
                    c0 = c00 * (1 - wy) + c10 * wy
                    c1 = c01 * (1 - wy) + c11 * wy
                    
                    output[x, y, z] = c0 * (1 - wz) + c1 * wz
                end
            end
        end
    end
    
    return output
end

"""
    spatial_attention_transform(input, attention_map)

Aplica una transformación atencional ponderando regiones según un mapa de atención.
"""
function spatial_attention_transform(
    input::Array{T,3}, 
    attention_map::Array{S,3}
) where {T <: AbstractFloat, S <: AbstractFloat}
    # Asegurar que las dimensiones coincidan
    if size(input) != size(attention_map)
        attention_map = tensor_interpolation(attention_map, size(input))
    end
    
    # Aplicar atención (multiplicación elemento a elemento)
    return input .* attention_map
end

# Funciones auxiliares

"""
    zero_pad(tensor, padding)

Añade padding de ceros alrededor del tensor.
"""
function zero_pad(tensor::Array{T,3}, padding::Int) where T <: AbstractFloat
    dim_x, dim_y, dim_z = size(tensor)
    
    padded = zeros(T, dim_x + 2*padding, dim_y + 2*padding, dim_z + 2*padding)
    
    # Copiar tensor original al centro del padded
    padded[padding+1:padding+dim_x, 
           padding+1:padding+dim_y, 
           padding+1:padding+dim_z] = tensor
    
    return padded
end

"""
    distance_weights(region_size)

Genera pesos basados en distancia al centro para pooling ponderado.
"""
function distance_weights(region_size::NTuple{3,Int})
    dim_x, dim_y, dim_z = region_size
    
    # Centro de la región
    center_x = (dim_x + 1) / 2
    center_y = (dim_y + 1) / 2
    center_z = (dim_z + 1) / 2
    
    # Inicializar matriz de pesos
    weights = zeros(Float32, region_size)
    
    # Calcular pesos basados en distancia al centro
    for x in 1:dim_x
        for y in 1:dim_y
            for z in 1:dim_z
                # Distancia al centro
                dist = sqrt((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2)
                
                # Peso inversamente proporcional a la distancia
                weights[x, y, z] = 1.0f0 / (1.0f0 + dist)
            end
        end
    end
    
    # Normalizar pesos
    weights ./= sum(weights)
    
    return weights
end
end