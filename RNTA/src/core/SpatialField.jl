# core/SpatialField.jl
# Define el campo espacial que representa una región en el espacio 3D

"""
    SpatialField

Representa una región tridimensional en el espacio cerebral.
"""
# Añadir al inicio del archivo:
module SpatialField

export Spatial_Field, contains, distance, overlap, shift_field!, expand_field, extract_tensor, clone
struct Spatial_Field
    # Centro del campo
    center::NTuple{3,Int}
    
    # Tamaño/extensión del campo en cada dimensión
    size::NTuple{3,Int}
    
    # Bordes del campo (región cubierta)
    bounds::NTuple{3,UnitRange{Int}}
end

"""
Constructor que calcula los límites a partir del centro y tamaño
"""
function Spatial_Field(center::NTuple{3,Int}, size::NTuple{3,Int})
    # Calcular los límites del campo
    half_x = div(size[1], 2)
    half_y = div(size[2], 2)
    half_z = div(size[3], 2)
    
    bounds = (
        (center[1] - half_x):(center[1] + half_x),
        (center[2] - half_y):(center[2] + half_y),
        (center[3] - half_z):(center[3] + half_z)
    )
    
    return SpatialField(center, size, bounds)
end

"""
Constructor alternativo que crea el campo directamente a partir de los límites
"""
function Spatial_Field(bounds::NTuple{3,UnitRange{Int}})
    # Calcular el centro y tamaño a partir de los límites
    center = (
        div(bounds[1].start + bounds[1].stop, 2),
        div(bounds[2].start + bounds[2].stop, 2),
        div(bounds[3].start + bounds[3].stop, 2)
    )
    
    size = (
        length(bounds[1]),
        length(bounds[2]),
        length(bounds[3])
    )
    
    return SpatialField(center, size, bounds)
end

"""
    contains(field, position)

Verifica si una posición está contenida dentro del campo.
"""
function contains(field::Spatial_Field, position::NTuple{3,Int})
    return position[1] in field.bounds[1] &&
           position[2] in field.bounds[2] &&
           position[3] in field.bounds[3]
end

"""
    distance(field, position)

Calcula la distancia desde una posición al centro del campo.
"""
function distance(field::Spatial_Field, position::NTuple{3,Int})
    return sqrt(
        (position[1] - field.center[1])^2 +
        (position[2] - field.center[2])^2 +
        (position[3] - field.center[3])^2
    )
end

"""
    overlap(field1, field2)

Calcula la superposición entre dos campos espaciales.
"""
function overlap(field1::Spatial_Field, field2::Spatial_Field)
    # Calcular intersección de límites
    x_overlap = max(0, min(field1.bounds[1].stop, field2.bounds[1].stop) - 
                       max(field1.bounds[1].start, field2.bounds[1].start) + 1)
    
    y_overlap = max(0, min(field1.bounds[2].stop, field2.bounds[2].stop) - 
                       max(field1.bounds[2].start, field2.bounds[2].start) + 1)
    
    z_overlap = max(0, min(field1.bounds[3].stop, field2.bounds[3].stop) - 
                       max(field1.bounds[3].start, field2.bounds[3].start) + 1)
    
    # Volumen de superposición
    return x_overlap * y_overlap * z_overlap
end

"""
    shift_field!(field, direction, amount)

Desplaza el campo en la dirección especificada.
"""
function shift_field(field::Spatial_Field, direction::NTuple{3,Int}, amount::Float32)
    # Calcular el nuevo centro
    new_center = (
        field.center[1] + round(Int, direction[1] * amount),
        field.center[2] + round(Int, direction[2] * amount),
        field.center[3] + round(Int, direction[3] * amount)
    )
    
    # Actualizar los límites
    half_x = div(field.size[1], 2)
    half_y = div(field.size[2], 2)
    half_z = div(field.size[3], 2)
    
    new_bounds = (
        (new_center[1] - half_x):(new_center[1] + half_x),
        (new_center[2] - half_y):(new_center[2] + half_y),
        (new_center[3] - half_z):(new_center[3] + half_z)
    )
    
    # Crear un nuevo campo con los valores actualizados
    return SpatialField(new_center, field.size, new_bounds)
end

"""
    expand_field(field, factor)

Expande el campo por un factor dado.
"""
function expand_field(field::Spatial_Field, factor::Float32)
    # Calcular nuevo tamaño
    new_size = (
        round(Int, field.size[1] * factor),
        round(Int, field.size[2] * factor),
        round(Int, field.size[3] * factor)
    )
    
    # Crear nuevo campo con mismo centro pero mayor tamaño
    return SpatialField(field.center, new_size)
end

"""
    extract_tensor(tensor, field)

Extrae la región del tensor correspondiente al campo.
"""
function extract_tensor(tensor::Array{T,3}, field::Spatial_Field) where T <: AbstractFloat
    # Asegurarse de que los límites están dentro del tensor
    valid_x_range = max(1, field.bounds[1].start):min(size(tensor, 1), field.bounds[1].stop)
    valid_y_range = max(1, field.bounds[2].start):min(size(tensor, 2), field.bounds[2].stop)
    valid_z_range = max(1, field.bounds[3].start):min(size(tensor, 3), field.bounds[3].stop)
    
    # Extraer la región
    return tensor[valid_x_range, valid_y_range, valid_z_range]
end

"""
    clone(field)

Crea una copia profunda del campo.
"""
function clone(field::Spatial_Field)
    return SpatialField(field.center, field.size, field.bounds)
end
end