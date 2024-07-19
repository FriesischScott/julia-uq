get_union_types(x::Union) = (x.a, get_union_types(x.b)...)
get_union_types(x::Type) = (x,)