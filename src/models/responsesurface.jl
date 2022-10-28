"""
    ResponseSurface(data::DataFrame, dependendVarName::Symbol, deg::Int64, dim::Int64)

Creates a response surface using polynomial least squares regression with given degree.

# Examples
```jldoctest
julia> data = DataFrame(x = 1:10, y = [1, 4, 10, 15, 24, 37, 50, 62, 80, 101])
10×2 DataFrame
 Row │ x      y     
     │ Int64  Int64 
─────┼──────────────
   1 │     1      1
   2 │     2      4
   3 │     3     10
   4 │     4     15
   5 │     5     24
   6 │     6     37
   7 │     7     50
   8 │     8     62
   9 │     9     80
  10 │    10    101

julia> rs = ResponseSurface(data, :y, 2)
ResponseSurface([1.018939393939398, -0.23863636363631713, 0.4833333333332348], :y, [:x], 2, Monomial{true}[x₁², x₁, 1])
```
"""
struct ResponseSurface <: UQModel
    β::Array
    y::Symbol
    names::Array{Symbol}
    p::Int64
    monomials::MonomialVector{true}

    function ResponseSurface(data::DataFrame, output::Symbol, p::Int64)
        if p < 0
            error("Degree(p) of ResponseSurface must be non-negative.")
        end

        @polyvar x[1:size(data, 2) - 1]
        m = monomials(x, 0:p)

        names = propertynames(data[:, Not(output)])

        X = Matrix{Float64}(data[:, names]) # convert to matrix, sort by rs.names
        y = Vector{Float64}(data[:, output])

        β = multi_dimensional_polynomial_regression(X, y, m)

        return new(β, output, names, p, m)
    end
end



# only to be called internally by constructor
function multi_dimensional_polynomial_regression(X::Matrix, y::Vector, monomials::MonomialVector{true})

    #fill monomials with the given x values for each row
    M = mapreduce(row -> begin
        return map(m -> m(row), monomials)'
    end, vcat, eachrow(X))

    return M \ y   #β
end



#called internally by evaluate!
#evaluates one datapoint using a given ResponseSurface
function calc(row::Array, rs::ResponseSurface)
    map(m -> m(row), rs.monomials') * rs.β
end



"""
    evaluate!(rs::ResponseSurface, data::DataFrame)

evaluating data by using a previously trained ResponseSurface.




# Examples

```jldoctest

julia> data = DataFrame(x = 1:10, y = [1, 4, 10, 15, 24, 37, 50, 62, 80, 101])
10×2 DataFrame
 Row │ x      y     
     │ Int64  Int64 
─────┼──────────────
   1 │     1      1
   2 │     2      4
   3 │     3     10
   4 │     4     15
   5 │     5     24
   6 │     6     37
   7 │     7     50
   8 │     8     62
   9 │     9     80
  10 │    10    101

julia> rs = ResponseSurface(data, :y, 2, 2)
ResponseSurface([0.4833333333331211, -0.23863636363637397, 1.018939393939391], :y, 2, 2)
julia> evaluate!(rs, [2.5, 11, 15])
```
"""
function evaluate!(rs::ResponseSurface, data::DataFrame)
    x = Matrix{Float64}(data[:, rs.names]) # convert to matrix, sort by rs.names
    out = map(row -> (calc(convert(Array, row), rs)), eachrow(x)) # fill monomial, evaluate given data with ResponseSurface
    data[!, rs.y] = out
    return nothing
end
