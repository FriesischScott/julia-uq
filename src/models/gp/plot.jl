@recipe function f(gp::GaussianProcess; β=0.95, obsv=true)
    if gp.gpBase.dim == 1
        X = [to_physical(v, gp.std.min, gp.std.max) for v in gp.gpBase.x]
        xlims --> (minimum(X), maximum(X))
        xmin, xmax = plotattributes[:xlims]
        x = range(0, 1, 100)
        μ, Σ = predict_f(gp.gpBase, x)

        y = μ
        err = invΦ((1 + β) / 2) * sqrt.(Σ)

        x = [to_physical(v, gp.std.min, gp.std.max) for v in x]

        @series begin
            seriestype := :path
            ribbon := err
            fillcolor --> :lightblue
            seriescolor --> :black
            label --> "Mean with $β credible bands"
            x, y
        end
        if obsv
            @series begin
                seriestype := :scatter
                markershape := :circle
                markercolor := :black
                label --> "Data"
                X', gp.gpBase.y
            end
        end
    else
        error("Selected Gaussian Procces has dimension $gp.gpBase.dim grater than 1")
    end
end