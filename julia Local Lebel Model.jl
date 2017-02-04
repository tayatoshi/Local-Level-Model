
using Distributions
using Gadfly

α = Vector{Float64}(50)
η = Normal(0,5)
ɛ = Normal(0,5)
α[1] = exp(2)+80.0
for i in 2:50
    α[i] = α[i-1] + rand(η)
end
y = α .+ rand(ɛ, length(α))

plot(
    layer(y=y,Geom.line,Geom.point),
    layer(y=α,Geom.line,Theme(default_color=color("red"))),
    Guide.manual_color_key("Legend", ["観測値", "状態"],
                           [color("deepskyblue"), color("red")])
    )

function kalmanfilter(a0::Float64, p0::Float64, Var_ɛ::Float64, Var_η::Float64)
    #a0:一期先状態予測(Float64)
    #p0:一期先状態分散(Float64)
    a = a0
    p = p0
    A_t = Vector{Float64}(50)
    P_t = Vector{Float64}(50)
    for i in 1:length(y)
        υ = y[i] - a
        f = p + Var_ɛ
        K = p/f
        L = 1 - K
        a_t = a + K*υ
        A_t[i] = a_t
        p_t = p*L
        P_t[i] = p_t
        a = A_t[i]
        p = P_t[i] + Var_η
    end
    plot(
    layer(y=y,Geom.line),
    layer(y=α,Geom.line,Theme(default_color=color("red"))),
    layer(y=A_t,Geom.line,Theme(default_color=color("green"))),
    Guide.manual_color_key("Legend", ["観測値", "真の状態","フィルタ化推定量"],
    [color("deepskyblue"), color("red"),color("green")])
    )
end

kalmanfilter(mean(α),5000.0,5.0,5.0)
