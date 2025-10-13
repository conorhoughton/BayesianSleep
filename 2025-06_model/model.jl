using Printf

using Turing, Distributions, Random, Plots

# c-process function
function c_process(t, phi)
    sin.(2π .* (t .+ phi) ./ 24)
end

# s-process during wakefulness
function s_process_wake(s0, dt, tau_w, s_inf_w)
    s_inf_w .+ (s0 - s_inf_w) .* exp(-dt / tau_w)
end

# s-process during sleep
function s_process_sleep(s0, dt, tau_s, s_inf_s)
    s_inf_s .+ (s0 - s_inf_s) .* exp(-dt / tau_s)
end

# logistic (sigmoid) function
σ(x, slope) = 1 / (1 + exp(-slope * x))

# Priors (with better mean values)
σ_ϕ = 4.0
μ_ϕ = -10.0  # aligns sleepiness at 23:00

μ_τ_w = 8.0
μ_τ_s = 8.0

s_α = 5.0 #these aren't used at the moment
s_β = 2.5 #these aren't used at the moment

σ_θ = 1.0
μ_θ_w = 0.2
μ_θ_s = 0.0

μ_λ_w = 4.0
μ_λ_s = 10.0

# Bayesian prior model (unchanged for later use)
@model sleep_model_prior(t) = begin
    ϕ ~ Normal(μ_ϕ, σ_ϕ)
    τ_w ~ Exponential(1/μ_τ_w)
    τ_s ~ Exponential(1/μ_τ_s)
    s_inf ~ Beta(s_α, s_β) # I don't think this is correct!
    θ_w ~ Normal(μ_θ_w, σ_θ)
    θ_s ~ Normal(μ_θ_s, σ_θ)
    λ_w ~ Exponential(1/μ_λ_w)
    λ_s ~ Exponential(1/μ_λ_s)
    
    c = c_process(t, ϕ)
    s0 = s_inf
    s_wake = s_process_wake(s0, t, τ_w, s_inf)
    s_sleep = s_process_sleep(s0, t, τ_s, s_inf)
    p_wake = σ.(c .- s_wake .+ θ_w, λ_w)
    p_sleep = σ.(s_sleep .- c .+ θ_s, λ_s)
    return (c=c, s_wake=s_wake, s_sleep=s_sleep, p_wake=p_wake, p_sleep=p_sleep)
end

# Set deterministic parameters to their prior means for simulation
ϕ = μ_ϕ
τ_w = μ_τ_w
τ_s = μ_τ_s
s_inf_s = -1.0
s_inf_w = 0.4
θ_w = μ_θ_w
θ_s = μ_θ_s
λ_w = μ_λ_w
λ_s = μ_λ_s

maxT=10*24
plotWindow=3*24

# Simulation parameters
t = 0:0.1:240 
global state = "sleep"
global s = s_inf_s
s_array = fill(s_inf_s, length(t))  # initialize cleanly
c_array = c_process(t, ϕ)
state_array = zeros(Int, length(t))

Random.seed!(1234)

for i in 2:length(t)
    global state, s
    dt = t[i] - t[i-1]

    if state == "wake"
        s = s_process_wake(s, dt, τ_w, s_inf_w)
        p_sleep = σ(s - c_array[i] + θ_s, λ_s)
        state_array[i] = 1  # awake

        if rand() < p_sleep * dt
            state = "sleep"
        end
    else
        s = s_process_sleep(s, dt, τ_s, s_inf_s)
        p_wake = σ(c_array[i] - s + θ_w, λ_w)
        state_array[i] = 0  # asleep

        if rand() < p_wake * dt
            state = "wake"
        end
    end

    s_array[i] = s
end



# --- restrict to final 3 days ---
t_end   = last(t)
t_start = t_end - plotWindow            # last 72 h
idx     = findall(x -> x >= t_start, t) # indices for the window

t_plot = t[idx]
c_plot = c_array[idx]
s_plot = s_array[idx]
state_plot = state_array[idx]

# --- plot ---
plot(t_plot, c_plot, label="c-process (circadian)", linewidth=2, color=:black)
plot!(t_plot, s_plot, label="s-process", linewidth=2)

# y-lims first so vlines span full height
ylims!(-1.1, 1.2)

# dotted vertical lines at each midnight within the window
midnights = collect(ceil(Int, t_start/24)*24 : 24 : floor(Int, t_end/24)*24)
vline!(midnights, linestyle=:dot, color=:gray, linewidth=1, label="midnight")

# sleep/wake indicator along the bottom (blue sleep, yellow wake)
sleep_indicator = ifelse.(state_plot .== 0, -1.05, NaN)
wake_indicator  = ifelse.(state_plot .== 1, -1.05, NaN)
plot!(t_plot, wake_indicator, linewidth=5, color=:yellow, label="Wake", legend=:topright)
plot!(t_plot, sleep_indicator, linewidth=5, color=:blue,   label="Sleep")

# x-ticks every 12 h across the visible window
xt = collect(ceil(Int, t_start/12)*12 : 12 : floor(Int, t_end/12)*12)
xtlbl = [@sprintf("%02d:00", mod(x, 24)) for x in xt]  # clock-time labels
xticks!(xt, xtlbl)

xlabel!("Time (hours)")
ylabel!("Process value")
title!("Sleep-Wake Cycle Simulation (last 3 of 10 days)")
savefig("sleep_wake_simulation.png")
