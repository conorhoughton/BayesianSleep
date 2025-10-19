# turing_sleep.jl
# Posterior inference for subject-specific parameters from event times using Turing.jl
# Fits each subject independently with the same priors used to generate the data.

using Random
using Distributions
using DataFrames
using CSV
using Turing
using MCMCChains

# ---------- model helpers (must match your simulator) ----------
include("priors.jl")

σ(x, slope) = 1 / (1 + exp(-slope * x))  # logistic
c_process(tt, ϕ) = sin.(2π .* (tt .+ ϕ) ./ 24)

# single-step updates (scalar Δt)
s_process_wake(s0, Δt, τ_w, s_inf_w) = s_inf_w + (s0 - s_inf_w) * exp(-Δt / τ_w)
s_process_sleep(s0, Δt, τ_s, s_inf_s) = s_inf_s + (s0 - s_inf_s) * exp(-Δt / τ_s)

# ---------- data prep ----------
const dt = 0.1                 # hours, must match simulator
const record_h = 3 * 24        # 3 days
const tt = collect(0:dt:record_h)   # time grid (length T)
const Tsteps = length(tt)

"""
Construct the binary event-indicator vector y (length Tsteps) for one subject,
AND return the initial discrete state at t=0 ("wake" or "sleep").
The rule matches the simulator: an observed indicator "w" is sleep->wake at that time; "s" is wake->sleep.
The state between events is constant; state flips at event indices where y[i]==1.
"""
function subject_y_and_initial_state(df::DataFrame, subject::Int)
    sdf = sort(df[df.subject .== subject, :], :time)
    y = zeros(Int, Tsteps)
    init_state = "sleep"  # fallback
    if nrow(sdf) > 0
        first_ind = sdf.indicator[1]
        init_state = first_ind == "w" ? "sleep" : "wake"
        for r in eachrow(sdf)
            # Map event time to nearest grid index
            idx = clamp(round(Int, r.time / dt) + 1, 1, Tsteps)
            y[idx] = 1
        end
    end
    return y, init_state
end

# ---------- Turing model for ONE subject ----------
@model function sleep_transition_model(y::AbstractVector{<:Integer}, init_state::AbstractString, tt::AbstractVector{<:Real})
    # Priors (match priors.jl used in the sim)
    ϕ   ~ Normal(μ_ϕ,  σ_ϕ)
    τ_w ~ Exponential(1/μ_τ_w)
    τ_s ~ Exponential(1/μ_τ_s)

    s_inf_w ~ Normal(μ_inf_w, σ_inf_w)
    s_inf_s ~ Normal(μ_inf_s, σ_inf_s)

    θ_w ~ Normal(μ_θ_w, σ_θ)
    θ_s ~ Normal(μ_θ_s, σ_θ)

    λ_w ~ Exponential(1/μ_λ_w)
    λ_s ~ Exponential(1/μ_λ_s)

    # Deterministic forward pass over the grid
    c = c_process(tt, ϕ)

    # Initial homeostatic level (uninformative but stable choice): start at asymptote of starting state
    s = init_state == "wake" ? s_inf_w : s_inf_s

    state_is_wake = (init_state == "wake")

    @inbounds for i in 2:length(tt)
        Δt = tt[i] - tt[i-1]
        # Propagate s given current state
        if state_is_wake
            s = s_process_wake(s, Δt, τ_w, s_inf_w)
            p = σ(s - c[i] + θ_s, λ_s)             # hazard of switching to sleep
        else
            s = s_process_sleep(s, Δt, τ_s, s_inf_s)
            p = σ(c[i] - s + θ_w, λ_w)             # hazard of switching to wake
        end
        # Discrete-time hazard to Bernoulli step probability
        pstep = clamp(p * Δt, 1e-9, 1 - 1e-9)
        y[i] ~ Bernoulli(pstep)
        # If an event is observed at this step, the state flips for subsequent steps
        if y[i] == 1
            state_is_wake = !state_is_wake
        end
    end
end

# ---------- POOLED model over ALL subjects ----------
# Shared parameters across subjects: φ̄, τ̄_w, τ̄_s, θ̄_w, θ̄_s, λ̄_w, λ̄_s, s̄_∞w, s̄_∞s
# We keep weakly-informative priors centered on priors.jl values.
@model function pooled_transition_model(Y::Vector{<:AbstractVector{<:Integer}}, init_states::Vector{String}, tt::AbstractVector{<:Real})
    # Shared parameters (renamed to avoid collisions with priors constants)
    φ̄   ~ Normal(μ_ϕ,  σ_ϕ)                      # circadian phase (hours)

    # Positive scales: use LogNormal or Exponential. Here LogNormal around prior means.
    τ̄_w ~ LogNormal(log(μ_τ_w), 0.5)
    τ̄_s ~ LogNormal(log(μ_τ_s), 0.5)

    s̄_∞w ~ Normal(μ_inf_w, σ_inf_w)
    s̄_∞s ~ Normal(μ_inf_s, σ_inf_s)

    θ̄_w ~ Normal(μ_θ_w, σ_θ)
    θ̄_s ~ Normal(μ_θ_s, σ_θ)

    λ̄_w ~ LogNormal(log(μ_λ_w), 0.5)
    λ̄_s ~ LogNormal(log(μ_λ_s), 0.5)

    c = c_process(tt, φ̄)

    @inbounds for sidx in eachindex(Y)
        y = Y[sidx]
        state_is_wake = (init_states[sidx] == "wake")
        s = state_is_wake ? s̄_∞w : s̄_∞s
        for i in 2:length(tt)
            Δt = tt[i] - tt[i-1]
            if state_is_wake
                s = s_process_wake(s, Δt, τ̄_w, s̄_∞w)
                p = σ(s - c[i] + θ̄_s, λ̄_s)
            else
                s = s_process_sleep(s, Δt, τ̄_s, s̄_∞s)
                p = σ(c[i] - s + θ̄_w, λ̄_w)
            end
            pstep = clamp(p * Δt, 1e-9, 1 - 1e-9)
            y[i] ~ Bernoulli(pstep)
            if y[i] == 1
                state_is_wake = !state_is_wake
            end
        end
    end
end

# ---------- fitting helpers ----------
function fit_subject!(chains_by_subject::Dict{Int,Chains}, df::DataFrame; subject::Int,
                      n_samples::Int=1000, n_adapt::Int=1000, n_chains::Int=4, seed::Int=2025)
    y, init_state = subject_y_and_initial_state(df, subject)
    # Ensure at least one event exists; otherwise the model becomes weakly identified.
    if sum(y) == 0
        @warn "Subject $subject has no events in the window; skipping."
        return chains_by_subject
    end
    Random.seed!(seed + subject)
    model = sleep_transition_model(y, init_state, tt)
    sampler = NUTS(n_adapt, 0.8)
    chns = sample(model, sampler, MCMCThreads(), n_samples, n_chains; progress=true)
    chains_by_subject[subject] = chns
    return chains_by_subject
end

function fit_all(; results_csv::AbstractString="results.csv", subjects::Union{Nothing,AbstractVector{Int}}=nothing,
                 n_samples::Int=1000, n_adapt::Int=1000, n_chains::Int=4)
    df = CSV.read(results_csv, DataFrame)
    subj_list = isnothing(subjects) ? sort!(unique(df.subject)) : subjects
    chains_by_subject = Dict{Int,Chains}()
    for s in subj_list
        fit_subject!(chains_by_subject, df; subject=s, n_samples=n_samples, n_adapt=n_adapt, n_chains=n_chains)
    end
    return chains_by_subject
end

# Fit the pooled model over all subjects and return a single Chains object
function fit_pooled(; results_csv::AbstractString="results.csv", n_samples::Int=2000, n_adapt::Int=1500, n_chains::Int=4, seed::Int=2025)
    df = CSV.read(results_csv, DataFrame)
    subj_list = sort!(unique(df.subject))
    Y = Vector{Vector{Int}}()
    init_states = String[]
    for s in subj_list
        y, init_state = subject_y_and_initial_state(df, s)
        push!(Y, y); push!(init_states, init_state)
    end
    Random.seed!(seed)
    model = pooled_transition_model(Y, init_states, tt)
    sampler = NUTS(n_adapt, 0.8)
    chns = sample(model, sampler, MCMCThreads(), n_samples, n_chains; progress=true)
    return chns
end

# ---------- quick demo when run as a script ----------
if abspath(PROGRAM_FILE) == @__FILE__
    chains = fit_all()
    for (s, ch) in sort(collect(chains); by=first)
        println("\nSubject ", s, ":")
        display(describe(ch))
    end
end
