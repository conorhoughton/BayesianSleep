using Random
using Distributions
using DataFrames
using CSV

include("priors.jl")

c_process(t, ϕ) = sin.(2π .* (t .+ ϕ) ./ 24)
s_process_wake(s0, dt, τ_w, s_inf_w) = s_inf_w + (s0 - s_inf_w) * exp(-dt / τ_w)
s_process_sleep(s0, dt, τ_s, s_inf_s) = s_inf_s + (s0 - s_inf_s) * exp(-dt / τ_s)
σ(x, slope) = 1 / (1 + exp(-slope * x))  # logistic

# --- simulation controls ---
const dt = 0.1                 # hours
const burnin_h = 7 * 24        # 1 week
const record_h = 3 * 24        # 3 days
const t = 0:dt:(burnin_h + record_h)
const nSubjects = 10

Random.seed!(1234)

function draw_subject_params()
    ϕ   = rand(Normal(μ_ϕ,  σ_ϕ))
    τ_w = rand(Exponential(1/μ_τ_w))
    τ_s = rand(Exponential(1/μ_τ_s))
    s_inf_w = rand(Normal(μ_inf_w, σ_inf_w))
    s_inf_s = rand(Normal(μ_inf_s, σ_inf_s))
    θ_w = rand(Normal(μ_θ_w, σ_θ))
    θ_s = rand(Normal(μ_θ_s, σ_θ))
    λ_w = rand(Exponential(1/μ_λ_w))
    λ_s = rand(Exponential(1/μ_λ_s))
    return (; ϕ, τ_w, τ_s, s_inf_w, s_inf_s, θ_w, θ_s, λ_w, λ_s)
end

function simulate_subject(subject_id::Int)
    p = draw_subject_params()
    c = c_process(t, p.ϕ)

    state = "sleep"                 # start asleep
    s = p.s_inf_s                   # start near sleep asymptote
    events = NamedTuple[]           # (subject, event, indicator, time)

    event_no = 1
    for i in 2:length(t)
        Δt = t[i] - t[i-1]

        if state == "wake"
            s = s_process_wake(s, Δt, p.τ_w, p.s_inf_w)
            p_sleep = σ(s - c[i] + p.θ_s, p.λ_s)         # per-step prob; scaled by dt below
            if rand() < p_sleep * Δt
                state = "sleep"
                # record wake->sleep as "s"
                if t[i] >= burnin_h && t[i] <= burnin_h + record_h
                    push!(events, (subject=subject_id, event=event_no, indicator="s", time=t[i]-burnin_h))
                    event_no += 1
                end
            end
        else
            s = s_process_sleep(s, Δt, p.τ_s, p.s_inf_s)
            p_wake = σ(c[i] - s + p.θ_w, p.λ_w)
            if rand() < p_wake * Δt
                state = "wake"
                # record sleep->wake as "w"
                if t[i] >= burnin_h && t[i] <= burnin_h + record_h
                    push!(events, (subject=subject_id, event=event_no, indicator="w", time=t[i]-burnin_h))
                    event_no += 1
                end
            end
        end

        if t[i] > burnin_h + record_h
            break
        end
    end

    return events
end


all_events = reduce(vcat, (simulate_subject(i) for i in 1:nSubjects))
# columns: subject, event, indicator, time (hours since start of 3-day window)
df = DataFrame(all_events)

filename="results.csv"

CSV.write(filename, df)

println("saved results to "*filename)
