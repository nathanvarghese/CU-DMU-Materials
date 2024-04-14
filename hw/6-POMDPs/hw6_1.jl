using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, Uniform
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using Statistics: mean
using LinearAlgebra


##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    bp_vec = zeros(length(states(up.m)))

    # Fill in code for belief update
    for sp in states(up.m)
        i = stateindex(up.m,sp)
        for s in states(up.m)
            bp_vec[i] += T(up.m,s, a, sp)* pdf(b,s)
        end
        bp_vec[i] *= Z(up.m,a,sp,o)
    end

    bp_vec /= sum(bp_vec)
    # Note that the ordering of the entries in bp_vec must be consistent with stateindex(m, s) (the container returned by states(m) does not necessarily obey this order)

    return DiscreteBelief(up.m, bp_vec)
end

# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
# Z(o | a, s') can be programmed with
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    belief_vec = beliefvec(b)
    i = argmax([dot(α, belief_vec) for α in p.alphas])
    return p.alpha_actions[i]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

#------
# QMDP
#------

function qmdp_solve(m, discount=discount(m))


    function value_iteration(m, epsilon=1e-6, max_iter=100000000000)
        A = collect(actions(m))
        V = ones(length(states(m)))*-100
        T = transition_matrices(m, sparse=true)
        R = reward_vectors(m)
        Q = zeros(length(states(m)), length(A))

    
        g = 0.99
    
        for _ in 1:max_iter
            Vp = copy(V)   
    
            for (i,a) in enumerate(A)
                Q[:,i]  = R[a] + g * T[a] * V
                # Vp = max.(V_p, EV_a)
            end
            Q_vectors = [Q[:,i] for i in 1:size(Q,2)]
            Vp = max.(Q_vectors...)
    
            if norm(Vp - V) < epsilon
                break
            else
                V = Vp
            end
        end
    
        return V
    end

    V = value_iteration(m, discount)

    acts = actiontype(m)[]
    alphas = Vector{Float64}[]
    T = transition_matrices(m,sparse = true)
    R = reward_vectors(m)
    
    for a in actions(m)
        push!(acts,a)
        Q_sa = R[a] .+ discount .* (T[a]*V)
        push!(alphas, Q_sa)
        # Fill in alpha vector calculation
        # Note that the ordering of the entries in the alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)       
    end
    return HW6AlphaVectorPolicy(alphas, acts)
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

@show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
@show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)


# using Plots

# n = length(sarsop_p.alphas[1])
# x = collect(1:n)
# for alpha_vector in sarsop_p.alphas
#     SARSOP_plot = plot!(x, alpha_vector, marker=:circle, label="", xlabel = "State", ylabel = "Alpha from SARSOP")
#     savefig(SARSOP_plot, "SARSOP_plot.png")  # Saving the SARSOP plot
# end

# j = length(qmdp_p.alphas[1])
# y = collect(1:j)
# for alpha_vector in qmdp_p.alphas
#     QMDP_plot = plot!(y, alpha_vector, marker=:circle, label="", xlabel = "State", ylabel = "Alpha from QMDP")
#     savefig(QMDP_plot, "QMDP_plot.png")  # Saving the QMDP_plot
# end




###################
# Problem 2: Cancer
###################

cancer = QuickPOMDP(

    # Fill in your actual code from last homework here

    states = [:healthy, :in_situ, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [true, false],
    isterminal = s->s==:death,

    # Define the initial state distribution
    initialstate = Deterministic(:healthy),
    discount = 0.99,

    # Define the transition function
    transition = function (s, a)
        if s == :healthy
            return SparseCat([:healthy, :in_situ], [0.98, 0.02])
        elseif s == :in_situ
            if a == :treat
                return SparseCat([:healthy, :in_situ], [0.6,0.4])
            else
                return SparseCat([:in_situ, :invasive], [0.9,0.1])
            end
        elseif s == :invasive
            if a == :treat
                return SparseCat([:healthy, :invasive, :death], [0.2, 0.6, 0.2])
            else
                return SparseCat([:invasive, :death], [0.4, 0.6])
            end
        else
            return Deterministic(s)
        end
    end,

    # Define the observation function
    observation = function (a, sp)
        if a == :test && sp == :healthy
            return SparseCat([true, false], [0.05, 0.95])
        elseif a == :test && sp == :in_situ 
            return SparseCat([true, false], [0.8, 0.2])
        elseif a == :test && sp == :invasive
            return Uniform([true])
        elseif a == :treat && (sp == :in_situ || sp == :invasive)
            return Uniform([true])
        end
        return Uniform([false])
    end,

    # Define the reward function
    reward = function (s, a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        else
            return 0.0
        end
    end,


)

# @assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)
    # Extract probabilities of different states from the belief b
    prob_healthy = pdf(b, :healthy)
    prob_in_situ = pdf(b, :in_situ)
    prob_invasive = pdf(b, :invasive)
    prob_death = pdf(b, :death)

    # If there's a high probability of being healthy, wait
    if prob_healthy > 0.9
        return :wait
    end

    # If there's a high probability of being in situ or invasive, treat
    if prob_in_situ + prob_invasive > 0.6
        return :treat
    end

    # If there's a high probability of death, wait
    if prob_death > 0.8
        return :wait
    end

    # Otherwise, default to testing
    return :test
end
)

@show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
@show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79

#####################
# Problem 3: LaserTag
#####################

m = LaserTagPOMDP()

qmdp_p = qmdp_solve(m)
up = DiscreteUpdater(m) # you may want to replace this with your updater to test it

# Use this version with only 100 episodes to check how well you are doing quickly
# @show HW6.evaluate((qmdp_p, up), n_episodes=100)
@show HW6.evaluate((qmdp_p, up), n_episodes=5000, "nathan.varghese@colorado.edu")
