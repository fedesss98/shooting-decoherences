using DecoKiller.BB84Circuit
using Revise
using Random

include("../src/utils.jl")


config_path = joinpath(@__DIR__, "..", "..", "configs", "config.toml")

config = read_config(normpath(config_path))
N = config["general"]["n_qubits"]
if N > 20
    println("Too many qubits, cutting to 12.")
    N = 12
end

seed = config["general"]["seed"]
Random.seed!(seed)
dim = 2^N

alice_basis = rand(["X","Z"], N)
bob_basis = rand(["X","Z"], N)

# Alice encodes her (random) key
key = rand([0, 1], N)

alice_states = [encode_bit(bit, basis) for (bit, basis) in zip(key, alice_basis)]
ρa = reduce(kron, alice_states)

bob_bits = [decode_bit(state, basis) for (state, basis) in zip(alice_states, bob_basis)]

# Take the first half of the key to check for eavesdroppers: 
# here everything is shared publicly and the fidelity is measured.
# Alice share half of her bases to Bob and they check matches
n_check = get(config["general"], "eave_check", div(N, 2))
shared_a_basis = alice_basis[1:n_check]
shared_b_basis = bob_basis[1:n_check]
bob_matches = [x == y for (x, y) in zip(shared_a_basis, shared_b_basis)]
# Alice share her encoding for the matched bits
shared_encoding = alice_states[1:n_check][bob_matches]
# Bob creates an encoding from his measurements
shared_b_bits = bob_bits[1:n_check]
bob_encoding = [encode_bit(bit, basis) 
    for (bit, basis) 
    in zip(shared_b_bits[bob_matches], shared_b_basis[bob_matches])
]
# They compare the fidelity of their states
ψa = reduce(kron, shared_encoding)
ψb = reduce(kron, bob_encoding)
ρa = ψa * ψa'
ρb = ψb * ψb'

f = fidelity(ρa, ρb)

# Now Alice only shares her remaining bases and Bob checks matches
shared_a_basis = alice_basis[n_check+1:end]
shared_b_basis = bob_basis[n_check+1:end]
bob_matches = [x==y for (x, y) in zip(shared_a_basis, shared_b_basis)]
sifted_key = bob_bits[n_check+1:end][bob_matches]  # Now they both know this is a common private key

# Utility function to pretty print the bit strings 
stringadd(x, y) = string(x)*string(y)

println("""
Key     $(reduce(stringadd, key[1:n_check])) $(reduce(stringadd, key[n_check+1:end]))
______________________________________
Alice   $(reduce(*, alice_basis[1:n_check])) $(reduce(*, alice_basis[n_check+1:end]))
Bob     $(reduce(*, bob_basis[1:n_check])) $(reduce(*, bob_basis[n_check+1:end]))
Result  $(reduce(stringadd, bob_bits[1:n_check])) $(reduce(stringadd, bob_bits[n_check+1:end]))
______________________________________
Shared  $(reduce(stringadd, sifted_key))
"""
)


println("Eavesdropper Check Fidelity: $f.")


