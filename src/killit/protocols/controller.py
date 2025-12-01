from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # NEW: Independent import
from qiskit_aer.noise import NoiseModel, depolarizing_error

from .bb84_circuit import BB84Elements


class BB84Controller:
    def __init__(self, config):
        self.config = config
        
        gen_conf = self.config.get('general', self.config)
        self.n = gen_conf['n_qubits']
        self.shots = gen_conf['shots']
        
        self.qc = QuantumCircuit(self.n, self.n)
        self.builder = BB84Elements()

        # Setup optional Noise
        noise_model = None        
        noise_conf = self.config.get('noise', {})

        if noise_conf.get('enabled', False):
            prob = noise_conf.get('depolarizing_prob', 0.0)
            # Apply an 'error gate' to all 1-qubit gates
            error_gate = depolarizing_error(prob, 1)
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_gate, ['x','h','id'])
            
        # Options: "automatic", "statevector", "density_matrix", "matrix_product_state"
        method = gen_conf.get('sim_method', 'automatic')
        
        self.backend = AerSimulator(
            method=method,
            noise_model=noise_model
        ) 
        
    def run_alice(self, bits, bases):
        for i in range(self.n):
            self.builder.encode_bit(self.qc, i, bits[i])
            self.builder.rotate_basis(self.qc, i, bases[i])
        self.qc.barrier()

    def run_bob(self, bases):
        for i in range(self.n):
            self.builder.rotate_basis(self.qc, i, bases[i])
            self.builder.add_measure(self.qc, i)

    def execute_circuit(self):
        transpiled_qc = transpile(self.qc, self.backend)
        # shots=1 gives us one instance of the N-bit string
        job = self.backend.run(transpiled_qc, shots=1)
        
        # Get the bitstring
        raw_counts = job.result().get_counts()
        # Reverse the bit string 
        raw_bitstring = list(raw_counts.keys())[0][::-1]
        
        return [int(bit) for bit in raw_bitstring]

    def calculate_metrics(self, alice_bits, alice_bases, bob_bases, bob_results):
        """
        Sifts the key and calculates QBER.
        """
        sifted_key = []
        errors = 0
        sifted_length = 0

        for i in range(self.n):
            # Sifting: Only keep bits where bases matched
            if alice_bases[i] == bob_bases[i]:
                sifted_length += 1
                sifted_key.append(bob_results[i])
                
                # Metric: Did the bit survive correctly?
                if alice_bits[i] != bob_results[i]:
                    errors += 1
        
        qber = (errors / sifted_length) if sifted_length > 0 else 0.0
        
        return {
            "total_qubits": self.n,
            "sifted_key_length": sifted_length,
            "error_count": errors,
            "QBER": qber,
            "final_key": sifted_key
        }
