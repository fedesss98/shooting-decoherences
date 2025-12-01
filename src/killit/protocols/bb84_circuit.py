from qiskit import QuantumCircuit

class BB84Elements:
    def encode_bit(self, qc: QuantumCircuit, index: int, bit_val: int):
        if bit_val == 1:
            qc.x(index)

    def rotate_basis(self, qc: QuantumCircuit, index: int, basis_val: int):
        # basis_val 0: Z-basis (do nothing), 1: X-basis (apply H)
        if basis_val == 1:
            qc.h(index)

    def add_measure(self, qc: QuantumCircuit, index: int):
        qc.measure(index, index)
