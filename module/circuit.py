from qiskit import (
    QuantumCircuit,
    transpile,
)  # Stellen Sie sicher, dass 'transpile' importiert ist
import numpy as np
from qiskit_aer import Aer  # Importiere den Simulator hier


class LGate:
    """Represents an L-Gate applied to a single qubit."""

    def __init__(self, qubit, tp_phases, ip_phases):
        self.qubit = qubit
        self.tp_phases = tp_phases  # List of 3 training phases
        self.ip_phases = ip_phases  # List of 3 input phases

    def apply(self, circuit):
        """Apply the L-Gate to the specified qubit in the given circuit."""
        for i in range(3):
            # Apply the training phase (TP)
            circuit.p(self.tp_phases[i], self.qubit)
            # Apply the input phase (IP)
            circuit.p(self.ip_phases[i], self.qubit)

            # Apply Hadamard gate between TP1-IP1 and TP2-IP2, and after TP3-IP3
            if i < 2:
                circuit.h(self.qubit)


class Layer:
    """Represents a layer of L-Gates applied to all qubits."""

    def __init__(self, qubits, tp_matrix, ip_matrix):
        self.qubits = qubits
        self.tp_matrix = tp_matrix  # Speichern der TP-Matrix
        self.ip_matrix = ip_matrix  # Speichern der IP-Matrix

        self.l_gates = [
            LGate(qubit, tp_matrix[:, qubit], ip_matrix[:, qubit])
            for qubit in range(qubits)
        ]

    def apply(self, circuit):
        """Apply the layer of L-Gates to all qubits in the circuit."""
        for l_gate in self.l_gates:
            l_gate.apply(circuit)


class Circuit:
    """Represents a quantum circuit composed of multiple layers."""

    def __init__(self, qubits, layers, shots):
        self.qubits = qubits
        self.layers = layers  # List of Layer objects
        self.shots = shots
        self.circuit = QuantumCircuit(qubits, qubits)
        self.simulation_result = None

        self.build_circuit()
        self.measure()

    def build_circuit(self):
        """Build the quantum circuit by applying each layer in sequence."""
        for layer in self.layers:
            layer.apply(self.circuit)

    def measure(self):
        """Add measurement operations to all qubits."""
        self.circuit.measure(range(self.qubits), range(self.qubits))

    def run(self, simulator=None):
        """Run the quantum circuit simulation and return the result."""
        if simulator is None:
            simulator = Aer.get_backend(
                "aer_simulator"
            )  # Standardmäßig den Aer Simulator verwenden
        compiled_circuit = transpile(
            self.circuit, simulator
        )  # Ensure transpile is used here
        self.simulation_result = simulator.run(
            compiled_circuit, shots=self.shots
        ).result()
        return self.simulation_result

    def get_counts(self):
        """Return the counts from the last simulation run."""
        if self.simulation_result is not None:
            return self.simulation_result.get_counts(self.circuit)
        else:
            raise RuntimeError("The circuit has not been run yet.")

    def train(self, target_state, optimizer):
        """
        Train the circuit to optimize the TP matrix for a given target state.
        :param target_state: The desired target state to maximize.
        :param optimizer: An instance of the optimizer to use.
        """
        # Iterate over each layer and optimize the training phases
        for layer in self.layers:
            # Set the initial training phases
            initial_tp_matrix = layer.tp_matrix

            # Optimize using the provided optimizer
            optimized_phases, _ = optimizer.optimize()

            # Update the layer's TP matrix with optimized phases
            layer.tp_matrix = np.array(optimized_phases)

            # Debugging: Print optimized training phases
            print(f"Optimized TP Matrix for Layer:\n{layer.tp_matrix}\n")

            # Apply the updated layer to the circuit
            self.circuit = QuantumCircuit(self.qubits, self.qubits)  # Reset circuit
            self.build_circuit()  # Rebuild with new training phases
            self.measure()

            # Run the optimized circuit and evaluate results
            self.run(simulator)
            counts = self.get_counts()
            max_state = max(counts, key=counts.get)
            probability = counts[max_state] / sum(counts.values())

            print(
                f"Target state: {target_state}, Max state: {max_state}, Probability: {probability}"
            )

    def __repr__(self):
        return self.circuit.draw(output="text").__str__()
