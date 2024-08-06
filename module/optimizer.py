import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from module.circuit import Circuit  # Importiere die Circuit-Klasse

class Optimizer:
    def __init__(self, circuit, target_state, learning_rate, max_iterations):
        self.circuit = circuit
        self.target_state = target_state
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.initial_distribution = None
        self.initial_probability = 0.0
        self.optimized_phases = None

    def loss_function(self, counts):
        """Calculate the loss as the negative probability of the target state."""
        total_shots = sum(counts.values())
        target_probability = counts.get(self.target_state, 0) / total_shots
        loss = -target_probability  # Minimiere die negative Wahrscheinlichkeit
        return loss

    def optimize(self):
        # Initialisiere beste Phasen und Verlust
        best_phases = np.array(self.circuit.layers[0].tp_matrix)  # Zugriff auf das Layer
        best_loss = float("inf")
        losses = []

        # Initialer Lauf und Verteilung
        self.circuit.run()
        initial_counts = self.circuit.get_counts()
        self.initial_distribution = self.get_distribution(initial_counts)
        if self.initial_distribution is None:
            print("Warning: Initial distribution is None. Check circuit run and get_counts().")
        self.initial_probability = self.initial_distribution.get(self.target_state, 0.0)

        for iteration in range(self.max_iterations):
            # Evaluiere aktuellen Verlust
            current_loss = self.evaluate(best_phases)
            losses.append(current_loss)

            # Aktualisiere Phasen für neue Zustände
            new_phases = self.update_phases(best_phases)
            new_loss = self.evaluate(new_phases)

            # Akzeptiere neue Phasen bei besserem Verlust
            if new_loss < best_loss:
                best_phases = new_phases
                best_loss = new_loss

            print(f"Iteration {iteration}, Loss: {best_loss}")

        # Setze die optimierten Trainingsphasen
        self.circuit.layers[0].tp_matrix = best_phases.tolist()
        self.optimized_phases = best_phases.tolist()

        return best_phases.tolist(), losses

    def evaluate(self, training_phases):
        # Update des Circuits mit neuen Trainingsphasen
        self.circuit.layers[0].tp_matrix = training_phases.tolist()
        self.circuit.run()

        # Erhalte Resultatzählungen
        counts = self.circuit.get_counts()
        return self.loss_function(counts)

    def update_phases(self, current_phases):
        # Erzeuge kleine zufällige Änderungen an den Trainingsphasen
        new_phases = current_phases + np.random.normal(
            0, self.learning_rate, current_phases.shape
        )
        return new_phases

    def get_distribution(self, counts):
        """Erhalte eine sortierte Wahrscheinlichkeitsverteilung aus den Zählwerten."""
        total_shots = sum(counts.values())
        if total_shots == 0:
            print("Warning: Total shots is zero. Counts may be incorrect.")
            return {}
        distribution = {state: counts[state] / total_shots for state in counts}
        return dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))

    def plot_distribution(self, counts, title):
        """Plotten Sie ein Histogramm der Zustandsverteilung."""
        distribution = self.get_distribution(counts)
        df = pd.DataFrame(distribution.items(), columns=["State", "Probability"])
        df = df.sort_values(by="Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("tight")
        ax.axis("off")
        ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )
        ax.set_title(title, fontsize=16)
        plt.show()

class AdamOptimizer(Optimizer):
    def __init__(self, *args, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Verwende das erste Layer für Trainingsphasen
        self.m = np.zeros_like(self.circuit.layers[0].tp_matrix)
        self.v = np.zeros_like(self.circuit.layers[0].tp_matrix)
        self.t = 0

    def update_phases(self, current_phases):
        self.t += 1
        gradient = np.random.normal(0, self.learning_rate, current_phases.shape)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        new_phases = current_phases + self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return new_phases

