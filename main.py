import json
import numpy as np
import pandas as pd
from qiskit_aer import Aer
from module.circuit import Circuit, Layer
from module.tokenizer import Tokenizer  # Importiere die Tokenizer-Klasse
from module.optimizer import AdamOptimizer  # Importiere den AdamOptimizer
from module.visual import Visual  # Importiere die Visual-Klasse


class LLYGLLM:
    """LLY-GLLM class that reads configuration from a JSON file and creates a quantum circuit."""

    def __init__(self, config_file, learning_rate=0.01, max_iterations=100):
        self.config_file = config_file
        self.qubits = 0
        self.l_gates = 0
        self.circuit = None
        self.tokenizer = Tokenizer()  # Initialisiere den Tokenizer
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tp_matrix = None
        self.initial_summary = []  # Speichere initiale Zustände
        self.final_summary = []  # Speichere finale Zustände
        self.iterations = 0  # Iterationen
        self.shots = 0  # Anzahl der Schüsse

    def load_configuration(self):
        """Load the number of qubits, L-gates, iterations, and shots from a JSON file."""
        try:
            with open(self.config_file, "r") as file:
                data = json.load(file)
                self.single_words = data.get("single_words", [])
                self.word_combinations = data.get("word_combinations", {})

                # Anzahl der Qubits entspricht der Anzahl der Wörter
                self.qubits = len(self.single_words)
                self.l_gates = len(self.single_words) + 2 * len(
                    self.word_combinations
                )  # Ein Layer pro Wort + Zwei Layer pro Kombination

                # Lade Iterationen und Shots
                self.iterations = data.get("iterations", self.max_iterations)
                self.shots = data.get("shots", 1024)

                # Debug-Ausgabe zur Überprüfung der geladenen Werte
                print(
                    f"Loaded configuration: {self.qubits} qubits, {self.l_gates} L-gates, {self.iterations} iterations, {self.shots} shots"
                )

        except FileNotFoundError:
            print(f"Configuration file {self.config_file} not found.")
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from file {self.config_file}. Please ensure it is correctly formatted."
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def tokenize_word(self, word):
        """Tokenize a single word and return a 3x20 matrix of tokens."""
        token = self.tokenizer.tokenize(word)
        return np.array(
            token
        ).T  # Transponiere, um die Form (3, token_length) zu erhalten

    def create(self):
        """Create a quantum circuit with the specified number of qubits and L-gates."""
        # Load configuration
        self.load_configuration()

        # Check for valid configuration
        if self.qubits <= 0 or self.l_gates <= 0:
            raise ValueError(
                "Invalid configuration: Number of qubits and L-gates must be greater than 0."
            )

        layers = []

        # Generiere einmalige Trainingsphasen
        self.tp_matrix = np.random.rand(3, self.qubits) * 2 * np.pi
        print(f"TP Matrix (constant):\n{self.tp_matrix}\n")

        # Erste Schleife: Einzelwörter
        for word in self.single_words:
            # Tokenize das aktuelle Wort
            ip_matrix = self.tokenize_word(word)

            # Erzeuge ein neues Layer mit dem konstanten TP und dem aktuellen IP
            layers.append(Layer(self.qubits, self.tp_matrix, ip_matrix))

            # Debug-Ausgabe des aktuellen Layers
            print(f"IP Matrix for word '{word}':\n{ip_matrix}\n")

            # Erzeuge Circuit und führe ihn aus
            circuit = Circuit(self.qubits, [layers[-1]], self.shots)
            state, probability, counts = self.run_single_layer(circuit)

            # Zustand speichern zusammen mit dem Wort
            self.initial_summary.append(
                {
                    "Wort": word,
                    "Zustand": state,
                    "Wahrscheinlichkeit": probability,
                    "Counts": counts,
                }
            )

        # Initiale Tabelle mit Layer-Informationen anzeigen
        self.display_summary(
            self.initial_summary, title="Initial Summary of Circuit Layers"
        )

    def run_single_layer(self, circuit):
        """Run a single layer of the quantum circuit and return the result state and its probability."""
        simulator = Aer.get_backend("aer_simulator")
        circuit.run(simulator)
        counts = circuit.get_counts()

        # Finde den Zustand mit der höchsten Wahrscheinlichkeit
        total_shots = sum(counts.values())
        max_state = max(counts, key=counts.get)
        probability = counts[max_state] / total_shots  # Wahrscheinlichkeit berechnen

        return max_state, probability, counts

    def display_summary(self, summary, title="Summary of Circuit Layers"):
        """Display a summary table of the circuit layers and their words."""
        df = pd.DataFrame(summary)
        print(f"\n{title}:")
        print(df.to_string(index=False))

    def train(self):
        """Train the quantum circuit to optimize the TP matrix for each word in single_words."""
        simulator = Aer.get_backend(
            "aer_simulator"
        )  # Den Simulator hier initialisieren

        # Erste Schleife: Training mit einzelnen Wörtern
        for summary in self.initial_summary:
            word = summary["Wort"]
            initial_state = summary["Zustand"]

            # Tokenize das aktuelle Wort
            ip_matrix = self.tokenize_word(word)

            # Erzeuge ein neues Layer mit dem konstanten TP und dem aktuellen IP
            layer = Layer(self.qubits, self.tp_matrix, ip_matrix)

            # Erzeuge Circuit
            circuit = Circuit(self.qubits, [layer], self.shots)

            # Verwende den AdamOptimizer zur Optimierung der TP-Matrix
            optimizer = AdamOptimizer(
                circuit=circuit,
                target_state=initial_state,
                learning_rate=self.learning_rate,
                max_iterations=self.iterations,
            )

            # Optimiere die Trainingsphasen
            optimized_phases, losses = optimizer.optimize()

            # Ausgabe der Ergebnisse der Optimierung
            print(f"\nOptimierung für Wort: {word}")
            print(f"Optimierte Trainingsphasen:\n{optimized_phases}\n")
            print(f"Verlustverlauf:\n{losses}\n")

            # Aktualisiere die TP-Matrix mit den optimierten Phasen
            layer.tp_matrix = optimized_phases

            # Führe den Circuit mit den optimierten Phasen erneut aus
            state_optimized, probability_optimized, counts_optimized = (
                self.run_single_layer(circuit)
            )

            # Speichere den optimierten Zustand zusammen mit dem Wort
            self.final_summary.append(
                {
                    "Wort": word,
                    "Zustand": state_optimized,
                    "Wahrscheinlichkeit": probability_optimized,
                    "Counts": counts_optimized,
                    "Loss": losses,
                }
            )

        # Zweite Schleife: Training mit Wortkombinationen
        for combination, result in self.word_combinations.items():
            first_word, second_word = combination.split()

            # Tokenize das erste und zweite Wort der Kombination
            first_ip_matrix = self.tokenize_word(first_word)
            second_ip_matrix = self.tokenize_word(second_word)

            # Erzeuge Layer für beide Wörter mit der gleichen TP-Matrix
            first_layer = Layer(self.qubits, self.tp_matrix, first_ip_matrix)
            second_layer = Layer(self.qubits, self.tp_matrix, second_ip_matrix)

            # Erzeuge Circuit mit zwei Layern
            circuit = Circuit(self.qubits, [first_layer, second_layer], self.shots)

            # Finde den Zielzustand (state) für das Ergebnis der Kombination
            _, target_probability, _ = self.run_single_layer(circuit)

            # Suche den erwarteten Zustand des resultierenden Wortes
            expected_state = None
            for summary in self.initial_summary:
                if summary["Wort"] == result:
                    expected_state = summary["Zustand"]
                    break

            # Verwende den AdamOptimizer zur Optimierung der TP-Matrix des zweiten Layers
            optimizer = AdamOptimizer(
                circuit=circuit,
                target_state=expected_state,  # Der Zielzustand ist das resultierende Wort
                learning_rate=self.learning_rate,
                max_iterations=self.iterations,
            )

            # Optimiere die Trainingsphasen des zweiten Layers
            optimized_phases, losses = optimizer.optimize()

            # Ausgabe der Ergebnisse der Optimierung
            print(f"\nOptimierung für Kombination: {combination} = {result}")
            print(
                f"Optimierte Trainingsphasen für zweites Layer:\n{optimized_phases}\n"
            )
            print(f"Verlustverlauf:\n{losses}\n")

            # Aktualisiere die TP-Matrix des zweiten Layers mit den optimierten Phasen
            second_layer.tp_matrix = optimized_phases

            # Führe den Circuit mit den optimierten Phasen erneut aus
            state_optimized, probability_optimized, counts_optimized = (
                self.run_single_layer(circuit)
            )

            # Speichere den optimierten Zustand zusammen mit dem Ergebniswort
            self.final_summary.append(
                {
                    "Wort": f"{combination} = {result}",
                    "Zustand": state_optimized,
                    "Wahrscheinlichkeit": probability_optimized,
                    "Counts": counts_optimized,
                    "Loss": losses,
                }
            )

        # Finalisierte Tabelle mit Layer-Informationen anzeigen
        self.display_summary(
            self.final_summary, title="Final Summary of Circuit Layers"
        )

        # Vergleiche initiale und finale Zustände
        self.compare_summaries(self.initial_summary, self.final_summary)

    def compare_summaries(self, initial_summary, final_summary):
        """Compare initial and final summaries to show the improvement."""
        initial_df = pd.DataFrame(initial_summary)
        final_df = pd.DataFrame(final_summary)

        comparison_df = initial_df.copy()
        comparison_df.columns = [
            "Wort",
            "Initial Zustand",
            "Initial Wahrscheinlichkeit",
            "Initial Counts",
        ]

        final_df.columns = [
            "Wort",
            "Final Zustand",
            "Final Wahrscheinlichkeit",
            "Final Counts",
            "Loss",
        ]

        # Merge the dataframes to show comparisons
        comparison_df = pd.merge(comparison_df, final_df, on="Wort")

        # Display comparison
        print("\nComparison of Initial and Final Circuit Layers:")
        print(comparison_df.to_string(index=False))

        # Plot the comparison using Visual class
        visual = Visual(self.final_summary, comparison_df, self.iterations, self.shots)
        visual.generate_report()

    def __repr__(self):
        """Return a string representation of the circuit."""
        if self.circuit is not None:
            return repr(self.circuit)
        return "Circuit not created."


# Beispiel für die Nutzung der LLY-GLLM-Klasse

# Erstelle eine Instanz von LLY-GLLM mit dem Pfad zur Konfigurationsdatei
lly_gllm = LLYGLLM("var/train.json", learning_rate=0.01, max_iterations=100)

# Erstelle den Quantum Circuit
lly_gllm.create()

# Trainiere den Quantum Circuit
lly_gllm.train()

# Die Ausgabe des Quanten-Circuits ist in der Zusammenfassung enthalten
