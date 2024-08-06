![GLLM-removebg-preview](https://github.com/user-attachments/assets/b586cbb8-3019-4ec0-8654-3e76922512f8)
## Introduction: LLY-GLLM

LLY-GLLM, which stands for Generative Large Language Model, is derived from the [LILY Project](http://www.lilyqml.de) and serves as a precursor to a Quantum Large Language Model. It is based on the LLY-GP model and primarily uses quantum circuits to represent words, combining them to form connections. For example, Circuit(King) + Circuit(Woman) = State(Queen). Similar to LLY-GP, these circuits consist of a series of L-gates that are filled with tokens.


LLY-GLLM is available on the [LILY QML platform](https://www.lilyqml.de), making it accessible for researchers and developers.

For inquiries or further information, please contact: [info@lilyqml.de](mailto:info@lilyqml.de).

## Contributors

| Role                     | Name          | Links                                                                                                                |
|--------------------------|---------------|----------------------------------------------------------------------------------------------------------------------|
| Project Lead             | Leon Kaiser   | [ORCID](https://orcid.org/0009-0000-4735-2044), [GitHub](https://github.com/xleonplayz)                              |
| Inquiries and Management | Raul Nieli    | [Email](mailto:raul.nieli@lilyqml.de)                                                                                |
| Supporting Contributors  | Eileen Kühn   | [GitHub](https://github.com/eileen-kuehn), [KIT Profile](https://www-kseta.ttp.kit.edu/fellows/Eileen.Kuehn/)        |
| Supporting Contributors  | Max Kühn      | [GitHub](https://github.com/maxfischer2781)                                                                          |


## Functionality


The LLY-GLLM model is based on the property of representing a multi-qubit system using a Q-Sphere. A Q-Sphere is a sphere that contains all possible states of a multi-qubit system on its surface, making the principle easier to illustrate.

By pre-configuring words for each state on this sphere, such as "King" and "Woman," these states are represented by a series of quantum circuits that use the tokens as input phases.

When these circuits are placed in sequence, the tuning phases can be optimized to indicate a specific state. As demonstrated in this example, the vector of "King" plus the vector of "Woman" would result in the vector "Queen." Technically, this is the circuit with the input token "King" followed by the circuit with the input token "Woman," which together represent the state associated with "Queen."





![thumbnail_QM-105__1_-removebg-preview (2)](https://github.com/user-attachments/assets/46d8378b-a7c1-4ed9-9306-7a195d558bf1)
![thumbnail_QM-106-removebg-preview](https://github.com/user-attachments/assets/1c4101b6-4c6d-4aed-aa39-09b3b18b8135)

## Workflow

A word is essentially represented as a token, which is then placed into the input phase of the gate. The tuning phases in this case are pre-trained sets. When we have multiple circuits representing tokens arranged in sequence, we refer to these as layers. Each layer has its own tuning phases, which have been previously trained. In practical applications, the pre-trained tuning phases are utilized.

## Outlook

Another point is the prospect of training an additional model, LLY-LLM. This is intended to be the large language model that emerges from the LILY project.
