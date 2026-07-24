import json
import os

def read_json_files(directory):
    results = []
    # Read all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                results.append(data)
    
    # Sort results based on 'id' key 
    results.sort(key=lambda x: x['id'])
    return results

def generate_markdown_table(data):
    # Table headers
    headers = data[0].keys()
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "|---" * len(headers) + "|\n"

    # Add data rows
    for entry in data:
        row = "| " + " | ".join(str(value) for key, value in entry.items()) + " |"
        markdown += row + "\n"

    return markdown

def save_to_readme(markdown, introduction, filename="/home/reyguadarrama/GSoC/tests/ECAL_tests/README.md"):
    with open(filename, 'w') as file:
        file.write(introduction + "\n\n" + markdown)

# Use the defined functions to read, generate, and save the data
directory = "/home/reyguadarrama/GSoC/tests/ECAL_tests/log"
introduction_text = """
<div align="center">

# **Summary of ECAL Channel Test Results**
---

<div align="justify">

This README file includes a summary of test results for the various parameters explored in my experiments.
Each entry in the table represents a specific configuration and its outcomes. The model used in the training is the proposed by 
[He-Liang et.al.](https://arxiv.org/abs/2010.06201), this model consist in a set of feature qubits which will represent the distribution
and a set of auxiliar qubits which gives the model more freedom, a post-processing is performed over the circuit output, first is divided by
a number $y \in [0, 1]$ which allows the circuit output to take values larger than 1 and fix the limitation of the maximum sum of the output 
probabilities.

<div align="center">

<img src="../../images/Quantum_generator-2.png" alt="PQC architecture" width="400" height="200"/>

<div align="justify">


- The first 5 test show that a smaller generator lr produces a better convergence, range tested: $gen\,\,lr \in [0.005, 0.4]$.
"""

data = read_json_files(directory)
markdown_table = generate_markdown_table(data)
save_to_readme(markdown_table, introduction_text)

print("The README.md has been updated with the results table.")
