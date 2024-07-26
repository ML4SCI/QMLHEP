from IPython.display import Markdown, display
from datetime import date
import json
import numpy as np
import h5py
import os

class ModelTest:
    def __init__(self, test_id, num_qubits, num_aux_qubits, circuit_depth, 
                 rotations, num_generators, generator_lr, discriminator_lr, 
                 batch_size, num_samples, num_epochs, y, channel, optimizer, 
                 resolution) -> None:
        
        self.test_id = test_id
        self.num_qubits = num_qubits
        self.num_aux_qubits = num_aux_qubits
        self.circuit_depth = circuit_depth
        self.rotations = rotations
        self.num_generators = num_generators
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_epochs = num_epochs
        self.y = y
        self.channel = channel
        self.optimizer = optimizer
        self.date = date.today()

        # Setting the path relative to the current file
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(self.current_dir, '..', '..', 'tests', channel+'_tests', 'log')


        # Check for existing test ID
        self._check_existing_test_id()
        self._check_identical_test()

        # Create Specs JSON
        self._create_specs_json()

    def _create_specs_json(self) -> None:
        # Data to save
        data = self._get_specs_dictionary()

        # file name
        filename = self.path + f"/{self.test_id}_specs.json"

        # save data in a JSON file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        
        print(f"Data saved in {filename}")


    def _check_existing_test_id(self) -> None:
        # List all files in the specified path
        existing_files = os.listdir(self.path)
        # Check if any file matches the test_id in its filename
        for file in existing_files:
            if file.startswith(f"{self.test_id}_") and file.endswith(".json"):
                raise ValueError(f"A test with ID {self.test_id} already exists. Please use a unique test ID.")
            
    
    def _check_identical_test(self) -> None:
        # List all files in the specified path
        existing_files = os.listdir(self.path)
        # Current test parameters as a dictionary
        current_test_params = self._get_specs_dictionary()

        # Check each file for identical parameters
        for file in existing_files:
            if file.endswith(".json"):
                with open(os.path.join(self.path, file), 'r') as json_file:
                    file_params = json.load(json_file)
                    if self._compare_params(current_test_params, file_params):
                        existing_id = file_params['id']
                        raise ValueError(f"A test with identical parameters has already been conducted under ID {existing_id}.")


    def _get_specs_dictionary(self) -> dict:
        return {
            "id": self.test_id,
            "qubits": self.num_qubits,
            "auxiliar qubits": self.num_aux_qubits,
            "circuit depth": self.circuit_depth,
            "generators": self.num_generators,
            "rotations": self.rotations,
            "lr gen": self.generator_lr,
            "lr disc": self.discriminator_lr,
            "batch size": self.batch_size,
            "resolution": self.resolution,
            "optimizer": self.optimizer,
            "samples": self.num_samples,
            "epochs": self.num_epochs,
            "y": self.y      
        }
    
    
    def _compare_params(self, params1, params2):
        # Compare two parameter dictionaries
        keys_to_compare = set(params1.keys()) & set(params2.keys()) - {'id'}
        return all(params1[key] == params2[key] for key in keys_to_compare)

             

    def display_specs(self) -> None:

        markdown_content = f"""
# qGAN Individual Test Specification

## General Information
- **Test ID**: {self.test_id}
- **Date**: {self.date}

### **Training dataset features**
| Parameter            | Value  |
|----------------------|--------|
| Resolution     | {self.resolution}   |
| Interpolation  | Sumpool   |
| Channel        | {self.channel}      |
| Transformation | None      |

### **Quantum Circuit Parameters**
| Parameter            | Value  |
|----------------------|--------|
| Number of Qubits     | {self.num_qubits}      |
| Number of Auxiliary Qubits | {self.num_aux_qubits}      |
| Circuit Depth        | {self.circuit_depth}      |
| Rotations per layer  | {self.rotations}   |
| output transf denominator | {self.y}        |

### **GAN Parameters**
| Parameter            | Value  |
|----------------------|--------|
| Number of Generators | {self.num_generators}      |
| Generator Learning Rate  | {self.generator_lr} |
| Discriminator Learning Rate  | {self.discriminator_lr} |
| Batch Size           | {self.batch_size}     |
| Number of Samples    | {self.num_samples}   |
| Number of epochs     | {self.num_epochs} |
| Optimizer            | {self.optimizer}  |

## Hardware and Software
- **Hardware used**: CPU AMD Ryzen 5 5600G
- **Framework**: Pytorch and Pennylane
"""

        display(Markdown(markdown_content))
        
   
    def save_results_to_json(self, metrics:dict[str:float]) -> None:
        # Data to save
        data = self._get_specs_dictionary()
        data["FID"] = metrics["FID"]
        data["RMSE"] = metrics["RMSE"]
        data["disc loss"] = metrics["discriminator loss"]
        data["gen loss"] = metrics["generator loss"]
        data["notes"] = metrics["notes"]

        # file name
        filename = self.path + f"/{self.test_id}_specs.json"

        # save data in a JSON file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        
        print(f"Data saved in {filename}")


    def save_trained_params(self, generator) -> None:
        params = np.stack((generator.q_params[0].detach().numpy(), generator.q_params[1].detach().numpy()))

        filename = self.path+f'/../trained_params/test-{self.test_id}_generator_params.h5'

        with h5py.File(filename, 'w') as file:
            file.create_dataset(f'{self.test_id}', data=params)
        
        print(f'Trained parameters saved in {filename}')

    