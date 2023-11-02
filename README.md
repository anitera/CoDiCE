# trustCE

## Introduction
`trustCE` is a Python package designed for counterfactual explanation search. By accounting for cognitive biases, `trustCE` aims to generate trustworthy explanations, making it a valuable tool for data scientists and researchers in the field of explainable AI.

## Installation
To install `trustCE`, use the following pip command:
```
pip install trustce
```
It is possible to enable different model supports and pre download test datasets. FOr example to enable `dev` version with `sklearn` and the `homeloan` dataset, use the following pip command:
```
pip install trustce[dev,sklearn,homeloan]
```

## Configuration

### Constraints Configuration
`trustCE` allows users to specify constraints for the search. You can define which features are immutable, have a monotonic relationship, or have causal or rule dependencies. Here's an example configuration located in `config/constraints_conf.json`:

```json
{
    "features": {
        "ApplicantIncome": {
            "type": "monotonic",
            "direction": "increasing"  
        },
        "CoapplicantIncome": {
            "type": "monotonic",
            "direction": "decreasing"  
        },
        "Gender": {
            "type": "immutable"
        },
        "Credit_History": {
            "type": "dependency",
            "dependencyType": "causal",
            "root": "Credit_History",
            "child": "ApplicantIncome"
        }
    }
}
```

## General Configuration
The general configuration can be found in `config/conf.yaml`. This file contains dataset information, model specifications, and settings for the counterfactual search. Here's a brief overview:

- **dataset**: Specifies the dataset type, path, features, and target variable.
- **constraints_file**: Path to the constraints configuration file.
- **output_folder**: Directory for the results.
- **model**: Details about the model type, backend, state, and more.
- **cfsearch**: Parameters for the counterfactual search optimizer, loss type, distance metrics, etc.

## Setting Up an Instance for Explanation
To provide an instance for explanation, fill out the `datasets/instance.json` file. Here's an example structure:
```json
{
    "Gender": "Male",
    "Married": "Yes",
    "ApplicantIncome": 4616,
    ...
    "Property_Area": "Urban"
}
```

## Running the Framework
You can run the framework using the following steps:

1. Set the path for the target instance.
2. Load the dataset and set up the necessary objects.
3. Create and configure the `CFsearch` object.
4. Load the target instance and find counterfactuals.
5. Evaluate and visualize the counterfactuals.

For a detailed example, refer to `examples_notebook_demos`.

## Features
- **Cognitive Bias Handling**: `trustCE` incorporates cognitive biases to provide more intuitive explanations.
- **Versatile**: Compatible with various machine learning frameworks like scikit-learn, TensorFlow, PyTorch, and gpgomea.
- **Performance**: Efficient algorithms ensure quick generation of explanations even for complex models.
- **Evaluation**: The framework return with every counterfactual instance an evaluation file which has list of metrics.

## Evaluation Metrics

`trustCE` doesn't just provide counterfactual instances but also offers a comprehensive evaluation of the generated counterfactuals. After the counterfactual search, an evaluation file is generated that contains various explainability metrics.

Here's a breakdown of the metrics provided:

- **distance_continuous**: A measure of the distance between the original and counterfactual instance for continuous features.
- **distance_categorical**: A measure of the distance for categorical features.
- **sparsity_cont**: The number of continuous features that have been changed.
- **sparsity_cat**: The number of categorical features that have been changed.
- **coherence_score**: A score indicating how coherent the counterfactual is with respect to the original instance.
- **incoherent_features**: A list of features that are deemed incoherent in the counterfactual.
- **validity**: Indicates if the counterfactual is valid (i.e., meets the constraints and conditions set).
- **new_outcome**: The outcome of the counterfactual instance.

Here's an example of the evaluation output:

```json
{
    "distance_continuous": 20.120539021267003,
    "distance_categorical": 4,
    "sparsity_cont": 4,
    "sparsity_cat": 4,
    "coherence_score": 0.5454545454545454,
    "incoherent_features": [
        "ApplicantIncome",
        "Self_Employed"
    ],
    "validity": true,
    "new_outcome": 1
}
```

These metrics provide a deeper understanding of the counterfactuals and their quality, aiding users in interpreting and trusting the generated explanations.


## Contribution
We welcome contributions to `trustCE`! If you'd like to contribute, please fork the repository and submit a pull request. For major changes or feature requests, please open an issue first to discuss your ideas.

## License
`trustCE` is available under the MIT License. See the `LICENSE` file for more details.
