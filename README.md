# Enhanced Neural Network with PyTorch

This repository is an upgraded implementation of a simple fully connected neural network using PyTorch, designed for training and evaluating on the MNIST dataset. This enhanced version introduces features like batch normalization, dropout, learning rate scheduling, and modular code structure.


## Features

- **Batch Normalization**: Improves training stability and convergence.
- **Dropout Regularization**: Reduces overfitting and enhances generalization.
- **Learning Rate Scheduler**: Dynamically adjusts the learning rate during training.
- **Validation Dataset**: Provides better insights into model performance.
- **Command-Line Arguments**: Enables easy customization of hyperparameters like learning rate, batch size, hidden layer size, etc.
- **Code Modularity**: Organized into reusable functions for clarity and maintainability.


## Installation

Clone this repository to your local machine:
```bash
git clone https://github.com/cxycode32/enhanced_nn.git
cd enhanced_nn
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```


## Usage

Run the training script with default parameters:
```bash
python main.py
```

Customize hyperparameters using command-line arguments:
```bash
python main.py --learning_rate 0.005 --batch_size 128 --num_epochs 20 --hidden_size 100 --dropout 0.3
```


## File Structure

```
├── main.py  # Main script for training and evaluation
├── README.md               # Project documentation
└── requirements.txt        # Required dependencies
```


## Results

This implementation achieves high accuracy on the MNIST dataset while offering flexibility to experiment with various configurations. The modular structure ensures easy extensibility for custom datasets and architectures.


## Contribution

Feel free to fork this repository and submit pull requests to improve the project or add new features.


## License

This project is licensed under the MIT License.


## Acknowledgments

Inspired by the PyTorch community and the foundational work in neural networks.

---

Happy experimenting with neural networks! 🎉
