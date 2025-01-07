# Enhanced Neural Network with PyTorch

Welcome to the **Enhanced Neural Network** repository! This project is an upgraded and flexible implementation of a simple fully connected neural network using PyTorch, designed for training and evaluating on the MNIST dataset. This enhanced version introduces features like batch normalization, dropout, learning rate scheduling, and modular code structure, making it a great starting point for both beginners and experienced developers to explore neural network training.

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

## Examples
Train the model with a hidden layer size of 100 and dropout rate of 0.4:
```bash
python main.py --hidden_size 100 --dropout 0.4
```

Results:
```
Epoch [1/10] | Loss: 0.5678 | Train Acc: 89.56% | Val Acc: 88.45%
...
Final Test Accuracy: 91.34%
```

## Results
This implementation achieves high accuracy on the MNIST dataset while offering flexibility to experiment with various configurations. The modular structure ensures easy extensibility for custom datasets and architectures.

## Contribution
Feel free to fork this repository, create issues, or submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Inspired by the PyTorch community and the foundational work in neural networks.

---

Happy experimenting with neural networks! 🎉
