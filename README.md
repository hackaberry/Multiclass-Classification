## Overview:
This project demonstrates a simple multiclass classification neural network built using PyTorch. It includes an interactive GUI that visualizes how the model learns to classify data into four categories or classes. The models decision boundaries, training loss, and accuracy are updated in real-time, providing an intuitive view of classification in machine learning.

---
## Features:
- Multiclass classification model implemented with PyTorch’s nn.Linear
- Interactive GUI built with Tkinter for real-time visualization
- Real-time decision boundary updates after each training cycle
- Real-time loss curve tracking
- Device-agnostic code supporting both CPU and GPU (if available)

---
## Installation:

1. Clone the repository
```bash
git clone https://github.com/yourusername/Multiclass-Classification-Visualizer.git
```
2. Make sure these dependencies are installed
```bash
pip install torch matplotlib scikit-learn tk
```
3. Run the program
```bash
python main.py
```