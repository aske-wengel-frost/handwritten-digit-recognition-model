## Handwritten Digit Recognition

This script allows you to recognize handwritten digits using a pre-trained neural network model. The model is trained on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0 through 9).

### Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.9 (or a specific version that supports TensorFlow)
- Libraries: `opencv-python`, `numpy`, `matplotlib`, `tensorflow`

### Usage

1. **Dataset Preparation:**
   - The MNIST dataset is automatically loaded by the script. No additional steps are required for dataset preparation.

2. **Model Loading:**
   - The pre-trained model is loaded from the file `handwritten-digits.model`.

3. **Providing Input:**
   - Place your handwritten digit images in the `digits_written_on_paper` directory.
   - Ensure that the images are named sequentially, e.g., `digit1.png`, `digit2.png`, and so on.

4. **Running the Script:**
   - Run the script in your Python environment using Python 3.9.
   - The script will iterate through all the digit images in the `digits_written_on_paper` directory and attempt to recognize the digits.

5. **Output:**
   - For each input image, the script will display the image and print the predicted digit.

### Notes

- The input images should be in PNG format and contain black digits on a white background.
- If an error occurs during processing an image, the script will continue to the next image.
- The model's prediction accuracy is based on its training on the MNIST dataset.
- The script assumes that the pre-trained model file `handwritten-digits.model` is available in the same directory.
- TensorFlow may require Python 3.9 or a specific version to work properly. Ensure that you use Python 3.9 or the compatible version.

### Additional Information

- The model architecture consists of a neural network with multiple layers, including input, hidden, and output layers.
- The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.
- During prediction, the softmax activation function is applied to the output layer to obtain probabilities for each digit class.

### Example

```bash
$ python3.9 handwritten_digit_recognition.py