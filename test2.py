import numpy as np

# Original data point (simplified to one dimension)
original_data = np.array([5.0])

# Function to add Gaussian noise
def add_noise(data, noise_level):
    return data + np.random.normal(0, noise_level, data.shape)

# Placeholder model for noise prediction
class NoisePredictor:
    def __init__(self):
        self.noise_factor = 0.25  # Initial guess

    def train(self, noisy_data, original_data):
        # Learn to estimate the noise added
        estimated_noise = noisy_data - original_data
        self.noise_factor = np.mean(np.abs(estimated_noise / noisy_data))

    def predict_noise(self, data):
        return self.noise_factor * data

# Training Phase: Simulate the noise addition process and train to predict noise
def train_diffusion(data, steps=10):
    noise_levels = np.linspace(1, 0.1, steps)  # Decreasing noise level
    noisy_data = add_noise(data, noise_levels[0])  # Add initial large noise
    predictor = NoisePredictor()  # Initialize a simple noise predictor model

    # Simulate training by adjusting the model's noise prediction factor
    for t in range(1, steps):
        current_noisy_data = add_noise(data, noise_levels[t])
        predictor.train(current_noisy_data, data)

    return noisy_data, predictor

# Inference Phase: Simulate the denoising process
def inference_diffusion(noisy_data, predictor, steps=10):
    for t in range(steps):
        predicted_noise = predictor.predict_noise(noisy_data)
        noisy_data -= predicted_noise  # Remove predicted noise
        print(f"After step {t+1}, data: {noisy_data}")

    return noisy_data

# Run the training phase to simulate noise addition and train the predictor
noisy_data, predictor = train_diffusion(original_data)
print("Starting noisy data:", noisy_data)

# Start inference from the noisy state using the trained predictor
recovered_data = inference_diffusion(noisy_data, predictor)
print("Recovered data:", recovered_data)
print("Original data:", original_data)

