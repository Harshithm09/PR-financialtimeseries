from src.data_loader import load_data
from src.preprocessing import preprocess
from src.signal_processing import generate_spectrogram, plot_fft
from src.train import train_model
from src.evaluate import evaluate
import matplotlib.pyplot as plt

# Step 1: Load Data
df = load_data()

# Step 2: Preprocess
df_scaled = preprocess()

# Step 3: Select one signal
signal = df_scaled.iloc[:, 0].values

# Step 4: Signal Processing
plot_fft(signal)
Sxx = generate_spectrogram(signal)

# Step 5: Train Model
target = signal[-1]
prediction = train_model(Sxx, target)

# Step 6: Evaluate
evaluate(target, prediction)




plt.figure()
df.plot()
plt.title("Time Series")
plt.savefig("results/time_series.png")