using CSV, DataFrames, Wavelets, Plots, Flux, Flux.Recurrent, ProgressMeter

# Read the data
df = CSV.read("main.csv", DataFrame)

# Convert the timestamps
df.openTime = DateTime.(df.openTime, Dates.unix2datetime)
df.closeTime = DateTime.(df.closeTime, Dates.unix2datetime)

# Perform the wavelet transform on all price features
open_wavelet = wavedec(df.open, 5, "db4")
close_wavelet = wavedec(df.close, 5, "db4")
high_wavelet = wavedec(df.high, 5, "db4")
low_wavelet = wavedec(df.low, 5, "db4")

# Visualize the wavelet coefficients
plt = plot(layout=(9, 1), size=(800, 800))

for i = 1:5
    plot!(plt[i], open_wavelet[i], label="Open Price Level $i")
    plot!(plt[i+4], close_wavelet[i], label="Close Price Level $i")
end

plot!(plt[9], df.open, label="Open Prices")
plot!(plt[9], df.close, label="Close Prices")
plot!(plt[9], df.high, label="High Prices")
plot!(plt[9], df.low, label="Low Prices")

plot!(plt, xlabel="Time", ylabel="Coefficient Value")
display(plt)

# Concatenate the wavelet coefficients into a single input matrix
X = hcat(open_wavelet..., close_wavelet..., high_wavelet..., low_wavelet...)

# Split the data into training and testing sets
train_size = Int(floor(0.8 * size(X, 2)))
X_train = X[:, 1:train_size]
X_test = X[:, train_size+1:end]

# Convert to batches for the RNN
X_train_batches = [X_train[:, i:i+99] for i in 1:100:size(X_train, 2)-99]
X_test_batches = [X_test[:, i:i+99] for i in 1:100:size(X_test, 2)-99]

#define model and loss function
model = Chain(
    LSTM(size(X, 1), 128),
    Dense(128, 4)
)

loss(x, y) = mse(model(x), y)

opt = ADAM()

#train by backpropagation through time
epochs = 50
@showprogress for epoch = 1:epochs
    for (x, y) in zip(X_train_batches, X_test_batches)
        grads = Flux.gradient(() -> loss(x, y), params(model))
        update!(opt, params(model), grads)
    end
end

#obtain and evaluate predicted Prices
y_pred = hcat([model(x) for x in X_test_batches]...)

# Evaluate the model's performance
println("Test MSE: ", mse(y_pred, X_test))