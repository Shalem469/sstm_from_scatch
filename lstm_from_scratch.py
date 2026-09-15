import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Dataset
# =========================================================

X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]]
], dtype=np.float32)

y = np.array([
    [4],
    [5],
    [6],
    [7],
    [8]
], dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================================================
# LSTM Dimensions
# =========================================================

input_size = 1
hidden_size = 10
output_size = 1

print("Input size:", input_size)
print("Hidden size:", hidden_size)
print("Output size:", output_size)


# =========================================================
# Initialize Parameters
# =========================================================

np.random.seed(42)

# LSTM has 4 transformations:
#
# forget gate
# input gate
# candidate cell state
# output gate
#
# Therefore we combine them into one matrix.

W = np.random.randn(
    4 * hidden_size,
    input_size + hidden_size
) * 0.1

b = np.zeros((4 * hidden_size, 1))

# Output layer

Wy = np.random.randn(
    output_size,
    hidden_size
) * 0.1

by = np.zeros((output_size, 1))


print("W :", W.shape)
print("b :", b.shape)
print("Wy:", Wy.shape)
print("by:", by.shape)


# =========================================================
# Activation Functions
# =========================================================

def sigmoid(x):

    return 1 / (1 + np.exp(-x))


# =========================================================
# Forward Pass
# =========================================================

def forward(x):

    # Initial hidden state

    h = np.zeros((hidden_size, 1))

    # Initial cell state

    c = np.zeros((hidden_size, 1))

    # Store states for BPTT

    cache = []

    for t in range(len(x)):

        # Current input

        xt = x[t].reshape(input_size, 1)

        # Combine input and previous hidden state

        combined = np.vstack((h, xt))

        # Four LSTM transformations

        z = W @ combined + b

        # Split into four parts

        f = sigmoid(
            z[:hidden_size]
        )

        i = sigmoid(
            z[hidden_size:2 * hidden_size]
        )

        g = np.tanh(
            z[2 * hidden_size:3 * hidden_size]
        )

        o = sigmoid(
            z[3 * hidden_size:]
        )

        # Cell state

        c_new = f * c + i * g

        # Hidden state

        h_new = o * np.tanh(c_new)

        # Store everything needed for BPTT

        cache.append(
            (
                xt,
                h,
                c,
                f,
                i,
                g,
                o,
                c_new,
                h_new
            )
        )

        h = h_new
        c = c_new

    # Final output

    y_pred = Wy @ h + by

    return y_pred, cache


# =========================================================
# Test Forward Pass
# =========================================================

prediction, cache = forward(X[0])

print("\nInitial prediction:")
print(prediction)

print("\nNumber of timesteps:")
print(len(cache))


# =========================================================
# Training
# =========================================================

epochs = 30000
learning_rate = 0.001

loss_history = []


for epoch in range(epochs):

    total_loss = 0

    for sample in range(len(X)):

        x = X[sample]

        target = y[sample].reshape(
            output_size,
            1
        )

        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        prediction, cache = forward(x)

        # -------------------------------------------------
        # MSE
        # -------------------------------------------------

        error = prediction - target

        loss = np.mean(error ** 2)

        total_loss += loss

        # -------------------------------------------------
        # Output gradients
        # -------------------------------------------------

        dy = 2 * error

        # Last hidden state

        h_last = cache[-1][-1]

        dWy = dy @ h_last.T

        dby = dy

        # Gradient flowing into final hidden state

        dh_next = Wy.T @ dy

        # Gradient flowing into cell state

        dc_next = np.zeros(
            (hidden_size, 1)
        )

        # -------------------------------------------------
        # LSTM gradients
        # -------------------------------------------------

        dW = np.zeros_like(W)

        db = np.zeros_like(b)

        # -------------------------------------------------
        # Backpropagation Through Time
        # -------------------------------------------------

        for t in reversed(range(len(cache))):

            (
                xt,
                h_prev,
                c_prev,
                f,
                i,
                g,
                o,
                c,
                h
            ) = cache[t]

            tanh_c = np.tanh(c)

            # -------------------------------------------------
            # Hidden state gradient
            # -------------------------------------------------

            dh = dh_next

            # h = o * tanh(c)

            do = dh * tanh_c

            dc = (
                dh *
                o *
                (1 - tanh_c ** 2)
            )

            dc += dc_next

            # -------------------------------------------------
            # Cell state
            # -------------------------------------------------

            # c = f*c_prev + i*g

            df = dc * c_prev

            di = dc * g

            dg = dc * i

            dc_next = dc * f

            # -------------------------------------------------
            # Gate activation derivatives
            # -------------------------------------------------

            # sigmoid derivative:
            # sigmoid(x) * (1 - sigmoid(x))

            df_pre = (
                df *
                f *
                (1 - f)
            )

            di_pre = (
                di *
                i *
                (1 - i)
            )

            # tanh derivative

            dg_pre = (
                dg *
                (1 - g ** 2)
            )

            do_pre = (
                do *
                o *
                (1 - o)
            )

            # -------------------------------------------------
            # Combine gate gradients
            # -------------------------------------------------

            dz = np.vstack(
                (
                    df_pre,
                    di_pre,
                    dg_pre,
                    do_pre
                )
            )

            # -------------------------------------------------
            # Parameter gradients
            # -------------------------------------------------

            combined = np.vstack(
                (
                    h_prev,
                    xt
                )
            )

            dW += dz @ combined.T

            db += dz

            # -------------------------------------------------
            # Gradient to previous hidden state
            # -------------------------------------------------

            dcombined = W.T @ dz

            dh_next = dcombined[
                :hidden_size
            ]


        # =====================================================
        # Gradient Clipping
        # =====================================================

        np.clip(
            dW,
            -1,
            1,
            out=dW
        )

        np.clip(
            db,
            -1,
            1,
            out=db
        )

        np.clip(
            dWy,
            -1,
            1,
            out=dWy
        )

        np.clip(
            dby,
            -1,
            1,
            out=dby
        )


        # =====================================================
        # Update Parameters
        # =====================================================

        W -= learning_rate * dW

        b -= learning_rate * db

        Wy -= learning_rate * dWy

        by -= learning_rate * dby


    # =========================================================
    # Training Loss
    # =========================================================

    if epoch % 500 == 0:

        average_loss = (
            total_loss / len(X)
        )

        loss_history.append(
            average_loss
        )

        print(
            f"Epoch {epoch} | "
            f"Loss: {average_loss:.4f}"
        )


# =========================================================
# Evaluate Training Data
# =========================================================

print("\nTraining Results")

for i in range(len(X)):

    prediction, _ = forward(X[i])

    print(
        f"Input: {X[i].flatten()} "
        f"Actual: {y[i][0]:.2f} "
        f"Prediction: {prediction[0][0]:.2f}"
    )


# =========================================================
# Test
# =========================================================

test = np.array(
    [[2], [3], [4]],
    dtype=np.float32
)

prediction, _ = forward(test)

print("\nTest")

print(
    f"Input: {test.flatten()} "
    f"Prediction: {prediction[0][0]:.2f}"
)


# =========================================================
# Plot Training Loss
# =========================================================

plt.plot(loss_history)

plt.xlabel("Training Checkpoint")
plt.ylabel("Loss")

plt.title("LSTM Training Loss")

plt.savefig(
    "training_loss.png"
)

plt.show()