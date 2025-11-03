import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import matplotlib.pyplot as plt
import optuna
from datetime import datetime

# Load the CSV file into a DataFrame
file_path = 'C:/Users/Howie/Documents/Jupyter Files/contacts.csv'
df = pd.read_csv(file_path)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Calculate days between consecutive issues for each person
df['days_between_issues'] = df.groupby('person')['date'].diff().dt.days

# Drop rows with missing or invalid values
df.dropna(subset=['days_between_issues'], inplace=True)

# Define features and target variable
X = df[['person']]  # Only person as feature
y = df['days_between_issues']

# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the PyTorch model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Store learning rates and MAEs from each trial
trial_numbers = []
learning_rates = []
maes = []

# Objective function for Optuna to optimize the learning rate and hidden units
def objective(trial):
    hidden_units = trial.suggest_int('hidden_units', 32, 512, step=32)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

    model = NeuralNet(X_train.shape[1], hidden_units)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training the model
    model.train()
    for epoch in range(100):  # Number of epochs
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mae = mean_absolute_error(y_test_tensor.numpy(), y_pred.numpy())
    
    # Track the trial number, learning rate, and corresponding MAE
    trial_numbers.append(trial.number)
    learning_rates.append(learning_rate)
    maes.append(mae)
    
    return mae  # We minimize the MAE

# Optuna study for hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Get the best trial (lowest MAE)
best_trial = study.best_trial
best_learning_rate = best_trial.params['learning_rate']
best_hidden_units = best_trial.params['hidden_units']

# Retrain the model with the best hyperparameters
best_model = NeuralNet(X_train.shape[1], best_hidden_units)
optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)
criterion = nn.MSELoss()

# Training the model again with the best hyperparameters
best_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = best_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Prediction interface with Streamlit widgets
person_dropdown = st.selectbox('Select Caller:', df['person'].unique())
predict_button = st.button("Predict")

output_text = st.empty()
output_graph = st.empty()

# Function to update graph and MAE for the selected user
def predict_days_until_issue():
    selected_person = person_dropdown
    input_data = pd.DataFrame([[selected_person]], columns=['person'])
    input_data_encoded = encoder.transform(input_data)  # Use the same encoder for consistency
    
    # Filter the DataFrame to include only data for the selected person
    df_selected_person = df[df['person'] == selected_person]
    X_selected_person = df_selected_person[['person']]
    y_selected_person = df_selected_person['days_between_issues']
    
    # Convert the input data to a tensor
    input_data_tensor = torch.tensor(input_data_encoded, dtype=torch.float32)
    
    # Make predictions using the best trained model
    predicted_days_tensor = best_model(input_data_tensor).item()  # Convert tensor to scalar
    predicted_days = max(predicted_days_tensor, 1)  # Ensure predicted days are at least 1 (to avoid 0 days)
    
    # Calculate the date of the next issue
    current_date = pd.to_datetime(datetime.now().date())  # <-- Using the datetime module here
    last_issue_date = df_selected_person['date'].max()

    # Ensure there's historical data for the selected person
    if pd.notnull(last_issue_date):
        next_issue_date = last_issue_date + pd.Timedelta(days=predicted_days)
        if next_issue_date <= current_date:
            next_issue_date = current_date + pd.Timedelta(days=1)
    else:
        next_issue_date = current_date + pd.Timedelta(days=predicted_days)

    # Update the days until issue prediction based on the current date
    days_until_issue = (next_issue_date - current_date).days

    # Calculate the Mean Absolute Error for the selected person
    selected_person_data = df[df['person'] == selected_person]
    X_selected_encoded = encoder.transform(selected_person_data[['person']])
    y_pred_person = best_model(torch.tensor(X_selected_encoded, dtype=torch.float32)).detach().numpy()
    mae_person = mean_absolute_error(selected_person_data['days_between_issues'], y_pred_person)

    # Display results for predicted days and predicted date
    output_text.empty()  # Clear previous output
    st.markdown(f"**Predicted days until next issue for {selected_person}:** {days_until_issue} days")
    st.markdown(f"**Predicted date of next issue:** {next_issue_date.date()}")
    st.markdown(f"**Mean Absolute Error for {selected_person}:** {mae_person}")

    # After the predictions, plot the interaction graph for MAE and learning rate
    with output_graph:
        output_graph.empty()
        
        # Plotting frequency of calls (dates) and issue counts
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot 1: Frequency of Calls (Dates)
        axes[0].hist(df_selected_person['date'], bins=20, alpha=0.7)
        axes[0].set_title(f"Date Distribution for {selected_person}")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Frequency")
        axes[0].tick_params(axis="x", rotation=45)

        # Subplot 2: Issue Distribution
        axes[1].hist(df_selected_person['days_between_issues'], bins=20, alpha=0.7)
        axes[1].set_title(f"Issue Distribution for {selected_person}")
        axes[1].set_xlabel("Days Between Issues")
        axes[1].set_ylabel("Frequency")
        axes[1].tick_params(axis="x", rotation=45)

        st.pyplot(fig)

    # Now show the graph for MAE and learning rate interaction after predictions
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot showing how each trial's MAE relates to its learning rate
    scatter = ax.scatter(trial_numbers, maes, c=learning_rates, cmap='viridis', s=100, edgecolor='black')

    # Add color bar to show the learning rates
    fig.colorbar(scatter, ax=ax, label='Learning Rate')

    # Add labels and title
    ax.set_xlabel("Trial Number (Iteration)")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("Interaction of Iterations, MAE, and Learning Rate")

    st.pyplot(fig)

    # Finally, show the best learning rate and best hidden units
    st.markdown(f"**Best Learning Rate:** {best_learning_rate}")
    st.markdown(f"**Best Hidden Units:** {best_hidden_units}")

# Trigger prediction on button click
if predict_button:
    predict_days_until_issue()
