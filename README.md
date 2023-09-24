# deepsy
Deep learning framework built with Python and NumPy. Supports building neural networks, training them with gradient descent, using different activation and loss functions as well as regularization and dropout.
  
### Use:  
```python
nn = NeuralNetwork(layers=[
    Layer(nr_neurons=8, nr_in_features=6, activation_func=ReLU()),
    Layer(nr_neurons=12, nr_in_features=8, activation_func=ReLU(), dropout_rate=0.2),
    Layer(nr_neurons=1, nr_in_features=12, activation_func=Sigmoid())
])
model = Model(nn, loss_func=MSE(), optimizer=GradientDescent(learning_rate=0.2))

# Train
model.train(X_train, Y_train, nr_epochs=400, reg_lambda=0.2)

# Validate
Y_validation_predicted, cost_validation = model.validate(X_validation, Y_validation)

# Test
Y_test_predicted = model.predict(X_test)
```
  
### Todo:
- add convolutional layers
- add adam and other optimizers
- add mini batch grad descent
