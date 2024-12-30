# Example message to predict
new_message = ["Congratulations! You've won a $1000 gift card. Claim it now."]

# Convert the message into the same format as the training data
new_message_vectorized = vectorizer.transform(new_message)

# Predict whether the message is spam (1) or ham (0)
prediction = model.predict(new_message_vectorized)
print("Prediction (0 = Ham, 1 = Spam):", prediction[0])
