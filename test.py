from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# Step 1: Load trained model
# -------------------------------
print("Loading model...")

tokenizer = T5Tokenizer.from_pretrained("recipe_generator_model")
model = T5ForConditionalGeneration.from_pretrained("recipe_generator_model")

print("Model loaded successfully!")

# -------------------------------
# Step 2: Continuous input loop
# -------------------------------
while True:
    print("\n----------------------------------")
    user_input = input("Enter ingredients (comma separated) or type 'exit': ")

    # Exit condition
    if user_input.lower() == "exit":
        print("Exiting program...")
        break

    # -------------------------------
    # Step 3: Format input
    # -------------------------------
    input_text = f"ingredients: {user_input}"

    # -------------------------------
    # Step 4: Tokenization
    # -------------------------------
    inputs = tokenizer(input_text, return_tensors="pt")

    # -------------------------------
    # Step 5: Generate recipe
    # -------------------------------
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=2.0
    )

    # -------------------------------
    # Step 6: Decode output
    # -------------------------------
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -------------------------------
    # Step 7: Display result
    # -------------------------------
    print("\n🍽️ Generated Recipe:\n")
    print(recipe)