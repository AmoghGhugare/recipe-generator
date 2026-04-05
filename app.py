import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🍳 Smart Recipe Generator",
    page_icon="🍽️",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------
st.title("🍳 Smart Recipe Generator")
st.markdown("Generate recipes using ingredients you have!")

# -------------------------------
# Load Model (cached for speed)
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("recipe_generator_model")
    model = T5ForConditionalGeneration.from_pretrained("recipe_generator_model")
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# Input Section
# -------------------------------
ingredients = st.text_area(
    "📝 Enter ingredients (comma separated):",
    placeholder="e.g. chicken, onion, rice"
)

# -------------------------------
# Generate Button
# -------------------------------
if st.button("🍽️ Generate Recipe"):

    if ingredients.strip() == "":
        st.warning("⚠️ Please enter some ingredients!")
    else:
        with st.spinner("Cooking your recipe... 🍳"):

            # Format input
            input_text = f"ingredients: {ingredients}"

            # Tokenize
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate
            outputs = model.generate(
                **inputs,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=2.0
            )

            # Decode
            recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # -------------------------------
        # Display Output
        # -------------------------------
        st.success("✅ Recipe Generated!")

# -------------------------------
# Structured Output Formatting
# -------------------------------

# Try to split recipe into lines
lines = recipe.split(". ")

# Recipe Title (first line)
title = lines[0] if len(lines) > 0 else "Generated Recipe"

# Ingredients & Steps (basic assumption)
ingredients_list = []
steps_list = []

for line in lines[1:]:
    if any(word in line.lower() for word in ["ingredient", "add", "mix", "use"]):
        ingredients_list.append(line.strip())
    else:
        steps_list.append(line.strip())

# -------------------------------
# Display Structured Output
# -------------------------------

st.subheader(f"🍲 {title}")

st.markdown("### 📝 Ingredients")
for item in ingredients_list[:6]:  # limit for cleaner UI
    st.write(f"- {item}")

st.markdown("### 👨‍🍳 Steps")
for i, step in enumerate(steps_list[:8], 1):
    st.write(f"{i}. {step}")