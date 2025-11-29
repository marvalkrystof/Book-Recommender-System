import streamlit as st

st.title("Book Recommender System")

st.write("Enter your preferences to get book recommendations")

# Text input
user_input = st.text_input("What kind of books are you looking for?", 
                           placeholder="e.g., fantasy novels with strong female characters")

# Display the input
if user_input:
    st.write(f"You're looking for: {user_input}")
    
    # Placeholder for recommendations
    st.subheader("Recommendations")
    st.info("Recommendation system will be integrated here")
