import streamlit as st
import json

def main():
    st.image(image="chatbot maker.png",use_column_width="auto")
    st.title("🤖 Chatbot Data Entry")
    st.header("Enter Prompts, Patterns, and Responses")
    st.write("📝 Create prompts with associated patterns and responses.")
    st.write("🚀 Click the 'Submit' button to save the data.")

    num_prompts = st.number_input("Number of Prompts", min_value=1, max_value=10, value=1)
    
    prompts_data = []
    for i in range(num_prompts):
        tag = st.text_input(f"Prompt {i+1} Tag", key=f"tag_{i}")
        
        num_patterns = st.number_input(f"Number of Patterns for {tag}", min_value=1, max_value=10, value=1)
        patterns = []
        for j in range(num_patterns):
            pattern = st.text_input(f"Pattern {j+1}", key=f"pattern_{i}_{j}")
            patterns.append(pattern)
        
        num_responses = st.number_input(f"Number of Responses for {tag}", min_value=1, max_value=10, value=1)
        responses = []
        for k in range(num_responses):
            response = st.text_area(f"Response {k+1}", key=f"response_{i}_{k}")
            responses.append(response)
        
        prompt_data = {
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        }
        
        prompts_data.append(prompt_data)
    
    if st.button("Save the data"):
        data_to_save = {
            "prompts": prompts_data
        }
        with open("chatbot_data.json", "w") as json_file:
            json.dump(data_to_save, json_file, indent=4)
        st.success("Now you can train the model!")

if __name__ == "__main__":
    main()