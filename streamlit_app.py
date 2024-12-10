import os
import streamlit as st
from nlp_crew.src.nlp_crew.crew import NlpCrew  # Ensure this import is correct

# Access the API key from Streamlit secrets
groq_api_key = st.secrets["GROQ"]["API_KEY"]

os.environ["SERPER_API_KEY"] = "a579ab50ddbfb327acc282c659c95c8a6cc420a5"
# os.environ["OPENAI_API_KEY"] = "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"

# Check if the API key exists
if not groq_api_key:
    st.error("Groq API key is missing! Please set it in the Streamlit secrets.")
else:
    # Set the API key in the environment for use by the external service
    os.environ["GROQ_API_KEY"] = groq_api_key  # If the API client uses environment variables

# Custom context class to store task parameters
class CrewContext:
    """Custom context class to hold dynamic inputs."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        # Convert the object attributes to a dictionary
        return self.__dict__

nlp_crew = NlpCrew()

st.sidebar.title("Navigation")
pages = ["Home", "Upload Materials"]
selection = st.sidebar.radio("Go to", pages)

if selection == "Home":
    # Streamlit app interface
    st.title("NLP Crew Dashboard")
    st.write("Interact with CrewAI tasks with custom inputs.")

    # Task selection dropdown
    task_option = st.selectbox(
        "Select a Task to Run",
        ["Question Answering", "Cheat Sheet"]
    )

    # Dynamic input fields for tasks
    if task_option == "Question Answering":
        query = st.text_input("Enter your query:", placeholder="Type your research topic here...")
    elif task_option == "Cheat Sheet":
        cheat_sheet_topic = st.text_input("Enter report topic:", placeholder="Type your report topic here...")
        upload_file = st.file_uploader("Upload Course Materials (PDFs)", type="pdf", accept_multiple_files=False)

    # Run task on button click
    if st.button("Run Task"):
        if task_option == "Question Answering":
            if not query.strip():
                st.warning("Please enter a query before running the question answering task!")
            else:
                with st.spinner("Running the Question Answerer..."):
                    # Context for the Question Answering task
                    # Initialize the CrewAI crew
                    context = CrewContext(topic=query)

                    try:
                        # Execute Question Answerer task
                        result = nlp_crew.crew().kickoff(inputs=context.to_dict())

                        # Read the output file
                        output_file = "question_answer_report.md"
                        if os.path.exists(output_file):
                            with open(output_file, "r", encoding="utf-8") as file:
                                report_content = file.read()
                            st.success("Task Completed!")
                            st.write("**Task Output:**")
                            st.markdown(report_content)
                        else:
                            st.warning("Output file not found. Please check the task execution.")
                    except Exception as e:
                        st.error(f"An error occurred while running the task: {str(e)}")

        elif task_option == "Cheat Sheet":
            if not cheat_sheet_topic.strip():
                st.warning("Please enter a topic for the cheat sheet!")
            elif not upload_file:
                st.warning("Please upload a PDF file!")
            else:
                with st.spinner("Generating Cheat Sheet..."):
                    try:
                        # Save the uploaded file temporarily
                        temp_file_path = f"temp_{upload_file.name}"
                        with open(temp_file_path, "wb") as f:
                            f.write(upload_file.read())

                        # Context for the Cheat Sheet task
                        context = CrewContext(
                            topic=cheat_sheet_topic,
                            uploaded_file=temp_file_path
                        )

                        # Execute Cheat Sheet task
                        result = nlp_crew.crew().kickoff(inputs=context.to_dict())

                        # Read the output file
                        output_file = "report.md"
                        if os.path.exists(output_file):
                            with open(output_file, "r", encoding="utf-8") as file:
                                cheat_sheet_content = file.read()
                            st.success("Cheat Sheet Generated!")
                            st.markdown(cheat_sheet_content)
                        else:
                            st.warning("Output file not found. Please check the task execution.")
                        
                        # Clean up temporary file
                        os.remove(temp_file_path)
                    except Exception as e:
                        st.error(f"An error occurred while running the task: {str(e)}")

                    
elif selection == "Upload Materials":
    st.title("Upload Section")
    
    uploaded_files = st.file_uploader("Upload Course Materials (PDFs)", type="pdf", accept_multiple_files=False)
    if uploaded_files:
        # Create a temporary directory to save the files
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded files temporarily and store their paths
        temp_file_paths = []
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Save the content of the uploaded file
            temp_file_paths.append(temp_file_path)  # Add path to list

        # Pass the list of file paths to the task
        result = nlp_crew.content_ingestion_task(uploaded_files=temp_file_paths[0])
        st.write(result)
