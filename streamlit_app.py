import os
import streamlit as st
from nlp_crew.src.nlp_crew.crew import NlpCrew  # Ensure this import is correct

# Access the API key from Streamlit secrets
groq_api_key = st.secrets["GROQ"]["API_KEY"]

os.environ["SERPER_API_KEY"] = "a579ab50ddbfb327acc282c659c95c8a6cc420a5"

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

# Initialize the CrewAI crew
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
        # Custom input fields for the "Research Task"
        query = st.text_input("Enter your query:", placeholder="Type your research topic here...")
    elif task_option == "Cheat Sheet":
        # Custom input fields for the "Reporting Task"
        cheat_sheet_topic = st.text_input("Enter report topic:", placeholder="Type your report topic here...")

    # Run task on button click
    if st.button("Run Task"):
        # Validation to ensure inputs are filled
        if task_option == "Question Answering" and query.strip() == "":
            st.warning("Please enter a query before running the question answering task!")
        elif task_option == "Cheat Sheet" and cheat_sheet_topic.strip() == "":
            st.warning("Please enter a query before running the cheat sheet task!")
        else:
            with st.spinner("Running the selected task..."):
                # Create a CrewContext for running tasks
                context = CrewContext(
                    topic=query if task_option == "Question Answering" else cheat_sheet_topic
                )

                # Execute the task by triggering the crew's process
                try:
                    result = nlp_crew.crew().kickoff(inputs=context.to_dict())
                    
                    # Access the 'raw' field of the result (since 'output' does not exist)
                    if hasattr(result, 'raw'):
                        st.success("Task Completed!")
                        st.write("**Task Output (HTML Format):**")
                        
                        # Read the output file and display its contents in Streamlit
                        with open('question_ansmwer_report.d', 'r') as file:
                            report = file.read()

                        st.write(report)  # Display the contents in the Streamlit app

                    else:
                        st.warning("The task returned an unexpected result format.")
                
                except Exception as e:
                    # Handle any errors that might occur during task execution
                    st.error(f"An error occurred while running the task: {str(e)}")
                    
elif selection == "Upload Materials":
    st.title("Upload Section")
    
    # Let the user upload PDF files
    uploaded_files = st.file_uploader("Upload Course Materials (PDFs)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
        
        # Convert uploaded files into a list of file-like objects
        file_names = [file.name for file in uploaded_files]
        
        # Call the content ingestion task via the Crew agent
        crew_instance = NlpCrew()
        # This task will handle the files and process them accordingly
        results = crew_instance.content_ingestion_task(uploaded_files)

        # Display the results from the task
        for result in results:
            st.write(result)  # Display each result in the Streamlit app


