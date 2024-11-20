import os
import streamlit as st
from nlp_crew.src.nlp_crew.crew import NlpCrew  # Ensure this import is correct

# Access the API key from Streamlit secrets
groq_api_key = st.secrets["GROQ"]["API_KEY"]

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

# Streamlit app interface
st.title("NLP Crew Dashboard")
st.write("Interact with CrewAI tasks with custom inputs.")

# Task selection dropdown
task_option = st.selectbox(
    "Select a Task to Run",
    ["Research Task", "Reporting Task"]
)

# Dynamic input fields for tasks
if task_option == "Research Task":
    # Custom input fields for the "Research Task"
    query = st.text_input("Enter your query:", placeholder="Type your research topic here...")
elif task_option == "Reporting Task":
    # Custom input fields for the "Reporting Task"
    report_topic = st.text_input("Enter report topic:", placeholder="Type your report topic here...")

# Run task on button click
if st.button("Run Task"):
    # Validation to ensure inputs are filled
    if task_option == "Research Task" and query.strip() == "":
        st.warning("Please enter a query before running the research task!")
    elif task_option == "Reporting Task" and report_topic.strip() == "":
        st.warning("Please enter a report topic before running the reporting task!")
    else:
        with st.spinner("Running the selected task..."):
            # Create a CrewContext for running tasks
            context = CrewContext(
                topic=query if task_option == "Research Task" else report_topic
            )

            # Execute the task by triggering the crew's process
            try:
                result = nlp_crew.crew().kickoff(inputs=context.to_dict())
                
                # Access the 'raw' field of the result (since 'output' does not exist)
                if hasattr(result, 'raw'):
                    st.success("Task Completed!")
                    st.write("**Task Output (HTML Format):**")
                    
                    # Render the result in HTML format
                    st.markdown(result.raw, unsafe_allow_html=True)
                else:
                    st.warning("The task returned an unexpected result format.")
            
            except Exception as e:
                # Handle any errors that might occur during task execution
                st.error(f"An error occurred while running the task: {str(e)}")
