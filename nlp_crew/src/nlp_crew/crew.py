import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, PDFSearchTool
import os

os.environ["OPENAI_API_KEY"] = "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"

# Specify the path
db_path = "../../db_storage"

# Ensure the directory exists
os.makedirs(db_path, exist_ok=True)

# Create the PersistentClient
client = chromadb.PersistentClient(path=db_path)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

pdf_tool = PDFSearchTool()

@CrewBase
class NlpCrew():
    """NlpCrew crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def question_answerer(self) -> Agent:
        return Agent(
            config=self.agents_config['question_answerer'],
            tools=[SerperDevTool()],
            verbose=True,
            max_iter=1,
            cache=False
        )

    @agent
    def cheat_sheet(self) -> Agent:
        return Agent(
            config=self.agents_config['cheat_sheet'],
            tools=[pdf_tool],
            verbose=True,
            cache=False
        )

    @agent
    def content_ingestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['content_ingestion_agent'],
            tools=[pdf_tool],
            verbose=True
        )

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a single PDF file."""
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def format_with_references(self, task_output):
        """
        Postprocess the task output to include references.
        """
        try:
            # Extract the main answer and references from the output
            answer = task_output.get("answer", "No answer provided.")
            references = task_output.get("references", [])

            # Format the references as a markdown-friendly list
            formatted_references = "\n".join(
                [f"{i+1}. {ref['title']} - {ref['url']}" for i, ref in enumerate(references)]
            )

            # Combine answer and references
            result = f"### Answer\n\n{answer}\n\n### References\n\n{formatted_references}"

            return result
        except Exception as e:
            return f"An error occurred while formatting the output: {e}"
    
    @task
    def question_answerer_task(self) -> Task:
        return Task(
            config=self.tasks_config['question_answerer_task'],
            output_file='question_answer_report.md',
            postprocess=self.format_with_references  # Use a custom postprocess step
        )


    @task
    def cheat_sheet_task(self) -> Task:

        return Task(
            config=self.tasks_config['cheat_sheet_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NlpCrew crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
