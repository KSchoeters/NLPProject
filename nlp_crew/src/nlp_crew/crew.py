import chromadb
import streamlit as st
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import os

# Specify the path
db_path = "../../db_storage"

# Ensure the directory exists
os.makedirs(db_path, exist_ok=True)

# Create the PersistentClient
client = chromadb.PersistentClient(path=db_path)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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
            verbose=True
        )

    @agent
    def cheat_sheet(self) -> Agent:
        return Agent(
            config=self.agents_config['cheat_sheet'],
            verbose=True
        )

    # @agent
    # def content_ingestion_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['content_ingestion_agent'],
    #         verbose=True
    #     )

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a single PDF file."""
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def preprocess_text(self, text):
        """Preprocess text by splitting it into smaller chunks."""
        chunk_size = 500  # Split text into chunks of 500 words
        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def create_embeddings(self, chunks):
        """Generate embeddings for text chunks."""
        embeddings = model.encode(chunks)
        return embeddings

    def store_in_chromadb(self, chunks, embeddings, pdf_name):
        """Store the chunks and their embeddings in ChromaDB."""
        collection = client.get_or_create_collection("course_materials")
        metadata = [{"source": pdf_name, "text": chunk} for chunk in chunks]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=[f"{pdf_name}_{i}" for i in range(len(chunks))]
        )
        return f"Successfully stored {len(chunks)} chunks from {pdf_name} in ChromaDB."
    
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
            output_format="json",  # Ensure the task outputs a structured format
            output_file='question_answer_report.md',
            postprocess=self.format_with_references  # Use a custom postprocess step
        )


    @task
    def cheat_sheet_task(self) -> Task:
        return Task(
            config=self.tasks_config['cheat_sheet_task'],
            output_file='report.md'
        )
    
    
    # def content_ingestion_task(self, uploaded_files):
    #     """Handle the content ingestion task."""
    #     # Convert the list to a tuple (hashable)
    #     uploaded_files_tuple = tuple(uploaded_files)

    #     results = []
    #     for uploaded_file in uploaded_files_tuple:
    #         text = self.extract_text_from_pdf(uploaded_file)
    #         chunks = self.preprocess_text(text)
    #         embeddings = self.create_embeddings(chunks)
    #         result = self.store_in_chromadb(chunks, embeddings, uploaded_file.name)
    #         results.append(result)
    #     return results


    # def query_chromadb(self, query):
    #     """Query ChromaDB for relevant documents."""
    #     query_embedding = model.encode([query])  # Convert query to embedding
    #     collection = client.get_collection("course_materials")
    #     results = collection.query(
    #         query_embeddings=query_embedding,
    #         n_results=5  # Get top 5 results
    #     )
    #     return results

    @crew
    def crew(self) -> Crew:
        """Creates the NlpCrew crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
