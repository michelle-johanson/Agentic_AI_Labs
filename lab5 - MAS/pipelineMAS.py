"""
Lab: Pipeline Pattern -- Resume Screening
IS 590R -- AI and Agentic Systems
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("ollama:qwen2.5:3b") #openai:gpt-5-mini ollama:qwen2.5:1.5b


class ApplicationState(TypedDict):
    resume_text: str
    job_requirements: str
    extracted_skills: str
    score: str
    email_draft: str


def extract_skills(state: ApplicationState) -> dict:
    response = model.invoke([
        {"role": "system", "content": (
            "You extract skills and relevant experience from resumes. "
            "Return a short bullet list of skills and a one-line summary "
            "of years of relevant experience. Be concise."
        )},
        {"role": "user", "content": state["resume_text"]},
    ])
    return {"extracted_skills": response.text}


def score_candidate(state: ApplicationState) -> dict:
    response = model.invoke([
        {"role": "system", "content": (
            "You score how well a candidate's extracted skills fit a job's "
            "requirements. Respond with a fit score from 1-10 followed by "
            "a one-sentence rationale, e.g. '7/10 -- strong SQL and BI "
            "experience, light on stakeholder presentation.'"
        )},
        {"role": "user", "content": (
            f"Job requirements:\n{state['job_requirements']}\n\n"
            f"Candidate's extracted skills:\n{state['extracted_skills']}"
        )},
    ])
    return {"score": response.text}


def draft_response(state: ApplicationState) -> dict:
    response = model.invoke([
        {"role": "system", "content": (
            "You draft short, professional candidate-facing emails for an "
            "HR team. If the fit score is 6 or higher, write a brief "
            "interview invitation. If it's below 6, write a brief, "
            "respectful rejection. Do not restate the numeric score in the "
            "email itself."
        )},
        {"role": "user", "content": f"Fit assessment:\n{state['score']}"},
    ])
    return {"email_draft": response.text}


graph = StateGraph(ApplicationState)
graph.add_node("extract_skills", extract_skills)
graph.add_node("score_candidate", score_candidate)
graph.add_node("draft_response", draft_response)
graph.add_edge(START, "extract_skills")
graph.add_edge("extract_skills", "score_candidate")
graph.add_edge("score_candidate", "draft_response")
graph.add_edge("draft_response", END)
pipeline_graph = graph.compile()
pipeline_graph.get_graph().draw_mermaid_png(output_file_path="pipeline_workflow.png")


JOB_REQUIREMENTS = """
Role: Junior Data Analyst
Requirements: SQL, at least 1 year working with spreadsheets or BI tools,
comfortable presenting findings to non-technical stakeholders.
"""

SAMPLE_RESUME = """
Maria Chen -- B.S. Information Systems, BYU, May 2026.
Internship: built weekly sales dashboards in Tableau for a 40-person retail
team; wrote SQL queries against a Snowflake warehouse to pull source data;
presented dashboard walkthroughs to store managers monthly.
"""

if __name__ == "__main__":
    result = pipeline_graph.invoke({
        "resume_text": SAMPLE_RESUME,
        "job_requirements": JOB_REQUIREMENTS,
        "extracted_skills": "",
        "score": "",
        "email_draft": "",
    })

    print("\n--- Extracted skills ---")
    print(result["extracted_skills"])
    print("\n--- Score ---")
    print(result["score"])
    print("\n--- Email draft ---")
    print(result["email_draft"])
