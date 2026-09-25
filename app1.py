
import os
import json

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from langserve import add_routes


# -----------------------------
# JOB DATABASE
# -----------------------------

JOBS = [
    {
        "title": "Python Developer Intern",
        "company": "TechNova",
        "location": "Hyderabad",
        "type": "Internship",
        "experience": "Fresher",
        "skills": ["Python", "SQL", "Git"],
        "salary": "₹10,000-₹20,000/month",
        "link": "https://www.linkedin.com/jobs/"
    },
    {
        "title": "Data Analyst Intern",
        "company": "DataWorks",
        "location": "Bengaluru",
        "type": "Internship",
        "experience": "Fresher",
        "skills": ["Python", "SQL", "Excel", "Power BI"],
        "salary": "₹15,000-₹25,000/month",
        "link": "https://www.linkedin.com/jobs/"
    },
    {
        "title": "Machine Learning Intern",
        "company": "AI Labs",
        "location": "Hyderabad",
        "type": "Internship",
        "experience": "Fresher",
        "skills": ["Python", "Machine Learning", "Pandas"],
        "salary": "₹15,000-₹30,000/month",
        "link": "https://www.linkedin.com/jobs/"
    },
    {
        "title": "Java Developer",
        "company": "CodeWorks",
        "location": "Chennai",
        "type": "Full-time",
        "experience": "0-2 years",
        "skills": ["Java", "Spring Boot", "SQL"],
        "salary": "₹3-6 LPA",
        "link": "https://www.linkedin.com/jobs/"
    },
    {
        "title": "Frontend Developer Intern",
        "company": "WebSpark",
        "location": "Remote",
        "type": "Internship",
        "experience": "Fresher",
        "skills": ["HTML", "CSS", "JavaScript", "React"],
        "salary": "₹10,000-₹20,000/month",
        "link": "https://www.linkedin.com/jobs/"
    },
    {
        "title": "Full Stack Developer Intern",
        "company": "AppForge",
        "location": "Hyderabad",
        "type": "Internship",
        "experience": "Fresher",
        "skills": ["JavaScript", "Node.js", "React", "MongoDB"],
        "salary": "₹12,000-₹25,000/month",
        "link": "https://www.linkedin.com/jobs/"
    }
]


# -----------------------------
# GEMINI MODEL
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0,
    max_retries=5
)


# -----------------------------
# TOOL 1: SEARCH JOBS
# -----------------------------

@tool
def search_jobs(role: str, location: str = "") -> str:
    """Search sample job listings by job role and location."""

    results = []

    for job in JOBS:

        role_match = (
            role.lower() in job["title"].lower()
            or role.lower() in " ".join(job["skills"]).lower()
        )

        location_match = (
            not location
            or location.lower() in job["location"].lower()
            or job["location"].lower() == "remote"
        )

        if role_match and location_match:
            results.append(job)

    return json.dumps(results, indent=2)


# -----------------------------
# TOOL 2: MATCH SKILLS
# -----------------------------

@tool
def match_skills(user_skills: str, job_title: str) -> str:
    """Compare user skills with the requirements of a job."""

    job = next(
        (
            j for j in JOBS
            if j["title"].lower() == job_title.lower()
        ),
        None
    )

    if not job:
        return "Job not found in the sample database."

    skills = [
        s.strip().lower()
        for s in user_skills.split(",")
    ]

    required = [
        s.lower()
        for s in job["skills"]
    ]

    matched = [
        s for s in required
        if s in skills
    ]

    missing = [
        s for s in required
        if s not in skills
    ]

    score = (
        round(len(matched) / len(required) * 100)
        if required else 0
    )

    return json.dumps({
        "job_title": job["title"],
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": score
    }, indent=2)


# -----------------------------
# TOOL 3: FILTER JOBS
# -----------------------------

@tool
def filter_jobs(
    location: str = "",
    job_type: str = "",
    experience: str = ""
) -> str:
    """Filter jobs by location, job type and experience."""

    results = []

    for job in JOBS:

        if location:
            if (
                location.lower() not in job["location"].lower()
                and job["location"].lower() != "remote"
            ):
                continue

        if job_type:
            if job_type.lower() not in job["type"].lower():
                continue

        if experience:
            if experience.lower() not in job["experience"].lower():
                continue

        results.append(job)

    return json.dumps(results, indent=2)


# -----------------------------
# CREATE AGENT
# -----------------------------

tools = [
    search_jobs,
    match_skills,
    filter_jobs
]

agent = create_react_agent(
    llm,
    tools
)


# -----------------------------
# AGENT FUNCTION
# -----------------------------

def job_recommendation(data):

    user_request = data["input"]

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_request
            }
        ]
    })

    answer = response["messages"][-1].content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return {
        "output": answer
    }


# -----------------------------
# API INPUT AND OUTPUT
# -----------------------------

class JobInput(BaseModel):
    input: str = Field(
        description="Job search request"
    )


class JobOutput(BaseModel):
    output: str


chain = RunnableLambda(
    job_recommendation
).with_types(
    input_type=JobInput,
    output_type=JobOutput
)


# -----------------------------
# FASTAPI APPLICATION
# -----------------------------

app = FastAPI(
    title="Job Search Agent",
    version="1.0",
    description="AI-powered Job Search and Skill Matching Agent"
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agent": "Job Search Agent"
    }


add_routes(
    app,
    chain,
    path="/agent"
)


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
