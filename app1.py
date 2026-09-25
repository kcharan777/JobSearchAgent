import os
import json
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# -----------------------------
# API CONFIGURATION
# -----------------------------

INDIAN_API_URL = "https://jobs.indianapi.in/jobs"

INDIAN_API_KEY = os.environ.get("INDIAN_API_KEY")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# -----------------------------
# GEMINI MODEL
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    max_retries=5
)


# -----------------------------
# FETCH LIVE JOBS
# -----------------------------

def fetch_live_jobs(limit=50):

    if not INDIAN_API_KEY:
        raise Exception(
            "INDIAN_API_KEY is missing in Render Environment."
        )

    headers = {
        "X-Api-Key": INDIAN_API_KEY
    }

    params = {
        "limit": str(limit)
    }

    response = requests.get(
        INDIAN_API_URL,
        headers=headers,
        params=params,
        timeout=30
    )

    if response.status_code == 401:
        raise Exception(
            "Invalid Indian API key. Check Render Environment."
        )

    response.raise_for_status()

    data = response.json()

    # API may return a list or a dictionary containing jobs

    if isinstance(data, list):
        jobs = data

    elif isinstance(data, dict):
        jobs = data.get("jobs", [])

    else:
        jobs = []

    return jobs


# -----------------------------
# TOOL 1: SEARCH LIVE JOBS
# -----------------------------

@tool
def search_jobs(role: str, location: str = "") -> str:
    """
    Search real job vacancies using Indian Jobs API.
    Search by job role and preferred location.
    """

    try:

        jobs = fetch_live_jobs(limit=50)

        results = []

        for job in jobs:

            title = str(
                job.get("title", "")
                or job.get("job_title", "")
            )

            company = str(
                job.get("company", "")
            )

            job_location = str(
                job.get("location", "")
            )

            description = str(
                job.get("job_description", "")
            )

            skills = str(
                job.get("education_and_skills", "")
            )

            role_text = (
                title + " " +
                description + " " +
                skills
            ).lower()

            location_text = job_location.lower()

            # Match role

            role_match = (
                not role
                or role.lower() in role_text
            )

            # Match location

            location_match = (
                not location
                or location.lower() in location_text
                or "remote" in location_text
            )

            if role_match and location_match:

                results.append({
                    "title": title,
                    "company": company,
                    "location": job_location,
                    "job_type": job.get("job_type", ""),
                    "experience": job.get("experience", ""),
                    "salary": job.get("salary", ""),
                    "description": description,
                    "required_skills": skills,
                    "apply_link": job.get("apply_link", ""),
                    "posted_date": job.get("posted_date", "")
                })

        return json.dumps(
            {
                "total_found": len(results),
                "jobs": results[:15]
            },
            ensure_ascii=False
        )

    except Exception as e:

        return json.dumps({
            "error": str(e)
        })


# -----------------------------
# TOOL 2: MATCH SKILLS
# -----------------------------

@tool
def match_skills(
    user_skills: str,
    required_skills: str
) -> str:
    """
    Compare user skills with the skills required for a job.
    Return matched skills, missing skills and match percentage.
    """

    user_list = [
        s.strip().lower()
        for s in user_skills.split(",")
        if s.strip()
    ]

    required_list = [
        s.strip().lower()
        for s in required_skills.split(",")
        if s.strip()
    ]

    matched = [
        skill for skill in required_list
        if any(
            skill in user_skill
            or user_skill in skill
            for user_skill in user_list
        )
    ]

    missing = [
        skill for skill in required_list
        if skill not in matched
    ]

    percentage = (
        round(len(matched) / len(required_list) * 100)
        if required_list else 0
    )

    return json.dumps({
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": percentage
    })


# -----------------------------
# CREATE LANGGRAPH AGENT
# -----------------------------

tools = [
    search_jobs,
    match_skills
]

agent = create_react_agent(
    llm,
    tools
)


# -----------------------------
# FASTAPI APPLICATION
# -----------------------------

app = FastAPI(
    title="Job Search Agent",
    version="2.0",
    description="Live Job Search and Skill Matching Agent"
)


# -----------------------------
# CORS CONFIGURATION
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "Live Job Search Agent"
    }


# -----------------------------
# TEST LIVE JOB API
# -----------------------------

@app.get("/live-jobs")
def live_jobs():

    try:

        jobs = fetch_live_jobs(limit=10)

        return {
            "total": len(jobs),
            "jobs": jobs
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# INPUT MODEL
# -----------------------------

class JobInput(BaseModel):

    input: str = Field(
        description="Job role, location and user skills"
    )


# -----------------------------
# AGENT INVOKE ENDPOINT
# -----------------------------

@app.post("/agent/invoke")
def invoke_agent(data: JobInput):

    try:

        user_request = data.input

        prompt = """
You are a Live Job Search Agent.

IMPORTANT RULES:

1. Use the search_jobs tool to retrieve real vacancies.
2. Do not invent companies, jobs, salaries or application links.
3. Use the match_skills tool to compare the user's skills.
4. Only recommend jobs returned by the live API.
5. If no jobs are found, clearly say no matching jobs were found.
6. If the API returns an error, report the error instead of
   showing sample jobs.
7. Display actual company, title, location, experience,
   salary if available, required skills, and application link.
8. If salary or any information is missing, say Not provided.
9. Never claim a job is currently open unless the API
   listing indicates it is available.

For each job, provide:

Company:
Job Role:
Location:
Experience:
Salary:
Required Skills:
Application Link:

Then give the skill match analysis.

User request:
""" + user_request

        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
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

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
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
