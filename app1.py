
import os
import re
import html
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------- CONFIGURATION ----------------

INDIAN_API_URL = "https://jobs.indianapi.in/jobs"

INDIAN_API_KEY = os.environ.get("INDIAN_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ---------------- FASTAPI APP ----------------

app = FastAPI(title="Live Job Search Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- GEMINI MODEL ----------------

llm = None

if GEMINI_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )


# ---------------- FETCH LIVE JOBS ----------------

def fetch_live_jobs(limit=50):

    if not INDIAN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="INDIAN_API_KEY is missing in Render environment variables."
        )

    headers = {
        "X-Api-Key": INDIAN_API_KEY
    }

    params = {
        "limit": limit
    }

    try:
        response = requests.get(
            INDIAN_API_URL,
            headers=headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):
            jobs = data

        elif isinstance(data, dict):
            jobs = (
                data.get("jobs")
                or data.get("data")
                or data.get("results")
                or []
            )

        else:
            jobs = []

        if not isinstance(jobs, list):
            jobs = []

        return jobs

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The job API took too long to respond."
        )

    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Job API returned an error: {e.response.status_code}"
        )

    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the live job API."
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="The job API returned invalid JSON."
        )


# ---------------- CLEAN JOB DATA ----------------

def clean(value):

    if value is None:
        return ""

    if isinstance(value, list):
        value = ", ".join(str(x) for x in value)

    if isinstance(value, dict):
        value = str(value)

    return html.unescape(str(value)).strip()


def get_job_details(job):

    return {
        "title": clean(
            job.get("job_title")
            or job.get("title")
            or "Not provided"
        ),

        "company": clean(
            job.get("company") or "Not provided"
        ),

        "location": clean(
            job.get("location") or "Not provided"
        ),

        "job_type": clean(
            job.get("job_type") or "Not provided"
        ),

        "experience": clean(
            job.get("experience") or "Not specified"
        ),

        "description": clean(
            job.get("job_description")
            or job.get("about_company")
            or "Not provided"
        ),

        "skills": clean(
            job.get("education_and_skills")
            or "Not specified"
        ),

        "responsibilities": clean(
            job.get("role_and_responsibility")
            or "Not specified"
        ),

        "apply_link": clean(
            job.get("apply_link") or ""
        ),

        "posted_date": clean(
            job.get("posted_date") or "Not specified"
        )
    }


# ---------------- SKILL MATCHING ----------------

def calculate_skill_match(user_skills, job):

    if not user_skills:
        return [], [], 0

    requirements = (
        job["skills"] + " " +
        job["description"] + " " +
        job["responsibilities"] + " " +
        job["title"]
    ).lower()

    skills = [
        skill.strip()
        for skill in re.split(r"[,;\n]", user_skills)
        if skill.strip()
    ]

    matched = []
    missing = []

    for skill in skills:

        if skill.lower() in requirements:
            matched.append(skill)

        else:
            missing.append(skill)

    percentage = round(
        len(matched) / len(skills) * 100
    )

    return matched, missing, percentage


# ---------------- SEARCH LIVE JOBS ----------------


def search_jobs(role, location="", skills="", limit=50):

    jobs = fetch_live_jobs(limit)

    role = role.strip().lower()
    location = location.strip().lower()

    results = []

    for raw_job in jobs:

        job = get_job_details(raw_job)

        # Use the actual job title first
        job_title = (
            job["title"]
        ).lower()

        description = (
            job["description"] + " " +
            job["skills"] + " " +
            job["responsibilities"]
        ).lower()

        searchable_text = (
            job_title + " " + description
        )

        # Role matching: check title OR description
        if role:

            role_words = [
                word for word in role.split()
                if word not in [
                    "job", "jobs", "fresher",
                    "freshers", "for", "in",
                    "intern", "internship"
                ]
            ]

            if role_words and not any(
                word in searchable_text
                for word in role_words
            ):
                continue

        # Location matching
        if location:

            job_location = job["location"].lower()

            location_aliases = {
                "bangalore": ["bangalore", "bengaluru"],
                "bengaluru": ["bangalore", "bengaluru"],
                "hyderabad": ["hyderabad"],
                "chennai": ["chennai"],
                "pune": ["pune"],
                "mumbai": ["mumbai"],
                "delhi": ["delhi"],
                "remote": ["remote", "work from home"]
            }

            locations_to_check = location_aliases.get(
                location,
                [location]
            )

            if not any(
                loc in job_location
                for loc in locations_to_check
            ) and "remote" not in job_location:
                continue

        # Calculate skill match
        matched, missing, percentage = (
            calculate_skill_match(skills, job)
        )

        job["matched_skills"] = matched
        job["missing_skills"] = missing
        job["match_percentage"] = percentage

        results.append(job)

    return results

# ---------------- FORMAT RESULTS ----------------

def format_results(results, role, location, skills):

    if not results:

        return (
            "No matching live jobs were found.\n\n"
            "Try another job role or location. "
            "The API may currently provide only a limited "
            "number of vacancies."
        )

    output = []

    output.append("LIVE JOB SEARCH RESULTS")
    output.append("=" * 35)

    output.append(f"Requested Role: {role or 'Any'}")
    output.append(f"Location: {location or 'Any'}")
    output.append(f"Live Jobs Found: {len(results)}")

    output.append(
        "\nThese results are retrieved from the live job API."
    )

    for index, job in enumerate(results, start=1):

        output.append("\n" + "-" * 40)

        output.append(f"\n{index}. {job['title']}")

        output.append(
            f"Company: {job['company']}"
        )

        output.append(
            f"Location: {job['location']}"
        )

        output.append(
            f"Job Type: {job['job_type']}"
        )

        output.append(
            f"Experience: {job['experience']}"
        )

        output.append(
            f"Posted Date: {job['posted_date']}"
        )

        output.append(
            f"Skills Match: {job['match_percentage']}%"
        )

        if job["matched_skills"]:
            output.append(
                "Matching Skills: "
                + ", ".join(job["matched_skills"])
            )

        if job["missing_skills"]:
            output.append(
                "Skills Not Found in Listing: "
                + ", ".join(job["missing_skills"])
            )

        output.append(
            "\nDescription: "
            + job["description"][:600]
        )

        output.append(
            "\nRequired Skills: "
            + job["skills"][:400]
        )

        output.append(
            "\nResponsibilities: "
            + job["responsibilities"][:400]
        )

        output.append(
            "\nApply Here: "
            + (
                job["apply_link"]
                if job["apply_link"]
                else "Application link not provided"
            )
        )

    output.append(
        "\n\nNote: Skill match is based on keywords "
        "in your profile and the available job listing. "
        "It is not a guarantee of eligibility."
    )

    return "\n".join(output)


# ---------------- HEALTH CHECK ----------------

@app.get("/")
def home():

    return {
        "message": "Live Job Search Agent is running!"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "job_api_key_configured": bool(INDIAN_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY)
    }


# ---------------- LIVE JOB API TEST ----------------

@app.get("/live-jobs")
def live_jobs():

    jobs = fetch_live_jobs(50)

    return {
        "total": len(jobs),
        "jobs": jobs
    }


# ---------------- AGENT INVOKE ----------------

class AgentRequest(BaseModel):

    input: str


@app.post("/agent/invoke")
def invoke_agent(request: AgentRequest):

    user_input = request.input.strip()

    if not user_input:
        raise HTTPException(
            status_code=400,
            detail="Please enter a job role and location."
        )

    # Extract job role

    role_match = re.search(
        r"(?:find|search(?: for)?)\s+(.+?)\s+jobs?\s+in\s+",
        user_input,
        re.IGNORECASE
    )

    if role_match:
        role = role_match.group(1).strip()

    else:
        role = ""

    # Extract location

    location_match = re.search(
        r"\bjobs?\s+in\s+(.+?)"
        r"(?=\s+for\s+freshers|\s+for\s+beginners|"
        r"\s+my skills|\s+skills are|[.!?\n]|$)",
        user_input,
        re.IGNORECASE
    )

    location = (
        location_match.group(1).strip()
        if location_match
        else ""
    )

    # Extract user skills

    skills_match = re.search(
        r"(?:my skills are|my skills include|skills are)\s+(.+)",
        user_input,
        re.IGNORECASE
    )

    skills = ""

    if skills_match:

        skills = skills_match.group(1).strip()

        skills = re.split(
            r"\.\s|\.?$",
            skills
        )[0].strip()

    # Search the actual job API

    results = search_jobs(
        role=role,
        location=location,
        skills=skills,
        limit=50
    )

    answer = format_results(
        results,
        role,
        location,
        skills
    )

    return {
        "output": answer,
        "total_jobs": len(results),
        "role": role,
        "location": location,
        "skills": skills
    }


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
