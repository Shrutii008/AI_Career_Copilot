"""
gemini_helper.py
-----------------
All Gemini API integration for the AI Resume Analyzer lives here.

- Reads GEMINI_API_KEY from a .env file (python-dotenv).
- Uses the current Google Gen AI SDK (`google-genai`, imported as `from google import genai`).
- Every public function fails soft: if the key is missing/invalid or the
  API call errors out, it raises a RuntimeError with a clear message that
  app.py catches and turns into a friendly on-screen warning + fallback,
  instead of crashing the whole app.
"""

import os
import json

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load variables from a .env file in the project root (GEMINI_API_KEY=...)
load_dotenv()

# Pinned to a fixed, currently-supported model version (not an
# auto-updating alias) so behavior stays consistent across Google's
# model updates. Update this single constant if Google deprecates it,
# or if you hit free-tier quota routing issues with a specific model
# (as of July 2026, gemini-3.5-flash's free tier is affected by a
# known Google-side bug that misroutes it against the old
# gemini-2.5-flash quota bucket — gemini-2.5-flash-lite is currently
# unaffected and a good fallback if that recurs).
MODEL_NAME = "gemini-3.5-flash"

# --- Diagnostic guard -------------------------------------------------
# Proves, at runtime, exactly which file was loaded and which model is
# active. Prints on every fresh process start (not on Streamlit
# reruns within an already-running process, since the module is only
# imported once per process). Safe to delete once you've confirmed
# things are working.
print(f"[gemini_helper] loaded from: {os.path.abspath(__file__)} | MODEL_NAME = {MODEL_NAME}")


def _init_client():
    """Create the Gemini client once. Returns (client, error_message)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        return None, (
            "GEMINI_API_KEY not found. Add it to a .env file in your project "
            "root, e.g.:\n\nGEMINI_API_KEY=your_key_here"
        )
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as exc:  # bad key format, network/init issues, etc.
        return None, f"Failed to initialize Gemini client: {exc}"


_client, _client_error = _init_client()


def gemini_available() -> bool:
    """True if a Gemini client was successfully created from the .env key."""
    return _client is not None


def gemini_status_message() -> str:
    """Human-readable status for the sidebar / banner."""
    if _client is not None:
        return "✅ Gemini API connected"
    return f"⚠️ Gemini API not available — {_client_error}"


def _call_gemini(prompt: str, json_output: bool = False) -> str:
    """
    Low-level call to Gemini. Raises RuntimeError on any failure
    (missing key, invalid key, network error, empty response, etc.)
    so callers can catch a single exception type.
    """
    if _client is None:
        raise RuntimeError(_client_error)

    config = None
    if json_output:
        config = types.GenerateContentConfig(response_mime_type="application/json")

    try:
        response = _client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        # Covers invalid/revoked API keys (401/403), rate limits, network errors, etc.
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    if not response or not getattr(response, "text", None):
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()


def _parse_json_list(raw_text: str):
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise RuntimeError("Gemini JSON response was not a list as expected.")
    return [str(item).strip() for item in data if str(item).strip()]


# ------------------------------------------------------------------
# COMBINED ANALYSIS — ONE Gemini call returns everything:
#   1. Resume Summary
#   2. ATS Improvement Suggestions
#   3. Resume Improvement Recommendations
#   4. Interview Questions (technical / hr / project)
#   5. Missing Skill Suggestions
#   6. Tailored Resume (only meaningful when a JD is provided)
#   7. Cover Letter
#   8. Grammar Quality + Readability Suggestions
#   9. Project Quality Review
#   10. ATS Keyword Optimizer (natural keyword integration suggestions)
# This is the ONLY Gemini call made per resume analysis.
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_full_analysis(
    resume_text: str,
    jd_text: str,
    found_skills: list,
    ats_score: float,
    checklist: dict,
    job_role: str,
    candidate_name: str,
    readability_stats: dict = None,
    passive_voice_examples: list = None,
    repeated_phrases: list = None,
) -> dict:
    missing_checklist_items = [k for k, present in checklist.items() if not present]
    jd_block = f'Job description:\n"""{jd_text[:4000]}"""' if jd_text else "No job description was provided."
    name_for_letter = candidate_name if candidate_name and candidate_name != "Not Found" else "the candidate"

    readability_stats = readability_stats or {}
    passive_voice_examples = passive_voice_examples or []
    repeated_phrases = repeated_phrases or []

    readability_context = f"""Locally computed readability metrics (for context — do not recompute, just use these):
- Average sentence length: {readability_stats.get('avg_sentence_length', 'n/a')} words/sentence
- Flesch Reading Ease score: {readability_stats.get('readability_score', 'n/a')}/100
- Passive-voice sentence examples detected: {"; ".join(passive_voice_examples) if passive_voice_examples else "none detected"}
- Repeated 3-word phrases detected: {", ".join(p for p, c in repeated_phrases) if repeated_phrases else "none detected"}"""

    missing_skills_instructions = (
        """7. "missing_skills": Compare the resume against the job
   description above and list important skills the job description
   implies or requires that are absent from the resume. Return as a
   list of 3-8 short skill names (no explanations)."""
        if jd_text else
        """7. "missing_skills": No job description was provided. Suggest
   3-8 important, in-demand skills for this candidate's apparent
   field/role that are missing from their resume, as a list of short
   skill names (no explanations)."""
    )

    tailored_resume_instructions = (
        f"""6. "tailored_resume": Rewrite the ORIGINAL RESUME so it is fully
   tailored to the job description above. Preserve all factual
   information exactly (no invented employers, titles, dates, degrees,
   certifications, or metrics). Improve ATS keyword alignment, rewrite
   weak bullet points with strong action verbs, retarget the
   professional summary, reorder skills so the most relevant appear
   first, and sharpen project descriptions — while preserving the
   original section order and formatting style. Return the full
   rewritten resume text as a single string."""
        if jd_text else
        """6. "tailored_resume": No job description was provided, so set
   this to an empty string "".""" 
    )

    ats_keyword_optimizer_instructions = (
        """10. "ats_keyword_optimizer": Compare the job description
   against the resume and suggest 3-8 additional keywords/phrases from
   the JD that should naturally be worked into the resume. For each,
   return ONE string that names the keyword AND explains where/how to
   add it naturally based on the candidate's actual experience — do
   NOT suggest keyword stuffing or adding terms the candidate has no
   real basis for claiming. Example format: "Add 'Kubernetes' to your
   Skills section — your Docker experience suggests likely exposure to
   container orchestration." Return as a list of strings."""
        if jd_text else
        """10. "ats_keyword_optimizer": No job description was provided,
   so set this to an empty list []."""
    )

    prompt = f"""You are an expert technical recruiter, ATS optimization
specialist, interview coach, career advisor, professional writing
coach, and technical project reviewer, all at once. Perform ALL TEN
of the following tasks for the candidate below, using their resume
(and job description, if given) as context. Do not skip any task.

Detected skills: {", ".join(found_skills) if found_skills else "none clearly detected"}
Computed ATS score: {ats_score}%
Missing resume checklist items: {", ".join(missing_checklist_items) if missing_checklist_items else "none"}
Target job role: {job_role}
{jd_block}

{readability_context}

Original resume:
\"\"\"{resume_text[:8000]}\"\"\"

TASKS:

1. "summary": Write a concise, professional 3-4 sentence third-person
   summary of this candidate, suitable to show directly to them.

2. "ats_suggestions": Give 3-6 short, specific, actionable suggestions
   to improve this resume's ATS score (sections to add, keywords,
   formatting fixes) — return as a list of strings.

3. "resume_recommendations": Give 4-6 general resume-writing
   improvement recommendations tailored to THIS resume's actual
   content (wording, structure, quantified achievements, action verbs,
   formatting, clarity) — return as a list of strings.

4. "interview_questions": Generate personalized interview questions as
   an object with exactly these keys: "technical" (3-5 questions),
   "hr" (3-5 questions), "project" (3-5 questions) — each a list of
   strings.

5. "cover_letter": Write a tailored, concise (250-350 word) cover
   letter for {name_for_letter} applying for the "{job_role}" position,
   with a greeting and sign-off, as a single string.

{tailored_resume_instructions}

{missing_skills_instructions}

8. "grammar_quality" and "readability_suggestions": Using the locally
   computed readability metrics above as grounding, write:
   - "grammar_quality": a 2-3 sentence qualitative assessment of this
     resume's grammar and writing quality (tense consistency, clarity,
     wording issues you notice by reading the resume text itself).
   - "readability_suggestions": a list of 3-5 specific suggestions to
     improve readability, referencing the sentence length, passive
     voice, and repeated phrases data above where relevant (e.g. "Your
     average sentence length is X words — consider breaking up longer
     sentences" or naming a specific repeated phrase to vary).

9. "project_quality": Review the Projects section of the resume (if
   present) and return an object with exactly these keys, each a list
   of strings (empty list if not applicable):
   - "better_titles": suggested stronger, more specific project titles
   - "better_descriptions": suggestions for improving weak project
     descriptions (reference the actual project if possible)
   - "missing_technologies": technologies/tools that seem like they
     should be mentioned but aren't (based on what the project
     appears to involve)
   - "missing_impact_metrics": suggestions for what kind of
     quantifiable impact/metric could be added to each project
   - "github_presentation_tips": suggestions for presenting the
     projects better on GitHub (READMEs, demos, pinned repos, etc.)
   If the resume has no discernible Projects section, return empty
   lists for all five keys.

{ats_keyword_optimizer_instructions}

Return ONLY valid JSON (no markdown fences, no commentary) with
exactly this shape:
{{
  "summary": "...",
  "ats_suggestions": ["...", "..."],
  "resume_recommendations": ["...", "..."],
  "interview_questions": {{"technical": ["..."], "hr": ["..."], "project": ["..."]}},
  "tailored_resume": "...",
  "cover_letter": "...",
  "missing_skills": ["...", "..."],
  "grammar_quality": "...",
  "readability_suggestions": ["...", "..."],
  "project_quality": {{
    "better_titles": ["..."],
    "better_descriptions": ["..."],
    "missing_technologies": ["..."],
    "missing_impact_metrics": ["..."],
    "github_presentation_tips": ["..."]
  }},
  "ats_keyword_optimizer": ["...", "..."]
}}"""

    raw = _call_gemini(prompt, json_output=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc

    required_keys = (
        "summary", "ats_suggestions", "resume_recommendations",
        "interview_questions", "tailored_resume", "cover_letter",
        "missing_skills", "grammar_quality", "readability_suggestions",
        "project_quality", "ats_keyword_optimizer",
    )
    for key in required_keys:
        if key not in data:
            raise RuntimeError(f"Gemini response missing '{key}'.")

    iq = data["interview_questions"]
    for key in ("technical", "hr", "project"):
        if key not in iq or not isinstance(iq[key], list) or not iq[key]:
            raise RuntimeError(f"Gemini response missing interview_questions.'{key}'.")

    pq = data["project_quality"]
    pq_keys = (
        "better_titles", "better_descriptions", "missing_technologies",
        "missing_impact_metrics", "github_presentation_tips",
    )
    for key in pq_keys:
        if key not in pq or not isinstance(pq[key], list):
            raise RuntimeError(f"Gemini response missing project_quality.'{key}'.")

    return {
        "summary": str(data["summary"]).strip(),
        "ats_suggestions": [str(s).strip() for s in data["ats_suggestions"] if str(s).strip()],
        "resume_recommendations": [str(s).strip() for s in data["resume_recommendations"] if str(s).strip()],
        "interview_questions": {
            "technical": [str(q).strip() for q in iq["technical"]],
            "hr": [str(q).strip() for q in iq["hr"]],
            "project": [str(q).strip() for q in iq["project"]],
        },
        "tailored_resume": str(data["tailored_resume"]).strip(),
        "cover_letter": str(data["cover_letter"]).strip(),
        "missing_skills": [str(s).strip() for s in data["missing_skills"] if str(s).strip()],
        "grammar_quality": str(data["grammar_quality"]).strip(),
        "readability_suggestions": [str(s).strip() for s in data["readability_suggestions"] if str(s).strip()],
        "project_quality": {
            key: [str(s).strip() for s in pq[key] if str(s).strip()] for key in pq_keys
        },
        "ats_keyword_optimizer": [str(s).strip() for s in data["ats_keyword_optimizer"] if str(s).strip()],
    }


# ------------------------------------------------------------------
# REWRITE INDIVIDUAL BULLET POINT
# On-demand, per-bullet call (only fires when the user clicks
# "Rewrite" on a specific bullet) — not part of the combined analysis,
# since bullets are numerous and rewriting all of them automatically
# on every upload would multiply API usage unnecessarily.
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_bullet_rewrite(bullet_text: str, found_skills: list = None) -> str:
    skills_context = ", ".join(found_skills) if found_skills else "none specified"

    prompt = f"""You are an expert resume writer. Rewrite the single resume
bullet point below into a stronger, achievement-oriented version.

Guidelines:
- Start with a strong action verb.
- Make it specific and results-oriented. If the original implies a
  measurable outcome, make that outcome explicit; do NOT invent
  specific numbers, employers, tools, or facts that aren't implied by
  the original.
- Keep it to a single sentence, roughly 15-30 words.
- Candidate's known skills (for context only, don't force all of them
  in): {skills_context}

Original bullet point:
"{bullet_text}"

Return ONLY the rewritten bullet point text — no quotes, no bullet
symbol, no explanation, no "Before/After" labels."""

    result = _call_gemini(prompt)
    return result.strip().strip('"').strip("'")