import os
import sys
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from agno.agent import Agent
from agno.models.google.gemini import Gemini
from agno.models.groq.groq import Groq

# Make sure we can import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.appconfig_cloud import GOOGLE_API_KEY, GROQ_API_KEY  # type: ignore


# -------------------------------------------------------------------
# Logging & Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "config" / "logs"
DATA_DIR = PROJECT_ROOT / "data"
STATE_FILE = DATA_DIR / "app_state.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "config.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_health_fitness_v2")


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------
@dataclass
class HealthProfile:
    age: int
    height_cm: float
    weight_kg: float
    sex: str
    activity_level: str
    fitness_goal: str
    dietary_preference: str
    health_conditions: List[str]
    time_available_min: int

    @property
    def height_m(self) -> float:
        return self.height_cm / 100.0

    @property
    def bmi(self) -> float:
        if self.height_m <= 0:
            return 0.0
        return round(self.weight_kg / (self.height_m ** 2), 1)

    @property
    def bmi_category(self) -> str:
        if self.bmi == 0:
            return "Unknown"
        if self.bmi < 18.5:
            return "Underweight"
        if self.bmi < 25:
            return "Normal weight"
        if self.bmi < 30:
            return "Overweight"
        return "Obese"

    @property
    def calorie_target(self) -> int:
        """
        Very rough daily calorie estimate using Mifflin-St Jeor + activity factor.
        This is just to guide the LLM; user should confirm with a professional.
        """
        if self.sex.lower().startswith("m"):
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161

        activity_factor_map = {
            "Sedentary": 1.2,
            "Lightly Active": 1.375,
            "Moderately Active": 1.55,
            "Very Active": 1.725,
            "Extremely Active": 1.9,
        }
        factor = activity_factor_map.get(self.activity_level, 1.2)
        maintenance = bmr * factor

        goal = self.fitness_goal
        if "Lose" in goal:
            maintenance -= 300
        elif "Gain" in goal:
            maintenance += 250

        return int(round(maintenance))


# -------------------------------------------------------------------
# Simple persistence (per app instance)
# -------------------------------------------------------------------
def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"profiles": {}, "plans": {}, "chat": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return {"profiles": {}, "plans": {}, "chat": {}}


def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


# -------------------------------------------------------------------
# Model helpers
# -------------------------------------------------------------------
GEMINI_MODEL_NAME = "gemini-2.5-pro-exp-03-25"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"


def get_model(model_choice: str):
    try:
        if model_choice == "Gemini 2.5 Pro":
            return Gemini(id=GEMINI_MODEL_NAME, api_key=GOOGLE_API_KEY)
        elif model_choice == "Llama 3.3 70B":
            return Groq(id=GROQ_MODEL_NAME, api_key=GROQ_API_KEY)
        else:
            st.error("Invalid model selection.")
            return None
    except Exception as e:
        logger.error(f"Error initializing model: {e}", exc_info=True)
        st.error(f"Error initializing model: {e}")
        return None


# -------------------------------------------------------------------
# Plan generation logic
# -------------------------------------------------------------------
def build_profile_prompt(profile: HealthProfile) -> str:
    return f"""
User Health Profile:
- Age: {profile.age}
- Sex: {profile.sex}
- Height: {profile.height_cm} cm
- Weight: {profile.weight_kg} kg
- BMI: {profile.bmi} ({profile.bmi_category})
- Activity level: {profile.activity_level}
- Fitness goal: {profile.fitness_goal}
- Dietary preference: {profile.dietary_preference}
- Health conditions: {", ".join(profile.health_conditions)}
- Time available per day: {profile.time_available_min} minutes
- Estimated daily calorie target: ~{profile.calorie_target} kcal

IMPORTANT:
- Assume the user has no acute medical emergency.
- Be conservative and safe. When in doubt, recommend consulting a healthcare professional.
"""


def generate_diet_plan(profile: HealthProfile, model) -> str:
    agent = Agent(
        name="Diet Planner",
        role="Create safe, realistic daily diet plans.",
        model=model,
        markdown=True,
        instructions=[
            "Use the provided profile.",
            "Respect dietary preferences and health conditions.",
            "Target roughly the given calorie target.",
            "Respond in clear Markdown with the following sections:",
            "1. Summary (2–3 sentences)",
            "2. Daily Calorie & Macro Overview (rough, not exact science)",
            "3. Meal Plan (Breakfast, Snack, Lunch, Snack, Dinner)",
            "4. Hydration & General Tips",
            "Avoid promising medical cures. Suggest seeing a professional for major issues.",
        ],
    )

    prompt = build_profile_prompt(profile)
    response = agent.run(prompt)
    return getattr(response, "content", str(response))


def generate_workout_plan(profile: HealthProfile, model) -> str:
    agent = Agent(
        name="Workout Coach",
        role="Design beginner-friendly fitness routines.",
        model=model,
        markdown=True,
        instructions=[
            "Use the provided profile.",
            "Respect joint pain or serious conditions by avoiding high-impact moves.",
            "Adapt the volume to the available time per day.",
            "Respond in clear Markdown with the following sections:",
            "1. Overview & Goals",
            "2. Weekly Structure (e.g., Day 1–Day 5)",
            "3. Warm-up & Cool-down",
            "4. Safety Notes",
            "Use bodyweight or simple equipment unless profile clearly suggests gym access.",
        ],
    )

    prompt = build_profile_prompt(profile)
    response = agent.run(prompt)
    return getattr(response, "content", str(response))


def answer_chat_question(
    profile: HealthProfile,
    diet_plan: str,
    workout_plan: str,
    history: List[Dict[str, str]],
    question: str,
    model,
) -> str:
    agent = Agent(
        name="Health & Fitness Assistant",
        role="Answer questions about diet and training safely.",
        model=model,
        markdown=True,
        instructions=[
            "Use the user's profile and their generated diet & workout plan.",
            "Use previous chat turns only for context, not strict memory.",
            "If the question is clearly medical, encourage consulting a professional.",
            "Be supportive, specific, and practical.",
        ],
    )

    # Build short text transcript of last few messages
    last_turns = history[-8:]
    history_lines = []
    for m in last_turns:
        prefix = "User" if m["role"] == "user" else "Assistant"
        history_lines.append(f"{prefix}: {m['content']}")
    history_text = "\n".join(history_lines)

    prompt = f"""
{build_profile_prompt(profile)}

Existing Diet Plan:
{diet_plan}

Existing Workout Plan:
{workout_plan}

Recent Conversation:
{history_text}

User Question:
{question}
"""
    response = agent.run(prompt)
    return getattr(response, "content", str(response))


# -------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------
def init_session_state():
    if "profile_id" not in st.session_state:
        st.session_state.profile_id = None
    if "plans" not in st.session_state:
        st.session_state.plans = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []


def render_css():
    st.markdown(
        """
<style>
    * {
        font-family: "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4E5FFF;
        margin-bottom: 0.5rem;
    }
    .plan-card {
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        background-color: #ffffff;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        border-left: 4px solid #4ECDC4;
    }
    .diet-card {
        border-left-color: #FF6B6B;
    }
    .meta-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background-color: #F3F4FF;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .chat-box {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.4rem;
    }
    .chat-user {
        background-color: #E0E7FF;
    }
    .chat-assistant {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_profile_summary(profile: HealthProfile):
    st.markdown("### 🧍 Your Profile Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Age", f"{profile.age} yrs")
        st.metric("Sex", profile.sex)
    with col2:
        st.metric("Height", f"{profile.height_cm} cm")
        st.metric("Weight", f"{profile.weight_kg} kg")
    with col3:
        st.metric("BMI", f"{profile.bmi} ({profile.bmi_category})")
        st.metric("Calorie Target", f"~{profile.calorie_target} kcal")

    st.markdown(
        f"""
<span class="meta-pill">Activity: {profile.activity_level}</span>
<span class="meta-pill">Goal: {profile.fitness_goal}</span>
<span class="meta-pill">Diet: {profile.dietary_preference}</span>
""",
        unsafe_allow_html=True,
    )

    if profile.health_conditions and "None" not in profile.health_conditions:
        st.markdown(
            f"<span class='meta-pill'>Health: {', '.join(profile.health_conditions)}</span>",
            unsafe_allow_html=True,
        )


def render_plan_cards(diet_markdown: str, workout_markdown: str):
    st.markdown("### 📊 Your Personalized Plans")
    with st.container():
        st.markdown('<div class="plan-card diet-card">', unsafe_allow_html=True)
        st.markdown("#### 🍽️ Dietary Plan", unsafe_allow_html=True)
        st.markdown(diet_markdown, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="plan-card">', unsafe_allow_html=True)
        st.markdown("#### 💪 Fitness Plan", unsafe_allow_html=True)
        st.markdown(workout_markdown, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_chat():
    st.markdown("### 💬 Ask Your Assistant")

    # Display messages
    for m in st.session_state.chat_history:
        css_class = "chat-user" if m["role"] == "user" else "chat-assistant"
        with st.container():
            st.markdown(
                f"<div class='chat-box {css_class}'>{m['content']}</div>",
                unsafe_allow_html=True,
            )

    # Chat input
    user_msg = st.chat_input("Ask about your plan, adjustments, or next steps...")
    return user_msg


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Health & Fitness Planner v2",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_css()
    init_session_state()
    state = load_state()

    st.markdown("<h1 class='main-header'>🏋️ AI Health & Fitness Planner</h1>", unsafe_allow_html=True)
    st.caption("Personalized diet and workout guidance powered by Gemini / Llama — rebuilt & optimized.")

    # Sidebar: model & profile management
    with st.sidebar:
        st.subheader("⚙️ Model Settings")
        model_choice = st.radio(
            "AI Model",
            ["Gemini 2.5 Pro", "Llama 3.3 70B"],
            index=0,
        )

        st.subheader("👤 Profile")
        profiles = state.get("profiles", {})
        profile_names = list(profiles.keys())

        selected_profile_name = None
        if profile_names:
            selected_profile_name = st.selectbox(
                "Existing profiles",
                options=["(Create new)"] + profile_names,
                index=0 if st.session_state.profile_id is None else
                       profile_names.index(st.session_state.profile_id) + 1
                       if st.session_state.profile_id in profile_names else 0
            )

        if selected_profile_name and selected_profile_name != "(Create new)":
            st.session_state.profile_id = selected_profile_name
        elif selected_profile_name == "(Create new)":
            st.session_state.profile_id = None

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            if st.session_state.profile_id:
                state.setdefault("chat", {})[st.session_state.profile_id] = []
                save_state(state)
            st.experimental_rerun()

    model = get_model(model_choice)
    if model is None:
        return

    # Layout: left = form/plans, right = chat/exports
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("<h2 class='sub-header'>1️⃣ Build or Load Your Profile</h2>", unsafe_allow_html=True)

        # If profile exists, load it
        current_profile: HealthProfile | None = None
        if st.session_state.profile_id and st.session_state.profile_id in state.get("profiles", {}):
            raw = state["profiles"][st.session_state.profile_id]
            current_profile = HealthProfile(**raw)

        with st.form("profile_form"):
            if current_profile:
                age = st.number_input("Age", 10, 100, value=current_profile.age)
                height_cm = st.number_input("Height (cm)", 100.0, 250.0, value=float(current_profile.height_cm))
                weight_kg = st.number_input("Weight (kg)", 20.0, 300.0, value=float(current_profile.weight_kg))
                sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(current_profile.sex))
                activity_level = st.selectbox(
                    "Activity Level",
                    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                    index=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"].index(current_profile.activity_level),
                )
                fitness_goal = st.selectbox(
                    "Fitness Goal",
                    ["Lose Weight", "Gain Muscle", "Endurance", "Stay Fit", "Strength Training"],
                    index=["Lose Weight", "Gain Muscle", "Endurance", "Stay Fit", "Strength Training"].index(current_profile.fitness_goal),
                )
                dietary_preference = st.selectbox(
                    "Dietary Preference",
                    ["No Restrictions", "Vegetarian", "Vegan", "Keto", "Gluten Free", "Low Carb", "Dairy Free"],
                    index=["No Restrictions", "Vegetarian", "Vegan", "Keto", "Gluten Free", "Low Carb", "Dairy Free"].index(current_profile.dietary_preference),
                )
                health_conditions = st.multiselect(
                    "Health Conditions",
                    options=["None", "Diabetes", "Hypertension", "Heart Disease", "Joint Pain", "Obesity", "Other"],
                    default=current_profile.health_conditions or ["None"],
                )
                time_available = st.slider(
                    "Time Available for Exercise (minutes/day)", 15, 120, value=current_profile.time_available_min, step=15
                )
                profile_name = st.text_input(
                    "Profile Name (for saving/loading)",
                    value=st.session_state.profile_id,
                    help="e.g., 'Joshua_Cut_Phase' or 'Bulk_Program_Summer'",
                )
            else:
                age = st.number_input("Age", 10, 100, value=25)
                height_cm = st.number_input("Height (cm)", 100.0, 250.0, value=175.0)
                weight_kg = st.number_input("Weight (kg)", 20.0, 300.0, value=70.0)
                sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                activity_level = st.selectbox(
                    "Activity Level",
                    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                )
                fitness_goal = st.selectbox(
                    "Fitness Goal",
                    ["Lose Weight", "Gain Muscle", "Endurance", "Stay Fit", "Strength Training"],
                )
                dietary_preference = st.selectbox(
                    "Dietary Preference",
                    ["No Restrictions", "Vegetarian", "Vegan", "Keto", "Gluten Free", "Low Carb", "Dairy Free"],
                )
                health_conditions = st.multiselect(
                    "Health Conditions",
                    options=["None", "Diabetes", "Hypertension", "Heart Disease", "Joint Pain", "Obesity", "Other"],
                    default=["None"],
                )
                time_available = st.slider(
                    "Time Available for Exercise (minutes/day)", 15, 120, value=45, step=15
                )
                profile_name = st.text_input(
                    "Profile Name (for saving/loading)",
                    value="My_Profile_1",
                )

            submitted = st.form_submit_button("✅ Save Profile & Generate Plans")

        if submitted:
            profile = HealthProfile(
                age=int(age),
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                sex=str(sex),
                activity_level=str(activity_level),
                fitness_goal=str(fitness_goal),
                dietary_preference=str(dietary_preference),
                health_conditions=list(health_conditions),
                time_available_min=int(time_available),
            )

            # Save profile
            state.setdefault("profiles", {})[profile_name] = asdict(profile)
            st.session_state.profile_id = profile_name

            with st.spinner("Generating smarter diet & workout plans..."):
                diet_plan = generate_diet_plan(profile, model)
                workout_plan = generate_workout_plan(profile, model)

            state.setdefault("plans", {})[profile_name] = {
                "diet": diet_plan,
                "workout": workout_plan,
            }

            # Reset chat for this profile
            state.setdefault("chat", {})[profile_name] = []
            st.session_state.chat_history = []

            save_state(state)
            st.session_state.plans = state["plans"][profile_name]
            st.success("Profile saved & plans generated!")

        # Show summary & plans if available
        if st.session_state.profile_id and st.session_state.profile_id in state.get("profiles", {}):
            current_profile = HealthProfile(**state["profiles"][st.session_state.profile_id])
            render_profile_summary(current_profile)

            existing_plans = state.get("plans", {}).get(st.session_state.profile_id)
            if existing_plans:
                st.session_state.plans = existing_plans
                render_plan_cards(existing_plans["diet"], existing_plans["workout"])
            else:
                st.info("No plans found yet for this profile. Save the form to generate.")

    with col_right:
        st.markdown("<h2 class='sub-header'>2️⃣ Chat & Export</h2>", unsafe_allow_html=True)

        # Load chat for current profile
        if st.session_state.profile_id:
            profile_chat = state.setdefault("chat", {}).get(st.session_state.profile_id, [])
            if not st.session_state.chat_history and profile_chat:
                st.session_state.chat_history = profile_chat

        user_msg = render_chat()

        # Handle user message
        if user_msg and st.session_state.profile_id and st.session_state.plans:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            # Keep last 30 messages
            st.session_state.chat_history = st.session_state.chat_history[-30:]

            profile = HealthProfile(**state["profiles"][st.session_state.profile_id])
            diet = st.session_state.plans["diet"]
            workout = st.session_state.plans["workout"]

            with st.spinner("Thinking about your question..."):
                answer = answer_chat_question(
                    profile,
                    diet,
                    workout,
                    st.session_state.chat_history,
                    user_msg,
                    model,
                )

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.chat_history = st.session_state.chat_history[-30:]

            # Persist chat
            state.setdefault("chat", {})[st.session_state.profile_id] = st.session_state.chat_history
            save_state(state)
            st.experimental_rerun()

        st.markdown("---")
        if st.session_state.profile_id and st.session_state.plans:
            if st.download_button(
                "📥 Download Full Plan (Markdown)",
                data=(
                    f"# AI Health & Fitness Plan\n\n"
                    f"Profile: {st.session_state.profile_id}\n\n"
                    f"## Dietary Plan\n\n{st.session_state.plans['diet']}\n\n"
                    f"## Fitness Plan\n\n{st.session_state.plans['workout']}\n"
                ),
                file_name=f"{st.session_state.profile_id}_plan.md",
                mime="text/markdown",
            ):
                logger.info("User downloaded plan markdown.")


if __name__ == "__main__":
    main()
