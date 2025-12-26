🏋️ AI Health & Fitness Planner – By Joshua

An intelligent web application that provides personalized health and fitness plans using advanced AI models.
This upgraded version is redesigned, optimized, and enhanced by Joshua, delivering smarter plans, better UX, and improved reliability.

🌟 Key Features
🔹 Personalized Health Profiles

Create and manage rich user profiles including:

Age, weight, height, sex

Activity level

Fitness goals

Dietary preferences

Health conditions

Available workout time

🤖 AI-Powered Health Plans

Automatically generate:

Custom Dietary Plans

Personalized Fitness & Workout Plans

Driven by:

Google Gemini 2.5 Pro

Llama 3.3 70B (via Groq)

💬 Smart Chat Assistant

Ask questions and receive:

Real-time responses

Context-aware guidance

Fitness & nutrition explanations

Safe and helpful recommendations

🧠 Persistent Experience

Save & load profiles

Retain AI-generated plans

Maintain chat history

Smooth, optimized performance

Modern & beautiful UI

🛠️ Tech Stack
Component	Technology
Frontend	Streamlit
AI Models	Gemini 2.5 Pro / Groq Llama 3.3 70B
Framework	Agno AI
State Management	Streamlit Session State + Local Persistence
Styling	Modern Custom CSS
📋 Prerequisites

Python 3.8+

Google Gemini API Key

Groq API Key

Project dependencies (requirements.txt)

🚀 Installation
1️⃣ Clone the Repository
git clone https://github.com/olatowojujoshua/ai-health-and-fitness-planner.git
cd ai-health-and-fitness-planner


2️⃣ Create & Activate a Virtual Environment
python -m venv venv
.\venv\Scripts\activate     # Windows
# or
source venv/bin/activate   # Mac / Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure API Keys

Go to:

config/appconfig_cloud.py


Add your keys:

GOOGLE_API_KEY = "your-gemini-api-key"
GROQ_API_KEY = "your-groq-api-key"


(Alternatively, use .streamlit/secrets.toml depending on setup)

💻 Usage

Start the app:

streamlit run src/main.py


Then open your browser at:

http://localhost:8501

📱 Detailed Functionalities
🧍 Health Profile Creation

✔️ Age, Weight, Height, Sex
✔️ Activity Level
✔️ Fitness Goals
✔️ Diet Preferences
✔️ Health Conditions
✔️ Available Exercise Time

🥗 AI-Generated Dietary Plan

Personalized meal suggestions

Nutrition guidance

Key health considerations

Hydration reminders

💪 AI-Generated Fitness Plan

Custom workout programs

Warm-up and cool-down guidance

Progress and consistency tips

Safety recommendations

💬 Chat Assistant

Real-time interaction

Streaming chatbot responses

Understands plans and user context

Persistent conversation memory

🔒 Security

✔️ Secure API key handling
✔️ Safe user data handling
✔️ Session-based storage

🤝 Contributing

Contributions are welcome!
Feel free to fork, improve, and submit a pull request.

🙏 Acknowledgment

Originally inspired by collaborative development.
Refactored, enhanced, and optimized by Joshua.

Powered by:

Google Gemini

Llama via Groq

Streamlit + Agno

📞 Support

For assistance or collaboration inquiries, please contact olatowoju@gmail.com
