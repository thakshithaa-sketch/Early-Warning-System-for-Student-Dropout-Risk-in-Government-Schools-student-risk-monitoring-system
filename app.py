import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
import os
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Student Risk Monitoring System",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>
body{
background: linear-gradient(120deg,#eef7ff,#f5fbff);
}
.high-risk{color:#ff4b4b;font-weight:bold;}
.medium-risk{color:#ffa500;font-weight:bold;}
.low-risk{color:#28a745;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# FILE PATH SETUP
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
students_file = os.path.join(BASE_DIR, "students.csv")
intervention_file = os.path.join(BASE_DIR, "intervention.csv")
HIGH_RISK_THRESHOLD = 50

# ---------------------------------
# CREATE FILES IF MISSING
# ---------------------------------
if not os.path.exists(students_file):
    df = pd.DataFrame(columns=[
        "reg_no","name","attendance_percentage","exam_marks",
        "mid_day_meal_participation","distance_home_school","sibling_dropout"
    ])
    df.to_csv(students_file, index=False)

if not os.path.exists(intervention_file):
    pd.DataFrame(columns=[
        "reg_no","date_of_counselling","intervention_taken",
        "before_attendance","after_attendance","before_exam_marks","after_exam_marks"
    ]).to_csv(intervention_file, index=False)

df = pd.read_csv(students_file)

# ---------------------------------
# GOVERNMENT POLICIES
# ---------------------------------
gov_policies = {
    "Low attendance": [
        "National Education Policy 2020: Mandatory attendance for all students",
        "Right to Education Act: Schools must track and ensure 75% minimum attendance"
    ],
    "Low exam marks": [
        "Samagra Shiksha Scheme: Extra tutoring & remedial classes",
        "Mid-day meal incentive: Rewards for academic improvement"
    ],
    "Irregular mid-day meal": [
        "Mid-Day Meal Scheme: Provides free meals to encourage attendance"
    ],
    "Long distance from school": [
        "Transport Allowance Scheme: Financial support for travel",
        "Online Education Policy: Access to digital classes"
    ],
    "Sibling dropout history": [
        "Beti Bachao Beti Padhao: Counseling and support programs",
        "RTE Act: Priority support for at-risk students"
    ]
}

# ---------------------------------
# RISK ENGINE
# ---------------------------------
def calculate_risk(student):
    risk = 0
    reasons = []
    if student["attendance_percentage"] < 65:
        risk += 20
        reasons.append("Low attendance")
    if student["exam_marks"] < 40:
        risk += 20
        reasons.append("Low exam marks")
    if student["mid_day_meal_participation"] == 0:
        risk += 15
        reasons.append("Irregular mid-day meal")
    if student["distance_home_school"] > 5:
        risk += 15
        reasons.append("Long distance from school")
    if student["sibling_dropout"] == 1:
        risk += 20
        reasons.append("Sibling dropout history")
    return risk, reasons

# ---------------------------------
# INTERVENTIONS
# ---------------------------------
def assign_interventions(student, risk):
    interventions = []
    if risk > HIGH_RISK_THRESHOLD:
        if student["attendance_percentage"] < 65:
            interventions.append("Counseling")
            interventions.append("Home visit")
        if student["exam_marks"] < 40:
            interventions.append("Extra tutoring")
        if student["distance_home_school"] > 5:
            interventions.append("Transport support")
    return "; ".join(interventions)

# ---------------------------------
# UPDATE ATTENDANCE
# ---------------------------------
def update_attendance(reg_no, present):
    df_local = pd.read_csv(students_file)
    student = df_local[df_local["reg_no"] == reg_no]
    total_days = 100
    present_days = int(student["attendance_percentage"].values[0] * total_days / 100)
    if present == 1:
        present_days += 1
    attendance_percentage = (present_days / total_days) * 100
    df_local.loc[df_local["reg_no"] == reg_no, "attendance_percentage"] = attendance_percentage
    df_local.to_csv(students_file, index=False)
    return attendance_percentage

# ---------------------------------
# INTERVENTION LOG
# ---------------------------------
def log_intervention(reg_no, action, after_att, after_marks):
    students = pd.read_csv(students_file)
    student = students[students["reg_no"] == reg_no].iloc[0]
    before_att = student["attendance_percentage"]
    before_marks = student["exam_marks"]
    students.loc[students["reg_no"] == reg_no, "attendance_percentage"] = after_att
    students.loc[students["reg_no"] == reg_no, "exam_marks"] = after_marks
    students.to_csv(students_file, index=False)
    inter = pd.read_csv(intervention_file)
    new_row = {
        "reg_no": reg_no,
        "date_of_counselling": date.today(),
        "intervention_taken": action,
        "before_attendance": before_att,
        "after_attendance": after_att,
        "before_exam_marks": before_marks,
        "after_exam_marks": after_marks
    }
    inter = pd.concat([inter, pd.DataFrame([new_row])], ignore_index=True)
    inter.to_csv(intervention_file, index=False)
    return before_att, before_marks

# ---------------------------------
# AI MODEL TRAINING
# ---------------------------------
def train_ai_model(df):
    X = df[["attendance_percentage","exam_marks","mid_day_meal_participation",
            "distance_home_school","sibling_dropout"]]
    y = df["risk_score"] > HIGH_RISK_THRESHOLD
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ---------------------------------
# AI PREDICTION
# ---------------------------------
def predict_student_risk(student):
    features = [[student["attendance_percentage"], student["exam_marks"],
                 student["mid_day_meal_participation"], student["distance_home_school"],
                 student["sibling_dropout"]]]
    prediction = ai_model.predict(features)[0]
    return "🔴 High Dropout Risk" if prediction else "🟢 Low Risk"

# ---------------------------------
# PARENT MESSAGE GENERATOR
# ---------------------------------
def generate_parent_message(student, risk, reasons):

    name = student["name"]
    interventions = student.get("suggested_intervention", "Teacher guidance and extra support")

    strengths = []
    improvements = []

    if student["attendance_percentage"] >= 75:
        strengths.append("good attendance")

    if student["exam_marks"] >= 50:
        strengths.append("academic progress")

    if student["mid_day_meal_participation"] == 1:
        strengths.append("active participation in school programs")

    if student["attendance_percentage"] < 75:
        improvements.append("regular attendance")

    if student["exam_marks"] < 50:
        improvements.append("extra academic practice")

    if student["distance_home_school"] > 5:
        improvements.append("ensuring consistent school travel")

    if risk > 50:
        msg = f"""
Dear Parent,

Your child {name} has great potential and we are committed to helping them grow and succeed.

To support their learning journey, the school is providing additional guidance such as:
{interventions}

With encouragement at home and support from teachers, focusing on {", ".join(improvements)} can help {name} achieve even better results.

Together we can ensure a bright and successful future for your child. Thank you for your continued support.

Warm regards  
School Support Team
"""

    elif 30 <= risk <= 50:
        msg = f"""
Dear Parent,

{name} is progressing well in school and has many opportunities to grow even further.

By continuing to support areas like {", ".join(improvements)}, we can help them build stronger confidence and learning habits.

Your encouragement plays an important role in your child's success.

Warm regards  
School Support Team
"""

    else:
        msg = f"""
Dear Parent,

{name} is doing very well in school. We appreciate the encouragement and support you provide at home.

Maintaining good habits like {", ".join(strengths) if strengths else "consistent learning"} will help them continue achieving great progress.

Thank you for being a wonderful partner in your child's education.

Warm regards  
School Support Team
"""

    return msg
# ---------------------------------
# LOGIN SYSTEM (FIXED)
# ---------------------------------
teachers = {
    "t001": "pass123",
    "t002": "pass456",
    "admin": "admin123",
    "teacher": "123"
}

if "login" not in st.session_state:
    st.session_state.login = False
if "teacher" not in st.session_state:
    st.session_state.teacher = ""

if not st.session_state.login:
    st.title("🎓 Student Risk Monitoring System")
    teacher_id = st.text_input("Teacher ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if teacher_id in teachers and teachers[teacher_id] == password:
            st.session_state.login = True
            st.session_state.teacher = teacher_id
            st.success(f"Welcome {teacher_id}!")
        else:
            st.error("Invalid Login")

    st.stop()  # stop until login is successful

# ---------------------------------
# SIDEBAR & DASHBOARD CONTINUES
# ---------------------------------
st.sidebar.title(f"Welcome, {st.session_state.teacher}")
menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard","Student Data","Risk Analysis","Update Attendance",
     "Intervention","Parent Communication","Progress Comparison","AI Counselor"]
)

# ---------------------------------
# CALCULATE RISK AND INTERVENTIONS
# ---------------------------------
risks, interventions = [], []
for _, row in df.iterrows():
    risk, _ = calculate_risk(row)
    risks.append(risk)
    interventions.append(assign_interventions(row, risk))
df["risk_score"] = risks
df["suggested_intervention"] = interventions

# ---------------------------------
# TRAIN AI MODEL
# ---------------------------------
if len(df) > 2:
    ai_model = train_ai_model(df)

# ----------------------- 
# The rest of the code (dashboard, update attendance, parent messages, AI counselor) remains unchanged
# You can paste all your existing dashboard code here as is.
# -----------------------
# DASHBOARD
# ---------------------------------
if menu == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Low Attendance", len(df[df['attendance_percentage'] < 65]))
    col3.metric("Low Marks", len(df[df['exam_marks'] < 40]))
    col4.metric("High Risk Students", len(df[df['risk_score'] > HIGH_RISK_THRESHOLD]))
    fig = px.bar(df, x="name", y="risk_score", color="risk_score", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# STUDENT DATA
# ---------------------------------
if menu == "Student Data":
    st.dataframe(df)

# ---------------------------------
# RISK ANALYSIS
# ---------------------------------
if menu == "Risk Analysis":
    for _, row in df.iterrows():
        risk, reasons = calculate_risk(row)
        st.write(f"### {row['name']}")
        st.write("Risk Score:", risk)
        st.write("Reasons:", ", ".join(reasons) if reasons else "None")
        st.write("Suggested Interventions:", assign_interventions(row, risk))

        st.write("**Government Policies:**")
        if risk > 0 and reasons:
            for reason in reasons:
                if reason in gov_policies:
                    for policy in gov_policies[reason]:
                        st.markdown(f"- {policy}")
        else:
            st.markdown("- No suggestions — student is performing well")

        if len(df) > 2:
            st.write("AI Prediction:", predict_student_risk(row))
# ---------------------------------
# UPDATE ATTENDANCE
# ---------------------------------
if menu == "Update Attendance":
    reg_no = st.selectbox("Student", df["reg_no"])
    present = st.radio("Present Today?", [1, 0])
    if st.button("Update Attendance"):
        new_att = update_attendance(reg_no, present)
        st.success(f"Attendance Updated: {new_att:.2f}%")

# ---------------------------------
# INTERVENTION
# ---------------------------------
if menu == "Intervention":
    reg_no = st.selectbox("Select Student", df["reg_no"])
    student = df[df["reg_no"] == reg_no].iloc[0]
    risk, reasons = calculate_risk(student)
    st.metric("Current Attendance", student["attendance_percentage"])
    st.metric("Current Marks", student["exam_marks"])
    st.metric("Risk Score", student["risk_score"])
    suggested = student["suggested_intervention"]
    st.info(f"Suggested Intervention: {suggested}")
    st.subheader("Government Support / Policies")
    for reason in reasons:
        if reason in gov_policies:
            for policy in gov_policies[reason]:
                st.markdown(f"- {policy}")
    action = st.text_area("Intervention Taken", value=suggested)
    after_att = st.number_input("New Attendance %", value=float(student["attendance_percentage"]))
    after_marks = st.number_input("New Marks", value=float(student["exam_marks"]))
    if st.button("Save Intervention"):
        log_intervention(reg_no, action, after_att, after_marks)
        st.success("Intervention Saved Successfully")

# ---------------------------------
# PARENT COMMUNICATION
# ---------------------------------
if menu == "Parent Communication":
    reg_no = st.selectbox("Select Student", df["reg_no"])
    student = df[df["reg_no"] == reg_no].iloc[0]
    risk, reasons = calculate_risk(student)
    message = generate_parent_message(student, risk,reasons)
    st.text_area("Parent Message", message, height=300)
    st.download_button("Download Message", message, file_name=f"parent_message_{student['reg_no']}.txt")

# ---------------------------------
# PROGRESS COMPARISON
# ---------------------------------
if menu == "Progress Comparison":
    inter = pd.read_csv(intervention_file)
    if inter.empty:
        st.warning("No intervention data available")
    else:
        reg_no = st.selectbox("Student", inter["reg_no"].unique())
        student = inter[inter["reg_no"] == reg_no].iloc[-1]
        fig1 = px.bar(x=["Before Attendance", "After Attendance"], y=[student["before_attendance"], student["after_attendance"]], title="Attendance Improvement")
        fig2 = px.bar(x=["Before Marks", "After Marks"], y=[student["before_exam_marks"], student["after_exam_marks"]], title="Marks Improvement")
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
#AI COUNSELOR
# ---------------------------------

# ---------------------------------
# AI COUNSELOR
# ---------------------------------
if menu == "AI Counselor":

    st.title("🤖 AI Teacher Counselor")

    reg_no = st.selectbox("Select Student", df["reg_no"])

    student = df[df["reg_no"] == reg_no].iloc[0]

    risk, reasons = calculate_risk(student)

    st.subheader(student["name"])

    st.metric("Risk Score", risk)

    st.subheader("Possible Dropout Causes")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("No major risk factors")

    # Policy suggestions without repetition

    policies = set()

    if "Low attendance" in reasons:
        policies.add("Right to Education Act (2009)")
        policies.add("Vidyanjali Mentorship Programme")

    if "Low exam marks" in reasons:
        policies.add("National Education Policy 2020 – Bridge Learning")

    if "Irregular mid-day meal" in reasons:
        policies.add("PM POSHAN Mid-Day Meal Scheme")

    if "Long distance from school" in reasons:
        policies.add("Transport / Bicycle Support Schemes")

    if "Sibling dropout history" in reasons:
        policies.add("Kasturba Gandhi Balika Vidyalaya")

    st.subheader("Relevant Government Policies")

    if policies:
        for p in sorted(policies):
            st.write("•", p)
    else:
        st.write("No specific policy recommendation")

    st.subheader("AI Suggested Interventions")

    recommended = []

    if "Low attendance" in reasons:
        recommended.append("Counseling")

    if "Low exam marks" in reasons:
        recommended.append("Extra tutoring")

    if "Long distance from school" in reasons:
        recommended.append("Transport support")

    if "Sibling dropout history" in reasons:
        recommended.append("Parent meeting")

    for r in recommended:
        st.write("✔", r)

    st.subheader("Teacher Custom Action Plan")

    tutoring = st.checkbox("Extra tutoring")
    counseling = st.checkbox("Student counseling")
    parent = st.checkbox("Parent meeting")
    scheme = st.checkbox("Government scheme referral")

    teacher_note = st.text_area("Teacher Notes")

    if st.button("Save Teacher Plan"):
        st.success("Teacher plan recorded")

    st.subheader("Suggested Conversation")

    if risk >= 60:

        st.markdown("""
"I noticed you have been struggling recently.  
Is there anything making it difficult for you to attend school?  
We want to support you and help you succeed."
""")

    elif risk >= 40:

        st.markdown("""
"You have great potential but your attendance and marks need attention.  
Let's work together to improve them."
""")

    else:

        st.markdown("""
"You're doing well. Keep maintaining your effort."
""")