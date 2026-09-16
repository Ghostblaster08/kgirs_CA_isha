"""
Virtual Laboratory Experiment Template (Streamlit)
A completely generic, modular template partitioned into 4 core sections:
  1. Theory: Concepts, objectives, procedure, and terminology.
  2. Simulation: Interactive parameters, execution model, visual plots, and trial logger.
  3. Quiz: Self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student info, recorded trials, observations, and downloadable PDF report.

Note: No custom CSS is used so that Streamlit native light and dark themes render seamlessly.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT (CUSTOMIZE THIS BLOCK)
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Virtual Laboratory Experiment",
    "objectives": [
        "State and understand the primary conceptual objectives of the experiment.",
        "Systematically analyze how varying input parameters influences output behavior.",
        "Compare empirical simulation results with expected theoretical outcomes.",
        "Evaluate experimental observations, operational constraints, and potential error sources."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview & Principles
Describe the core concepts, principles, and theoretical background of your experiment here.
You can use standard Markdown to explain the operating mechanisms, fundamental rules, and 
practical context of the study.

### Workflow & System Overview
1. **Input Configuration**: Define baseline parameters and operational constraints.
2. **Execution & Transformation**: Execute the experimental model across selected test conditions.
3. **Data Analysis**: Observe outputs, calculate key metrics, and verify consistency.
    """,
    "procedure": [
        "Step 1: Review the theoretical background, objectives, and key terminology.",
        "Step 2: Navigate to the Simulation section in the sidebar menu.",
        "Step 3: Adjust the primary input controls to configure your experiment.",
        "Step 4: Run the simulation to view real-time metrics and dynamic response curves.",
        "Step 5: Click 'Record Current Trial' to log data into your experimental session table.",
        "Step 6: Repeat for at least 3-4 distinct parameter configurations.",
        "Step 7: Complete the assessment Quiz to test your conceptual understanding.",
        "Step 8: Open Report Generation, enter your student information, and download your PDF report."
    ],
    "key_terms": {
        "Input Parameter 1": "Primary independent variable controlled during the trial.",
        "Input Parameter 2": "Secondary operational factor or configuration mode.",
        "Output Response": "Measured dependent variable resulting from system execution.",
        "Efficiency / Metric": "Calculated performance ratio or characteristic value."
    }
}

SIMULATION_CONFIG = {
    "param_1_label": "Parameter 1 (Scale / Input Magnitude)",
    "param_1_min": 10,
    "param_1_max": 500,
    "param_1_default": 100,
    "param_1_step": 10,
    "param_2_label": "Parameter 2 (Operational Strategy)",
    "param_2_options": ["Strategy A", "Strategy B", "Strategy C"],
    "param_3_label": "Parameter 3 (Rate / Factor)",
    "param_3_min": 1.0,
    "param_3_max": 10.0,
    "param_3_default": 2.5,
    "param_3_step": 0.5
}

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "How does increasing Parameter 1 influence the system output response?",
        "options": [
            "A) It scales the output magnitude proportionally",
            "B) It causes the output to immediately drop to zero",
            "C) It has no measurable impact on system behavior",
            "D) It reverses the operational direction completely"
        ],
        "answer_index": 0,
        "explanation": "Parameter 1 controls the input scale, producing a proportional increase in output magnitude."
    },
    {
        "id": 2,
        "question": "Which operational strategy yields the highest baseline efficiency in this model?",
        "options": [
            "A) Strategy A",
            "B) Strategy B",
            "C) Strategy C",
            "D) All strategies yield identical results"
        ],
        "answer_index": 2,
        "explanation": "Strategy C employs optimized execution routines that maximize output efficiency."
    },
    {
        "id": 3,
        "question": "What is the primary purpose of recording multiple experimental trials?",
        "options": [
            "A) To arbitrarily fill up computer memory",
            "B) To establish repeatable trends and observe behavior across parameter ranges",
            "C) To reset all variables back to default states",
            "D) Multiple trials are unnecessary for experimental analysis"
        ],
        "answer_index": 1,
        "explanation": "Multiple trials across different parameter points reveal system trends and reduce random variation."
    },
    {
        "id": 4,
        "question": "What defines an independent variable in an experimental investigation?",
        "options": [
            "A) The variable intentionally changed or controlled by the experimenter",
            "B) The measured outcome that changes in response to inputs",
            "C) The unintended background noise in measurement equipment",
            "D) The final conclusion written in the lab report"
        ],
        "answer_index": 0,
        "explanation": "An independent variable is the factor deliberately manipulated to observe its effect on the dependent outcome."
    },
    {
        "id": 5,
        "question": "Why is baseline calibration important prior to conducting experimental runs?",
        "options": [
            "A) To permanently fix all output values to zero",
            "B) To eliminate or account for systematic offsets and measurement bias",
            "C) To bypass the need for recording trial data",
            "D) Calibration is optional and does not affect accuracy"
        ],
        "answer_index": 1,
        "explanation": "Calibration aligns measurement readings with known reference standards, minimizing systematic errors."
    },
    {
        "id": 6,
        "question": "What does the term 'steady state' indicate in a dynamic system simulation?",
        "options": [
            "A) The condition where system variables remain constant or follow a stable periodic pattern",
            "B) The initial instant when the simulation begins execution",
            "C) A condition where all inputs and outputs are completely disabled",
            "D) The state that occurs only when an error is thrown"
        ],
        "answer_index": 0,
        "explanation": "Steady state refers to equilibrium where transient fluctuations have dissipated and variables stabilize."
    },
    {
        "id": 7,
        "question": "How does Parameter 3 (rate/factor) influence the transition speed of the response curve?",
        "options": [
            "A) It has no effect on the transition rate",
            "B) Lower values cause instantaneous stabilization",
            "C) Higher values accelerate the rate of convergence toward steady state",
            "D) It converts the curve into a static horizontal line"
        ],
        "answer_index": 2,
        "explanation": "Parameter 3 governs the exponential rate factor; higher values increase the rate of approach to steady state."
    },
    {
        "id": 8,
        "question": "What is the primary benefit of comparing empirical simulation data against theoretical predictions?",
        "options": [
            "A) To automatically discard experimental data that does not match theory",
            "B) To avoid having to calibrate instruments or sensors",
            "C) To prove that theoretical formulas never require validation",
            "D) To identify discrepancies, evaluate assumptions, and validate model fidelity"
        ],
        "answer_index": 3,
        "explanation": "Comparing empirical results against theoretical models tests underlying assumptions and quantifies model accuracy."
    },
    {
        "id": 9,
        "question": "Which of the following is an example of random error rather than systematic error?",
        "options": [
            "A) Fluctuations caused by ambient temperature changes and electrical noise",
            "B) A scale consistently reading 5 grams too high due to improper zeroing",
            "C) An incorrect formula used in software data processing",
            "D) A stopwatch running 10% slow throughout the entire experiment"
        ],
        "answer_index": 0,
        "explanation": "Random errors arise from unpredictable environmental fluctuations and precision limits, unlike constant systematic biases."
    },
    {
        "id": 10,
        "question": "In scientific and engineering reporting, what should the discussion section primarily emphasize?",
        "options": [
            "A) A raw copy-paste of computer code without explanation",
            "B) Analysis of observations, comparison with expected outcomes, and limitations",
            "C) Speculation unrelated to the collected trial data",
            "D) Only the student name and experiment date"
        ],
        "answer_index": 1,
        "explanation": "The discussion synthesizes key findings, interprets trends, evaluates limitations, and connects results to core principles."
    }
]


# ======================================================================================
# 2. SIMULATION ENGINE (CUSTOMIZE YOUR LOGIC HERE)
# ======================================================================================

def run_simulation(param_1: int, param_2: str, param_3: float) -> dict:
    """
    Generic simulation model. Replace this with your domain-specific calculations.
    """
    if param_2 == "Strategy A":
        factor = 1.0
        efficiency = 82.0
    elif param_2 == "Strategy B":
        factor = 1.6
        efficiency = 91.5
    else:  # Strategy C
        factor = 2.4
        efficiency = 97.8

    output_metric = round(param_1 * factor * (param_3 / 2.0), 2)
    response_time_ms = round((param_1 / (factor * 5.0)) + 3.5, 2)

    # Progression curve
    t_points = np.linspace(0, 10, 25)
    curve_values = [round(output_metric * (1.0 - np.exp(-param_3 * (t / 5.0))), 2) for t in t_points]

    return {
        "output_metric": output_metric,
        "efficiency": efficiency,
        "response_time_ms": response_time_ms,
        "time_points": t_points,
        "curve_values": curve_values
    }


# ======================================================================================
# 3. LAB REPORT PDF EXPORTER
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str) -> bytes:
    """Compiles experiment benchmark records into a proper, formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 22, 190, 22, "FD")

    pdf.set_xy(14, 24)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 32)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score/quiz_total)*100 if quiz_total else 0)}%)", 1)

    pdf.ln(12)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials & Data", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        cols = list(trials_df.columns)
        num_cols = len(cols)
        col_w = max(18, int(190 / max(1, num_cols)))

        for c in cols:
            pdf.cell(col_w, 6, str(c)[:14], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        fill = False

        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, val_str[:14], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    # 3. Discussion & Notes
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The experimental trials demonstrated consistent behavior across varying parameter configurations, "
        "matching expected analytical outcomes."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 4. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
# ======================================================================================

def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, and Procedure."""
    st.header("Theoretical Framework & Background")
    st.markdown(THEORY_CONTENT["background"])

    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.write(f"- {step}")

    st.divider()
    with st.expander("Key Terminology & Variable Reference"):
        var_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term / Variable", "Definition & Role"]
        )
        st.table(var_df)


def render_simulation_section():
    """Renders Section 2: Interactive Execution Sandbox and Plots."""
    st.header("Interactive Simulation Sandbox")
    st.info("Adjust the experimental parameters below to run the simulation and log trial data.")

    cfg = SIMULATION_CONFIG
    col1, col2, col3 = st.columns(3)

    with col1:
        param_1 = st.slider(
            cfg["param_1_label"],
            min_value=cfg["param_1_min"],
            max_value=cfg["param_1_max"],
            value=cfg["param_1_default"],
            step=cfg["param_1_step"]
        )

    with col2:
        param_2 = st.selectbox(
            cfg["param_2_label"],
            options=cfg["param_2_options"],
            index=0
        )

    with col3:
        param_3 = st.slider(
            cfg["param_3_label"],
            min_value=cfg["param_3_min"],
            max_value=cfg["param_3_max"],
            value=cfg["param_3_default"],
            step=cfg["param_3_step"]
        )

    sim_res = run_simulation(param_1, param_2, param_3)

    st.divider()

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Output Metric", f"{sim_res['output_metric']}")
    with m2:
        st.metric("Efficiency", f"{sim_res['efficiency']}%")
    with m3:
        st.metric("Response Time", f"{sim_res['response_time_ms']} ms")

    # Plot
    st.subheader("Dynamic Response Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_res["time_points"],
        y=sim_res["curve_values"],
        mode="lines+markers",
        name="Output Response",
        line=dict(width=2.5)
    ))
    fig.update_layout(
        title="System Response Progression",
        xaxis_title="Time / Progression Index",
        yaxis_title="Output Value",
        hovermode="x unified",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data Logger
    st.divider()
    st.subheader("Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture current parameters and metrics into your session trial table:")
        if st.button("Record Current Trial", type="primary", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Param 1": param_1,
                "Param 2": param_2,
                "Param 3": param_3,
                "Output": sim_res["output_metric"],
                "Efficiency (%)": sim_res["efficiency"],
                "Time (ms)": sim_res["response_time_ms"],
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully saved!")

        if st.button("Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, use_container_width=True, hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="experiment_trials.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to begin collecting experimental data.")


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.header("Concept Assessment Quiz")
    st.write("Answer the conceptual questions below to evaluate your understanding of the experiment.")

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            selected = st.radio(
                label=f"Options for Question {q['id']}:",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Feedback")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            if user_ans == correct_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                         f"**Correct Answer:** {q['options'][correct_ans]}\n\n"
                         f"**Reasoning:** _{q['explanation']}_")

        st.session_state["quiz_score"] = score
        perc = (score / len(QUIZ_QUESTIONS)) * 100
        st.info(f"Final Score: **{score} / {len(QUIZ_QUESTIONS)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already submitted. Current score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with Guaranteed PDF Export."""
    st.header("Report Generation")
    st.write("Compile your student details, recorded trials, and quiz evaluation into an official PDF report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-001"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")
    student_notes = st.text_area(
        "Enter your interpretation of results, observations, and conclusions:",
        value=st.session_state.get("student_notes", (
            "The experimental trials demonstrated consistent behavior across varying parameter configurations, "
            "matching expected analytical outcomes."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.")

    # Generate PDF bytes and write file to disk
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes
    )

    # Save to local files for guaranteed download
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        # Direct static link ending in .pdf
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        # Standard Streamlit download button
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


# ======================================================================================
# 5. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-001",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""


def main():
    st.set_page_config(
        page_title="Virtual Lab Experiment",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    # Native Streamlit Title (No custom CSS)
    st.title(EXPERIMENT_CONFIG["title"])

    # Navigation Sidebar
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")

    # Section Dispatcher
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
