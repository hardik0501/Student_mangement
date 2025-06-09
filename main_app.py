import gradio as gr
import pandas as pd
import openai
import requests
import json

openai.api_key = "Your Api"

DATA_FILE = "student_data.csv"

def load_data():
    try:
        return pd.read_csv(DATA_FILE)
    except:
        df = pd.DataFrame(columns=["roll_no", "name", "attendance", "marks", "remarks"])
        df.to_csv(DATA_FILE, index=False)
        return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def analyze_student_performance(name, attendance, marks):
    api_key = "AIzaSyBhpRkOnmcurInRavgzB2nc0UkLulq7yYo"
    url = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText"
    headers = {
        "Content-Type": "application/json",
    }
    prompt = (f"Provide a concise performance remark for a student named {name} "
              f"who has an attendance of {attendance}% and scored {marks} marks out of 100. "
              f"Be encouraging and constructive.")
    data = {
        "prompt": {
            "text": prompt
        },
        "temperature": 0.7,
        "maxOutputTokens": 100,
        "candidateCount": 1,
        "topP": 0.8,
        "topK": 40
    }
    params = {"key": api_key}
    response = requests.post(url, headers=headers, params=params, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            remark = result['candidates'][0]['output']
            return remark
        except (KeyError, IndexError):
            return "Error: Unexpected response format from AI API."
    else:
        return f"Error: API returned status code {response.status_code}"

def add_student(roll_no, name, attendance, marks):
    df = load_data()
    if roll_no in df['roll_no'].values:
        return df
    new_row = {"roll_no": roll_no, "name": name, "attendance": attendance, "marks": marks, "remarks": ""}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)
    return df

def delete_student(roll_no):
    df = load_data()
    df = df[df['roll_no'] != roll_no]
    save_data(df)
    return df

def search_student(query):
    df = load_data()
    return df[df['name'].str.contains(query, case=False) | df['roll_no'].astype(str).str.contains(query)]

def generate_remark(roll_no):
    df = load_data()
    student = df[df['roll_no'] == roll_no]
    if student.empty:
        return df
    student = student.iloc[0]
    remark = analyze_student_performance(student['name'], student['attendance'], student['marks'])
    df.loc[df['roll_no'] == roll_no, 'remarks'] = remark
    save_data(df)
    return df

def get_student_info(name):
    df = load_data()
    row = df[df['name'] == name]
    if row.empty:
        return 0, "", 0, 0
    student = row.iloc[0]
    return student['roll_no'], student['name'], student['attendance'], student['marks']

def update_student_full(old_name, new_roll, new_name, new_attendance, new_marks):
    df = load_data()
    idx = df[df['name'] == old_name].index
    if idx.empty:
        return df
    df.at[idx[0], 'roll_no'] = new_roll
    df.at[idx[0], 'name'] = new_name
    df.at[idx[0], 'attendance'] = new_attendance
    df.at[idx[0], 'marks'] = new_marks
    save_data(df)
    return df

with gr.Blocks() as app:
    gr.Markdown("# 🎓 AI-Powered Student Management System")

    with gr.Tab("📋 View / Search"):
        query = gr.Textbox(label="Search by Name or Roll No")
        output_table = gr.Dataframe()
        query.submit(search_student, inputs=query, outputs=output_table)

    with gr.Tab("➕ Add Student"):
        roll = gr.Number(label="Roll No", precision=0)
        name = gr.Textbox(label="Name")
        attendance = gr.Number(label="Attendance (%)")
        marks = gr.Number(label="Marks")
        add_btn = gr.Button("Add Student")
        add_output = gr.Dataframe()
        add_btn.click(add_student, inputs=[roll, name, attendance, marks], outputs=add_output)

    with gr.Tab("✏️ Edit Full Record by Name"):
        student_names = load_data()['name'].tolist()
        dropdown = gr.Dropdown(label="Select Student by Name", choices=student_names)

        old_roll = gr.Number(label="Current Roll No", interactive=False)
        new_roll = gr.Number(label="New Roll No", precision=0)
        new_name = gr.Textbox(label="New Name")
        new_attendance = gr.Number(label="New Attendance (%)")
        new_marks = gr.Number(label="New Marks")
        update_btn = gr.Button("Update Student")
        update_output = gr.Dataframe()

        dropdown.change(get_student_info, inputs=dropdown,
                        outputs=[old_roll, new_name, new_attendance, new_marks])
        update_btn.click(update_student_full,
                         inputs=[dropdown, new_roll, new_name, new_attendance, new_marks],
                         outputs=update_output)

    with gr.Tab("❌ Delete Student"):
        roll_delete = gr.Number(label="Roll No to Delete", precision=0)
        del_btn = gr.Button("Delete")
        del_output = gr.Dataframe()
        del_btn.click(delete_student, inputs=roll_delete, outputs=del_output)

    with gr.Tab("🧠 AI Performance Review"):
        roll_review = gr.Number(label="Roll No", precision=0)
        ai_btn = gr.Button("Generate Remark")
        ai_output = gr.Dataframe()
        ai_btn.click(generate_remark, inputs=roll_review, outputs=ai_output)

app.launch()
