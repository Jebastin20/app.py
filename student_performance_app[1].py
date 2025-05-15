import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr

# Load dataset
df = pd.read_csv('student-mat.csv', sep=';')

# Preprocessing and encoding
target = 'G3'
df_encoded = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop(target, axis=1))
y = df_encoded[target]

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function for Gradio interface
def predict_grade(
    school, sex, age, address, famsize, Pstatus, Medu, Fedu,
    Mjob, Fjob, reason, guardian, traveltime, studytime,
    failures, schoolsup, famsup, paid, activities, nursery,
    higher, internet, romantic, famrel, freetime, goout,
    Dalc, Walc, health, absences, G1, G2
):
    input_data = {
        'school': school, 'sex': sex, 'age': int(age), 'address': address, 'famsize': famsize,
        'Pstatus': Pstatus, 'Medu': int(Medu), 'Fedu': int(Fedu), 'Mjob': Mjob, 'Fjob': Fjob,
        'reason': reason, 'guardian': guardian, 'traveltime': int(traveltime), 'studytime': int(studytime),
        'failures': int(failures), 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': int(famrel), 'freetime': int(freetime), 'goout': int(goout),
        'Dalc': int(Dalc), 'Walc': int(Walc), 'health': int(health), 'absences': int(absences),
        'G1': int(G1), 'G2': int(G2)
    }

    input_df = pd.DataFrame([input_data])

    # Combine original data (without target) + new input for consistent encoding
    df_temp = pd.concat([df.drop(target, axis=1), input_df], ignore_index=True)
    df_temp_encoded = pd.get_dummies(df_temp, drop_first=True)
    # Reindex to ensure columns match the training data
    df_temp_encoded = df_temp_encoded.reindex(columns=df_encoded.drop(target, axis=1).columns, fill_value=0)

    scaled_input = scaler.transform(df_temp_encoded.tail(1))
    prediction = model.predict(scaled_input)

    return round(prediction[0], 2)

# Define Gradio input components
inputs = [
    gr.Dropdown(['GP', 'MS'], label="School (GP=Gabriel Pereira, MS=Mousinho da Silveira)"),
    gr.Dropdown(['M', 'F'], label="Gender (M=Male, F=Female)"),
    gr.Number(label="Student Age"),
    gr.Dropdown(['U', 'R'], label="Residence Area (U=Urban, R=Rural)"),
    gr.Dropdown(['LE3', 'GT3'], label="Family Size (LE3=≤3, GT3=>3 members)"),
    gr.Dropdown(['A', 'T'], label="Parent Cohabitation Status (A=Apart, T=Together)"),
    gr.Number(label="Mother's Education Level (0-4)"),
    gr.Number(label="Father's Education Level (0-4)"),
    gr.Dropdown(['teacher', 'health', 'services', 'at_home', 'other'], label="Mother's Job"),
    gr.Dropdown(['teacher', 'health', 'services', 'at_home', 'other'], label="Father's Job"),
    gr.Dropdown(['home', 'reputation', 'course', 'other'], label="Reason for Choosing School"),
    gr.Dropdown(['mother', 'father', 'other'], label="Guardian"),
    gr.Number(label="Travel Time to School (1-4)"),
    gr.Number(label="Weekly Study Time (1-4)"),
    gr.Number(label="Past Class Failures (0-3)"),
    gr.Dropdown(['yes', 'no'], label="Extra School Support"),
    gr.Dropdown(['yes', 'no'], label="Family Support"),
    gr.Dropdown(['yes', 'no'], label="Extra Paid Classes"),
    gr.Dropdown(['yes', 'no'], label="Participates in Activities"),
    gr.Dropdown(['yes', 'no'], label="Attended Nursery"),
    gr.Dropdown(['yes', 'no'], label="Aspires Higher Education"),
    gr.Dropdown(['yes', 'no'], label="Internet Access at Home"),
    gr.Dropdown(['yes', 'no'], label="Currently in a Relationship"),
    gr.Number(label="Family Relationship Quality (1-5)"),
    gr.Number(label="Free Time After School (1-5)"),
    gr.Number(label="Going Out Frequency (1-5)"),
    gr.Number(label="Workday Alcohol Consumption (1-5)"),
    gr.Number(label="Weekend Alcohol Consumption (1-5)"),
    gr.Number(label="Health Status (1=Very Bad to 5=Excellent)"),
    gr.Number(label="Number of Absences"),
    gr.Number(label="Grade in 1st Period (G1: 0-20)"),
    gr.Number(label="Grade in 2nd Period (G2: 0-20)")
]

output = gr.Number(label="🎯 Predicted Final Grade (G3)")

# Launch the Gradio app
gr.Interface(
    fn=predict_grade,
    inputs=inputs,
    outputs=output,
    title="🎓 Student Performance Predictor",
    description="Enter academic and demographic info to predict the final grade (G3) of a student."
).launch()