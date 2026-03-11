import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

st.title("⚙️ AI Predictive Maintenance Dashboard")
st.subheader("ระบบ ML ทำนายความเสี่ยงเครื่องจักรขัดข้อง")
st.markdown("---")

# --- โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return joblib.load('predictive_model.pkl')

model = load_model()

# --- แถบด้านข้าง(Sidebar) สำหรับรับค่าเซนเซอร์ ---
st.sidebar.header("🎛️ ปรับค่าเซนเซอร์เครื่องจักร")
st.sidebar.markdown("จำลองการรับค่าจาก IoT Sensor เพื่อทำนายผลแบบ Real-time")

# แปลงเป็นตัวเลขให้ตรงกับตอนเทรนโมเดล  L=0, M=1, H=2
type_input = st.sidebar.selectbox("คุณภาพสินค้า (Product Type)", ["Low (L)", "Medium (M)", "High (H)"])
type_dict = {"Low (L)": 0, "Medium (M)": 1, "High (H)": 2}
type_val = type_dict[type_input]

# รับค่าเซนเซอร์ต่างๆ ด้วย Slider
air_temp = st.sidebar.slider("Air temperature(อุณหภูมิอากาศ)[K]", 290.0, 310.0, 298.0)
process_temp = st.sidebar.slider("Process temperature(อุณหภูมิ)[K]", 300.0, 320.0, 308.0)
rpm = st.sidebar.slider("Rotational speed (ความเร็วรอบ)[rpm]", 1100, 2900, 1500)
torque = st.sidebar.slider("Torque (แรงบิด)[Nm]", 10.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool wear(การสึกหรอเครื่อง)[min]", 0, 300, 50) # ยิ่งใช้เยอะยิ่งเสี่ยงพัง

# --- ข้อมูลที่ให้ AI ทำนาย ---
input_data = pd.DataFrame({
    'Type': [type_val],
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rpm],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear]
})

# --- แบ่งเป็น 2 Column ---
col1, col2 = st.columns([1, 1])

# ให้โมเดลทำนายผล
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]
failure_probability = prediction_proba[1] * 100

with col1:
    st.subheader("🔮ผลการวิเคราะห์จาก AI (Prediction Result)")
    if prediction == 0:
        st.success("✅ สถานะ: ปกติ (Normal) - เครื่องจักรทำงานได้อย่างปลอดภัย")
    else:
        st.error("🚨 สถานะ: เสี่ยงขัดข้อง (Failure Warning) - พบแนวโน้มเครื่องจักรเสียหาย!")
        
    # กราฟแท่งแนวนอน ดูว่าเซนเซอร์ตัวไหนมีผลต่อการตัดสินใจของ AI มากที่สุด
    st.markdown("**📊 ปัจจัยที่ส่งผลต่อความเสี่ยงมากที่สุด**")
    feature_imp = pd.DataFrame({
        'Feature': ['Product Type', 'Air Temp', 'Process Temp', 'RPM', 'Torque', 'Tool Wear'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    # สร้างกราฟแท่งแนวนอน
    fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                     color_discrete_sequence=["#EC6E48"])
    fig_imp.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), 
                          xaxis_title="ระดับส่งผลต่อเครื่องจักร", yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)

with col2:
    st.subheader("🚥 ระดับความเสี่ยง (Risk Probability)")
    # กราฟหน้าปัด (Gauge Chart)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = failure_probability,
        title = {'text': "โอกาสที่เครื่องจะพัง (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 30], 'color': "#2ECC71"},   # เขียว
                {'range': [30, 70], 'color': "#F1C40F"},  # เหลือง
                {'range': [70, 100], 'color': "#E74C3C"}  # แดง
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)


st.markdown("---")

# --- ฟีเจอร์ระบบ AI แนะนำวิธีแก้ปัญหา (Prescriptive Action Plan) ---
st.subheader("🛠️ ข้อเสนอแนะการซ่อมบำรุง (Prescriptive Action Plan)")
action_plan_text = ""

# ใช้ Logic ตรวจสอบค่าเซนเซอร์เพื่อกำหนดงานให้ช่าง
if failure_probability < 30 and tool_wear < 150 and torque < 50:
    st.success("**🟢 คำแนะนำ:** เครื่องจักรอยู่ในสภาพสมบูรณ์ ให้ดำเนินการผลิตต่อและบำรุงรักษาตามรอบปกติ")
    action_plan_text = "ปกติ (Normal) - บำรุงรักษาตามรอบปกติ"
else:
    actions_for_ui = []
    actions_for_csv = []
    
    # เช็คเรื่องหัวเจาะสึกหรอ (Check tool wear)
    if tool_wear >= 200:
        actions_for_ui.append("**Tool Wear Warning (การสึกหรอ):** หัวเจาะมีการสึกหรอสูงมาก แนะนำให้ช่างเทคนิคเตรียมอะไหล่และทำการ **เปลี่ยนหัวเจาะ (Tool Replacement)** ทันที เพื่อป้องกันชิ้นงานเสียหาย")
        actions_for_csv.append("Tool Wear Warning: แนะนำเปลี่ยนหัวเจาะ (Tool Replacement) ทันที")
    elif tool_wear >= 150:
        actions_for_ui.append("**Tool Wear Alert:** หัวเจาะเริ่มสึกหรอ ควรจัดตารางตรวจสอบในกะถัดไป")
        actions_for_csv.append("Tool Wear Alert: ควรจัดตารางตรวจสอบในกะถัดไป")
        
    # เช็คเรื่องแรงบิดและมอเตอร์ (Check torque)
    if torque >= 60:
        actions_for_ui.append("**Torque Overload (แรงบิดสูง):** พบแรงบิดสูงผิดปกติ แนะนำให้ตรวจสอบ **โหลดของมอเตอร์ (Motor Load)** และตรวจเช็คระบบหล่อลื่น")
        actions_for_csv.append("Torque Overload: แนะนำตรวจสอบโหลดของมอเตอร์และระบบหล่อลื่น")

    # เช็คเรื่องความร้อนสะสม (Check process temp & air temp)
    if (process_temp - air_temp) > 8.6 or process_temp > 315:
        actions_for_ui.append("**Heat Dissipation (ความร้อนสะสม):** ระบบระบายความร้อนทำงานหนักเกินไป แนะนำให้ตรวจสอบ **พัดลมระบายอากาศ (Cooling System)** หรือหยุดพักเครื่องชั่วคราว")
        actions_for_csv.append("Heat Dissipation: แนะนำตรวจสอบพัดลมระบายอากาศหรือหยุดพักเครื่อง")

    # แสดงผลลัพธ์ออกมา (Result)
    if len(actions_for_ui) > 0:
        for action in actions_for_ui:
            st.warning(f"- {action}")
        action_plan_text = " | ".join(actions_for_csv)
    else:
        st.warning("⚠️ สภาวะการทำงานมีความเสี่ยงแฝง ควรส่งวิศวกรเข้าตรวจสอบหน้าเครื่องจักรเพื่อประเมินสถานการณ์จริง")
        action_plan_text = "ความเสี่ยงแฝง ควรส่งวิศวกรเข้าตรวจสอบหน้าเครื่องจักร"

# --- ฟีเจอร์ แสดงตารางข้อมูล และ ปุ่มดาวน์โหลดรายงาน ---
st.markdown("---")
st.subheader("📋 ตารางรายงานข้อมูลสถานะปัจจุบัน")

# เอาผลลัพธ์จาก AI มารวมกับข้อมูล input แล้วให้ตอนโหลดไฟล์ไปจะได้มีผลทำนายติดไปด้วย
export_data = input_data.copy()
export_data['Prediction_Result'] = "Failure Warning" if prediction == 1 else "Normal"
export_data['Failure_Probability_%'] = round(failure_probability, 2)

export_data['Action_Plan'] = action_plan_text

st.dataframe(export_data, use_container_width=True)
csv = export_data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download the analysis results.",
    data=csv,
    file_name='predictive_maintenance_report.csv',
    mime='text/csv',
)

st.markdown("<br>", unsafe_allow_html=True)
