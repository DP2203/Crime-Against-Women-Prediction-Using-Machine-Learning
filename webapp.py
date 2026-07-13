import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Crime Against Women Prediction",
    page_icon="fevicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR CSS ---------------- #
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    position: fixed;
    height: 100vh;
    width: 260px !important;
    background: linear-gradient(180deg, #1E1E2F, #2C3E50);
}
[data-testid="collapsedControl"] {
    display: none !important;
}
.main {
    margin-left: 260px;
}
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2E8B57);
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATABASE ---------------- #
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY,
             email TEXT UNIQUE,
             password TEXT)''')
conn.commit()


# ---------------- ADMIN ---------------- #
ADMIN_EMAIL = "admin@admin.com"
ADMIN_PASS = "admin123"


# ---------------- SESSION ---------------- #
if "login_status" not in st.session_state:
    st.session_state.login_status = False

if "role" not in st.session_state:
    st.session_state.role = None

# 🔥 NEW: menu state control
if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Home"


# ---------------- SIDEBAR ---------------- #
menu_options = ["🏠 Home", "✨ Sign Up", "🔐 Login", "📊 Dashboard"]

menu = st.sidebar.selectbox(
    "",
    menu_options,
    index=menu_options.index(st.session_state.menu)
)

# 🔥 update selected menu
st.session_state.menu = menu


if st.session_state.login_status:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.login_status = False
        st.session_state.role = None
        st.session_state.menu = "🏠 Home"
        st.rerun()

st.sidebar.markdown("""
<style>
.sidebar-footer {
    position: fixed;
    bottom: 10px;
    left: 20px;
    font-size: 15px;
    color: gray;
}
</style>
<div class="sidebar-footer">
    © 2026 Crime ML System
</div>
""", unsafe_allow_html=True)


# ---------------- HOME ---------------- #
if menu == "🏠 Home":

    st.title("Women’s Safety in India: Crime Trend Forecasting Using ML")

    st.markdown("""
    ### 📌 Project Overview
    
    The project **“Women’s Safety in India: Crime Trend Forecasting Using Machine Learning”** focuses on analyzing crimes against women using historical data and predicting future trends using ML models.
    
    It uses crime datasets (NCRB & public sources) to:
    - Understand crime patterns across different states
    - Identify high-risk regions
    - Forecast future crime rates
    
    ---
    
    ### 🎯 Purpose of the Project
    
    - Identify crime trends  
    - Predict future crime rates  
    - Support decision-making for safety planning  
    - Provide a user-friendly analysis system  
    
    ---
    
    ### ⚙️ Technologies Used
    
    - **Language:** Python  
    - **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn  
    - **Web Framework:** Streamlit  
    - **Database:** SQLite  
    
    ---
    
    ### 🧠 Machine Learning Models Used
    
    - Linear Regression  
    - Decision Tree  
    - Random Forest  
    - Gradient Boosting  
    
    ---
    
    ### 🔄 System Workflow
    
    1. Data Collection (Crime Dataset)  
    2. Data Preprocessing  
    3. Region Selection  
    4. Feature Selection  
    5. Train-Test Split (80-20)  
    6. Model Training  
    7. Evaluation (R², MSE, MAE, RMSE)  
    8. Prediction  
    9. Visualization  
    
    ---
    
    ### 📊 Key Features
    
    - Upload CSV/Excel datasets  
    - Select state-wise data  
    - Choose features & targets  
    - Train multiple ML models  
    - View performance metrics  
    - Visualize predictions using graphs  
    
    ---
    
    ###  Conclusion
    
    This system helps in understanding crime trends and predicting future risks using machine learning.  
    It can support government and law enforcement agencies in improving women's safety through data-driven decisions.
    """)
# ---------------- SIGN UP ---------------- #
elif menu == "✨ Sign Up":

    st.markdown("<h1 style='text-align: center;'>✨ Create Your Account</h1>", unsafe_allow_html=True)

    with st.form("signup_form"):
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password")

        submitted = st.form_submit_button("🚀 Create Account")

        if submitted:
            if not all([email, password, confirm_password]):
                st.error("⚠️ Please fill all fields")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                try:
                    c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
                    conn.commit()
                    st.success("✅ Account created successfully!")
                except sqlite3.IntegrityError:
                    st.error("⚠️ Email already registered")


# ---------------- LOGIN ---------------- #
elif menu == "🔐 Login":

    st.markdown("<h1 style='text-align: center;'>🔐 Login</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        login_email = st.text_input("📧 Email")

    with col2:
        login_password = st.text_input("🔒 Password", type="password")

    if st.button("🚀 Login"):

        if login_email == ADMIN_EMAIL and login_password == ADMIN_PASS:
            st.session_state.login_status = True
            st.session_state.role = "admin"
            st.session_state.menu = "📊 Dashboard"   # 🔥 REDIRECT
            st.rerun()

        else:
            c.execute("SELECT * FROM users WHERE email=? AND password=?", (login_email, login_password))
            user = c.fetchone()

            if user:
                st.session_state.login_status = True
                st.session_state.role = "user"
                st.session_state.menu = "📊 Dashboard"   # 🔥 REDIRECT
                st.rerun()
            else:
                st.error("❌ Invalid credentials")


# ---------------- DASHBOARD ---------------- #
elif menu == "📊 Dashboard":

    if not st.session_state.login_status:
        st.warning("⚠️ Please login first")
        st.stop()

    # ---------------- ADMIN ---------------- #
    if st.session_state.role == "admin":

        st.success("Logged In as Admin")
        st.subheader("👨‍💼 Admin Dashboard")

        if st.button("📋 View All Users"):
            c.execute("SELECT email, password FROM users")
            users = c.fetchall()

            if users:
                df = pd.DataFrame(users, columns=["Email", "Password"])
                st.dataframe(df)
            else:
                st.info("No users found")

        st.markdown("### 🗑 Delete User")
        del_email = st.text_input("Enter Email")

        if st.button("Delete User"):
            if del_email:
                c.execute("DELETE FROM users WHERE email=?", (del_email,))
                conn.commit()
                st.success("User Deleted Successfully")
            else:
                st.warning("Enter email first")

    # ---------------- USER DASHBOARD ---------------- #
    elif st.session_state.role == "user":

        st.success("Logged in Successfully")
        st.header("Crime Against Women Prediction Using Machine Learning")

        data_file = st.file_uploader(
            "Upload Crime Data (CSV / Excel)",
            type=["csv", "xlsx", "xls"]
        )

        data = None

        if data_file is not None:
            try:
                if data_file.name.endswith(".csv"):
                    data = pd.read_csv(data_file)
                else:
                    data = pd.read_excel(data_file)

                st.write("### Data Preview")
                st.write(data.head())

            except Exception as e:
                st.error(f"Error reading file: {e}")

        if data is not None:

            data.replace("na", np.nan, inplace=True)
            data.dropna(inplace=True)

            if "Id" in data.columns:
                data = data.drop(["Id"], axis=1)

            state_column = None
            if 'State/Region' in data.columns:
                state_column = 'State/Region'
            elif 'State' in data.columns:
                state_column = 'State'
            elif 'Region' in data.columns:
                state_column = 'Region'

            if state_column:
                selected_state = st.selectbox("Select State", data[state_column].unique())
                data = data[data[state_column] == selected_state]

            st.write("### Processed Data")
            st.write(data.head())

            year_column = 'Year' if 'Year' in data.columns else None
            column_options = list(data.columns[2:])

            columns = st.multiselect("Select Crime Features", column_options, default=column_options[:5])
            target_options = st.multiselect("Select Target Variables", column_options, default=[column_options[0]])

            if len(columns) == 0 or len(target_options) == 0:
                st.warning("Please select features and target variables")
                st.stop()

            X = data[columns]
            y = data[target_options]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            selected_model = st.selectbox(
                "Select Model",
                ["Linear Regression","Random Forest","Decision Tree","Gradient Boosting"]
            )

            models = {
                "Linear Regression": MultiOutputRegressor(LinearRegression()),
                "Random Forest": MultiOutputRegressor(RandomForestRegressor(max_depth=10, n_estimators=50, random_state=0)),
                "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor()),
                "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor())
            }

            model = models[selected_model]
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)

            st.write(f"### Model : {selected_model}")

            for i, target in enumerate(target_options):
                actual = y_test[target].values
                predicted = y_pred[:, i]

                st.write(f"**{target} R2 Score:** {r2_score(actual, predicted):.4f}")
                st.write(f"**{target} MSE:** {mean_squared_error(actual, predicted):.4f}")
                st.write(f"**{target} MAE:** {mean_absolute_error(actual, predicted):.4f}")
                st.write(f"**{target} RMSE:** {np.sqrt(mean_squared_error(actual, predicted)):.4f}")

            fig, ax = plt.subplots(figsize=(10,6))

            pred_full = model.predict(X)
            if len(pred_full.shape) == 1:
                pred_full = pred_full.reshape(-1, 1)

            if year_column:
                years = data[year_column]
                for target in target_options:
                    idx = target_options.index(target)
                    ax.plot(years, data[target], marker='o', label=f'Actual {target}')
                    ax.plot(years, pred_full[:, idx], linestyle='--')
                ax.set_xlabel("Year")
            else:
                r = np.arange(len(X))
                for target in target_options:
                    idx = target_options.index(target)
                    ax.plot(r, data[target], label=f'Actual {target}')
                    ax.plot(r, pred_full[:, idx], linestyle='--')

            ax.legend()
            st.pyplot(fig)

            predefined_values = ', '.join(map(str, X.iloc[-1].values))

            st.write("### Predict Crimes for 2026")

            test_input = st.text_input("Enter values (comma separated)", predefined_values)

            if st.button("Predict 2026 Crime Count"):

                try:
                    test_values = list(map(float, test_input.split(',')))

                    if len(test_values) != len(columns):
                        st.error(f"Enter {len(columns)} values")
                    else:
                        test_df = pd.DataFrame([test_values], columns=columns)
                        predictions = model.predict(test_df)

                        if len(predictions.shape) == 1:
                            predictions = predictions.reshape(1, -1)

                        for i, target in enumerate(target_options):
                            st.write(f"Predicted {target} cases in 2026 : {predictions[0, i]:.2f}")

                except ValueError:
                    st.error("Invalid input")
