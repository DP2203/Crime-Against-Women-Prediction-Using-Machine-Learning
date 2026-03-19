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


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Crime Against Women Prediction",
    page_icon="🚨",
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

/* Button styling */
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
             name TEXT,
             city TEXT,
             Pincode TEXT,
             email TEXT UNIQUE,
             mobile TEXT,
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


# ---------------- SIDEBAR ---------------- #
menu = st.sidebar.selectbox("", ["🏠 Home", "✨ Sign Up", "🔐 Login"])

if st.session_state.login_status:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.login_status = False
        st.session_state.role = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Crime ML System")


# ---------------- HOME ---------------- #
if menu == "🏠 Home":

    st.title("Crime Against Women Prediction Using Machine Learning")

    st.markdown("""
    ### Project Overview
    
    This application predicts crimes against women using machine learning models.
    """)


# ---------------- SIGN UP ---------------- #
elif menu == "✨ Sign Up":

    st.markdown("""
        <h1 style='text-align: center;'>✨ Create Your Account</h1>
        <p style='text-align: center; color: gray;'>Join the Crime Prediction System</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.form("signup_form"):

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("👤 Full Name")
            city = st.text_input("🏙 City")
            Pincode = st.text_input("📍 Pincode")

        with col2:
            email = st.text_input("📧 Email")
            mobile = st.text_input("📱 Mobile Number")

        st.markdown("---")

        password = st.text_input("🔒 Password", type="password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password")

        submitted = st.form_submit_button("🚀 Create Account")

        if submitted:

            if not all([name, city, Pincode, email, mobile, password, confirm_password]):
                st.error("⚠️ Please fill all fields")

            elif password != confirm_password:
                st.error("❌ Passwords do not match")

            else:
                try:
                    c.execute(
                        "INSERT INTO users (name, city, Pincode, email, mobile, password) VALUES (?, ?, ?, ?, ?, ?)",
                        (name, city, Pincode, email, mobile, password)
                    )
                    conn.commit()
                    st.success("✅ Account created successfully! Please login.")

                except sqlite3.IntegrityError:
                    st.error("⚠️ Email already registered")


# ---------------- LOGIN ---------------- #
elif menu == "🔐 Login":

    st.markdown("""
        <h1 style='text-align: center;'>🔐 Login to Your Account</h1>
        <p style='text-align: center; color: gray;'>Access your dashboard</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        login_email = st.text_input("📧 Email")

    with col2:
        login_password = st.text_input("🔒 Password", type="password")

    if st.button("🚀 Login"):

        if login_email == ADMIN_EMAIL and login_password == ADMIN_PASS:
            st.session_state.login_status = True
            st.session_state.role = "admin"

        else:
            c.execute(
                "SELECT * FROM users WHERE email=? AND password=?",
                (login_email, login_password)
            )

            user = c.fetchone()

            if user:
                st.session_state.login_status = True
                st.session_state.role = "user"
            else:
                st.error("❌ Invalid credentials")


    # ---------------- DASHBOARD ---------------- #
    if st.session_state.login_status:

        # ---------------- ADMIN ---------------- #
        if st.session_state.role == "admin":

            st.success("Logged In as Admin")
            st.subheader("👨‍💼 Admin Dashboard")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📋 View All Users"):
                    c.execute("SELECT name, city, Pincode, email, mobile FROM users")
                    users = c.fetchall()

                    if users:
                        df = pd.DataFrame(users, columns=["Name","City","Pincode","Email","Mobile"])
                        st.dataframe(df)
                    else:
                        st.info("No users found")

            with col2:
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

            data_file = st.file_uploader("Upload Crime Data CSV", type=["csv"])
            data = None

            if data_file is not None:
                data = pd.read_csv(data_file)
                st.write("### Data Preview")
                st.write(data.head())

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

                selected_model = st.selectbox("Select Model", ["Linear Regression","Random Forest","Decision Tree","Gradient Boosting"])

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(max_depth=10, n_estimators=10, random_state=0),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor()
                }

                model = models[selected_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write(f"### Model : {selected_model}")

                for i, target in enumerate(target_options):
                    st.write(f"**{target} R2 Score:** {r2_score(y_test[target], y_pred[:, i]):.4f}")
                    st.write(f"**{target} MSE:** {mean_squared_error(y_test[target], y_pred[:, i]):.4f}")
                    st.write(f"**{target} MAE:** {mean_absolute_error(y_test[target], y_pred[:, i]):.4f}")
                    st.write(f"**{target} RMSE:** {np.sqrt(mean_squared_error(y_test[target], y_pred[:, i])):.4f}")

                fig, ax = plt.subplots(figsize=(10,6))

                if year_column:
                    years = data[year_column]
                    for target in target_options:
                        ax.plot(years, data[target], marker='o', label=f'Actual {target}')
                        ax.plot(years, model.predict(X)[:, target_options.index(target)], linestyle='--')
                    ax.set_xlabel("Year")
                else:
                    r = np.arange(len(X))
                    for target in target_options:
                        ax.plot(r, data[target], label=f'Actual {target}')
                        ax.plot(r, model.predict(X)[:, target_options.index(target)], linestyle='--')

                ax.legend()
                st.pyplot(fig)

                # Prediction
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

                            for i, target in enumerate(target_options):
                                st.write(f"Predicted {target} cases in 2026 : {predictions[0, i]:.2f}")

                    except ValueError:
                        st.error("Invalid input")
