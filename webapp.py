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


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Crime Against Women Prediction",
    page_icon="Fevicon.png",
    layout="centered"
)


# ---------------- BACKGROUND ---------------- #

def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://wallpaperboat.com/wp-content/uploads/2019/10/free-website-background-01.jpg");
            background-size: cover
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()


# ---------------- SESSION ---------------- #

if "login_status" not in st.session_state:
    st.session_state.login_status = False

if "role" not in st.session_state:
    st.session_state.role = None


# ---------------- SIDEBAR ---------------- #

menu = st.sidebar.selectbox("Navigate", ["Home", "Register", "Login"])

# Logout button
if st.session_state.login_status:
    if st.sidebar.button("Logout"):
        st.session_state.login_status = False
        st.session_state.role = None
        st.rerun()


# ---------------- HOME ---------------- #

if menu == "Home":

    st.title("Crime Against Women Prediction Using Machine Learning")

    st.image("https://cdn.dribbble.com/users/1787323/screenshots/11073040/data_visualization.gif")

    st.markdown("""
    ### Project Overview
    
    This application predicts crimes against women using machine learning models.
    """)


# ---------------- REGISTER ---------------- #

elif menu == "Register":

    st.title("User Registration")

    with st.form("register_form"):

        name = st.text_input("Name")
        city = st.text_input("City")
        Pincode = st.text_input("Pincode")
        email = st.text_input("Email")
        mobile = st.text_input("Mobile")

        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        submitted = st.form_submit_button("Register")

        if submitted:

            if not all([name, city, Pincode, email, mobile, password, confirm_password]):
                st.error("Please fill all fields")

            elif password != confirm_password:
                st.error("Passwords do not match")

            else:

                try:

                    c.execute(
                        "INSERT INTO users (name, city, Pincode, email, mobile, password) VALUES (?, ?, ?, ?, ?, ?)",
                        (name, city, Pincode, email, mobile, password)
                    )

                    conn.commit()

                    st.success("Registration successful. Please login.")

                except sqlite3.IntegrityError:

                    st.error("Email already registered")


# ---------------- LOGIN ---------------- #

elif menu == "Login":

    st.title("Login")

    login_email = st.text_input("Email")
    login_password = st.text_input("Password", type="password")

    if st.button("Login"):

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

                st.error("Invalid credentials")


    # ---------------- ADMIN DASHBOARD ---------------- #

    if st.session_state.login_status:

        # ---------------- ADMIN DASHBOARD ---------------- #

        if st.session_state.role == "admin":

            st.success("Logged In as Admin")
            st.subheader("Admin Dashboard")

            if st.button("Fetch All Users"):

                c.execute("SELECT name, city, Pincode, email, mobile FROM users")
                users = c.fetchall()

                if users:

                    df = pd.DataFrame(
                        users,
                        columns=[
                            "Name",
                            "City",
                            "Pincode",
                            "Email",
                            "Mobile"
                        ]
                    )

                    st.dataframe(df)

                else:
                    st.info("No users found")

            st.subheader("Delete User")

            del_email = st.text_input("Enter Email to Delete User")

            if st.button("Delete User"):

                if del_email:

                    c.execute("DELETE FROM users WHERE email=?", (del_email,))
                    conn.commit()

                    st.success("User Deleted Successfully")

                else:
                    st.warning("Please enter email")


        # ---------------- USER DASHBOARD ---------------- #

        elif st.session_state.role == "user":

            st.success("Logged in Successfully")

            st.header("Crime Against Women Prediction Using Machine Learning")

            data_file = st.file_uploader(
                "Upload Crime Data CSV",
                type=["csv"]
            )

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

                    selected_state = st.selectbox(
                        "Select State",
                        options=data[state_column].unique()
                    )

                    data = data[data[state_column] == selected_state]

                st.write("### Processed Data")
                st.write(data.head())

                year_column = 'Year' if 'Year' in data.columns else None

                column_options = list(data.columns[2:])

                columns = st.multiselect(
                    "Select Crime Features",
                    options=column_options,
                    default=column_options[:5]
                )

                target_options = st.multiselect(
                    "Select Target Variables",
                    options=column_options,
                    default=[column_options[0]]
                )

                X = data[columns]
                y = data[target_options]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model_names = [
                    "Linear Regression",
                    "Random Forest",
                    "Decision Tree",
                    "Gradient Boosting"
                ]

                selected_model = st.selectbox("Select Model", model_names)

                models = {

                    "Linear Regression": LinearRegression(),

                    "Random Forest": RandomForestRegressor(
                        max_depth=10,
                        n_estimators=10,
                        random_state=0
                    ),

                    "Decision Tree": DecisionTreeRegressor(),

                    "Gradient Boosting": GradientBoostingRegressor()

                }

                model = models[selected_model]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                st.write(f"### Model : {selected_model}")

                for i, target in enumerate(target_options):

                    st.write(
                        f"**{target} R2 Score:** {r2_score(y_test[target], y_pred[:, i]):.4f}"
                    )

                    st.write(
                        f"**{target} MSE:** {mean_squared_error(y_test[target], y_pred[:, i]):.4f}"
                    )

                    st.write(
                        f"**{target} MAE:** {mean_absolute_error(y_test[target], y_pred[:, i]):.4f}"
                    )

                    st.write(
                        f"**{target} RMSE:** {np.sqrt(mean_squared_error(y_test[target], y_pred[:, i])):.4f}"
                    )

                fig, ax = plt.subplots(figsize=(10,6))

                if year_column:

                    years = data[year_column]

                    for target in target_options:

                        ax.plot(
                            years,
                            data[target],
                            marker='o',
                            label=f'Actual {target}'
                        )

                        ax.plot(
                            years,
                            model.predict(X)[:, target_options.index(target)],
                            linestyle='--',
                            label=f'Predicted {target}'
                        )

                    ax.set_xlabel("Year")

                else:

                    r = np.arange(len(X))

                    for target in target_options:

                        ax.plot(r, data[target], label=f'Actual {target}')

                        ax.plot(
                            r,
                            model.predict(X)[:, target_options.index(target)],
                            linestyle='--',
                            label=f'Predicted {target}'
                        )

                ax.set_ylabel("Number of Cases")
                ax.set_title("Actual vs Predicted")
                ax.legend()

                st.pyplot(fig)

                predefined_values = ', '.join(map(str, X.iloc[-1].values))

                st.write("### Predict Crimes for 2025")

                test_input = st.text_input(
                    "Enter values (comma separated)",
                    predefined_values
                )

                if st.button("Predict 2025 Crime Count"):

                    try:

                        test_values = list(map(float, test_input.split(',')))

                        if len(test_values) != len(columns):
                            st.error(f"Enter {len(columns)} values")

                        else:

                            test_df = pd.DataFrame([test_values], columns=columns)

                            predictions = model.predict(test_df)

                            for i, target in enumerate(target_options):

                                st.write(
                                    f"Predicted {target} cases in 2025 : {predictions[0, i]:.2f}"
                                )

                    except ValueError:

                        st.error("Invalid input")
