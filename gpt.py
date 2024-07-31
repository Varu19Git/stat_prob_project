import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

def load_lottieurl(url):
    if LOTTIE_AVAILABLE:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    return None

def set_custom_style():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
    }
    .main {
        background-color: transparent !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
    }
    h1, h2, h3, p {
        color: white !important;
    }
    .stPlotlyChart {
        background-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="🧮 Math Magic IV", layout="wide")
    set_custom_style()

    st.title("🧮 Math Magic IV: Statistics & Probability Wonderland")

    menu = ["🏠 Home", "📈 Regression Wizard", "🔗 Correlation Explorer", "🎲 Probability Playground", "🧪 Hypothesis Lab", "🔬 ANOVA Arena", "📊 Visualization Vault"]
    choice = st.sidebar.selectbox("Navigate the Math Realm", menu)

    if choice == "🏠 Home":
        home()
    elif choice == "📈 Regression Wizard":
        regression_wizard()
    elif choice == "🔗 Correlation Explorer":
        correlation_explorer()
    elif choice == "🎲 Probability Playground":
        probability_playground()
    elif choice == "🧪 Hypothesis Lab":
        hypothesis_lab()
    elif choice == "🔬 ANOVA Arena":
        anova_arena()
    elif choice == "📊 Visualization Vault":
        visualization_vault()

def home():
    st.header("🌟 Welcome to the Math Magic IV Wonderland!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        Embark on a magical journey through the realms of Statistics and Probability! 🚀
        
        🧙‍♂️ Unleash your inner mathematician and explore:
        
        1. 📈 Regression Wizard - Predict the future!
        2. 🔗 Correlation Explorer - Uncover hidden connections!
        3. 🎲 Probability Playground - Master chance and fortune!
        4. 🧪 Hypothesis Lab - Test your theories!
        5. 🔬 ANOVA Arena - Compare and conquer!
        6. 📊 Visualization Vault - Bring data to life!
        
        Get ready for a colorful, interactive adventure in the world of numbers! 🌈✨
        """)
    
    with col2:
        if LOTTIE_AVAILABLE:
            lottie_url = "https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json, key="welcome")
        else:
            st.image("https://via.placeholder.com/400x300.png?text=Welcome+to+Math+Magic+IV", use_column_width=True)

def regression_wizard():
    st.header("📈 Regression Wizard")
    st.write("Predict the future with the power of regression! ✨")

    data_input = st.text_area("Enter your X and Y data (comma-separated, one pair per line):", 
                              "1,2\n2,4\n3,5\n4,4\n5,5")
    
    uploaded_file = st.file_uploader("Or upload a CSV file with 'X' and 'Y' columns", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        X = data.iloc[:, 0].values.reshape(-1, 1)
        y = data.iloc[:, 1].values
    elif data_input:
        data = [list(map(float, line.split(','))) for line in data_input.splitlines()]
        X = np.array([point[0] for point in data]).reshape(-1, 1)
        y = np.array([point[1] for point in data])
    
    if uploaded_file or data_input:
        model = LinearRegression()
        model.fit(X, y)
        
        st.write(f"🧙‍♂️ The crystal ball reveals:")
        st.write(f"✨ Intercept: {model.intercept_:.2f}")
        st.write(f"✨ Slope: {model.coef_[0]:.2f}")
        
        fig = px.scatter(x=X.flatten(), y=y, labels={'x': 'X', 'y': 'Y'}, title="🔮 Magical Regression Line")
        fig.add_traces(
            px.line(x=X.flatten(), y=model.predict(X), color_discrete_sequence=['red']).data[0]
        )
        fig.update_traces(marker=dict(size=10, color="purple", symbol="star"))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0.1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig)


def correlation_explorer():
    st.header("🔗 Correlation Explorer")
    st.write("Uncover the hidden connections in your data! 🕵️‍♀️")

    data_input = st.text_area("Enter your data (comma-separated values, one variable per line):",
                              "1,2,3,4,5\n2,4,5,4,5\n3,6,7,6,7")
    
    uploaded_file = st.file_uploader("Or upload a CSV file with columns for each variable", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    elif data_input:
        data = pd.DataFrame([list(map(float, line.split(','))) for line in data_input.splitlines()]).T
        data.columns = [f"Var{i+1}" for i in range(data.shape[1])]
    
    if uploaded_file or data_input:
        corr_matrix = data.corr()
        
        st.write("🔬 Correlation Matrix:")
        st.write(corr_matrix)
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="🌈 Correlation Heatmap")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig)

def probability_playground():
    st.header("🎲 Probability Playground")
    st.write("Master the art of chance and fortune! 🍀")

    dist_type = st.selectbox("Choose your probability adventure:", 
                             ["🔔 Normal", "🎰 Binomial", "☄️ Poisson"])
    
    if dist_type == "🔔 Normal":
        mu = st.slider("Mean", -10.0, 10.0, 0.0)
        sigma = st.slider("Standard Deviation", 0.1, 10.0, 1.0)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', 
                                 line=dict(color='purple', width=2)))
        fig.update_layout(title="🔔 Normal Distribution Magic",
                          xaxis_title="X", yaxis_title="Probability Density",
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"))
        st.plotly_chart(fig)
    
    elif dist_type == "🎰 Binomial":
        n = st.slider("Number of trials", 1, 100, 10)
        p = st.slider("Probability of success", 0.0, 1.0, 0.5)
        x = np.arange(0, n+1)
        y = stats.binom.pmf(x, n, p)
        
        fig = px.bar(x=x, y=y, title="🎰 Binomial Distribution Bonanza")
        fig.update_traces(marker_color='green', marker_line_color='white',
                          marker_line_width=1.5, opacity=0.6)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"))
        st.plotly_chart(fig)
    
    elif dist_type == "☄️ Poisson":
        lam = st.slider("Lambda (rate of occurrences)", 0.1, 10.0, 1.0)
        x = np.arange(0, 20)
        y = stats.poisson.pmf(x, lam)
        
        fig = px.bar(x=x, y=y, title="☄️ Poisson Distribution Power")
        fig.update_traces(marker_color='blue', marker_line_color='white',
                          marker_line_width=1.5, opacity=0.6)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"))
        st.plotly_chart(fig)

def hypothesis_lab():
    st.header("🧪 Hypothesis Lab")
    st.write("Test your theories with rigorous hypotheses! 🧠")

    test_type = st.selectbox("Choose your hypothesis test:", 
                             ["🔍 One-Sample T-Test", "🧑‍🤝‍🧑 Two-Sample T-Test", "🔬 Chi-Square Test"])

    if test_type == "🔍 One-Sample T-Test":
        data_input = st.text_area("Enter your sample data (comma-separated):", "2.3, 3.1, 2.9, 3.3, 3.0, 2.7, 3.2")
        
        uploaded_file = st.file_uploader("Or upload a CSV file with one column for sample data", type="csv")
        
        if uploaded_file:
            sample = pd.read_csv(uploaded_file).iloc[:, 0].values
        elif data_input:
            sample = np.array(list(map(float, data_input.split(','))))
        
        if uploaded_file or data_input:
            mu = st.number_input("Enter the population mean to test against:", value=3.0)
            t_stat, p_val = stats.ttest_1samp(sample, mu)
            
            st.write(f"🔍 T-Statistic: {t_stat:.2f}")
            st.write(f"🔍 P-Value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.write("🌟 Result: Reject the null hypothesis.")
            else:
                st.write("🌟 Result: Fail to reject the null hypothesis.")
    
    elif test_type == "🧑‍🤝‍🧑 Two-Sample T-Test":
        data_input1 = st.text_area("Enter your first sample data (comma-separated):", "2.3, 3.1, 2.9, 3.3, 3.0")
        data_input2 = st.text_area("Enter your second sample data (comma-separated):", "2.8, 3.0, 3.1, 3.2, 2.9")
        
        uploaded_file1 = st.file_uploader("Or upload a CSV file with one column for the first sample", type="csv", key="file1")
        uploaded_file2 = st.file_uploader("Or upload a CSV file with one column for the second sample", type="csv", key="file2")
        
        if uploaded_file1 and uploaded_file2:
            sample1 = pd.read_csv(uploaded_file1).iloc[:, 0].values
            sample2 = pd.read_csv(uploaded_file2).iloc[:, 0].values
        elif data_input1 and data_input2:
            sample1 = np.array(list(map(float, data_input1.split(','))))
            sample2 = np.array(list(map(float, data_input2.split(','))))
        
        if (uploaded_file1 and uploaded_file2) or (data_input1 and data_input2):
            t_stat, p_val = stats.ttest_ind(sample1, sample2)
            
            st.write(f"🧑‍🤝‍🧑 T-Statistic: {t_stat:.2f}")
            st.write(f"🧑‍🤝‍🧑 P-Value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.write("🌟 Result: Reject the null hypothesis.")
            else:
                st.write("🌟 Result: Fail to reject the null hypothesis.")
    
    elif test_type == "🔬 Chi-Square Test":
        observed = st.text_area("Enter the observed frequencies (comma-separated):", "10, 20, 30")
        expected = st.text_area("Enter the expected frequencies (comma-separated):", "15, 15, 30")
        
        uploaded_file_observed = st.file_uploader("Or upload a CSV file with one column for observed frequencies", type="csv", key="file_observed")
        uploaded_file_expected = st.file_uploader("Or upload a CSV file with one column for expected frequencies", type="csv", key="file_expected")
        
        if uploaded_file_observed and uploaded_file_expected:
            observed_freq = pd.read_csv(uploaded_file_observed).iloc[:, 0].values
            expected_freq = pd.read_csv(uploaded_file_expected).iloc[:, 0].values
        elif observed and expected:
            observed_freq = np.array(list(map(float, observed.split(','))))
            expected_freq = np.array(list(map(float, expected.split(','))))
        
        if (uploaded_file_observed and uploaded_file_expected) or (observed and expected):
            chi2_stat, p_val = stats.chisquare(observed_freq, expected_freq)
            
            st.write(f"🔬 Chi-Square Statistic: {chi2_stat:.2f}")
            st.write(f"🔬 P-Value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.write("🌟 Result: Reject the null hypothesis.")
            else:
                st.write("🌟 Result: Fail to reject the null hypothesis.")

def anova_arena():
    st.header("🔬 ANOVA Arena")
    st.write("Compare and conquer with ANOVA! ⚔️")

    data_input = st.text_area("Enter your data (comma-separated values, one group per line):",
                              "2.3, 3.1, 2.9\n3.3, 3.0, 2.7\n3.2, 3.4, 3.5")
    
    uploaded_file = st.file_uploader("Or upload a CSV file with columns for each group", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    elif data_input:
        data = pd.DataFrame([list(map(float, line.split(','))) for line in data_input.splitlines()]).T
    
    if uploaded_file or data_input:
        f_stat, p_val = stats.f_oneway(*[data[col] for col in data.columns])
        
        st.write(f"⚔️ F-Statistic: {f_stat:.2f}")
        st.write(f"⚔️ P-Value: {p_val:.4f}")
        
        if p_val < 0.05:
            st.write("🌟 Result: Reject the null hypothesis.")
        else:
            st.write("🌟 Result: Fail to reject the null hypothesis.")

def visualization_vault():
    st.header("📊 Visualization Vault")
    st.write("Bring your data to life with stunning visualizations! 🎨")

    uploaded_file = st.file_uploader("Upload a CSV file to visualize", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Data Preview:")
        st.write(data)
        
        chart_type = st.selectbox("Choose your chart type:", ["📈 Line Chart", "📊 Bar Chart", "📉 Scatter Plot", "🗺️ Heatmap"])
        
        if chart_type == "📈 Line Chart":
            fig = px.line(data, title="📈 Line Chart")
        elif chart_type == "📊 Bar Chart":
            fig = px.bar(data, title="📊 Bar Chart")
        elif chart_type == "📉 Scatter Plot":
            fig = px.scatter(data, title="📉 Scatter Plot")
        elif chart_type == "🗺️ Heatmap":
            fig = px.imshow(data.corr(), text_auto=True, title="🗺️ Heatmap")
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0.1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
