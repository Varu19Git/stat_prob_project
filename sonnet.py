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
    
    if data_input:
        data = [list(map(float, line.split(','))) for line in data_input.splitlines()]
        X = np.array([point[0] for point in data]).reshape(-1, 1)
        y = np.array([point[1] for point in data])
        
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
    
    if data_input:
        data = pd.DataFrame([list(map(float, line.split(','))) for line in data_input.splitlines()]).T
        data.columns = [f"Var{i+1}" for i in range(data.shape[1])]
        
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
        lambda_param = st.slider("Lambda", 0.1, 20.0, 5.0)
        x = np.arange(0, 20)
        y = stats.poisson.pmf(x, lambda_param)
        
        fig = px.bar(x=x, y=y, title="☄️ Poisson Distribution Portal")
        fig.update_traces(marker_color='orange', marker_line_color='white',
                          marker_line_width=1.5, opacity=0.6)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"))
        st.plotly_chart(fig)

def hypothesis_lab():
    st.header("🧪 Hypothesis Lab")
    st.write("Test your theories and challenge the unknown! 🔬")

    test_type = st.selectbox("Choose your hypothesis adventure:", 
                             ["🎯 One Sample t-test", "⚖️ Two Sample t-test", "🔲 Chi-square test"])
    
    if test_type == "🎯 One Sample t-test":
        st.write("Enter comma-separated values for the sample:")
        sample_input = st.text_input("Sample values", "5.2, 6.3, 4.8, 5.5, 5.9, 6.1, 5.7")
        if sample_input:
            sample = np.array([float(x.strip()) for x in sample_input.split(',')])
            hypothesized_mean = st.number_input("Hypothesized population mean", value=5.5)
            
            t_statistic, p_value = stats.ttest_1samp(sample, hypothesized_mean)
            
            st.write(f"🔬 Results of the experiment:")
            st.write(f"🧪 t-statistic: {t_statistic:.4f}")
            st.write(f"🧫 p-value: {p_value:.4f}")
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=sample, name="Sample", marker_color="purple"))
            fig.add_hline(y=hypothesized_mean, line_dash="dash", line_color="red", 
                          annotation_text="Hypothesized Mean", annotation_position="bottom right")
            fig.update_layout(title="🎯 One Sample t-test Visualization",
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig)
    
    elif test_type == "⚖️ Two Sample t-test":
        st.write("Enter comma-separated values for two samples:")
        sample1_input = st.text_input("Sample 1 values", "5.2, 6.3, 4.8, 5.5, 5.9")
        sample2_input = st.text_input("Sample 2 values", "5.7, 6.1, 5.4, 5.8, 6.2")
        
        if sample1_input and sample2_input:
            sample1 = np.array([float(x.strip()) for x in sample1_input.split(',')])
            sample2 = np.array([float(x.strip()) for x in sample2_input.split(',')])
            
            t_statistic, p_value = stats.ttest_ind(sample1, sample2)
            
            st.write(f"🔬 Results of the experiment:")
            st.write(f"🧪 t-statistic: {t_statistic:.4f}")
            st.write(f"🧫 p-value: {p_value:.4f}")
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=sample1, name="Sample 1", marker_color="blue"))
            fig.add_trace(go.Box(y=sample2, name="Sample 2", marker_color="green"))
            fig.update_layout(title="⚖️ Two Sample t-test Visualization",
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig)
    
    elif test_type == "🔲 Chi-square test":
        st.write("Enter comma-separated values for observed and expected frequencies:")
        observed_input = st.text_input("Observed frequencies", "10, 15, 20, 25, 30")
        expected_input = st.text_input("Expected frequencies", "20, 20, 20, 20, 20")
        
        if observed_input and expected_input:
            observed = np.array([float(x.strip()) for x in observed_input.split(',')])
            expected = np.array([float(x.strip()) for x in expected_input.split(',')])
            
            chi2_statistic, p_value = stats.chisquare(observed, expected)
            
            st.write(f"🔬 Results of the experiment:")
            st.write(f"🧪 Chi-square statistic: {chi2_statistic:.4f}")
            st.write(f"🧫 p-value: {p_value:.4f}")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(1, len(observed)+1)), y=observed, name="Observed", marker_color="purple"))
            fig.add_trace(go.Bar(x=list(range(1, len(expected)+1)), y=expected, name="Expected", marker_color="orange"))
            fig.update_layout(title="🔲 Chi-square test Visualization",
                              xaxis_title="Categories",
                              yaxis_title="Frequencies",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig)

def anova_arena():
    st.header("🔬 ANOVA Arena")
    st.write("Compare and conquer in the analysis of variance battleground! ⚔️")

    st.write("Enter comma-separated values for each group (one group per line):")
    groups_input = st.text_area("Group values", 
                                "5.2, 6.3, 4.8, 5.5, 5.9\n6.1, 5.7, 6.5, 6.2, 5.9\n4.9, 5.1, 5.3, 5.0, 5.2")
    
    if groups_input:
        groups = [np.array([float(x.strip()) for x in group.split(',')]) for group in groups_input.split('\n') if group]
        
        f_statistic, p_value = stats.f_oneway(*groups)
        
        st.write(f"⚔️ Results of the ANOVA battle:")
        st.write(f"🛡️ F-statistic: {f_statistic:.4f}")
        st.write(f"🏆 p-value: {p_value:.4f}")
        
        fig = go.Figure()
        for i, group in enumerate(groups):
            fig.add_trace(go.Box(y=group, name=f"Group {i+1}", 
                                 marker_color=px.colors.qualitative.Set3[i]))
        fig.update_layout(title="🔬 ANOVA: Box Plot of Groups",
                          yaxis_title="Values",
                          boxmode="group",
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"))
        st.plotly_chart(fig)

        # Add animated comparison
        fig_comparison = go.Figure()
        for i, group in enumerate(groups):
            fig_comparison.add_trace(go.Violin(y=group, name=f"Group {i+1}", 
                                               side="positive", line_color=px.colors.qualitative.Set3[i]))
        fig_comparison.update_layout(title="🎻 ANOVA: Violin Plot Comparison",
                                     yaxis_title="Values",
                                     violinmode="overlay",
                                     plot_bgcolor="rgba(0,0,0,0)",
                                     paper_bgcolor="rgba(0,0,0,0)",
                                     font=dict(color="white"))
        fig_comparison.update_traces(opacity=0.7)
        st.plotly_chart(fig_comparison)

def visualization_vault():
    st.header("📊 Visualization Vault")
    st.write("Bring your data to life with magical visualizations! ✨")

    data_input = st.text_area("Enter your data (comma-separated values, one variable per line):",
                              "1,2,3,4,5,6,7,8,9,10\n2,4,1,5,7,3,8,6,10,9\n3,6,9,4,1,8,2,7,5,10")
    
    if data_input:
        data = pd.DataFrame([list(map(float, line.split(','))) for line in data_input.splitlines()]).T
        data.columns = [f"Var{i+1}" for i in range(data.shape[1])]
        
        plot_type = st.selectbox("Choose your visualization spell:", 
                                 ["🌈 Rainbow Scatter", "🔮 Magic Bubbles", "🌪️ Tornado Plot", "🎭 Animated Histogram"])
        
        if plot_type == "🌈 Rainbow Scatter":
            fig = px.scatter(data, x="Var1", y="Var2", color="Var3", size="Var3",
                             hover_data=["Var1", "Var2", "Var3"],
                             title="🌈 Rainbow Scatter Plot")
            fig.update_traces(marker=dict(sizemin=5))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig)
        
        elif plot_type == "🔮 Magic Bubbles":
            fig = px.scatter_3d(data, x="Var1", y="Var2", z="Var3", size="Var1", color="Var2",
                                hover_data=["Var1", "Var2", "Var3"],
                                title="🔮 3D Magic Bubbles")
            fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig)
        
        elif plot_type == "🌪️ Tornado Plot":
            fig = go.Figure()
            for col in data.columns:
                fig.add_trace(go.Scatter(x=data[col], y=data.index, name=col, mode="lines+markers"))
            fig.update_layout(title="🌪️ Tornado Plot",
                              xaxis_title="Values",
                              yaxis_title="Index",
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig)
        
        elif plot_type == "🎭 Animated Histogram":
            fig = px.histogram(data, x="Var1", animation_frame="Var2", 
                               range_x=[data["Var1"].min(), data["Var1"].max()],
                               range_y=[0, data.shape[0]],
                               title="🎭 Animated Histogram")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig)

    st.write("✨ Pro tip: Try different combinations of variables for magical results!")

if __name__ == "__main__":
    main()