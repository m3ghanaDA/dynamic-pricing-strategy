import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Dynamic Pricing Strategy Dashboard",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main{
    background-color:#FFFFFF;
}

h1{
    color:#0F172A;
}

h2{
    color:#0F172A;
}

h3{
    color:#0F172A;
}

div[data-testid="metric-container"]{
    background:#F8FAFC;
    border-radius:12px;
    padding:20px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("dynamic_pricing.csv")
    return df

df = load_data()



@st.cache_data
def load_data():
    dff = pd.read_csv("dynamic_pricing_processed.csv")
    return dff

dff = load_data()


predictions_df = pd.read_csv("model_predictions.csv")



metrics_df = pd.read_csv("model_metrics.csv")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🚖 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Exploratory Data Analysis",
        "Dynamic Pricing Analysis",
        "Price Prediction",
        "Model Performance"
    ]
)


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("🚖 Dynamic Pricing Strategy Dashboard")

    st.markdown("---")

    st.write("""
This project demonstrates a Machine Learning based Dynamic Pricing Strategy
for ride-sharing services.

The dashboard includes:

- Exploratory Data Analysis
- Dynamic Pricing Analysis
- Interactive Visualizations
- Machine Learning Price Prediction
""")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", len(df))
    c2.metric("Features", df.shape[1])
    c3.metric("Vehicle Types", df["Vehicle_Type"].nunique())
    c4.metric("Average Historical Cost", f"₹ {df['Historical_Cost_of_Ride'].mean():.2f}")

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


# ==================================================
# EDA PAGE
# ==================================================

elif page == "Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.markdown("---")

    # -------------------------
    # Dataset Overview
    # -------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("Descriptive Statistics")

    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")




    st.subheader("Ride Duration vs Historical Cost")

    fig = px.scatter(
        df,
        x="Expected_Ride_Duration",
        y="Historical_Cost_of_Ride",
        color="Vehicle_Type",
        trendline="ols",
        #size="Number_of_Riders",
        hover_data=["Number_of_Drivers"],
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)



    st.subheader("Historical Ride Cost Distribution by Vehicle Type")

    fig = px.box(
    df,
    x="Vehicle_Type",
    y="Historical_Cost_of_Ride",
    color="Vehicle_Type",
    points="outliers",
    template="plotly_white",
    #title="Historical Cost of Ride Distribution by Vehicle Type"
    )

    fig.update_layout(
    xaxis_title="Vehicle Type",
    yaxis_title="Historical Ride Cost (₹)",
    showlegend=False,
    title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)




    st.subheader("Vehicle Type Distribution")

    vehicle = (
        df["Vehicle_Type"]
        .value_counts()
        .reset_index()
    )

    vehicle.columns = ["Vehicle Type", "Count"]

    fig = px.pie(
        vehicle,
        names="Vehicle Type",
        values="Count",
        hole=0.45,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)




    st.subheader("Number of Riders Distribution")

    fig = px.histogram(
        df,
        x="Number_of_Riders",
        nbins=25,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)




    st.subheader("Number of Drivers Distribution")

    fig = px.histogram(
        df,
        x="Number_of_Drivers",
        nbins=25,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)




    st.subheader("Historical Ride Cost Distribution")

    fig = px.histogram(
        df,
        x="Historical_Cost_of_Ride",
        nbins=30,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)




    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)









# =====================================================
# DYNAMIC PRICING DASHBOARD
# =====================================================

elif page == "Dynamic Pricing Analysis":

    st.title("📈 Dynamic Pricing Dashboard")

    st.markdown("---")



    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
    "Average Historical Cost",
    f"₹ {dff['Historical_Cost_of_Ride'].mean():.2f}"
    )

    c2.metric(
    "Average Adjusted Cost",
    f"₹ {dff['adjusted_ride_cost'].mean():.2f}"
    )

    c3.metric(
    "Average Profit %",
    f"{dff['profit_percentage'].mean():.2f}%"
    )

    c4.metric(
    "Maximum Profit %",
    f"{dff['profit_percentage'].max():.2f}%"
    )



    st.markdown("---")

    st.subheader("Expected Ride Duration vs Adjusted Ride Cost")

    fig = px.scatter(
    dff,
    x="Expected_Ride_Duration",
    y="adjusted_ride_cost",
    color="Vehicle_Type",
    color_continuous_scale="RdYlGn",
    #size="Number_of_Riders",
    trendline="ols",
    hover_data=[
        "Historical_Cost_of_Ride",
        "profit_percentage",
        "Number_of_Drivers"
    ],
    template="plotly_white"
    )

    fig.update_layout(
    xaxis_title="Expected Ride Duration",
    yaxis_title="Adjusted Ride Cost",
    legend_title="Vehicle Type"
    )

    st.plotly_chart(fig, use_container_width=True)



    st.subheader("Profit Percentage Distribution")

    fig = px.histogram(
    dff,
    x="profit_percentage",
    nbins=30,
    template="plotly_white"
    )

    st.plotly_chart(fig,use_container_width=True)




    st.subheader("Demand Multiplier")

    fig = px.histogram(
    dff,
    x="demand_multiplier",
    nbins=25,
    color_discrete_sequence=["green"],
    template="plotly_white"
    )

    st.plotly_chart(fig,use_container_width=True)




    st.subheader("Supply Multiplier")

    fig = px.histogram(
    dff,
    x="supply_multiplier",
    nbins=25,
    color_discrete_sequence=["orange"],
    template="plotly_white"
    )

    st.plotly_chart(fig,use_container_width=True)




    st.subheader("Top 20 Highest Dynamic Prices")

    top=dff.sort_values(
    "adjusted_ride_cost",
    ascending=False
    ).head(20)

    fig=px.bar(
    top,
    x=top.index,
    y="adjusted_ride_cost",
    color="Vehicle_Type",
    template="plotly_white"
    )

    st.plotly_chart(fig,use_container_width=True)




    st.subheader("Average Ride Cost by Vehicle Type")

    vehicle=dff.groupby(
    "Vehicle_Type"
    )["adjusted_ride_cost"].mean().reset_index()

    fig=px.bar(
    vehicle,
    x="Vehicle_Type",
    y="adjusted_ride_cost",
    color="Vehicle_Type",
    template="plotly_white"
    )

    st.plotly_chart(fig,use_container_width=True)




    st.subheader("Profit vs Loss")

    profit=dff.copy()

    profit["Status"]=profit["profit_percentage"].apply(
    lambda x:"Profit" if x>0 else "Loss"
    )

    status=profit["Status"].value_counts().reset_index()

    status.columns=["Status","Count"]

    fig=px.pie(
    status,
    names="Status",
    values="Count",
    hole=.55,
    template="plotly_white"
    )

    st.plotly_chart(fig,use_container_width=True)




# =====================================================
# PRICE PREDICTION
# =====================================================

elif page == "Price Prediction":

    st.title("🤖 Dynamic Ride Fare Prediction")

    st.markdown(
        "Predict the **dynamic ride price** based on ride demand, driver availability, vehicle type and expected ride duration."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        riders = st.number_input(
            "Number of Riders",
            min_value=1,
            max_value=200,
            value=50
        )

        drivers = st.number_input(
            "Number of Drivers",
            min_value=1,
            max_value=200,
            value=30
        )

    with col2:

        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["Economy", "Premium"]
        )

        duration = st.number_input(
            "Expected Ride Duration (minutes)",
            min_value=1,
            max_value=180,
            value=25
        )

    if vehicle_type == "Premium":
        vehicle = 1
    else:
        vehicle = 0

    st.markdown("---")

    if st.button("Predict Ride Price", use_container_width=True):

        prediction = model.predict(
            [[
                riders,
                drivers,
                vehicle,
                duration
            ]]
        )

        predicted_price = prediction[0]

        st.success("Prediction Successful!")

        c1, c2, c3 = st.columns(3)

        c1.metric(
        "Predicted Ride Price",
        f"₹ {predicted_price:.2f}"
        )

        average_price = df["Historical_Cost_of_Ride"].mean()

        difference = predicted_price - average_price

        c2.metric(
        "Average Historical Price",
        f"₹ {average_price:.2f}"
        )

        c3.metric(
        "Difference",
        f"₹ {difference:.2f}"
        )



        st.markdown("---")

        if predicted_price < 300:
            category = "Low Price"

        elif predicted_price < 700:
            category = "Medium Price"

        else:
            category = "High Price"

        st.info(f"### Estimated Price Category: **{category}**")



        fig = go.Figure(
        go.Indicator(
        mode="gauge+number",
        value=predicted_price,
        title={"text": "Predicted Dynamic Price"},
        gauge={
            "axis": {"range": [0, max(1000, predicted_price + 100)]},
            "bar": {"color": "green"}
        }
        )
        )

        st.plotly_chart(fig, use_container_width=True)





elif page == "Model Performance":

    st.title("📉 Model Performance")

    st.markdown("Evaluate the Random Forest model using actual vs predicted ride prices.")

    st.markdown("---")

    fig = go.Figure()

    # Actual vs Predicted points
    fig.add_trace(
        go.Scatter(
            x=predictions_df["Actual"],
            y=predictions_df["Predicted"],
            mode="markers",
            name="Predicted",
            marker=dict(
                size=6,
                opacity=0.7
            )
        )
    )

    # Ideal line
    minimum = min(predictions_df["Actual"].min(),
                  predictions_df["Predicted"].min())

    maximum = max(predictions_df["Actual"].max(),
                  predictions_df["Predicted"].max())

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Ideal Prediction",
            line=dict(
                color="red",
                dash="dash",
                width=2
            )
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Ride Cost",
        xaxis_title="Actual Ride Cost (₹)",
        yaxis_title="Predicted Ride Cost (₹)",
        template="plotly_white",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)


    c1, c2, c3 = st.columns(3)

    c1.metric(
    "MAE",
    f"{metrics_df.loc[0, 'Value']:.2f}"
    )

    c2.metric(
    "RMSE",
    f"{metrics_df.loc[1, 'Value']:.2f}"
    )

    c3.metric(
    "R² Score",
    f"{metrics_df.loc[2, 'Value']:.4f}"
    )