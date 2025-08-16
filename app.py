import pandas as pd
import calendar
import streamlit as st
import random
import plotly.express as px
import plotly.graph_objects as go


# ----------------------------- SIDEBAR / USER INPUTS -----------------------------------
st.set_page_config(page_title="Advisor Fee Model Comparison", layout="wide")

st.title("Advisor Fee Models' Impact on Fund Performance")

st.expander("Welcome & Instructions", expanded=True).markdown(
    """
    This dashboard demonstrates how different advisor fee models impact your portfolio over time.

    **Key Outputs:**
    - Year-by-year balances
    - Effective annual fee rates
    - Cumulative fees by type

    **How to use:**
    1. Set your initial balance, date range.
    2. Choose contribution parameters.
    3. State market & fund assumptions.
    4. Customize fee model details.
    5. Specify utilization of each model.
    6. Explore charts and tables to compare models. You may export table data, if desired.
    """
)

# --------------------------------- CONSTANTS -------------------------------------------
# Number of Months per Year
months = list(range(1, 13))

# Tiered AUM Fee Schedule
aum_tier_df = pd.DataFrame({
    "Tier": [1, 2, 3, 4],
    "Min AUM ($)": [0, 1_000_001, 5_000_001, 10_000_001],
    "Max AUM ($)": [1_000_000, 5_000_000, 10_000_000, float('inf')],
    "Fee": [1.00, 0.75, 0.50, 0.25]
})

# ------------------------------- UI SIDEBAR -------------------------------------------

# Main UI sidebar
with st.sidebar:
    st.image("images/EyesWideOpenLogo.png", use_container_width=False, width=300)
    st.markdown("Tool developed by Eric Hubbard. More details about charging models, and more, by navigating to the URL below:")
    st.markdown("[Navigate to: KnowTheCostFinancial.com](https://knowthecostfinancial.com)")

    st.markdown("---")
    st.header("1. Beginninng Balance & Range")
    st.info("Set your starting portfolio parameters and timeline.")
    initial_balance = st.number_input("Initial Balance ($)", value=250000, step=1000)
    start_year = st.number_input("Start Year", value=2025)
    end_year = st.number_input("End Year", value=2070)


    st.markdown("---")
    st.header("2. Contributions")
    st.info("Enter how much you contribute annually and how this grows over time.")
    annual_contribution = st.number_input(
        "Annual Contribution ($)", value=15000, step=1000)
    contribution_growth_rate = st.slider(
        "Annual Contribution Growth Rate (%)", 0.0, 10.0, value=3.0) / 100
    contribution_timing = st.selectbox("When do you contribute your funds?", [
        "Beginning of Year", "Spread Evenly Over the Year", "End of Year"], index = 1) #specifies even spread as default option

    st.markdown("---")
    st.header("3. Market & Fund Assumptions")
    st.info("Define the expected market return and fund expense ratio.")
    st.sidebar.subheader("Market RoR")
    return_rate = st.slider("Average Market Return (%)", 0.0, 15.0, value=8.0) / 100

    st.sidebar.subheader("Expense Ratios (Annual %)")

    with st.expander("Expense Rations (Annual %)"):
        expense_ratio_aum = st.number_input(
            "AUM Expense Ratio (%) (likely advisor-selected mutual funds)",
            min_value=0.0, max_value=2.0, value=0.50, step=0.01,
            help="Enter the expected expense ratio for the investment type used in the AUM model."
        ) / 100

        expense_ratio_flat = st.number_input(
            "Flat Fee Model Fund Expense Ratio (%)",
            min_value=0.0, max_value=2.0, value=0.05, step=0.01,
            help="Enter the expected expense ratio for the investment type used in the Flat Fee model."
        ) / 100

        expense_ratio_hourly = st.number_input(
            "Hourly Fee Model Fund Expense Ratio (%)",
            min_value=0.0, max_value=2.0, value=0.05, step=0.01,
            help="Enter the expected expense ratio for the investment type used in the Hourly model."
        ) / 100

    st.markdown("---")
    st.header("4. Fee Models")
    st.info("Set fees for different advisor fee models.")

    with st.expander("AUM Fee Model Settings"):
        st.markdown(
            "Select the contiguous range of years when AUM fees are active:")
        aum_years_range = st.slider(
            "AUM Active Years Range",
            min_value=start_year,
            max_value=end_year,
            value=(start_year, end_year)
        )
        # Convert to list for later use
        aum_years = list(range(aum_years_range[0], aum_years_range[1] + 1))
        st.write(
            f"**AUM fees active for {len(aum_years)} years: {aum_years_range[0]} to {aum_years_range[1]}**")
        
        st.write("---")
        st.write("Update AUM Tier Structure, if desired:")

        # ---------------------------AUM Editable Table--------------------------------------------

        edit_aum_tier_df = st.data_editor(
        aum_tier_df, # Use the exising df as the starting point for the user since it contains standard values
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Tier": st.column_config.NumberColumn(
                label="Tier", min_value=1, max_value=5, step=1, format="%d"
            ),
            "Min AUM ($)": st.column_config.NumberColumn(
                label="Min AUM ($)", min_value=0, format="$%d"
            ),
            "Max AUM ($)": st.column_config.NumberColumn(
                label="Max AUM ($)", min_value=0, format="$%d"
            ),
            "Fee (%)": st.column_config.NumberColumn(
                label="Fee (%)", min_value=0.0, max_value=10.0, step=0.01, format="%.2f%%"
            ),
        }
        )
        

    # Add in information icon @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    # Show editable table to user
            
    with st.expander("Flat Fee Model Settings"):
        flat_fee = st.number_input("Flat Fee ($ per year)", value=3000, step=100)
        flat_fee_growth = st.number_input("Annual Flat Fee Growth (%)", value=3.0, step=0.1) / 100

    with st.expander("Hourly Fee Model Settings"):
        hourly_fee = st.number_input("Hourly Rate ($)", value=250, step=10)
        hourly_fee_growth = st.number_input("Annual Hourly Rate Growth (%)", value=3.0, step=0.1) / 100
        hours_per_year = st.number_input("Hours per Year", value=15, step=1)

    st.markdown("---")
    st.header("5. Utilization")
    st.info("What percentage of years do you use each advisor model?")
    perc_years_flatfee_util = st.number_input("Percentage of Years Flat Fee Advisor Used (%)", value=75, step=5, min_value = 0, max_value = 100) / 100
    perc_years_hourlyfee_util = st.number_input("Percentage of Years Hourly Fee Advisor Used (%)", value=75, step=5, min_value = 0, max_value = 100) / 100

    utilization_distribution = st.selectbox(
    "Active Utilization Years Spread (Relevant for Flat & Hourly Fee Only)",
    options=["Random", "Alternating", "Block"],
    index=0,  # default to "Random"
    help="""
    Choose how active years are spread:
    - Random: Active years are randomly scattered.
    - Block: Active years are consecutive at the start.
    - Alternating: Active years evenly spaced with gaps based on total years / active years (e.g., 20 years, 5 active years = 4 years apart).
    """
    )


# ---------------------------- FUNCTION: CALCULATE FEE MODELS --------------------------


# Determines the aum fee based on the tier the balance falls in for a given period
def calculate_aum_fee_by_tier(balance, aum_tiers):
    total_fee = 0.0
    breakdown = {}
    for _, row in aum_tiers.iterrows():
        min_aum, max_aum = row["Min AUM ($)"], row["Max AUM ($)"]
        rate = row["Fee"] / 100.0
        amount_in_tier = max(0, min(balance, max_aum) - min_aum)
        cost = amount_in_tier * rate
        breakdown[f"Tier {int(row['Tier'])}"] = cost
        total_fee += cost
    return {"total_fee": total_fee, "breakdown": breakdown}

# Enable user to determine the % of the time that they use the flat or hourly fee Structure
# User may define Random, Block, or Alternating
def build_flags(total_years, total_active, style):
    flags = [0] * total_years
    if total_active == 0:
        return flags

    if style == "Random":
        flags = [1] * total_active + [0] * (total_years - total_active)
        random.shuffle(flags)
    elif style == "Block":
        for i in range(total_active):  # For all years where it's active
            flags[i] = 1
    elif style == "Alternating":
        interval = total_years / total_active
        for i in range(total_active):
            pos = round(i * interval)
            if pos >= total_years:
                pos = total_years - 1
            flags[pos] = 1
    else:
        # Default to block for unknown styles
        for i in range(total_active):
            flags[i] = 1
    return flags

# Reference output of build_flags to create actual cost of each model based on expected model usage
def build_utilization(
        start_year, 
        end_year, 
        perc_years_flatfee_util, 
        perc_years_hourlyfee_util, 
        distribution, 
        aum_active_years
    ):

    model_utilization = []
    total_years = end_year - start_year + 1

    # Calculate active years for each model
    aum_flags = [1 if year in aum_active_years else 0 for year in range(start_year, end_year + 1)]
    total_years_active_flat = min(round(total_years * perc_years_flatfee_util), total_years)
    total_years_active_hourly = min(round(total_years * perc_years_hourlyfee_util), total_years)

    # Construct a binary list for each model
    # Calling the build flags function
    active_flat_list = build_flags(total_years, total_years_active_flat, distribution)
    # Calling the build flags function
    active_hourly_list = build_flags(total_years, total_years_active_hourly, distribution)

    # Creates a df for each model so it's easier to multiply through total cost by model
    flat_data = [{"Year": year, "Active?": active_flat_list[i]} for i, year in enumerate(range(start_year, end_year + 1))]
    hourly_data = [{"Year": year, "Active?": active_hourly_list[i]} for i, year in enumerate(range(start_year, end_year + 1))]

    df_flat = pd.DataFrame(flat_data)
    df_flat["Model"] = "Flat Fee"

    df_hourly = pd.DataFrame(hourly_data)
    df_hourly["Model"] = "Hourly Fee"

    df_aum = pd.DataFrame({"Year": list(range(start_year, end_year + 1)), "Active?": aum_flags, "Model": "AUM"})

    return pd.concat([df_flat, df_hourly, df_aum], ignore_index=True)

# Main function: generates the projections for each model based on usage function outputs and other user inputs
def generate_all_model_projections(
    initial_balance,
    start_year,
    end_year,
    annual_contribution,
    contribution_growth_rate,
    return_rate,
    expense_ratio_aum,
    expense_ratio_flat,
    expense_ratio_hourly,
    flat_fee,
    hourly_fee,
    hours_per_year,
    contribution_timing,
    util_df, # Contains the utilization of each model, by year, based on user input
    edit_aum_tier_df,
    flat_fee_growth,
    hourly_fee_growth
):
    
    models = ["AUM", "Flat Fee", "Hourly Fee"]
    current_flat_fee = flat_fee
    current_hourly_fee = hourly_fee
    all_results = []

    for model in models:
        current_balance = initial_balance
        contribution = annual_contribution

        for year in range(start_year, end_year + 1):
            # Determine active flag
            active_flag = util_df.loc[(util_df["Year"] == year) & (util_df["Model"] == model), "Active?"].values[0]

            # Initialize record
            record = {
                "Year": year,
                "Model": model,
                "Start Balance": current_balance,
                "Contribution": contribution
            }

            # Calculate fees
            if model == "AUM":
                aum_fee_info = calculate_aum_fee_by_tier(current_balance, edit_aum_tier_df)
                advisor_fee = aum_fee_info["total_fee"] * active_flag
                fund_fee = current_balance * expense_ratio_aum
                # Add tier breakdown directly into record
                record.update({tier: fee * active_flag for tier, fee in aum_fee_info["breakdown"].items()})
            elif model == "Flat Fee":
                advisor_fee = flat_fee * active_flag
                fund_fee = current_balance * expense_ratio_flat
            elif model == "Hourly Fee":
                advisor_fee = hourly_fee * hours_per_year * active_flag
                fund_fee = current_balance * expense_ratio_hourly

            # Contribution timing and interest
            if contribution_timing == "Beginning of Year":
                avg_balance = current_balance + contribution
            elif contribution_timing == "Spread Evenly Over the Year":
                avg_balance = current_balance + contribution / 2
            else:
                avg_balance = current_balance

            total_cost = advisor_fee + fund_fee
            record.update({ # Adding fee-related columns to existing dictionary base
                "Advisor Fee": advisor_fee,
                "Fund Fee": fund_fee,
                "Total Cost": total_cost,
                "Average Balance": avg_balance,
                "Effective Fee Rate": total_cost / avg_balance
                })
            
            interest = avg_balance * return_rate
            end_balance = current_balance + contribution + interest - advisor_fee - fund_fee
            record["Interest"] = interest
            record["End Balance"] = end_balance
            # Could also use: record.update({"Interest": interest, "End Balance": end_balance}) # Adds multiple keys to dictionary at once

            # Round all numeric values
            record = {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in record.items()}

            # Append once
            all_results.append(record)

            # Update balances and contributions for next year
            current_balance = end_balance
            contribution *= (1 + contribution_growth_rate)
            if model == "Flat Fee":
                flat_fee *= (1 + flat_fee_growth)
            elif model == "Hourly Fee":
                hourly_fee *= (1 + hourly_fee_growth)

    return pd.DataFrame(all_results)

# ----------------------------- CALL FUNCTIONS W/STREAMLIT UI ARGUMENTS  ---------------------------------------
# Call build_utilization
util_df = build_utilization(
    start_year,
    end_year,
    perc_years_flatfee_util,
    perc_years_hourlyfee_util,
    distribution=utilization_distribution,
    aum_active_years=aum_years
)

# Call projected_df
projected_df = generate_all_model_projections(
    initial_balance,
    start_year,
    end_year,
    annual_contribution,
    contribution_growth_rate,
    return_rate,
    expense_ratio_aum,
    expense_ratio_flat,
    expense_ratio_hourly,
    flat_fee,
    hourly_fee,
    hours_per_year,
    contribution_timing,
    util_df,
    edit_aum_tier_df,
    flat_fee_growth,
    hourly_fee_growth
)

# ------------------------------------- MAIN DATAFRAME UPDATES -------------------------------------------

# Merge the projected df (main) with the utilization - this makes it so we can calculate the realized advisor fee based on selected use
merged_df = projected_df.merge(util_df, on=['Year', 'Model'], how='left')

print(merged_df.head())

# Account for the adjusted cost (whether or not a flat/hourly advisor is used in a given year)
merged_df["Adjusted Advisor Fee"] = merged_df["Advisor Fee"] * merged_df["Active?"]
merged_df["Adjusted Total Cost"] = merged_df["Total Cost"] * merged_df["Active?"]

# Add cumulative columns to the merged_df - these will be used later on
# Compute cumulative totals for each model and fee type
merged_df["Adjusted Advisor Fee Cumulative"] = merged_df.groupby("Model")["Adjusted Advisor Fee"].cumsum()
merged_df["Fund Fee Cumulative"] = merged_df.groupby("Model")["Fund Fee"].cumsum()

# ------------------------------------ CUSTOM DATAFRAMES --------------------------------------------------

# Create separate DataFrames for easier comparison based on the aggregated df
df_aum = merged_df[merged_df["Model"] == "AUM"][["Year", "End Balance"]].rename(columns={"End Balance": "AUM"})
df_flat = merged_df[merged_df["Model"] == "Flat Fee"][["Year", "End Balance"]].rename(columns={"End Balance": "Flat"})
df_hourly = merged_df[merged_df["Model"] == "Hourly Fee"][["Year", "End Balance"]].rename(columns={"End Balance": "Hourly"})

# Merge into one df to compare fees between models (absolute and % differences)
comparison_df = df_aum.merge(df_flat, on="Year").merge(df_hourly, on="Year")

# Create new columns comparing absolute model performance differences
comparison_df["AUM_vs_Flat"] = comparison_df["AUM"] - comparison_df["Flat"]
comparison_df["AUM_vs_Hourly"] = comparison_df["AUM"] - comparison_df["Hourly"]
comparison_df["Flat_vs_Hourly"] = comparison_df["Flat"] - comparison_df["Hourly"]

# Create new columns comparing % model performance differences
comparison_df["AUM_vs_Flat_%"] = (comparison_df["AUM_vs_Flat"] / comparison_df["Flat"]) * 100
comparison_df["AUM_vs_Hourly_%"] = (comparison_df["AUM_vs_Hourly"] / comparison_df["Hourly"]) * 100
comparison_df["Flat_vs_Hourly_%"] = (comparison_df["Flat_vs_Hourly"] / comparison_df["Hourly"]) * 100


# -------------------------------------------- VISUALIATIONS ----------------------------------------------

## --- UI Year Selector for Container Metrics

unique_years = sorted(merged_df["Year"].unique())
st.subheader("Evaluate in Year:")
selected_year = st.selectbox("", unique_years)

st.header("Annual & Total Costs by Year, Model")

## --- Metrics by Model (cards)
year_df = merged_df[merged_df["Year"] == selected_year]

# Create 3 card columns, one per model
metrics_cols = st.columns(3)
for col, model in zip(metrics_cols, ["AUM", "Flat Fee", "Hourly Fee"]):
    # Container for each model
    container = col.container(border=True)
    # Filter for the selected model/year
    year_row = year_df.loc[year_df["Model"] == model]
    # Assign annual values
    balance = year_row["End Balance"].values[0]
    advisor_fee_annual = year_row["Adjusted Advisor Fee"].values[0]
    fund_fee_annual = year_row["Fund Fee"].values[0]
    # Cumulative values
    advisor_fee_cum = year_row["Adjusted Advisor Fee Cumulative"].values[0]
    fund_fee_cum = year_row["Fund Fee Cumulative"].values[0]
    # Model title
    container.subheader(model)
    # Balance metric
    container.metric(f"End Balance ({selected_year})", f"${balance:,.2f}")
    # Advisor Fees: Annual + Cumulative in same row
    a_col1, a_col2 = container.columns(2)
    a_col1.metric(f"Advisor Fee ({selected_year})", f"${advisor_fee_annual:,.2f}")
    a_col2.metric("Advisor Fee (Total)", f"${advisor_fee_cum:,.2f}")
    # Fund Fees: Annual + Cumulative in same row
    f_col1, f_col2 = container.columns(2)
    f_col1.metric(f"Fund Fee ({selected_year})", f"${fund_fee_annual:,.2f}")
    f_col2.metric("Fund Fee (Total)", f"${fund_fee_cum:,.2f}")
    


# Prepare data for the selected year
year_row = year_df.loc[year_df["Year"] == selected_year]

models = ["AUM", "Flat Fee", "Hourly Fee"]
metrics = ["End Balance " + f"({selected_year})", 
           "Advisor Fee " + f"Year ({selected_year})", 
           "Advisor Fee (Total)", 
           "Fund Fee " + f"({selected_year})", 
           "Fund Fee (Total)"]

table_data = []
for model in models:
    row = year_row.loc[year_row["Model"] == model]
    table_data.append([
        f"${row['End Balance'].values[0]:,.0f}",
        f"${row['Adjusted Advisor Fee'].values[0]:,.0f}",
        f"${row['Adjusted Advisor Fee Cumulative'].values[0]:,.0f}",
        f"${row['Fund Fee'].values[0]:,.0f}",
        f"${row['Fund Fee Cumulative'].values[0]:,.0f}",
    ])

with st.expander("Tabular View of Costs"):
    fig5 = go.Figure(data=[go.Table(
        header=dict(values=["Model"] + metrics,
                    fill_color="#333333",
                    font=dict(color="white", size=15),
                    align="center", 
                    height = 26),
        cells=dict(values=[models] + list(zip(*table_data)),
                fill_color="#f5f5f5",
                font=dict(color="black", size=15),
                align="center",
                height = 24))
    ])
    
    # Remove extra space around the table
    fig5.update_layout(
        autosize = True,
        margin=dict(l=0, r=0, t=0, b=0),  # left, right, top, bottom
        height=210
    )

    st.plotly_chart(fig5, use_container_width=True)

st.markdown("----")

## --- Stacked bar charts by model, year

# Merged_df has : Year, Model, Advisor Fee, Fund Fee
fee_types = ["Advisor Fee", "Fund Fee"]
# Convert to long format so Fee Type is a column
long_df = merged_df.melt(
    id_vars=["Year", "Model"],
    value_vars = fee_types,
    var_name="Fee Type",
    value_name="Cost"
)

# Filter per model and make one chart per column
models = long_df["Model"].unique()
cols = st.columns(len(models))

for col, model in zip(cols, models):
    model_data = long_df[long_df["Model"] == model]
    fig = px.bar(
        model_data,
        x="Year",
        y="Cost",
        color="Fee Type",
        barmode="stack",
        title=f"{model} - Annual Fees",
        color_discrete_sequence=["#1D4E89", "#F5A623"]  # colors
    )

    # Match the refined style from your line charts
    fig.update_layout(
        paper_bgcolor="rgb(245, 245, 245)",
        plot_bgcolor="rgb(245, 245, 245)", 
        title_font_size=28,
        font=dict(size=14, color="white"),
        legend=dict(
            title="Fee Type",
            font=dict(size=12, color="black"),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=40, r=30, t=40, b=40)
    )

    col.plotly_chart(fig, use_container_width=True)

## --- Display Data Tables for Each Model

st.header("Detailed Year-by-Year Results")
with st.expander("Expand to View Detailed Tables"):
    tables_cols = st.columns(3)
    for col, model in zip(tables_cols, ["AUM", "Flat Fee", "Hourly Fee"]):
        model_df = merged_df[merged_df["Model"] == model].copy()
        # Duplicative since there's a title for each
        model_df = model_df.drop(columns=["Model", "Total Cost"])
        col.write(f"**{model} Model**")
        col.dataframe(model_df, hide_index=True)

## --- Line Chart: End Balance by Model, Year

end_balance_chart_data = merged_df.pivot(
    index="Year",
    columns="Model",
    values="End Balance"
).reset_index()

end_balance_melt = end_balance_chart_data.melt(
    id_vars="Year",
    value_vars=["AUM", "Flat Fee", "Hourly Fee"],
    var_name="Model",
    value_name="End Balance"
)

fig_end_balance = px.line(
    end_balance_melt,
    x="Year",
    y="End Balance",
    color="Model",
    markers=True,
    title="  End Balances by Model"
)

fig_end_balance.update_layout(
    plot_bgcolor ="rgb(245, 245, 245)",
    paper_bgcolor="rgb(245, 245, 245)",
    title_font_size=28,
    font=dict(size=14, color="black"),
    legend_title_text="Model"
)

st.plotly_chart(fig_end_balance, use_container_width=True)

## --- AUM Fee by Tier Stacked Bar Chart

# Define list of aum tiers
tier_cols = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]

aum_long_df = merged_df.melt(
    id_vars = ["Year", "Model"],
    value_vars = tier_cols,
    var_name = "Tier",
    value_name ="Cost"
)

# Filter for AUM only
aum_long_df = aum_long_df[aum_long_df["Model"] == "AUM"]

# Plotly stacked bar chart
fig3 = px.bar(
    aum_long_df,
    x="Year",
    y="Cost",
    color="Tier",
    title="   AUM Fee Tier, Year",
    labels={"Cost": "Fee ($)", "Year": "Year"},
    color_discrete_sequence=["#1D4E89", "#F5A623", "#27AE60","#8E44AD"]  # colors
)

fig3.update_layout(
    barmode="stack",
    paper_bgcolor="rgb(245, 245, 245)",
    plot_bgcolor="rgb(245, 245, 245)",
    title_font_size=28,
    font=dict(size=14, color="black"),
    legend_title_text="Model",

    xaxis=dict(
        showgrid=True,
        gridcolor='white',
        tickfont=dict(color='black'),
        title_font=dict(color='black')
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='white',
        tickfont=dict(color='black'),
        title_font=dict(color='black')
    ),
    legend=dict(
        font=dict(color='black')        
    ),
    title_font=dict(
        color='black',
        size=28,
    )       # Chart title
)
st.plotly_chart(fig3, use_container_width=True)

# Ensure Effective Fee Rate uses the Average Balance from projections
merged_df["Effective Fee Rate"] = (
    merged_df["Adjusted Advisor Fee"] + merged_df["Fund Fee"]
) / merged_df["Average Balance"]

# Plot stepped effective fee rate
fig_eff_rate = px.line(
    merged_df,
    x="Year",
    y="Effective Fee Rate",
    color="Model",
    line_shape="hv",  # horizontal-vertical steps
    title="   Effective Annual Fee Rate",
)

fig_eff_rate.update_layout(
    title="   Effective Annual Fee Rate (Stepped)",
    plot_bgcolor="rgb(245, 245, 245)",  # very light grey
    paper_bgcolor="rgb(245, 245, 245)",
    title_font_size=28,
    xaxis=dict(
        title="Year",
        tickmode="linear",  # evenly spaced ticks
        dtick=1,            # step size of 1 year
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
    ),
    yaxis=dict(
        title="Effective Rate",
        tickformat=".2%"  # format as percentage with 2 decimals
    )
)

# Show chart in Streamlit
st.plotly_chart(fig_eff_rate, use_container_width=True)

## --- Line Chart: % Difference in End Balances by Model

diff_df_melt = comparison_df.melt(
    id_vars="Year",
    value_vars=["AUM_vs_Flat_%", "AUM_vs_Hourly_%", "Flat_vs_Hourly_%"],
    var_name="Comparison",
    value_name="Percent Difference"
    )

fig_diff = px.line(
    diff_df_melt,
    x="Year",
    y="Percent Difference",
    color="Comparison",
    markers=True,  # optional: show points
    title="  % Difference in End Balances, by Model",
)

fig_diff.update_layout(
    plot_bgcolor="rgb(245, 245, 245)",  # very light grey
    paper_bgcolor="rgb(245, 245, 245)",
    title_font_size=28,
    font=dict(family="Arial", size=14, color="black"),
    legend_title_text="Comparison"
)

st.plotly_chart(fig_diff, use_container_width=True)