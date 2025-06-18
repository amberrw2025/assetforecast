import streamlit as st
import pandas as pd
import plotly.express as px
import copy
import math
import warnings

# Suppress FutureWarnings from plotly/pandas
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Page Configuration ---
st.set_page_config(
    page_title="Investment Impact Visualizer",
    page_icon="📈",
    layout="wide"
)

# --- Initialize Session State ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []
if 'editing_scenario_key' not in st.session_state:
    st.session_state.editing_scenario_key = None
if 'show_baseline' not in st.session_state:
    st.session_state.show_baseline = True
if 'show_live_scenario' not in st.session_state:
    st.session_state.show_live_scenario = True

# --- Helper Functions ---
def format_currency(amount):
    """Formats a number as a British Pound currency string."""
    return f"£{amount:,.0f}"

def calculate_growth_annual(initial, horizon_years, irr_perc, mgmt_fees_perc=0.0, perf_fees_perc=0.0):
    """
    Calculates investment growth on a simple, year-by-year basis.
    Calculations use two-decimal-place precision, but final results for display are rounded to whole numbers.
    Order of operations:
    1. Management fee is calculated and deducted from the start-of-year value.
    2. IRR (profit) is calculated on the remaining value.
    3. Performance fee is calculated on the profit generated in step 2.
    """
    num_years = math.floor(horizon_years)
    
    irr_annual_rate = irr_perc / 100
    mgmt_fees_annual_rate = mgmt_fees_perc / 100
    perf_fees_annual_rate = perf_fees_perc / 100

    years = [i for i in range(num_years + 1)]
    values = [round(float(initial))]
    gains_list = [0.0]
    mgmt_fees_list = [0.0]
    perf_fees_list = [0.0]

    # current_value maintains 2-decimal precision for calculations
    current_value = round(float(initial), 2)

    for year in range(1, num_years + 1):
        value_at_start_of_year = current_value

        # --- Calculations are done with two-decimal-place precision ---
        annual_mgmt_fee = round(value_at_start_of_year * mgmt_fees_annual_rate, 2)
        value_after_mgmt_fee = round(value_at_start_of_year - annual_mgmt_fee, 2)
        profit = round(value_after_mgmt_fee * irr_annual_rate, 2)
        annual_perf_fee = round(max(0, profit) * perf_fees_annual_rate, 2)
        final_value = round(value_after_mgmt_fee + profit - annual_perf_fee, 2)
        
        # --- Round the results to whole numbers just before appending for display ---
        values.append(round(final_value))
        gains_list.append(round(profit))
        mgmt_fees_list.append(round(annual_mgmt_fee))
        perf_fees_list.append(round(annual_perf_fee))
        
        # --- Carry over the 2-decimal-precision value for the next year's calculation ---
        current_value = final_value

    df = pd.DataFrame({
        'Year': years, 
        'Value': values,
        'Profit': gains_list,
        'Management Fee': mgmt_fees_list,
        'Performance Fee': perf_fees_list,
    })
    df['Total Fees'] = df['Management Fee'] + df['Performance Fee']

    if initial > 0:
        df['Multiple'] = df['Value'] / initial
    else:
        df['Multiple'] = 0
    return df

# --- UI: Title and Description ---
st.title("📈 Investment Impact Visualizer")
st.write(
    "This tool simulates long-term investment growth, showing how management and performance fees can impact your portfolio's final value. "
    "Use the sidebar to configure your investment, and add multiple scenarios to compare different fee structures and returns."
)

# --- Set Defaults for Sidebar ---
# Check if a scenario was loaded in the previous run to set widget defaults
if st.session_state.get("scenario_to_load"):
    defaults = st.session_state.scenario_to_load
    st.session_state.scenario_to_load = None # Consume the value
else:
    # Standard defaults
    defaults = { "title": "", "irr": 8.0, "mgmt_fees": 1.0, "perf_fees": 15.0 }

# --- UI: Sidebar for Inputs ---
with st.sidebar:
    st.header("Global Simulation Inputs")
    initial_investment = st.number_input(
        "Initial Investment (£)",
        min_value=100, max_value=100000, value=1000, step=100,
        help="The total amount of your initial investment. This value applies to all scenarios."
    )
    investment_horizon = st.number_input(
        "Investment Horizon (Years)",
        min_value=1.0, max_value=25.0, value=25.0, step=0.5, format="%.1f",
        help="The number of years you plan to keep the investment. This applies to all scenarios."
    )
    
    st.markdown("---")
    st.header("Live Scenario Parameters")
    st.info("Use the sliders below to configure a 'live' scenario. Its projections will update instantly on the charts.", icon="💡")
    irr = st.number_input(
        "Internal Rate of Return (IRR %)",
        min_value=0.0, max_value=40.0, value=8.0, step=0.1, format="%.1f",
        help="The estimated annual growth rate of the investment before any fees are deducted."
    )
    mgmt_fees = st.number_input(
        "Management Fees (%)",
        min_value=0.0, max_value=5.0, value=1.0, step=0.01, format="%.2f",
        help="The annual fee charged on the total value of the investment at the start of each year."
    )
    perf_fees = st.number_input(
        "Performance Fees (% of Gains)",
        min_value=0.0, max_value=50.0, value=15.0, step=0.01, format="%.2f",
        help="The fee charged on the annual profit. This is calculated after management fees have been deducted."
    )

    # --- Scenario Management UI ---
    st.markdown("---")
    st.header("Manage Scenarios")
    
    # Add toggles for baseline and live scenarios
    st.session_state.show_baseline = st.checkbox("Show Baseline (No Fees)", value=st.session_state.get('show_baseline', True), help="Toggle the visibility of the 'no fees' projection on the charts.")
    st.session_state.show_live_scenario = st.checkbox("Show Live Scenario", value=st.session_state.get('show_live_scenario', True), help="Toggle the visibility of the 'live scenario' projection on the charts.")
    
    scenario_title = st.text_input("Scenario Title", help="Give your scenario a name before saving it for comparison.")
    
    if st.button("Add Scenario to Comparison", use_container_width=True):
        # Use .strip() to handle titles that are just whitespace
        title = scenario_title.strip()
        if not title:
            # Default title format matches the input widget formats
            title = f"IRR {irr:.1f}%, Mgmt {mgmt_fees:.2f}%, Perf {perf_fees:.2f}%"

        # Enforce unique scenario titles
        if any(s['title'] == title for s in st.session_state.scenarios):
            st.error(f"A scenario with the title '{title}' already exists. Please choose a unique title.")
        else:
            new_scenario = {
                "title": title,
                "irr": irr,
                "mgmt_fees": mgmt_fees,
                "perf_fees": perf_fees,
                "key": f"{title}_{irr}_{mgmt_fees}_{perf_fees}",
                "visible": True
            }
            st.session_state.scenarios.append(new_scenario)
            st.success(f"Added '{title}'")
            st.rerun() # Rerun to clear inputs and show the new scenario card

    if st.button("Clear All Scenarios", use_container_width=True) and st.session_state.scenarios:
        st.session_state.scenarios = []

# --- Calculation Engine ---
# This dictionary will hold the dataframes for each scenario
scenario_data = {}

# Always calculate the live and baseline scenarios for the delta metrics
df_no_fees = calculate_growth_annual(initial_investment, investment_horizon, irr, 0.0, 0.0)
df_no_fees['Scenario'] = "Baseline (No Fees)"
scenario_data[df_no_fees['Scenario'].iloc[0]] = df_no_fees

df_live = calculate_growth_annual(initial_investment, investment_horizon, irr, mgmt_fees, perf_fees)
df_live['Scenario'] = "Live Scenario (updates with sliders)"
scenario_data[df_live['Scenario'].iloc[0]] = df_live

all_dfs = []

# 1. Baseline (No Fees) Scenario
if st.session_state.get('show_baseline', True):
    all_dfs.append(df_no_fees)

# 2. Live Scenario
if st.session_state.get('show_live_scenario', True):
    all_dfs.append(df_live)

# 3. Saved Scenarios
for s in st.session_state.scenarios:
    # Only include scenarios in the charts if they are marked as visible
    if s.get('visible', True):
        df_saved = calculate_growth_annual(initial_investment, investment_horizon, s['irr'], s['mgmt_fees'], s['perf_fees'])
        df_saved['Scenario'] = s['title'] # Use the custom title for the legend
        scenario_data[s['title']] = df_saved
        all_dfs.append(df_saved)

# Combine all dataframes for plotting
if all_dfs:
    master_df = pd.concat(all_dfs)
else:
    # If no scenarios are visible, create an empty dataframe to prevent crashes
    master_df = pd.DataFrame({'Year': [], 'Value': [], 'Scenario': [], 'Multiple': []})


# --- UI: Delta Metrics for Live Scenario ---
final_value_no_fees = df_no_fees["Value"].iloc[-1]
final_value_with_fees = df_live["Value"].iloc[-1]
shortfall_dollars = final_value_no_fees - final_value_with_fees
shortfall_percent = (shortfall_dollars / final_value_no_fees) * 100 if final_value_no_fees > 0 else 0

st.header("Live Scenario Projections")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Baseline Value (No Fees)", value=format_currency(final_value_no_fees))
with col2:
    st.metric(
        label="Live Scenario Value (With Fees)",
        value=format_currency(final_value_with_fees),
        delta=f"-{format_currency(shortfall_dollars)}",
        delta_color="inverse"
    )
with col3:
    st.metric(
        label="Total Fees Impact (vs. Baseline)",
        value=f"{shortfall_percent:.2f}%",
        help="The percentage of your potential final portfolio value lost to all fees in the live scenario."
    )

# --- UI: Charts and Data Tabs ---
tab_growth, tab_multiple, tab_data, tab_breakdown = st.tabs(["📈 Growth Chart", " multiplies  Multiple Chart", "📊 Detailed Data", "🔬 Annual Breakdown"])

with tab_growth:
    st.header("Growth Trajectories (Value)")
    st.info(
        "This chart displays the projected growth of your investment over time for each selected scenario. "
        "Hover over the lines to see detailed values for each year.",
        icon="ℹ️"
    )
    fig_growth = px.line(
        master_df,
        x="Year",
        y="Value",
        color="Scenario",
        title="Investment Value Over Time",
        labels={"Value": "Investment Value (£)", "Scenario": "Scenarios"},
        markers=True
    )
    fig_growth.update_traces(hovertemplate="<b>%{data.name}</b><br>Year: %{x:.0f}<br>Value: %{y:£,.0f}<extra></extra>")
    fig_growth.update_layout(
        yaxis_title="Investment Value (£)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_growth, use_container_width=True)

with tab_multiple:
    st.header("Money Multiple (MOIC)")
    st.info(
        "This chart shows the growth of your investment as a multiple of the initial capital (e.g., a value of 2.0x means your investment has doubled). "
        "This is also known as the Multiple on Invested Capital (MOIC).",
        icon="ℹ️"
    )
    fig_multiple = px.line(
        master_df,
    x="Year",
        y="Multiple",
        color="Scenario",
        title="Investment Multiple Over Time",
        labels={"Multiple": "Multiple (x)", "Scenario": "Scenarios"},
        markers=True
    )
    fig_multiple.update_traces(hovertemplate="<b>%{data.name}</b><br>Year: %{x:.0f}<br>Multiple: %{y:.2f}x<extra></extra>")
    fig_multiple.update_layout(
        yaxis_title="Multiple",
        yaxis_ticksuffix="x",
        hovermode="x unified"
    )
    st.plotly_chart(fig_multiple, use_container_width=True)

with tab_data:
    st.header("Detailed Data Table")
    st.info(
        "The tables below provide the underlying annual data for all visible scenarios, pivoted for easy comparison. "
        "You can sort the tables by clicking on the column headers.",
        icon="ℹ️"
    )

    st.subheader("Value Over Time (£)")
    pivot_value = master_df.pivot(index='Year', columns='Scenario', values='Value')
    formatters_value = {col: "£{:,.0f}" for col in pivot_value.columns}
    st.dataframe(pivot_value.style.format(formatters_value))

    st.subheader("Multiple Over Time")
    pivot_multiple = master_df.pivot(index='Year', columns='Scenario', values='Multiple')
    st.dataframe(pivot_multiple.style.format({col: "{:.2f}x" for col in pivot_multiple.columns}))

with tab_breakdown:
    st.header("Annual Profit and Fees Breakdown")

    if not all_dfs:
        st.info("No scenarios are visible. Please select a scenario to see its breakdown.", icon="ℹ️")
    else:
        st.info("These charts show the profit generated and fees paid each year for every visible scenario. This helps visualize how fees compound over time.", icon="ℹ️")

        for scenario_title in master_df['Scenario'].unique():
            if scenario_title == "Baseline (No Fees)":
                continue  # Skip the baseline as it has no fees to break down

            with st.container(border=True):
                st.subheader(f"Breakdown for '{scenario_title}'")
                
                df_breakdown = scenario_data[scenario_title]
                yearly_breakdown = df_breakdown[df_breakdown['Year'] > 0].copy()

                if not yearly_breakdown.empty:
                    df_for_plot = yearly_breakdown.melt(
                        id_vars='Year',
                        value_vars=['Profit', 'Management Fee', 'Performance Fee'],
                        var_name='Metric',
                        value_name='Amount'
                    )

                    fig_breakdown = px.bar(
                        df_for_plot,
                        x='Year',
                        y='Amount',
                        color='Metric',
                        barmode='group',
                        title=f"Annual Breakdown for {scenario_title}",
                        labels={"Amount": "Amount (£)", "Year": "Year", "Metric": "Metric"},
                        color_discrete_map={'Profit': '#2ca02c', 'Management Fee': '#d62728', 'Performance Fee': '#ff7f0e'}
                    )
                    fig_breakdown.update_layout(yaxis_title="Amount (£)", xaxis_dtick=1, title_x=0.5)
                    fig_breakdown.update_traces(hovertemplate="<b>%{data.name}</b><br>Year: %{x}<br>Amount: %{y:£,.0f}<extra></extra>")
                    st.plotly_chart(fig_breakdown, use_container_width=True)
                else:
                    st.info("No data to display for this scenario.")


# --- UI: Saved Scenarios Area ---
def set_scenario_to_load(index):
    """Callback to set the scenario to be loaded into the sidebar."""
    st.session_state.scenario_to_load = st.session_state.scenarios[index]

if st.session_state.scenarios:
    st.markdown("---")
    
    visible_scenarios = sum(1 for s in st.session_state.scenarios if s.get('visible', True))
    expander_title = f"Saved Scenarios for Comparison ({visible_scenarios} of {len(st.session_state.scenarios)} visible)"
    
    with st.expander(expander_title, expanded=True):
        for i, s in enumerate(st.session_state.scenarios):
            st.markdown("---")
            is_being_edited = (st.session_state.editing_scenario_key == s['key'])

            if is_being_edited:
                with st.form(key=f"edit_form_{s['key']}"):
                    st.subheader(f"Editing: {s['title']}")
                    new_title = st.text_input("Title", value=s['title'], key=f"edit_title_{s['key']}")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        new_irr = st.number_input("IRR (%)", value=s['irr'], min_value=0.0, max_value=40.0, step=0.1, format="%.1f", help="The estimated annual growth rate of the investment before any fees are deducted.")
                    with c2:
                        new_mgmt = st.number_input("Mgmt Fees (%)", value=s['mgmt_fees'], min_value=0.0, max_value=5.0, step=0.01, format="%.2f", help="The annual fee charged on the total value of the investment at the start of each year.")
                    with c3:
                        new_perf = st.number_input("Perf Fees (%)", value=s['perf_fees'], min_value=0.0, max_value=50.0, step=0.01, format="%.2f", help="The fee charged on the annual profit. This is calculated after management fees have been deducted.")
                    
                    form_c1, form_c2, _ = st.columns([1, 1, 4])
                    if form_c1.form_submit_button("Save Changes", use_container_width=True):
                        # Enforce unique titles on edit
                        is_duplicate = any(sc['title'] == new_title and sc['key'] != s['key'] for sc in st.session_state.scenarios)
                        if is_duplicate:
                            st.error(f"Another scenario with the title '{new_title}' already exists.")
                        else:
                            # Find the index of the scenario to update
                            idx_to_update = next((idx for (idx, d) in enumerate(st.session_state.scenarios) if d['key'] == s['key']), None)
                            if idx_to_update is not None:
                                st.session_state.scenarios[idx_to_update].update({
                                    'title': new_title, 'irr': new_irr, 'mgmt_fees': new_mgmt, 'perf_fees': new_perf,
                                    'key': f"{new_title}_{new_irr}_{new_mgmt}_{new_perf}"
                                })
                            st.session_state.editing_scenario_key = None
                            st.rerun()
                    
                    if form_c2.form_submit_button("Cancel", use_container_width=True):
                        st.session_state.editing_scenario_key = None
                        st.rerun()
            else:
                with st.container():
                    cols = st.columns([4, 1, 1])
                    with cols[0]:
                        c1, c2 = st.columns([1, 10])
                        with c1:
                            # Use a default of True for visibility if the key doesn't exist yet (for old scenarios)
                            is_visible = st.checkbox("", value=s.get('visible', True), key=f"visible_{s['key']}", help="Show/Hide this scenario on the charts")
                            if is_visible != s.get('visible', True):
                                st.session_state.scenarios[i]['visible'] = is_visible
                                st.rerun()
                        with c2:
                            st.subheader(s['title'])
                        
                        # Display metrics in a more compact way
                        p_col1, p_col2, p_col3 = st.columns(3)
                        p_col1.metric("Final Value", format_currency(calculate_growth_annual(initial_investment, investment_horizon, s['irr'], s['mgmt_fees'], s['perf_fees'])['Value'].iloc[-1]))
                        p_col2.markdown(f"**IRR:** {s['irr']:.1f}%")
                        p_col3.markdown(f"**Mgmt Fees:** {s['mgmt_fees']:.2f}%")
                        p_col3.markdown(f"**Perf Fees:** {s['perf_fees']:.2f}%")
                    
                    with cols[1]:
                        if st.button("Edit", key=f"edit_{s['key']}", use_container_width=True):
                            st.session_state.editing_scenario_key = s['key']
                            st.rerun()

                    with cols[2]:
                        if st.button("Remove", key=f"remove_{s['key']}", use_container_width=True):
                            st.session_state.scenarios.pop(i)
                            st.rerun()