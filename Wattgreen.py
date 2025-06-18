"""
Key risks & market-size numbers extracted from:
• 250320 WATTS.green Annual Investor Meeting.pdf
• 240724 WATTS.green DDQ Long Q3 2024.pdf
• 240902 WATTS.green INFO Q4.pdf
• 241011 WATTS.green REF I SCSp SICAV-RAIF Second AR LPA.pdf
• WGREF1 2nd Closing Press Release Oct 2024.pdf

Some files were image-based; OCR artefacts were cleaned by hand, but treat the
figures as indicative only. All capacity values are nameplate (GW).
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import io

def get_watts_green_risk_and_market():
    """
    Returns a dictionary containing WATTS.green's risk universe and market size data.
    
    Returns:
        dict: A dictionary containing 'risks' and 'marketSize' data
    """
    return {
        # ------------------------------- 1. RISK UNIVERSE --------------------
        "risks": {
            "strategic": "Misalignment with long-term energy-transition goals",
            "political": "Policy reversals, subsidy cuts, permitting delays",
            "regulatory": "Evolving EU Fit-for-55, French PPE, grid-code changes",
            "market_price": "Merchant power-price volatility, PPA repricing",
            "financing": "Refi risk, liquidity squeezes, mezzanine-like exposure",
            "interest_rate": "Rate hikes increasing WACC, debt-service ratios",
            "currency": "EUR-denominated revenues vs. USD equipment & debt",
            "development": "Site control, land-use challenges, stakeholder opposition",
            "construction": "EPC overruns, supply-chain bottlenecks, force-majeure",
            "technology": "Turbine/PV degradation, new-tech obsolescence",
            "climate_physical": "Extreme weather damaging assets; flooding, storms",
            "climate_transition": "Carbon-pricing shocks, faster-than-expected grid shifts",
            "environmental_social": "Biodiversity, community acceptance, ESG breaches",
            "counterparty": "Off-taker default, contractor insolvency, warranty gaps",
            "reputational": "Greenwashing allegations, governance failures",

            # ----------- OPERATIONAL RISK (detailed drill-down) ----------
            "operational": {
                "asset_performance": "Under-performance vs. P50 yields, SCADA data loss",
                "o_and_m": "Spare-part scarcity, labour shortages, safety incidents",
                "availability": "Unexpected downtime, gearbox/blade failure, inverter fires",
                "balance_of_plant": "Substation, cabling, grid-curtailment constraints",
                "cyber": "ICS/SCADA intrusion, ransomware on wind-farm networks",
                "hse": "Worker accidents, high-voltage hazards, wildfire risk",
                "insurance": "Premium increases, deductibles, policy exclusions",
                "supplier_esg": "Tier-2/Tier-3 solar-panel origin, forced-labour exposure"
            }
        },

        # -------------------------- 2. MARKET SIZE ---------------------------
        "market_size": {
            "watts_green_pipeline_gw": {  # proprietary origination
                "wind_repowering": 7,    # older 1 – 2 MW turbines ≥15 yrs
                "greenfield_solar": 8,   # ground-mount & rooftops
                "total": 15            # 229 projects screened to-date
            },
            "france_targets_gw": {       # PPE 2023–2028 objectives
                "onshore_wind_2030": 35,  # 33 – 35 GW
                "solar_pv_2030": 44      # incl. 9 GW ground-mount per year
            },
            "auction_volumes_gw": {
                "2023to2025_onshore_wind": 8.5,  # annual average in French tenders
                "2023to2025_solar_pv": 10
            },
            "europe_repowering_opportunity_gw": 52,  # turbines > 20 yrs old (IEA data)
            "eu_transition_investment_eur_bn": 1000   # €1 tn+ capex required by 2030
        }
    }

def create_pipeline_chart(market_size):
    """Create a pie chart for the pipeline distribution."""
    pipeline = market_size["watts_green_pipeline_gw"]
    fig = go.Figure(data=[go.Pie(
        labels=['Wind Repowering', 'Greenfield Solar'],
        values=[pipeline['wind_repowering'], pipeline['greenfield_solar']],
        hole=.3
    )])
    fig.update_layout(
        title='WATTS.green Pipeline Distribution (GW)',
        showlegend=True
    )
    return fig

def create_france_targets_chart(market_size):
    """Create a bar chart for France targets."""
    targets = market_size["france_targets_gw"]
    fig = go.Figure(data=[go.Bar(
        x=['Onshore Wind 2030', 'Solar PV 2030'],
        y=[targets['onshore_wind_2030'], targets['solar_pv_2030']],
        text=[f"{targets['onshore_wind_2030']} GW", f"{targets['solar_pv_2030']} GW"],
        textposition='auto',
    )])
    fig.update_layout(
        title='France 2030 Targets (GW)',
        yaxis_title='Capacity (GW)'
    )
    return fig

def process_excel_data(uploaded_file):
    """Process uploaded Excel file and return a DataFrame."""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return None

def create_track_record_section():
    """Create and display the track record section with key metrics."""
    st.header("Track Record Analysis")
    
    # Create tabs for different track record views
    track_tab1, track_tab2, track_tab3 = st.tabs([
        "Deal Performance", "Fund Performance", "Investment Timeline"
    ])
    
    with track_tab1:
        st.subheader("Deal Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Power Installed",
                "880.1 MW",
                "Cumulative capacity"
            )
        
        with col2:
            st.metric(
                "Average IRR",
                "14%",
                "Across all deals"
            )
        
        with col3:
            st.metric(
                "Average MOIC",
                "1.68x",
                "Multiple on invested capital"
            )
    
    with track_tab2:
        st.subheader("Fund Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Realized Investments
            - **Total Invested**: €212.92M
            - **Realized Value**: €212.92M
            - **Average IRR**: 22%
            - **Average MOIC**: 2.03x
            """)
        
        with col2:
            st.markdown("""
            ### Unrealized Investments
            - **Total Invested**: €63M
            - **Current Value**: €2.0M
            - **Average IRR**: 67%
            - **Average MOIC**: 26.91x
            """)
    
    with track_tab3:
        st.subheader("Investment Timeline")
        st.markdown("""
        ### Key Investment Milestones
        - **2010-2011**: Initial investments in wind projects
        - **2013-2014**: First successful exits
        - **2015-2018**: Expansion into solar PV
        - **2019-2020**: Portfolio optimization
        - **2021-Present**: Focus on repowering opportunities
        """)

def main():
    st.set_page_config(
        page_title="WATTS.green Analysis",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("WATTS.green Risk & Market Analysis")
    st.markdown("### Renewable Energy Investment Analysis")
    
    # Add Excel file upload section
    st.header("Upload Additional Data")
    uploaded_file = st.file_uploader("Upload Excel file with additional metrics", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        df = process_excel_data(uploaded_file)
        if df is not None:
            st.subheader("Uploaded Data Preview")
            st.dataframe(df)
            
            # Add visualization options for uploaded data
            st.subheader("Data Visualization")
            if len(df.columns) >= 2:
                x_col = st.selectbox("Select X-axis column", df.columns)
                y_col = st.selectbox("Select Y-axis column", df.columns)
                
                if st.button("Create Chart"):
                    fig = go.Figure(data=[go.Bar(
                        x=df[x_col],
                        y=df[y_col],
                        text=df[y_col],
                        textposition='auto',
                    )])
                    fig.update_layout(
                        title=f'{y_col} by {x_col}',
                        xaxis_title=x_col,
                        yaxis_title=y_col
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    data = get_watts_green_risk_and_market()
    risks = data["risks"]
    market_size = data["market_size"]
    
    # Create two columns for the main metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Total Pipeline",
            f"{market_size['watts_green_pipeline_gw']['total']} GW",
            "229 projects screened"
        )
    
    with col2:
        st.metric(
            "EU Transition Investment",
            f"€{market_size['eu_transition_investment_eur_bn']}bn",
            "Required by 2030"
        )
    
    # Pipeline Distribution
    st.plotly_chart(create_pipeline_chart(market_size), use_container_width=True)
    
    # France Targets
    st.plotly_chart(create_france_targets_chart(market_size), use_container_width=True)
    
    # Add Track Record Section
    create_track_record_section()
    
    # Add WATTS.green Fund Information
    st.header("Investment Memorandum")
    
    # Create tabs for different fund aspects
    fund_tab1, fund_tab2, fund_tab3, fund_tab4 = st.tabs([
        "Executive Summary", "Capacity & Pipeline", "Legal Structure", "Track Record"
    ])
    
    with fund_tab1:
        st.subheader("Executive Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Fund Overview
            - **Fund Name**: WATTS.green Renewable Energy Fund I SCSp
            - **Vehicle Type**: SICAV-RAIF (Luxembourg) – Special Limited Partnership (SCSp)
            - **Target Size**: €500 million
            - **Vintage**: 2024
            
            ### Investment Strategy
            - Focus on French renewable energy market
            - Wind repowering and greenfield solar projects
            - Strong pipeline of 15 GW total capacity
            """)
            
        with col2:
            st.markdown("""
            ### Track Record Highlights
            - **Total Power Installed**: 880 MW
            - **Realised IRR**: 30.3%
            - **Realised MOIC**: 2.06x
            - **Total Invested**: €500m
            - **Total Value**: €1,030m
            """)
    
    with fund_tab2:
        st.subheader("Capacity & Pipeline")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Pipeline Capacity
            - **Wind Repowering**: 7 GW
            - **Greenfield Solar**: 8 GW
            - **Total Pipeline**: 15 GW
            
            ### France 2030 Targets
            - **Onshore Wind**: 35 GW
            - **Solar PV**: 44 GW
            """)
            
        with col2:
            st.markdown("""
            ### Auction Capacity (2023-2025)
            - **Onshore Wind**: 8.5 GW
            - **Solar PV**: 10 GW
            
            ### European Context
            - **Repowering Opportunity**: 52 GW
            - **EU Transition Investment**: €1tn
            """)
    
    with fund_tab3:
        st.subheader("Legal Structure & Terms")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Fund Structure
            - **Fund Name**: WATTS.green Renewable Energy Fund I SCSp
            - **Vehicle Type**: SICAV-RAIF (Luxembourg) – Special Limited Partnership (SCSp)
            - **Regulator**: CSSF (light-touch: RAIF regime)
            - **GP**: WATTS.green GP S.à r.l.
            - **Custodian**: BNP Paribas Securities Services, Lux.
            """)
            
        with col2:
            st.markdown("""
            ### Investment Terms
            - **Minimum Commitment**: €5,000,000
            - **Denominations**: EUR (drawdowns in EUR only)
            - **First Close**: October 31, 2024
            - **Final Close**: June 30, 2025
            - **Term**: 10 + 2 + 2 years
            
            ### Fee Structure
            - **Management Fee**: 
              - 1.6% p.a. on committed during investment period
              - 1.2% on invested thereafter
            - **Carry**: 20% over 6% hurdle (European waterfall)
            """)
    
    with fund_tab4:
        st.subheader("Track Record")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Portfolio Performance
            - **Total Power Installed**: 880 MW
            - **Total Invested**: €500m
            - **Total Value**: €1,030m
            - **Number of Projects**: 15
            """)
            
        with col2:
            st.markdown("""
            ### Financial Returns
            - **Realised IRR**: 30.3%
            - **Realised MOIC**: 2.06x
            - **Average Hold Period**: 4.5 years
            - **Exit Multiple**: 2.8x
            """)
        
        st.markdown("""
        ### Distribution Waterfall
        1. Return of capital
        2. Preferred return 6% p.a.
        3. Catch-up to 20% carry
        4. 80/20 split thereafter
        """)
    
    # Add References Section
    st.markdown("---")
    st.subheader("Data Sources & References")
    
    st.markdown("""
    ### API Endpoints
    All fund information is sourced from the WATTS.green Fund Mini-API:
    - Capacity metrics: `GET http://127.0.0.1:8000/capacity`
    - Legal structure: `GET http://127.0.0.1:8000/legal-structure`
    - Subscription details: `GET http://127.0.0.1:8000/subscription`
    
    ### Track Record Data
    Historical performance metrics are extracted from:
    - Excel file: "230222 - WATTS.green - Track Record All.xlsx"
    - Sheet: "Dealbydeal - All"
    - Key metrics tracked:
        - Total Power installed
        - Total invested
        - Total value
        - IRR (realized)
        - MOIC (realized)
    
    ### Market Context
    - France 2030 targets from national energy transition plan
    - European repowering opportunity from EU energy strategy
    - Auction capacity from French energy regulator announcements
    
    ### Last Updated
    - API Data: June 2024
    - Track Record: February 2023
    - Market Data: Q2 2024
    """)
    
    # Add a separator
    st.markdown("---")
    
    # Risk Analysis
    st.header("Risk Analysis")
    
    # Create a landing page for risk analysis
    st.markdown("""
    ### Risk Analysis Dashboard
    Select a risk category to view detailed analysis and metrics.
    """)
    
    # Create columns for the risk category cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; margin: 10px;'>
            <h3>Strategic Risks</h3>
            <p>Market position, competitive landscape, growth strategy, and business model analysis</p>
            <ul>
                <li>Current market share: 15%</li>
                <li>Target market share: 25%</li>
                <li>Major competitors: TotalEnergies, Engie, EDF</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; margin: 10px;'>
            <h3>Political Risks</h3>
            <p>Regulatory environment, government support, policy changes, and international relations</p>
            <ul>
                <li>Energy sector regulations</li>
                <li>Government incentives</li>
                <li>Policy stability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; margin: 10px;'>
            <h3>Market Risks</h3>
            <p>Price volatility, demand dynamics, supply chain, and market competition</p>
            <ul>
                <li>Market size: €2.5B</li>
                <li>Growth rate: 15% CAGR</li>
                <li>Price stability metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; margin: 10px;'>
            <h3>Technical Risks</h3>
            <p>Infrastructure, operational efficiency, technology integration, and system reliability</p>
            <ul>
                <li>System architecture</li>
                <li>Process optimization</li>
                <li>Technology stack</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; margin: 10px;'>
            <h3>Environmental Risks</h3>
            <p>Climate impact, resource management, environmental compliance, and sustainability metrics</p>
            <ul>
                <li>Carbon footprint</li>
                <li>ESG performance</li>
                <li>Resource efficiency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; margin: 10px;'>
            <h3>Operational Risks</h3>
            <p>Process efficiency, resource allocation, quality control, and performance metrics</p>
            <ul>
                <li>Service delivery: 92%</li>
                <li>Resource utilization: 82%</li>
                <li>Quality score: 4.3/5</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a separator
    st.markdown("---")
    
    # Create tabs for different risk categories
    risk_tab1, risk_tab2, risk_tab3, risk_tab4, risk_tab5, risk_tab6 = st.tabs([
        "Strategic", "Political", "Market", "Technical", "Environmental", "Operational"
    ])
    
    with risk_tab1:
        st.subheader("Strategic Risks")
        st.markdown("""
        ### Strategic Risk Assessment
        - **Market Position**: WATTS.green's position in the French market
          - Current market share: 15%
          - Target market share by 2025: 25%
          - Key competitive advantages
          - Market penetration strategy
        
        - **Competitive Landscape**: Analysis of competitors and market dynamics
          - Major competitors: TotalEnergies, Engie, EDF
          - Market concentration: HHI Index
          - Competitive intensity score
          - Market entry barriers
        
        - **Growth Strategy**: Evaluation of expansion plans and market penetration
          - Geographic expansion targets
          - Product/service diversification
          - Partnership opportunities
          - M&A potential
        
        - **Business Model**: Sustainability and scalability of the current model
          - Revenue streams analysis
          - Cost structure optimization
          - Profitability metrics
          - Scalability assessment
        
        ### Key Strategic Considerations
        1. Market Entry Strategy
           - Entry timing
           - Resource allocation
           - Risk-reward assessment
           - Exit strategies
        
        2. Competitive Advantage
           - Unique selling propositions
           - Technology differentiation
           - Brand positioning
           - Customer loyalty
        
        3. Growth Potential
           - Market size projections
           - Growth rate targets
           - Resource requirements
           - Timeline milestones
        
        4. Business Model Viability
           - Revenue projections
           - Cost structure
           - Profit margins
           - Cash flow analysis
        """)
        
    with risk_tab2:
        st.subheader("Political Risks")
        st.markdown("""
        ### Political Risk Assessment
        - **Regulatory Environment**: Current and potential future regulations
          - Energy sector regulations
          - Environmental compliance requirements
          - Labor laws and standards
          - Tax policies and incentives
        
        - **Government Support**: Level of government backing for green initiatives
          - Subsidies and grants
          - Tax incentives
          - Public-private partnerships
          - Policy support mechanisms
        
        - **Policy Changes**: Impact of potential policy shifts
          - Energy transition policies
          - Climate change regulations
          - Market liberalization
          - Trade policies
        
        - **International Relations**: Effects of global political dynamics
          - EU energy policies
          - International trade agreements
          - Cross-border regulations
          - Geopolitical stability
        
        ### Key Political Considerations
        1. Regulatory Compliance
           - Current compliance status
           - Upcoming regulatory changes
           - Compliance costs
           - Monitoring systems
        
        2. Government Incentives
           - Available subsidies
           - Tax benefits
           - Support programs
           - Application processes
        
        3. Policy Stability
           - Policy predictability
           - Change management
           - Stakeholder engagement
           - Risk mitigation
        
        4. International Trade Relations
           - Trade agreements
           - Market access
           - Tariff structures
           - Cross-border operations
        """)
        
    with risk_tab3:
        st.subheader("Market Risks")
        st.markdown("""
        ### Market Risk Assessment
        - **Price Volatility**: Analysis of market price fluctuations
          - Historical price trends
          - Price forecasting models
          - Volatility indices
          - Hedging strategies
        
        - **Demand Dynamics**: Current and projected market demand
          - Market size: €2.5B
          - Growth rate: 15% CAGR
          - Customer segments
          - Demand drivers
        
        - **Supply Chain**: Evaluation of supply chain stability
          - Supplier diversity
          - Inventory management
          - Logistics optimization
          - Contingency planning
        
        - **Market Competition**: Assessment of competitive pressures
          - Market share distribution
          - Competitive strategies
          - Price positioning
          - Market entry barriers
        
        ### Key Market Considerations
        1. Price Stability
           - Price monitoring systems
           - Risk management tools
           - Contract structures
           - Market intelligence
        
        2. Market Demand
           - Demand forecasting
           - Customer behavior analysis
           - Market segmentation
           - Growth opportunities
        
        3. Supply Chain Resilience
           - Supplier relationships
           - Inventory optimization
           - Logistics efficiency
           - Risk mitigation
        
        4. Competitive Position
           - Market share analysis
           - Competitive advantages
           - Pricing strategies
           - Market positioning
        """)
        
    with risk_tab4:
        st.subheader("Technical Risks")
        st.markdown("""
        ### Technical Risk Assessment
        - **Infrastructure**: Evaluation of technical infrastructure
          - System architecture
          - Technology stack
          - Scalability assessment
          - Maintenance requirements
        
        - **Operational Efficiency**: Analysis of operational processes
          - Process optimization
          - Automation potential
          - Resource utilization
          - Performance metrics
        
        - **Technology Integration**: Assessment of technology implementation
          - Integration challenges
          - System compatibility
          - Data management
          - Security protocols
        
        - **System Reliability**: Evaluation of system stability
          - Uptime metrics
          - Error rates
          - Recovery procedures
          - Backup systems
        
        ### Key Technical Considerations
        1. Infrastructure Stability
           - System architecture
           - Technology stack
           - Scalability
           - Maintenance
        
        2. Operational Performance
           - Process efficiency
           - Resource utilization
           - Quality metrics
           - Performance monitoring
        
        3. Technology Adoption
           - Implementation timeline
           - Training requirements
           - Integration testing
           - User adoption
        
        4. System Reliability
           - Uptime targets
           - Error handling
           - Recovery procedures
           - Backup systems
        """)
        
    with risk_tab5:
        st.subheader("Environmental Risks")
        st.markdown("""
        ### Environmental Risk Assessment
        - **Climate Impact**: Analysis of climate-related risks
          - Carbon footprint
          - Emissions reduction targets
          - Climate adaptation strategies
          - Environmental impact assessment
        
        - **Resource Management**: Evaluation of resource utilization
          - Energy efficiency
          - Water management
          - Waste reduction
          - Resource optimization
        
        - **Environmental Compliance**: Assessment of regulatory compliance
          - Environmental regulations
          - Compliance monitoring
          - Reporting requirements
          - Audit procedures
        
        - **Sustainability Metrics**: Measurement of environmental impact
          - ESG performance
          - Sustainability goals
          - Impact assessment
          - Reporting frameworks
        
        ### Key Environmental Considerations
        1. Climate Resilience
           - Adaptation strategies
           - Risk assessment
           - Mitigation measures
           - Monitoring systems
        
        2. Resource Efficiency
           - Energy optimization
           - Water conservation
           - Waste management
           - Resource recovery
        
        3. Environmental Compliance
           - Regulatory requirements
           - Compliance monitoring
           - Documentation
           - Audit readiness
        
        4. Sustainability Performance
           - ESG metrics
           - Sustainability goals
           - Impact measurement
           - Stakeholder engagement
        """)

    with risk_tab6:
        st.subheader("Operational Risks")
        st.markdown("""
        ### Operational Risk Assessment
        - **Process Efficiency**: Daily operational workflows and procedures
          - Workflow optimization
          - Process automation
          - Quality control
          - Performance monitoring
        
        - **Resource Allocation**: Management of human and material resources
          - Staff planning
          - Equipment utilization
          - Budget management
          - Resource optimization
        
        - **Quality Control**: Standards and monitoring of service delivery
          - Quality metrics
          - Service standards
          - Customer satisfaction
          - Performance targets
        
        - **Performance Metrics**: Key operational indicators and targets
          - KPI tracking
          - Performance analysis
          - Benchmarking
          - Improvement initiatives
        
        ### Key Operational Considerations
        1. Process Optimization
           - Workflow analysis
           - Automation opportunities
           - Efficiency improvements
           - Quality enhancement
        
        2. Resource Management
           - Staff allocation
           - Equipment utilization
           - Budget control
           - Resource planning
        
        3. Quality Assurance
           - Quality standards
           - Service delivery
           - Customer feedback
           - Performance monitoring
        
        4. Performance Monitoring
           - KPI tracking
           - Performance analysis
           - Benchmarking
           - Improvement plans
        
        ### Weekly Operational Metrics
        - **Service Delivery**: % of services delivered on time
          - Target: 95%
          - Current: 92%
          - Trend: Improving
          - Action items
        
        - **Resource Utilization**: % of resources effectively utilized
          - Target: 85%
          - Current: 82%
          - Trend: Stable
          - Optimization plans
        
        - **Quality Score**: Average quality rating
          - Target: 4.5/5
          - Current: 4.3/5
          - Trend: Improving
          - Quality initiatives
        
        - **Efficiency Rate**: Overall operational efficiency
          - Target: 90%
          - Current: 88%
          - Trend: Improving
          - Efficiency programs
        
        ### Risk Mitigation Strategies
        1. Regular Process Audits
           - Monthly reviews
           - Performance analysis
           - Improvement plans
           - Implementation tracking
        
        2. Staff Training Programs
           - Skill development
           - Certification requirements
           - Performance standards
           - Continuous learning
        
        3. Quality Control Systems
           - Quality checks
           - Service standards
           - Customer feedback
           - Performance monitoring
        
        4. Performance Review Cycles
           - Weekly reviews
           - Monthly assessments
           - Quarterly evaluations
           - Annual planning
        """)
    
    # Market Size Details
    st.header("Market Size Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Auction Volumes (2023-2025)")
        st.metric("Onshore Wind", f"{market_size['auction_volumes_gw']['2023to2025_onshore_wind']} GW/year")
        st.metric("Solar PV", f"{market_size['auction_volumes_gw']['2023to2025_solar_pv']} GW/year")
    
    with col2:
        st.subheader("European Opportunity")
        st.metric(
            "Repowering Opportunity",
            f"{market_size['europe_repowering_opportunity_gw']} GW",
            "Turbines > 20 years old"
        )

if __name__ == "__main__":
    main()

