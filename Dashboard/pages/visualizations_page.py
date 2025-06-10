import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('pages/datasets/Dataset_Visualization.csv')

ATTRITION_COLORS = {'Yes': '#DC143C', 'No': '#2E8B57'}

FEATURE_SECTIONS = {
    'Demographics': {
        'Age Analysis': ('Age', 'Age Group'),
        'Gender': (None, 'Gender'),
        'Education Level': (None, 'Education Level'),
        'Education Field': (None, 'Education Field'),
        'Marital Status': (None, 'Marital Status'),
        'Distance Analysis': ('Distance From Home (km)', 'Distance Group'),
    },
    'Job Profile': {
        'Department': (None, 'Department'),
        'Job Role': (None, 'Job Role'),
        'Job Level': (None, 'Job Level'),
        'Business Travel Frequency': (None, 'Business Travel Frequency'),
        'Works Overtime': (None, 'Works Overtime'),
    },
    'Career & Experience': {
        'Company Tenure': ('Years at Company', 'Company Tenure Group'),
        'Role Tenure': ('Years in Current Role', 'Role Tenure Group'),
        'Promotion Recency': ('Years Since Last Promotion', 'Promotion Recency Group'),
        'Manager Relationship': ('Years with Current Manager', 'Years With Current Manager Group'),
        'Total Working Years': ('Total Working Years', 'Experience Group'),
    },
    'Performance & Growth': {
        'Income Analysis': ('Monthly Income ($)', 'Income Group'),
        'Salary Growth': ('Salary Hike (%)', 'Salary Hike Category'),
        'Training Analysis': ('Training Times Last Year', 'Training Frequency'),
        'Stock Option Level': (None, 'Stock Option Level'),
        'Job Involvement': (None, 'Job Involvement'),
    },
    'Work Environment & Satisfaction': {
        'Job Satisfaction': (None, 'Job Satisfaction'),
        'Environment Satisfaction': (None, 'Environment Satisfaction'),
        'Relationship Satisfaction': (None, 'Relationship Satisfaction'),
        'Work-Life Balance': (None, 'Work-Life Balance'),
    },
}

ALL_FEATURES = {k: v for section in FEATURE_SECTIONS.values() for k, v in section.items()}

CATEGORY_ORDERS = {
    'Attrition Status': ['Yes', 'No'],
    'Business Travel Frequency': ['Never Travels', 'Rarely Travels', 'Frequently Travels'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'Education Level': ['High School', 'Associate', 'Bachelors', 'Masters', 'Doctorate'],
    'Education Field': ['Other', 'Marketing', 'Human Resources', 'Life Sciences', 'Medical', 'Technical Degree'],
    'Environment Satisfaction': ['Low', 'Medium', 'High', 'Very High'],
    'Gender': ['Female', 'Male'],
    'Job Involvement': ['Low', 'Medium', 'High', 'Very High'],
    'Job Level': ['Entry Level', 'Mid Level', 'Senior Level', 'Manager', 'Executive'],
    'Job Satisfaction': ['Low', 'Medium', 'High', 'Very High'],
    'Job Role': [
        'Sales Executive', 'Sales Representative', 'Laboratory Technician',
        'Research Scientist', 'Research Director', 'Manufacturing Director',
        'Healthcare Representative', 'Manager', 'Human Resources'
    ],
    'Marital Status': ['Single', 'Married', 'Divorced'],
    'Works Overtime': ['No', 'Yes'],
    'Relationship Satisfaction': ['Low', 'Medium', 'High', 'Very High'],
    'Stock Option Level': ['No Stock', 'Level 1', 'Level 2', 'Level 3'],
    'Work-Life Balance': ['Bad', 'Fair', 'Good', 'Best'],
    'Age Group': ['18–25', '26–35', '36–45', '46–55', '56-60'],
    'Income Group': ['Low (1k-5k)', 'Medium (5k-10k)', 'High (10k-15k)', 'Very High (15k-20K)'],
    'Distance Group': ['0-5', '6-10', '11-20', '21-30'],
    'Companies Worked Group': ['0–1', '2–3', '4–5', '6+'],
    'Salary Hike Category': ['Low (11–15%)', 'Medium (16–20%)', 'High (21–25%)'],
    'Experience Group': ['0–5', '6–10', '11–20', '21–30', '31–40'],
    'Training Frequency': ['Low (0–1)', 'Medium (2–3)', 'High (4–6)'],
    'Performance Rating': ['Below Average', 'Average', 'Above Average', 'Outstanding'],
    'Company Tenure Group': ['<3 yrs', '3–5 yrs', '6–10 yrs', '11–20 yrs', '21+ yrs'],
    'Role Tenure Group': ['<3 yrs', '3–5 yrs', '6–10 yrs', '11–15 yrs', '16–18 yrs'],
    'Promotion Recency Group': ['Recent (<1yr)', '1–3 yrs', '4–5 yrs', '6–10 yrs', '11–15 yrs'],
    'Years With Current Manager Group': ['<3 yrs', '3–5 yrs', '6–10 yrs', '11–15 yrs', '16–18 yrs'],
}

def filter_dataframe(df, age=None, income=None, gender=None, level=None):
    filtered_df = df.copy()
    for value, column in [(age, 'Age Group'), (income, 'Income Group'), (gender, 'Gender'), (level, 'Job Level')]:
        if value and value != 'all': filtered_df = filtered_df[filtered_df[column] == value]
    return filtered_df

def get_risk_analysis(filtered_df, selected_feature, current_filters):
    if not selected_feature or selected_feature not in ALL_FEATURES:
        return None

    _, categorical_col = ALL_FEATURES[selected_feature]
    if categorical_col not in filtered_df.columns:
        return None

    try:
        stats = filtered_df.groupby(categorical_col).agg({'Attrition Status': ['count', lambda x: (x == 'Yes').sum()]})
        stats.columns = ['Total_Count', 'Attrition_Count']
        stats['Attrition_Rate'] = (stats['Attrition_Count'] / stats['Total_Count'] * 100).round(1)
        stats = stats.reset_index()
        if not stats.empty:
            highest = stats.loc[stats['Attrition_Rate'].idxmax()]
            lowest = stats.loc[stats['Attrition_Rate'].idxmin()]
            return {'highest': (highest[categorical_col], highest['Attrition_Rate']), 'lowest': (lowest[categorical_col], lowest['Attrition_Rate'])}
    except Exception as e:
        pass

    return None


def create_attrition_summary_card(filtered_df, theme='light', selected_feature=None, current_filters=None):
    total_employees = len(filtered_df)
    attrition_count = (filtered_df['Attrition Status'] == 'Yes').sum()
    attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0

    colors = {
        'total': '#00D4FF' if theme == 'dark' else '#007bff',
        'attrition': '#FF6B6B' if theme == 'dark' else '#dc3545',
        'rate': '#FFD93D' if theme == 'dark' else '#ffc107',
        'retention': '#4ECDC4' if theme == 'dark' else '#28a745',
        'card_bg': '#2c3e50' if theme == 'dark' else '#ffffff',
        'text_primary': 'white' if theme == 'dark' else '#2c3e50',
        'text_muted': '#bdc3c7' if theme == 'dark' else '#6c757d',
        'border': '#34495e' if theme == 'dark' else '#e9ecef'
    }

    risk_info = get_risk_analysis(filtered_df, selected_feature, current_filters or {})

    metrics = [(total_employees, "Total", colors['total'], "fas fa-users"),
               (total_employees - attrition_count, "Stayed", colors['retention'], "fas fa-user-check"),
               (attrition_count, "Left", colors['attrition'], "fas fa-user-times"),
               (f"{attrition_rate:.1f}%", "Rate", colors['rate'], "fas fa-percentage")]

    metric_cards = [
        dbc.Col([
            html.Div([
                html.I(className=f"{icon} mb-1", style={'color': color, 'fontSize': '1.2rem'}),
                html.H5(f"{value:,}" if isinstance(value, int) else value, style={'color': color, 'margin': '0', 'fontSize': '1.4rem', 'fontWeight': 'bold'}),
                html.Small(label, style={'color': colors['text_muted'], 'fontSize': '0.75rem'})
            ], className="text-center p-2", style={'backgroundColor': f"{color}10", 'borderRadius': '6px', 'border': f"1px solid {color}30"})
        ], md=3, sm=6, xs=6) for value, label, color, icon in metrics]

    risk_section = None
    if risk_info:
        risk_section = dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small("Highest Risk", className="text-muted d-block", style={'fontSize': '0.7rem'}),
                    html.Strong(f"{risk_info['highest'][0]} ({risk_info['highest'][1]:.1f}%)", style={'color': colors['attrition'], 'fontSize': '1.0rem', 'fontWeight': 'bold'})
                ], className="text-center p-1", style={'backgroundColor': f"{colors['attrition']}10", 'borderRadius': '4px'})
            ], md=6),
            dbc.Col([
                html.Div([
                    html.Small("Lowest Risk", className="text-muted d-block", style={'fontSize': '0.7rem'}),
                    html.Strong(f"{risk_info['lowest'][0]} ({risk_info['lowest'][1]:.1f}%)", style={'color': colors['retention'], 'fontSize': '1.0rem', 'fontWeight': 'bold'})
                ], className="text-center p-1", style={'backgroundColor': f"{colors['retention']}10", 'borderRadius': '4px'})], md=6)], className="g-1")

    return dbc.Card([
        dbc.CardHeader([
            html.H6([
                html.I(className="fas fa-chart-pie me-2"), "Attrition Overview"], className="mb-0", style={'color': colors['text_primary']})
        ], style={'backgroundColor': colors['card_bg'], 'padding': '8px 12px'}),

        dbc.CardBody([
            dbc.Row(metric_cards, className="g-2 mb-2"),
            html.Hr(style={'margin': '8px 0', 'borderColor': colors['border']}) if risk_info else None,
            risk_section], style={'padding': '10px'})
    ], style={'backgroundColor': colors['card_bg'], 'border': f"1px solid {colors['border']}", 'borderRadius': '8px'}, className="mb-2")

def create_dual_axis_chart(filtered_df, feature_name, theme='light'):

    if feature_name not in ALL_FEATURES:
        return go.Figure()
    
    _, categorical_col = ALL_FEATURES[feature_name]
    if categorical_col not in filtered_df.columns:
        return go.Figure()

    stats = filtered_df.groupby(categorical_col).agg(Total_Count=('Attrition Status', 'count'),Attrition_Count=('Attrition Status', lambda x: (x == 'Yes').sum())).round(2)
    stats['Attrition_Rate'] = (stats['Attrition_Count'] / stats['Total_Count'] * 100).round(1)
    stats = stats.reset_index()

    if categorical_col in CATEGORY_ORDERS:
        valid_categories = [cat for cat in CATEGORY_ORDERS[categorical_col] if cat in stats[categorical_col].values]
        stats[categorical_col] = pd.Categorical(stats[categorical_col], categories=valid_categories, ordered=True)
        stats = stats.sort_values(categorical_col)

    bar_color, line_color = ('#87CEEB', '#FF6B6B') if theme == 'dark' else ('#3498db', '#e74c3c')
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=stats[categorical_col], y=stats['Total_Count'], name='Employee Count', marker_color=bar_color,
            text=stats['Total_Count'], textposition='outside', textfont=dict(size=10, color=bar_color),
            hovertemplate=('<b>%{x}</b><br>''Employees: %{y}<br>''Attrition: %{customdata[0]}<extra></extra>'),
            customdata=stats[['Attrition_Count', 'Attrition_Rate']].values),secondary_y=False)

    fig.add_trace(
        go.Scatter(x=stats[categorical_col], y=stats['Attrition_Rate'], mode='lines+markers', name='Attrition Rate (%)',
            line=dict(color=line_color, width=2), marker=dict(size=6),
            text=[f'{rate:.1f}%' for rate in stats['Attrition_Rate']],
            textposition='top center', textfont=dict(size=9, color=line_color),
            hovertemplate='<b>%{x}</b><br>Attrition Rate: %{y:.1f}%<extra></extra>'), secondary_y=True)

    fig.update_layout(title=f"{feature_name}: Distribution & Attrition Rate", height=400,
                      template='plotly_dark' if theme == 'dark' else 'plotly_white',
                      legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
                                  bgcolor="rgba(255,255,255,0.9)" if theme == 'light' else "rgba(0,0,0,0.9)",
                                  bordercolor="rgba(0,0,0,0.2)" if theme == 'light' else "rgba(255,255,255,0.2)",
                                  borderwidth=1), margin=dict(t=60, b=5, l=60, r=40))

    fig.update_xaxes(title_text=categorical_col)
    fig.update_yaxes(title_text="Employee Count", secondary_y=False)
    fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)

    return fig

def create_distribution_plot(filtered_df, feature_name, theme='light'):

    if feature_name not in ALL_FEATURES:
        return None
    
    continuous_col, _ = ALL_FEATURES[feature_name]
    if continuous_col is None or continuous_col not in filtered_df.columns:
        return None

    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    fig = px.histogram(filtered_df, x=continuous_col, color='Attrition Status', marginal='box',
        color_discrete_map=ATTRITION_COLORS, title=f'{feature_name}: Distribution by Attrition Status',
        template=template, height=400)

    fig.update_layout(bargap=0.1, legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)" if theme == 'light' else "rgba(0,0,0,0.9)",
            bordercolor="rgba(0,0,0,0.2)" if theme == 'light' else "rgba(255,255,255,0.2)", borderwidth=1),
            margin=dict(t=60, b=5, l=60, r=40))

    return fig

def create_category_pie_charts(filtered_df, feature_name, theme='light', attrition_filter=None):

    if feature_name not in ALL_FEATURES:
        return go.Figure()

    _, categorical_col = ALL_FEATURES[feature_name]
    if categorical_col not in filtered_df.columns:
        return go.Figure()

    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    light_mode = theme == 'light'

    if attrition_filter in ['Yes', 'No']:
        display_df = filtered_df[filtered_df['Attrition Status'] == attrition_filter]

        if display_df.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data available for Attrition Status: {attrition_filter}", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig.update_layout(template=template, height=400)
            return fig

        category_counts = display_df[categorical_col].value_counts()

        if categorical_col in CATEGORY_ORDERS:
            ordered = [cat for cat in CATEGORY_ORDERS[categorical_col] if cat in category_counts.index]
            category_counts = category_counts.reindex(ordered)

        colors = px.colors.qualitative.Set3[:len(category_counts)]

        fig = go.Figure(data=[go.Pie(labels=category_counts.index, values=category_counts.values, marker_colors=colors,
                                     textinfo='label+percent+value', textposition='auto', textfont=dict(size=12),
                                     hovertemplate=("<b>%{label}</b><br>""Count: %{value}<br>""Percentage: %{percent}<extra></extra>"))])

        fig.update_layout(title=f"{feature_name}: Category Distribution for Attrition Status '{attrition_filter}'",
                          template=template, height=400, showlegend=True, 
                          legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05,
                                      bgcolor="rgba(255,255,255,0.9)" if light_mode else "rgba(0,0,0,0.9)",
                                      bordercolor="rgba(0,0,0,0.2)" if light_mode else "rgba(255,255,255,0.2)", borderwidth=1),
                        margin=dict(t=80, b=5, l=40, r=150))
        return fig


    categories = filtered_df[categorical_col].unique()
    if categorical_col in CATEGORY_ORDERS:
        categories = [cat for cat in CATEGORY_ORDERS[categorical_col] if cat in categories]
    else:
        categories = sorted(categories)

    n_categories = len(categories)
    cols = 2 if n_categories == 4 or feature_name == 'Department' else (5 if feature_name == 'Job Role' else min(3, n_categories))
    rows = (n_categories + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "domain"} for _ in range(cols)] for _ in range(rows)],
                        subplot_titles=categories, vertical_spacing=0.1, horizontal_spacing=0.05)

    for i, category in enumerate(categories):
        row, col = divmod(i, cols)
        row += 1
        col += 1

        category_data = filtered_df[filtered_df[categorical_col] == category]
        if category_data.empty:
            continue

        attrition_counts = category_data['Attrition Status'].value_counts()
        colors = [ATTRITION_COLORS.get(label, '#888888') for label in attrition_counts.index]

        fig.add_trace(go.Pie(labels=attrition_counts.index, values=attrition_counts.values, marker_colors=colors,
                             textinfo='label+percent', textposition='inside', 
                             hovertemplate=("<b>%{label}</b><br>""Count: %{value}<br>""Percentage: %{percent}<extra></extra>"),
                             showlegend=(i == 0)), row=row, col=col)

    fig.update_layout(title=f"{feature_name}: Attrition Status by Category", template=template, height=max(400, rows * 200),
                      showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                                                   bgcolor="rgba(255,255,255,0.9)" if light_mode else "rgba(0,0,0,0.9)",
                                                   bordercolor="rgba(0,0,0,0.2)" if light_mode else "rgba(255,255,255,0.2)", borderwidth=1),
                        margin=dict(t=80, b=5, l=40, r=40))

    return fig

def get_available_options(column):
    if column not in df.columns: return [{'label': 'All', 'value': 'all'}]
    unique_values = df[column].unique()
    if column in CATEGORY_ORDERS: all_values = [val for val in CATEGORY_ORDERS[column] if val in unique_values] + sorted([val for val in unique_values if val not in CATEGORY_ORDERS[column]])
    else: all_values = sorted(unique_values)
    return [{'label': 'All', 'value': 'all'}] + [{'label': str(val), 'value': val} for val in all_values]

def layout(theme='light'):
    bg_class, title_color = ('bg-dark', 'white') if theme == 'dark' else ('bg-light', '#2c3e50')
    first_section = list(FEATURE_SECTIONS.keys())[0]
    first_feature = list(FEATURE_SECTIONS[first_section].keys())[0]

    def filter_dropdown(label, feature_id, feature_name):
        return dbc.Col([
            html.Label(label, className="form-label mb-1", style={'fontSize': '0.85rem'}),
            dcc.Dropdown(id=feature_id, options=get_available_options(feature_name), value='all',clearable=False, style={'fontSize': '0.85rem'})
        ], md=3)

    return html.Div([
        dcc.Store(id='theme-store-viz', data=theme),
        dcc.Store(id='selected-feature-store', data=first_feature),
        dbc.Container([
            html.H3([
                html.I(className="fas fa-chart-bar me-2"), "Data Analysis & Visualizations"], className="text-center mb-3", style={'fontWeight': 'bold', 'color': title_color}),
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    filter_dropdown("Age", 'age-filter-viz', 'Age Group'),
                    filter_dropdown("Income", 'income-filter-viz', 'Income Group'),
                    filter_dropdown("Gender", 'gender-filter-viz', 'Gender'),
                    filter_dropdown("Level", 'level-filter-viz', 'Job Level'),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Analysis Section", className="form-label mb-1", style={'fontSize': '0.85rem'}),
                        dcc.Dropdown(id='section-dropdown-viz', options=[{'label': s, 'value': s} for s in FEATURE_SECTIONS], value=first_section, clearable=False, style={'fontSize': '0.85rem'})
                    ], md=3),
                    dbc.Col([
                        html.Label("Features", className="form-label mb-1", style={'fontSize': '0.85rem'}),
                        html.Div(id='feature-buttons-container', className="d-flex flex-wrap gap-1 mt-1")
                    ], md=9),
                ], className="align-items-end")], style={'padding': '15px'}), className="mb-3"),

            html.Div(id='summary-card-viz'),
            html.Div(id='charts-container')
        ], fluid=True, style={'padding': '15px'})], className=bg_class)

@callback(Output('charts-container', 'children'), [Input('selected-feature-store', 'data')])
def update_charts_layout(selected_feature):
    chart_height = '500px'
    chart_card = lambda title, chart_id, height='500px': dbc.Card([dbc.CardHeader([html.H6([html.I(className=f"fas fa-chart-{'bar' if 'main' in chart_id else 'pie'} me-2"), title], className="mb-0")], style={'padding': '6px 10px'}), dbc.CardBody([dcc.Loading([dcc.Graph(id=chart_id, style={'height': height}, config={'displayModeBar': False})])], style={'padding': '4px'})])
    if selected_feature == 'Job Role': return [dbc.Row([dbc.Col([chart_card("Distribution & Rate", 'main-chart-viz')], md=12)], className="mb-2"), dbc.Row([dbc.Col([chart_card("Detailed Analysis", 'secondary-chart-viz', '600px')], md=12)], className="mb-2")]
    else: return [dbc.Row([dbc.Col([chart_card("Distribution & Rate", 'main-chart-viz')], md=6), dbc.Col([chart_card("Detailed Analysis", 'secondary-chart-viz')], md=6)], className="mb-2")]

@callback(Output('summary-card-viz', 'children'), 
          [Input('age-filter-viz', 'value'), Input('income-filter-viz', 'value'), 
           Input('gender-filter-viz', 'value'), Input('level-filter-viz', 'value'), 
           Input('theme-store-viz', 'data'), Input('selected-feature-store', 'data')])
def update_summary_card(age, income, gender, level, theme, selected_feature):
    return create_attrition_summary_card(filter_dataframe(df, age, income, gender, level), theme, selected_feature, {'age': age, 'income': income, 'gender': gender, 'level': level})

@callback([Output('feature-buttons-container', 'children'), Output('selected-feature-store', 'data')], 
          [Input('section-dropdown-viz', 'value')], [State('selected-feature-store', 'data')])

def update_feature_buttons(selected_section, current_selected_feature):

    if not selected_section or selected_section not in FEATURE_SECTIONS: return [], None
    features = list(FEATURE_SECTIONS[selected_section].keys())
    feature_to_select = current_selected_feature if current_selected_feature and current_selected_feature in features else features[0] if features else None
    return [dbc.Button(feature, id={'type': 'feature-btn', 'index': feature}, 
                       color="primary" if feature == feature_to_select else "outline-primary", 
                       outline=not (feature == feature_to_select), size="sm", className="me-1 mb-1", 
                       style={'fontSize': '0.75rem', 'padding': '4px 10px'}) for feature in features], feature_to_select

@callback([Output({'type': 'feature-btn', 'index': dash.dependencies.ALL}, 'color'), 
           Output({'type': 'feature-btn', 'index': dash.dependencies.ALL}, 'outline'), 
           Output('selected-feature-store', 'data', allow_duplicate=True)], 
           [Input({'type': 'feature-btn', 'index': dash.dependencies.ALL}, 'n_clicks')], 
           [State('section-dropdown-viz', 'value')], prevent_initial_call=True)

def update_feature_button_styles(n_clicks_list, current_section):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update, dash.no_update
    clicked_feature = eval(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    if not current_section or current_section not in FEATURE_SECTIONS: return dash.no_update, dash.no_update, dash.no_update
    features = list(FEATURE_SECTIONS[current_section].keys())
    return (["primary" if feature == clicked_feature else "outline-primary" for feature in features], 
            [False if feature == clicked_feature else True for feature in features], clicked_feature)

@callback([Output('main-chart-viz', 'figure'), Output('secondary-chart-viz', 'figure')], 
          [Input('selected-feature-store', 'data'), Input('gender-filter-viz', 'value'),
           Input('level-filter-viz', 'value'), Input('age-filter-viz', 'value'), 
           Input('income-filter-viz', 'value'), Input('theme-store-viz', 'data')])

def update_charts(selected_feature, gender, level, age, income, theme):
    if not selected_feature:
        empty_fig = go.Figure().add_annotation(text="Loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(template='plotly_dark' if theme == 'dark' else 'plotly_white', height=400)
        return empty_fig, empty_fig
    filtered_df = filter_dataframe(df, age, income, gender, level)
    main_fig = create_dual_axis_chart(filtered_df, selected_feature, theme)
    secondary_fig = create_distribution_plot(filtered_df, selected_feature, theme) or create_category_pie_charts(filtered_df, selected_feature, theme)
    return main_fig, secondary_fig

@callback(Output('insights-panel-viz', 'children'), 
          [Input('selected-feature-store', 'data'), Input('gender-filter-viz', 'value'), 
           Input('level-filter-viz', 'value'), Input('age-filter-viz', 'value'), 
           Input('income-filter-viz', 'value')])

def update_insights(selected_feature, gender, level, age, income):

    if not selected_feature:
        return html.P("Loading insights...", className="text-info")
    filtered_df = filter_dataframe(df, age, income, gender, level)
    if len(filtered_df) == 0: 
        return html.P("No data available for the selected filters.", className="text-warning")
    if selected_feature not in ALL_FEATURES: 
        return html.P(f"Feature '{selected_feature}' not found.", className="text-warning")
    continuous_col, categorical_col = ALL_FEATURES[selected_feature]
    if categorical_col not in filtered_df.columns: 
        return html.P(f"Column '{categorical_col}' not found in dataset.", className="text-warning")
    
    try:
        stats = filtered_df.groupby(categorical_col).agg({'Attrition Status': ['count', lambda x: (x == 'Yes').sum()]}).round(2)
        stats.columns = ['Total_Count', 'Attrition_Count']
        stats['Attrition_Rate'] = (stats['Attrition_Count'] / stats['Total_Count'] * 100).round(1)
        stats = stats.reset_index()
        if len(stats) == 0: 
            return html.P("No statistics available for the selected feature.", className="text-warning")
        highest_attrition = stats.loc[stats['Attrition_Rate'].idxmax()]
        lowest_attrition = stats.loc[stats['Attrition_Rate'].idxmin()]
        overall_rate = (len(filtered_df[filtered_df['Attrition Status'] == 'Yes']) / len(filtered_df) * 100)
        current_section = None
        for section_name, features in FEATURE_SECTIONS.items():
            if selected_feature in features:
                current_section = section_name
                break

        section_insights = []
        if current_section == 'Performance & Growth' and 'Job Involvement' in categorical_col:
            high_involvement = filtered_df[filtered_df['Job Involvement'].isin(['High', 'Very High'])]
            if len(high_involvement) > 0:
                high_involve_rate = (len(high_involvement[high_involvement['Attrition Status'] == 'Yes']) / len(high_involvement) * 100)
                section_insights.append(f"High involvement employees: {high_involve_rate:.1f}% attrition rate")
        elif current_section == 'Work Environment & Satisfaction' and 'Job Satisfaction' in categorical_col:
            high_satisfaction = filtered_df[filtered_df['Job Satisfaction'].isin(['High', 'Very High'])]
            if len(high_satisfaction) > 0:
                high_sat_rate = (len(high_satisfaction[high_satisfaction['Attrition Status'] == 'Yes']) / len(high_satisfaction) * 100)
                section_insights.append(f"Highly satisfied employees: {high_sat_rate:.1f}% attrition rate")
        section_title = current_section if current_section else "Analysis"
    
    except Exception as e:
        return html.P(f"Error generating insights: {str(e)}", className="text-danger")