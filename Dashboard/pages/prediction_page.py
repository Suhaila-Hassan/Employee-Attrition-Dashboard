from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

with open('pages/models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('pages/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

MAPPINGS = {
    'business_travel': {'Never Travels': 0, 'Rarely Travels': 1, 'Frequently Travel': 2},
    'education': {'High School': 1, 'Associate': 2, 'Bachelors': 3, 'Masters': 4, 'Doctor': 5},
    'satisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'gender': {'Female': [1, 0], 'Male': [0, 1]},
    'performance': {'Low': 1, 'Good': 2, 'Excellent': 3, 'Outstanding': 4},
    'job_level': {'Entry Level': 1, 'Mid Level': 2, 'Senior Level': 3, 'Manager': 4, 'Executive': 5},
    'overtime': {'No': 0, 'Yes': 1},
    'balance': {'Bad': 1, 'Fair': 2, 'Good': 3, 'Best': 4},
    'marital_status': {'Single': [1, 0, 0], 'Married': [0, 1, 0], 'Divorced': [0, 0, 1]}
}

def create_input_field(field_id, label, field_type='number', value=None, min_val=None, max_val=None, width=6, label_style=None):
    return dbc.Col([
        dbc.Label(label, style=label_style, className="mb-1"),
        dbc.Input(id=field_id, type=field_type, value=value, min=min_val, max=max_val, className="mb-2", size="sm"),
    ], width=width)

def create_dropdown_field(field_id, label, options, default_value, width=6, label_style=None):
    return dbc.Col([
        dbc.Label(label, style=label_style, className="mb-1"),
        dcc.Dropdown(options, id=field_id, value=default_value, className="mb-2", style={'fontSize': '14px'}),
    ], width=width)

def create_slider_with_labels(slider_id, options_map, default_key, label_text, label_style):
    options_list = list(options_map.keys())
    default_value = options_list.index(default_key)
    return html.Div([
        dbc.Label(label_text, style=label_style, className="mb-1"),
        dcc.Slider(
            id=slider_id, min=0, max=len(options_list) - 1, step=1, value=default_value,
            marks={i: {'label': option, 'style': {'fontSize': '10px'}} for i, option in enumerate(options_list)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], className="mb-2")

def create_section_header(title, color):
    return html.H6(title, className="mb-2 mt-2", 
                   style={'color': color, 'borderBottom': f'2px solid {color}', 'paddingBottom': '3px', 'fontWeight': 'bold'})

def create_personal_details_section(label_style):
    return [
        create_section_header("Personal Details", '#007bff'),
        dbc.Row([
            create_input_field('age', 'Age', value=18, min_val=18, max_val=60, width=2, label_style=label_style),
            create_dropdown_field('gender', 'Gender', list(MAPPINGS['gender'].keys()), 'Male', width=2, label_style=label_style),
            create_dropdown_field('education', 'Education Level', list(MAPPINGS['education'].keys()), 'Bachelors', width=2, label_style=label_style),
            create_dropdown_field('marital_status', 'Marital Status', list(MAPPINGS['marital_status'].keys()), 'Single', width=2, label_style=label_style),
            create_input_field('distance', 'Distance From Home (km)', value=5, min_val=1, max_val=29, width=3, label_style=label_style)
        ], className="g-2")
    ]

def create_professional_details_section(label_style):
    return [
        create_section_header("Professional Details", '#28a745'),
        dbc.Row([
            create_dropdown_field('job_level', 'Job Level', list(MAPPINGS['job_level'].keys()), 'Entry Level', width=3, label_style=label_style),
            create_dropdown_field('business_travel', 'Business Travel Frequency', list(MAPPINGS['business_travel'].keys()), 'Never Travels', width=3, label_style=label_style),
            create_input_field('monthly_income', 'Monthly Income ($)', value=3000, min_val=1009, max_val=19999, width=3, label_style=label_style),
            create_input_field('salary_hike', 'Salary Hike (%)', value=11, min_val=11, max_val=25, width=3, label_style=label_style)
        ], className="g-2"),
        dbc.Row([
            create_input_field('total_working_years', 'Total Working Years', value=1, min_val=0, max_val=40, width=3, label_style=label_style),
            create_input_field('stock_option', 'Stock Option Level', value=0, min_val=0, width=3, label_style=label_style),
            create_input_field('years_at_company', 'Years at Company', value=1, min_val=0, max_val=40, width=3, label_style=label_style),
            create_input_field('years_in_role', 'Years in Current Role', value=1, min_val=0, max_val=18, width=3, label_style=label_style)
        ], className="g-2"),
        dbc.Row([
            create_input_field('years_since_promotion', 'Years Since Last Promotion', value=1, min_val=0, width=3, label_style=label_style),
            create_input_field('years_with_manager', 'Years with Current Manager', value=5, min_val=0, width=3, label_style=label_style),
            create_input_field('num_companies', 'Companies Worked at Previously', value=1, min_val=0, max_val=9, width=3, label_style=label_style),
            create_input_field('training_times', 'Training Times Last Year', value=2, min_val=0, width=3, label_style=label_style)
        ], className="g-2")
    ]

def create_work_environment_section(label_style):
    return [
        create_section_header("Work Environment & Satisfaction", '#dc3545'),
        dbc.Row([
            dbc.Col([create_slider_with_labels('env_satisfaction', MAPPINGS['satisfaction'], 'Medium', 'Environment Satisfaction', label_style)], width=4),
            dbc.Col([create_slider_with_labels('job_satisfaction', MAPPINGS['satisfaction'], 'Medium', 'Job Satisfaction', label_style)], width=4),
            dbc.Col([create_slider_with_labels('relationship_satisfaction', MAPPINGS['satisfaction'], 'Medium', 'Relationship Satisfaction', label_style)], width=4)
        ], className="g-2"),
        dbc.Row([
            dbc.Col([create_slider_with_labels('job_involvement', MAPPINGS['performance'], 'Good', 'Job Involvement', label_style)], width=4),
            dbc.Col([create_slider_with_labels('work_life_balance', MAPPINGS['balance'], 'Good', 'Work-Life Balance', label_style)], width=4),
            dbc.Col([
                html.Div([
                    dbc.Label('Work Schedule', style=label_style, className="mb-1"),
                    html.Br(),
                    dbc.Checkbox(id='overtime', label='Works Overtime', value=False, className="mt-1", label_style=label_style)
                ])
            ], width=4)
        ], className="g-2")
    ]

def get_risk_assessment(prob):
    risk_levels = [
        (0.20, "Very Low Risk", "#28a745", "fas fa-check-circle", "Employee will likely stay"),
        (0.40, "Low Risk", "#20c997", "fas fa-thumbs-up", "Employee shows good retention indicators"),
        (0.60, "Moderate Risk", "#ffc107", "fas fa-exclamation-triangle", "Monitor employee closely"),
        (0.80, "High Risk", "#fd7e14", "fas fa-exclamation-circle", "Apply retention strategies"),
        (1.00, "Very High Risk", "#dc3545", "fas fa-times-circle", "Immediate intervention recommended")
    ]
    for threshold, status, color, icon, message in risk_levels:
        if prob <= threshold:
            return status, color, icon, message
    return risk_levels[-1][1:]

def create_prediction_result(prob):
    status, color, icon, message = get_risk_assessment(prob)
    return html.Div([
        html.Div([
            html.I(className=icon, style={'fontSize': '3rem', 'color': color, 'marginBottom': '15px'}),
            html.H2(status, style={'color': color, 'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.P(message, style={'color': '#6c757d', 'fontSize': '1.2rem', 'marginBottom': '25px'})
        ], style={'textAlign': 'center'}),
        html.Div([
            daq.Gauge(
                showCurrentValue=True,
                color={"gradient": True, "ranges": {"#28a745": [0, 0.2], "#20c997": [0.2, 0.4], "#ffc107": [0.4, 0.6], "#fd7e14": [0.6, 0.8], "#dc3545": [0.8, 1]}},
                label={'label': 'Attrition Probability', 'style': {'fontSize': '16px', 'fontWeight': 'bold'}},
                max=1, min=0, value=round(prob, 4), size=150
            )
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Attrition Risk", className="text-center", style={'marginBottom': '10px'}),
                        html.H3(f"{prob:.1%}", className="text-center", style={'color': color, 'fontWeight': 'bold', 'margin': '0'})
                    ], style={'padding': '15px'})
                ], color="light", outline=True, style={'margin': '0 auto', 'maxWidth': '500px'})
            ], width=12, className="d-flex justify-content-center")
        ])
    ])
    
def create_prediction_modal():
    return dbc.Modal([
        dbc.ModalHeader([html.H3([html.I(className="fas fa-chart-line", style={'marginRight': '10px'}), "Attrition Risk Assessment"])]),
        dbc.ModalBody(id='modal_prediction_content'),
        dbc.ModalFooter([dbc.Button("Close", id="close_modal", className="ms-auto", color="secondary")])
    ], id='prediction_modal', size='lg', is_open=False)

def layout(theme='light'):
    container_class = "bg-dark" if theme == 'dark' else "bg-light"
    card_class = "bg-secondary text-light" if theme == 'dark' else "bg-white"
    label_style = {'color': 'white', 'fontWeight': '500', 'fontSize': '13px'} if theme == 'dark' else {'color': '#333', 'fontWeight': '500', 'fontSize': '13px'}
    btn_color = 'success' if theme == 'light' else 'outline-success'
    header_color = 'white' if theme == 'dark' else '#2c3e50'

    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2([
                        html.I(className="fas fa-magic-wand-sparkles", style={'marginRight': '10px'}),
                        'Employee Attrition Prediction'
                    ], className="text-center mb-3", style={'color': header_color, 'fontWeight': 'bold'})
                ])
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([html.H5("Employee Information", style={'color': header_color, 'margin': 0, 'fontWeight': 'bold'})], className="py-2"),
                        dbc.CardBody([
                            *create_personal_details_section(label_style),
                            *create_professional_details_section(label_style),
                            *create_work_environment_section(label_style),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button('Predict Attrition Risk', id='predict_button', 
                                             n_clicks=0, color=btn_color, size="lg",
                                             className='w-100 mt-2', style={'fontWeight': 'bold'})
                                ])
                            ])
                        ], className="py-3")
                    ], className=card_class, style={'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
                ], width=12)
            ])
        ], fluid=True, className=container_class, style={'minHeight': '90vh', 'padding': '15px'}),
        create_prediction_modal()
    ])

@callback(
    [Output('prediction_modal', 'is_open'), Output('modal_prediction_content', 'children')],
    [Input('predict_button', 'n_clicks'), Input('close_modal', 'n_clicks')],
    [State('prediction_modal', 'is_open')] + [State(field_id, 'value') for field_id in [
        'age', 'business_travel', 'distance', 'education', 'env_satisfaction', 'gender', 
        'job_involvement', 'job_level', 'job_satisfaction', 'monthly_income', 'num_companies', 
        'overtime', 'salary_hike', 'relationship_satisfaction', 'stock_option', 'total_working_years', 
        'training_times', 'work_life_balance', 'years_at_company', 'years_in_role', 
        'years_since_promotion', 'years_with_manager', 'marital_status'
    ]]
)
def handle_modal_and_prediction(predict_clicks, close_clicks, is_open, *args):
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == 'close_modal':
        return False, ""
    
    if triggered_id == 'predict_button' and predict_clicks > 0:
        try:
            # Check for None values and provide defaults
            values = list(args)
            
            # Validate that all required fields have values
            required_fields = [
                'age', 'business_travel', 'distance', 'education', 'env_satisfaction', 'gender', 
                'job_involvement', 'job_level', 'job_satisfaction', 'monthly_income', 'num_companies', 
                'overtime', 'salary_hike', 'relationship_satisfaction', 'stock_option', 'total_working_years', 
                'training_times', 'work_life_balance', 'years_at_company', 'years_in_role', 
                'years_since_promotion', 'years_with_manager', 'marital_status'
            ]
            
            # Check for None values
            for i, (field, value) in enumerate(zip(required_fields, values)):
                if value is None:
                    raise ValueError(f"Missing value for {field}")
            
            (age, business_travel, distance, education, env_satisfaction, gender, 
             job_involvement, job_level, job_satisfaction, monthly_income, 
             num_companies, overtime, salary_hike, relationship_satisfaction,
             stock_option, total_working_years, training_times, work_life_balance,
             years_at_company, years_in_role, years_since_promotion, 
             years_with_manager, marital_status) = values
            
            # Convert slider indices to actual values
            slider_mappings = {
                'env_satisfaction': (MAPPINGS['satisfaction'], env_satisfaction),
                'job_satisfaction': (MAPPINGS['satisfaction'], job_satisfaction),
                'job_involvement': (MAPPINGS['performance'], job_involvement),
                'work_life_balance': (MAPPINGS['balance'], work_life_balance),
                'relationship_satisfaction': (MAPPINGS['satisfaction'], relationship_satisfaction)
            }
            
            converted_values = {}
            for key, (mapping, value) in slider_mappings.items():
                options = list(mapping.keys())
                if value < len(options):
                    converted_values[key] = options[value]
                else:
                    converted_values[key] = options[0]  # Default to first option
            
            # Build feature array in the correct order expected by your model
            features = [
                age, 
                MAPPINGS['business_travel'][business_travel], 
                distance,
                MAPPINGS['education'][education], 
                MAPPINGS['satisfaction'][converted_values['env_satisfaction']],
                MAPPINGS['performance'][converted_values['job_involvement']], 
                MAPPINGS['job_level'][job_level],
                MAPPINGS['satisfaction'][converted_values['job_satisfaction']], 
                monthly_income, 
                num_companies,
                1 if overtime else 0, 
                salary_hike,
                MAPPINGS['satisfaction'][converted_values['relationship_satisfaction']], 
                stock_option,
                total_working_years, 
                training_times, 
                MAPPINGS['balance'][converted_values['work_life_balance']],
                years_at_company, 
                years_in_role, 
                years_since_promotion, 
                years_with_manager
            ]
            
            features.extend(MAPPINGS['marital_status'][marital_status])            
            features.extend(MAPPINGS['gender'][gender])
            
            # Create DataFrame and scale features
            X = pd.DataFrame([features])
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prob = model.predict_proba(X_scaled)[0][1]
            
            return True, create_prediction_result(prob)
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            error_content = html.Div([
                dbc.Alert([
                    html.H4("Error in Prediction", className="alert-heading"),
                    html.P(f"Error: {str(e)}"),
                    html.P("Please check all input values and try again."),
                ], color="danger")
            ])
            return True, error_content
    
    return is_open, ""