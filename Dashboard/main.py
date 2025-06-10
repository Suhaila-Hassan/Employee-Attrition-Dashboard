import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pages import model_page, prediction_page, visualizations_page
import warnings
warnings.filterwarnings("ignore")

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.ZEPHYR, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"])
app.title = "HR Analytics"

def create_navbar(theme='light'):
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-home me-2"), "Home"], href="/", className="text-white")),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-chart-line me-2"), "Analysis"], href="/visualizations", className="text-white")),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-cogs me-2"), "Model Info"], href="/model", className="text-white")),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-magic me-2"), "Prediction"], href="/predict", className="text-white")),
            html.Div([dbc.Switch(id="theme-toggle", label=html.Span([html.I(className="fas fa-moon me-1"), "Dark Mode"], className="text-white"), value=theme=='dark')], style={'marginLeft': '2rem'})
        ],
        brand=[html.I(className="fas fa-users-cog me-2"), "HR Analytics"],
        brand_href="/", color="dark" if theme=='dark' else "primary", dark=True, className="mb-0 shadow-lg"
    )

def create_card(icon, title, desc, href, colors):
    return html.A([dbc.Card([dbc.CardBody([html.Div([html.I(className=f"fas fa-{icon} fa-2x mb-2", style={'color': 'white'}), html.H5(title, className="card-title fw-bold text-white mb-2"), html.P(desc, className="card-text text-white small", style={'opacity': '0.9'})], className="text-center h-100 d-flex flex-column justify-content-center")], className="p-3")], style={'background': f"linear-gradient(135deg, {colors[0]}, {colors[1]})", 'border': 'none', 'borderRadius': '12px', 'minHeight': '140px'}, className="h-100")], href=href, style={'textDecoration': 'none'})

def landing_page_layout(theme='light'):
    bg, txt, sub = ('bg-dark', 'text-light', 'text-secondary') if theme=='dark' else ('bg-light', 'text-dark', 'text-muted')
    return html.Div([
        dbc.Container([
            html.Div([
                html.H1("Employee Attrition Analysis & Prediction", className=f"display-6 fw-bold text-center mb-3", style={'color': '#e74c3c'}),
                html.P("Harness the power of AI to understand, predict, and prevent employee attrition. Make data-driven decisions to improve retention and build stronger teams.", className=f"lead text-center {sub} mb-4", style={'fontSize': '1.1rem'})
            ], className="py-3 text-center"),
            html.Div([
                dbc.Row([
                    dbc.Col([create_card("chart-line", "Analysis & Visualizations", "Interactive dashboards with comprehensive employee analytics", "/visualizations", ["#e74c3c", "#c0392b"])], lg=4, md=4, className="mb-3"),
                    dbc.Col([create_card("cogs", "Model Information", "Feature importance and model performance metrics", "/model", ["#3498db", "#2980b9"])], lg=4, md=4, className="mb-3"),
                    dbc.Col([create_card("magic", "Attrition Prediction", "AI-powered attrition risk assessment", "/predict", ["#2ecc71", "#27ae60"])], lg=4, md=4, className="mb-3")
                ], className="g-3")
            ], className="pb-3"),
        ], className="py-3")
    ], className=bg, style={'minHeight': '85vh', 'display': 'flex', 'alignItems': 'center'})

app.layout = html.Div([
    dcc.Store(id='theme-store', data='light'),
    dcc.Location(id='url', refresh=False),
    html.Link(id='theme-link', rel='stylesheet', href=dbc.themes.ZEPHYR),
    html.Div(id='navbar-container'),
    html.Div(id='page-content', className='main-content')
])

@app.callback(Output('theme-link', 'href'), Input('theme-store', 'data'))
def update_theme(theme): 
    return dbc.themes.SLATE if theme=='dark' else dbc.themes.ZEPHYR

@app.callback(Output('navbar-container', 'children'), Input('theme-store', 'data'))
def update_navbar(theme): 
    return create_navbar(theme)

@app.callback(Output('theme-store', 'data'), Input('theme-toggle', 'value'), prevent_initial_call=True)
def toggle_theme(switch): 
    return 'dark' if switch else 'light'

@app.callback(Output('page-content', 'children'), [Input('url', 'pathname'), Input('theme-store', 'data')])
def display_page(pathname, theme):
    routes = {"/predict": prediction_page.layout, "/model": model_page.layout, "/visualizations": visualizations_page.layout}
    return routes.get(pathname, lambda t: landing_page_layout(t))(theme) if pathname in routes else landing_page_layout(theme)


if __name__ == '__main__':
    app.run(debug=True)
