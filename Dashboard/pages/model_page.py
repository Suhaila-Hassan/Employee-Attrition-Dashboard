import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('pages/datasets/Dataset_Machine_Learning.csv')
X, y = df.drop('Attrition Status', axis=1), df['Attrition Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_test_scaled = pd.read_csv('pages/datasets/Testing_Features.csv')
y_test = pd.read_csv('pages/datasets/Testing_Labels.csv')

with open('pages/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Calculate all metrics
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Feature analysis
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
importance_df = importance_df.iloc[:-1]
importance_df['Feature'].replace({'Gender: Female':'Gender'}, inplace=True)

correlation_with_attrition = df.corr(numeric_only=True)["Attrition Status"].drop("Attrition Status").sort_values(ascending=False)

def create_charts(theme='light'):
    template = "plotly_dark" if theme == 'dark' else "plotly_white"
    
    fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                           color='Importance', color_continuous_scale='viridis',
                           template=template, height=500)
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False,
                                title_x=0.5, margin=dict(l=120, r=30, t=20, b=30))
    
    fig_corr_bar = px.bar(x=correlation_with_attrition.values, y=correlation_with_attrition.index,
                         orientation='h', color=correlation_with_attrition.values,
                         color_continuous_scale='RdBu_r', template=template, height=500)
    fig_corr_bar.update_layout(showlegend=False, title_x=0.5, xaxis_title="Correlation",
                              margin=dict(l=120, r=30, t=20, b=30))
    
    return fig_importance, fig_corr_bar

def layout(theme='light'):
    fig_importance, fig_corr_bar = create_charts(theme)
    
    # Dynamic title color based on theme
    title_color = '#2c3e50' if theme == 'light' else 'white'
    
    metric_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{accuracy:.1%}", className="text-center mb-1", 
                    style={"color": "#007BFF", "font-weight": "bold"}),
            html.P("Accuracy", className="text-center mb-0 small")
        ], className="py-2"), className="shadow-sm"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{precision:.1%}", className="text-success text-center mb-1"),
            html.P("Precision", className="text-center mb-0 small")
        ], className="py-2"), className="shadow-sm"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{recall:.1%}", className="text-warning text-center mb-1"),
            html.P("Recall", className="text-center mb-0 small")
        ], className="py-2"), className="shadow-sm"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{f1:.1%}", className="text-info text-center mb-1"),
            html.P("F1 Score", className="text-center mb-0 small")
        ], className="py-2"), className="shadow-sm"), md=3)
    ], className="mb-3")
    
    data_info_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{len(df):,}", className="text-center mb-1", 
                    style={"font-weight": "bold", "font-size": "1.5rem", "color": "#FF4444"}),
            html.P("Total Employees", className="text-center mb-0 small")
        ], className="py-2"), className="shadow-sm"), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{len(X_train):,}", className="text-center mb-1", 
                    style={"font-weight": "bold", "font-size": "1.5rem", "color": "#00BFFF"}),
            html.P("Training Samples", className="text-center mb-0 small")
        ], className="py-2"), className="shadow-sm"), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{len(X_test):,}", className="text-center mb-1", 
                    style={"font-weight": "bold", "font-size": "1.5rem", "color": "#00FF7F"}),
            html.P("Test Samples", className="text-center mb-0 small")
        ], className="py-2"), className="shadow-sm"), md=4)
    ], className="mb-3")

    return dbc.Container([
        dbc.Row([
            html.H2([
                html.I(className="fas fa-cogs me-2"),
                "Machine Learning Model Information"
            ], className="text-center mb-2 fw-bold", style={"color": title_color}),
        ]),
        data_info_cards,
        metric_cards,
        dbc.Card([
            dbc.CardHeader(html.H5("Feature Importance", className="mb-0")),
            dbc.CardBody([
                dcc.Graph(figure=fig_importance),
                dbc.Alert([
                    html.P([html.Strong("Key Insight: "), f"The most influential factor for employee attrition is {importance_df.iloc[0]['Feature']} with an importance score of {importance_df.iloc[0]['Importance']:.3f}."], className="mb-0")
                ], color="info", className="mt-2")
            ], className="pb-2")
        ], className="shadow mb-3"),
        dbc.Card([
            dbc.CardHeader(html.H5("Correlation with Attrition", className="mb-0")),
            dbc.CardBody([
                dcc.Graph(figure=fig_corr_bar),
                dbc.Alert([
                    html.P([html.Strong("Key Insight: "), f"{correlation_with_attrition.index[0]} shows the strongest correlation with attrition ({correlation_with_attrition.iloc[0]:.3f}), while {correlation_with_attrition.index[-1]} has the weakest correlation ({correlation_with_attrition.iloc[-1]:.3f})."], className="mb-0")
                ], color="success", className="mt-2")
            ], className="pb-2")
        ], className="shadow")
    ], fluid=True, className="px-3 py-2")