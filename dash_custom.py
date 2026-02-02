# dash_custom.py
import os, uuid
from flask import session
import dash
from dash import html, dcc, Input, Output, State


from algo import check  

def create_custom_dash(server):
    dash_app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname="/custom_ui/",  
        suppress_callback_exceptions=True,
    )

    dash_app.layout = html.Div(
        [
            html.Link(href="/static/styles.css", rel="stylesheet"),

            html.H1("Set the parameters of the optimization algorithm", className="custom-header"),
            html.H2("The higher the values, the more time it will take to train."),

         
            dcc.Slider(id="slider-1", min=5, max=1000, step=1, value=50),
            html.Div(id="slider-output-1"),

            dcc.Slider(id="slider-2", min=5, max=1000, step=1, value=75),
            html.Div(id="slider-output-2"),

            html.Label("Enter the lower bound:"),
            dcc.Input(id="lb", type="number", value=0),
            html.Br(),
            html.Label("Enter the upper bound:"),
            dcc.Input(id="ub", type="number", value=0),
            html.Div(id="result", className="result"),

            html.Hr(),
            html.Div(id="display", className="display", children="0"),

          
            html.Button("1", id="btn-1"),
            html.Button("+", id="btn-+"),
            html.Button("X", id="btn-X"),
            html.Button("Submit", id="btn-equals", n_clicks=0),
            html.Button("C", id="btn-clear", n_clicks=0),

            html.Div(id="redirect-holder"),
        ]
    )

    @dash_app.callback(
        Output("slider-output-1", "children"),
        Output("slider-output-2", "children"),
        Input("slider-1", "value"),
        Input("slider-2", "value"),
    )
    def update_output(value1, value2):
        return f"Individuals: {value1}", f"Generations: {value2}"

    @dash_app.callback(
        Output("result", "children"),
        Input("lb", "value"),
        Input("ub", "value"),
    )
    def validate_bounds(lower_bound, upper_bound):
        if lower_bound is None or upper_bound is None:
            return ""
        if lower_bound >= upper_bound:
            return "Lower bound should be strictly smaller than the upper bound."
        return ""

    @dash_app.callback(
        Output("display", "children"),
        Output("redirect-holder", "children"),
        [
            Input("slider-1", "value"),
            Input("slider-2", "value"),
            Input("lb", "value"),
            Input("ub", "value"),
            Input("btn-1", "n_clicks"),
            Input("btn-+", "n_clicks"),
            Input("btn-X", "n_clicks"),
            Input("btn-clear", "n_clicks"),
            Input("btn-equals", "n_clicks"),
        ],
        State("display", "children"),
        prevent_initial_call=True
    )
    def update_display(it, gen, lb, ub,
                       btn1, btn_plus, btn_X, btn_clear, btn_equals,
                       current_display):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if current_display in (None, "0"):
            current_display = ""

        if trigger_id == "btn-clear":
            return "0", ""

        if trigger_id == "btn-1":
            current_display += "1"
            return current_display, ""

        if trigger_id == "btn-+":
            current_display += " + "
            return current_display, ""

        if trigger_id == "btn-X":
            current_display += " X "
            return current_display, ""

        if trigger_id == "btn-equals":
       
            if lb is None or ub is None or lb >= ub:
                session["custom_error"] = "Fix bounds first (lower < upper)."
                return current_display, dcc.Location(href="/custom_fit", id="redir")

        
            run_id = uuid.uuid4().hex[:10]

            try:
                res = check(it, gen, lb, ub, current_display, run_id=run_id)
                if res == "error":
                    session["custom_error"] = "Invalid function."
                    return current_display, dcc.Location(href="/custom_fit", id="redir")
            except Exception as e:
                session["custom_error"] = f"Run failed: {e}"
                return current_display, dcc.Location(href="/custom_fit", id="redir")

            session["custom_error"] = ""
            session["run_id"] = run_id
            session["custom_algo"] = "Custom"
            session["custom_preset"] = "User-defined"

         
            return "Redirecting ...", dcc.Location(href="/custom_fit", id="redir")

        return current_display, ""

    return dash_app
