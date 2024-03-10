import json
import logging
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from src.chain import get_rag_conversation_chain

logging.basicConfig(level=logging.INFO)

chain = get_rag_conversation_chain()


def header(name: str) -> dbc.Row:
    title = html.H1(name, style={"margin-top": 5})
    return dbc.Row([dbc.Col(title, md=8)])


def textbox(text_dict: dict) -> Any:
    text = text_dict["content"]
    role = text_dict["role"]
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
        "whiteSpace": "pre-wrap",
    }

    if role == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif role == "assistant":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        thumbnail = html.Img(
            src=app.get_asset_url("logo.jpg"),
            style={
                "border-radius": 50,
                "height": 36,
                "margin-right": 5,
                "float": "left",
            },
        )
        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)

        return html.Div([thumbnail, textbox])

    else:
        raise ValueError("Incorrect option for `box`.")


# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Define Layout
conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

controls = dbc.InputGroup(
    children=[
        dbc.Input(
            id="user-input",
            placeholder="Write your message...",
            value="",
            type="text",
        ),
        dbc.Button("Submit", id="submit"),
    ]
)

app.layout = dbc.Container(
    fluid=False,
    children=[
        header("COVID-19 Chatbot"),
        html.Hr(),
        dcc.Store(id="store-conversation", data="[]"),
        conversation,
        controls,
        dbc.Spinner(html.Div(id="loading-component")),
    ],
)


@app.callback(
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history: str) -> list:
    return [textbox(x) for i, x in enumerate(json.loads(chat_history))]


@app.callback(
    Output("user-input", "value"),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input(n_clicks: int, n_submit: int) -> str:
    return ""


@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
)
def run_chatbot(
    n_clicks: int, n_submit: int, user_input: str, chat_history: str
) -> tuple:
    if n_clicks == 0 and n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None

    chat_history_list = json.loads(chat_history)

    output = generate_response(chat_history_list, user_input)

    chat_history_list.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    chat_history_list.extend(
        [{"role": "assistant", "content": content} for content in output]
    )
    new_chat_history = json.dumps(chat_history_list)

    return new_chat_history, None


def generate_response(chat_history_list: list, user_input: str) -> list:
    output = []

    result = chain.invoke(user_input)

    documents = [doc.page_content for doc in result["context"]]
    docs_str = f"\n{'=' * 50}\n".join(
        f"{i}: {doc}" for i, doc in enumerate(documents, 1)
    )
    docs_str = f"Documents retrieved:\n{docs_str}"
    logging.info(docs_str)
    output.append(result["answer"])

    return output


if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0', port=8888)
