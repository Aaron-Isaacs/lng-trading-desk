import plotly.graph_objects as go

def plot_lines(df, columns, title, markers=False):
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            mode="lines+markers" if markers else "lines"
        ))
    fig.update_layout(title=title, template="plotly_dark")
    return fig

def plot_zscores(df, columns, title, markers=False):
    fig = plot_lines(df, columns, title, markers)
    fig.add_hline(y=0)
    fig.add_hline(y=2, line_dash="dash")
    fig.add_hline(y=-2, line_dash="dash")
    return fig