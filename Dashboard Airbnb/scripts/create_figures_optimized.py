import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List, Optional

ROOMS_PATH = ("Dashboard Airbnb/data/clean_data/rooms_clean.csv")
REVIEWS_PATH = ("Dashboard Airbnb/data/clean_data/reviews_clean.csv")
OUTPUT_DIR = "Dashboard Airbnb/results/presentation_figures"

THEME_COLORS = ["#feaf88", "#ff687d", "#cb6686", "#745984", "#265d84"]

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load rooms and reviews cleaned CSV files and fix coordinates."""
    rooms = pd.read_csv(ROOMS_PATH, sep=";")
    rooms["latitude"] = (
        rooms["latitude"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    rooms["longitude"] = (
        rooms["longitude"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    reviews = pd.read_csv(REVIEWS_PATH, sep=";")
    return rooms, reviews

def merge_rooms_reviews(rooms: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Return inner join of rooms and reviews on id."""
    return pd.merge(rooms, reviews, on="id", how="inner")

def save_figure(fig: go.Figure, name: str, folder: str = OUTPUT_DIR, scale: int = 3) -> None:
    """Export a figure as PNG and print the path."""
    path = f"{folder}/{name}.png"
    fig.write_image(path, scale=scale, width=1200, height=400)
    print(f"✅ Saved: {path}")

def apply_minimal_style(fig: go.Figure, percentage: bool = False) -> go.Figure:
    """Apply a consistent minimalist style to Plotly figures."""
    fig.update_traces(marker_line_width=0, opacity=1)
    fig.update_layout(
        height=800,
        width=800,
        font=dict(family="Fira Sans, Roboto, Arial", size=15, color="#2e2e2e"),
        title_font=dict(family="Fira Sans, Roboto, Arial", size=20, color="#111111"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.25,
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#E5E5E5",
            zeroline=False,
            tickformat=".0f" if percentage else "~s",
            range=[0, 100] if percentage else None,
        ),
        yaxis=dict(title=None, showgrid=False, zeroline=False, visible=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            title=None,
            font=dict(size=14),
        ),
        margin=dict(l=80, r=40, t=70, b=40),
    )
    return fig


def prepare_time_columns(df: pd.DataFrame, date_col: str = "last_review") -> pd.DataFrame:
    """Convert date column and add year/quarter/month/day columns."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    df["quarter"] = df[date_col].dt.quarter
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    return df


def compute_time_metrics(df: pd.DataFrame) -> dict:
    """Compute mean price and mean availability grouped by time units."""
    result = {}
    result["Day"] = (
        df.groupby("day", as_index=False)
        .agg(mean_price=("price", "mean"), mean_availability=("availability_365", "mean"))
        .sort_values("day")
    )

    result["Month"] = (
        df.groupby("month", as_index=False)
        .agg(mean_price=("price", "mean"), mean_availability=("availability_365", "mean"))
        .sort_values("month")
    )

    result["Quarter"] = (
        df.groupby("quarter", as_index=False)
        .agg(mean_price=("price", "mean"), mean_availability=("availability_365", "mean"))
        .sort_values("quarter")
    )

    result["Year"] = (
        df.groupby("year", as_index=False)
        .agg(mean_price=("price", "mean"), mean_availability=("availability_365", "mean"))
        .sort_values("year")
    )

    return result


def normalize_0_to_100(df: pd.DataFrame, cols: List[str], id_col: str) -> pd.DataFrame:
    """Scale columns to 0-100 and return long-format dataframe.

    Args:
        df: DataFrame in wide format.
        cols: Columns to scale.
        id_col: Column to keep as identifier when melting.
    """
    df_norm = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_norm[cols] = scaler.fit_transform(df[cols])
    df_long = df_norm.melt(id_vars=[id_col], var_name="metric", value_name="value_normalized")
    return df_long


def plot_time_metrics(
    df: pd.DataFrame,
    x: str,
    title: str,
    normalized: bool = False,
    colors: Optional[List[str]] = None,
) -> go.Figure:
    """Plot time series for price and availability. Optionally plot normalized values.

    If normalized is False, df is expected in wide format with columns
    ['mean_price', 'mean_availability'] and the identifier column x.
    If normalized is True, df should be long-format with 'metric' and 'value_normalized'.
    """
    colors = colors or ["#2E86C1", "#FFA07A"]

    if normalized:
        df_plot = df
        y_col = "value_normalized"
    else:
        df_plot = df.melt(id_vars=[x], var_name="metric", value_name="value")
        y_col = "value"

    fig = px.line(df_plot, x=x, y=y_col, color="metric", markers=True, color_discrete_sequence=colors, title=title)

    # Style lines and markers
    fig.update_traces(line_shape="spline", line=dict(width=5), marker=dict(size=7))

    # Add horizontal mean lines for each metric (using original df if not normalized)
    if not normalized:
        metrics = ["mean_price", "mean_availability"]
        colors_map = {"mean_price": colors[0], "mean_availability": colors[1]}
        for metric in metrics:
            mean_val = df[metric].mean()
            fig.add_hline(
                y=mean_val,
                line_dash="dot",
                line_color="#999999",
                annotation_text=f"Média geral: {mean_val:.2f}",
                annotation_position="top left",
                annotation_font=dict(size=12, color="#444"),
                opacity=0.8,
            )

    fig.update_layout(
        height=400,
        width=1000,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Fira Sans, Roboto", size=14, color="#2e2e2e"),
        title_font=dict(size=20, family="Fira Sans, Roboto", color="#111"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, title=None),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEE", zeroline=False, title=("Value normalized (0-100)" if normalized else "Value")),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def compute_review_metrics(df: pd.DataFrame) -> dict:
    """Compute average number_of_reviews grouped by time units."""
    result = {}
    result["Day"] = (
        df.groupby("day", as_index=False).agg(mean_reviews=("number_of_reviews", "mean")).sort_values("day")
    )
    result["Month"] = (
        df.groupby("month", as_index=False).agg(mean_reviews=("number_of_reviews", "mean")).sort_values("month")
    )
    result["Quarter"] = (
        df.groupby("quarter", as_index=False).agg(mean_reviews=("number_of_reviews", "mean")).sort_values("quarter")
    )
    result["Year"] = (
        df.groupby("year", as_index=False).agg(mean_reviews=("number_of_reviews", "mean")).sort_values("year")
    )
    return result


def plot_review_time_series(df: pd.DataFrame, x: str, title: str, color: str = THEME_COLORS[3]) -> go.Figure:
    """Plot average reviews over time and add overall mean line."""
    fig = px.line(df, x=x, y="mean_reviews", markers=True, title=title, color_discrete_sequence=[color])
    fig.update_traces(line_shape="spline", line=dict(width=5, color=color), marker=dict(size=7, color=color))

    overall_mean = df["mean_reviews"].mean()
    fig.add_hline(
        y=overall_mean,
        line_dash="dot",
        line_color="#999999",
        annotation_text=f"Média geral: {overall_mean:.2f}",
        annotation_position="top left",
        annotation_font=dict(size=13, color="#444"),
        opacity=0.8,
    )

    fig.update_layout(
        height=600,
        width=1000,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Fira Sans, Roboto", size=14, color="#2e2e2e"),
        title_font=dict(size=20, family="Fira Sans, Roboto", color="#111"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEE", zeroline=False),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def main() -> None:
    rooms, reviews = load_data()

    merged = merge_rooms_reviews(rooms, reviews)

    # Prepare time columns and compute metrics
    merged = prepare_time_columns(merged)
    time_metrics = compute_time_metrics(merged)

    # Normalize metrics for plotting on same scale (0-100)
    metrics = ["mean_price", "mean_availability"]
    normalized_plots = {}
    for period, df_period in time_metrics.items():
        id_map = {"Day": "day", "Month": "month", "Quarter": "quarter", "Year": "year"}
        id_col = id_map[period]
        normalized_plots[period] = normalize_0_to_100(df_period, metrics, id_col)

    # Create and save normalized figures
    fig_day_norm = plot_time_metrics(normalized_plots["Day"], "day", "Normalized daily mean", normalized=True)
    fig_month_norm = plot_time_metrics(normalized_plots["Month"], "month", "Normalized monthly mean", normalized=True)
    fig_quarter_norm = plot_time_metrics(normalized_plots["Quarter"], "quarter", "Normalized quarterly mean", normalized=True)
    fig_year_norm = plot_time_metrics(normalized_plots["Year"], "year", "Normalized yearly mean", normalized=True)

    save_figure(fig_day_norm, "normalized_day_price_availability")
    save_figure(fig_month_norm, "normalized_month_price_availability")
    save_figure(fig_quarter_norm, "normalized_quarter_price_availability")
    save_figure(fig_year_norm, "normalized_year_price_availability")

    # Create and save non-normalized figures (with mean lines)
    fig_day = plot_time_metrics(time_metrics["Day"], "day", "Daily mean — Price and Availability")
    fig_month = plot_time_metrics(time_metrics["Month"], "month", "Monthly mean — Price and Availability")
    fig_quarter = plot_time_metrics(time_metrics["Quarter"], "quarter", "Quarterly mean — Price and Availability")
    fig_year = plot_time_metrics(time_metrics["Year"], "year", "Yearly mean — Price and Availability")

    save_figure(fig_day, "day_price_availability")
    save_figure(fig_month, "month_price_availability")
    save_figure(fig_quarter, "quarter_price_availability")
    save_figure(fig_year, "year_price_availability")

    # Reviews metrics and plots (unchanged logic, renamed to English)
    review_metrics = compute_review_metrics(merged)

    fig_reviews_day = plot_review_time_series(review_metrics["Day"], "day", "Daily mean of reviews")
    fig_reviews_month = plot_review_time_series(review_metrics["Month"], "month", "Monthly mean of reviews")
    fig_reviews_quarter = plot_review_time_series(review_metrics["Quarter"], "quarter", "Quarterly mean of reviews")
    fig_reviews_year = plot_review_time_series(review_metrics["Year"], "year", "Yearly mean of reviews")

    save_figure(fig_reviews_day, "day_mean_reviews")
    save_figure(fig_reviews_month, "month_mean_reviews")
    save_figure(fig_reviews_quarter, "quarter_mean_reviews")
    save_figure(fig_reviews_year, "year_mean_reviews")


if __name__ == "__main__":
    main()


