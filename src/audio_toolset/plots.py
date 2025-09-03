from enum import Enum

import numpy as np
import plotly.io as pio
from plotly.graph_objects import Figure, Scatter
from plotly.subplots import make_subplots
from scipy import signal

from audio_toolset.audio_data import AudioData
from audio_toolset.constants.plotting import (
    COLUMN,
    FREQUENCY_PLOT_TICKS,
    LOGARITHMIC_AXES_TYPE,
    PLOT_COLUMNS,
    PLOT_RANGE_X_FREQUENCY,
    PLOT_RANGE_Y_BODE_AMP,
    PLOT_RANGE_Y_BODE_PHASE,
    PLOT_RANGE_Y_DYNAMICS,
    PLOT_RANGE_Y_PSD,
    PLOT_RANGE_Y_WAVEFORM,
    PLOT_ROWS,
    ROW_BODE_AMP,
    ROW_BODE_PHASE,
    ROW_DYNAMICS_PROCESSING,
    ROW_DYNAMICS_RESULT,
    ROW_PSD,
    ROW_WAVEFORM,
    WELCH_SEGMENT_SIZE_DEFAULT,
)
from audio_toolset.util import (
    convert_linear_to_db,
    convert_power_to_db,
)

pio.templates.default = "plotly_dark"


class PlotType(Enum):
    SIGNAL = "Audio Signal"
    BODE = "Bode Plot"
    DYNAMICS = "Dynamics Plot"


def _get_plot_title(
    audio_data: AudioData,
    plot_type: PlotType,
    title: str | None = None,
) -> str:
    if title is not None:
        return title
    if audio_data.info and audio_data.info.name:
        return audio_data.info.name
    return plot_type.value


def _apply_frequency_style_to_xaxes(figure: Figure, row: int, column: int) -> None:
    figure.update_xaxes(
        title_text="Frequency [Hz]",
        type=LOGARITHMIC_AXES_TYPE,
        tickvals=tuple(FREQUENCY_PLOT_TICKS.keys()),
        ticktext=tuple(FREQUENCY_PLOT_TICKS.values()),
        range=PLOT_RANGE_X_FREQUENCY,
        row=row,
        col=column,
    )


def get_signal_plot(audio_data: AudioData, title: str | None = None) -> Figure:
    t = np.arange(0, audio_data.get_duration_s(), audio_data.get_sample_period_s())
    frequencies, power_spectral_density = signal.welch(
        audio_data.data,
        audio_data.sample_rate,
        nperseg=min(WELCH_SEGMENT_SIZE_DEFAULT, audio_data.get_number_of_samples()),
    )

    figure = make_subplots(rows=PLOT_ROWS, cols=PLOT_COLUMNS)
    figure.add_traces(
        [
            Scatter(x=t, y=audio_data.data, name="Waveform"),
            Scatter(
                x=frequencies, y=convert_power_to_db(power_spectral_density), name="PSD"
            ),
        ],
        rows=[ROW_WAVEFORM, ROW_PSD],
        cols=[COLUMN, COLUMN],
    )

    figure.update_xaxes(title_text="Time [s]", row=ROW_WAVEFORM, col=COLUMN)
    figure.update_yaxes(
        title_text="Amplitude",
        range=PLOT_RANGE_Y_WAVEFORM,
        row=ROW_WAVEFORM,
        col=COLUMN,
    )
    _apply_frequency_style_to_xaxes(figure=figure, row=ROW_PSD, column=COLUMN)
    figure.update_yaxes(
        title_text="Power [dB]",
        range=PLOT_RANGE_Y_PSD,
        row=ROW_PSD,
        col=COLUMN,
    )
    figure.update_layout(title=_get_plot_title(audio_data, PlotType.SIGNAL, title))
    return figure


def get_bode_plot(
    audio_data: AudioData, second_order_filter: np.ndarray, title: str
) -> Figure:
    frequencies, frequency_response = signal.sosfreqz(
        sos=second_order_filter, fs=audio_data.sample_rate
    )
    frequency_response_db = convert_linear_to_db(frequency_response)

    figure = make_subplots(rows=PLOT_ROWS, cols=PLOT_COLUMNS)
    figure.add_traces(
        [
            Scatter(x=frequencies, y=frequency_response_db, name="Amplitude"),
            Scatter(
                x=frequencies, y=np.angle(frequency_response, deg=True), name="Phase"
            ),
        ],
        rows=[ROW_BODE_AMP, ROW_BODE_PHASE],
        cols=[COLUMN, COLUMN],
    )
    _apply_frequency_style_to_xaxes(figure=figure, row=ROW_BODE_AMP, column=COLUMN)
    figure.update_yaxes(
        title_text="Amplitude [dB]",
        range=PLOT_RANGE_Y_BODE_AMP,
        row=ROW_BODE_AMP,
        col=COLUMN,
    )
    _apply_frequency_style_to_xaxes(figure=figure, row=ROW_BODE_PHASE, column=COLUMN)
    figure.update_yaxes(
        title_text="Phase [deg]",
        range=PLOT_RANGE_Y_BODE_PHASE,
        row=ROW_BODE_PHASE,
        col=COLUMN,
    )
    figure.update_layout(title=_get_plot_title(audio_data, PlotType.BODE, title))
    return figure


def get_dynamics_plot(
    audio_data_pre: AudioData,
    audio_data_post: AudioData,
    attenuation_db: np.ndarray,
    threshold_db: float,
) -> Figure:
    t = np.arange(
        0,
        audio_data_pre.get_duration_s(),
        audio_data_pre.get_sample_period_s(),
    )
    threshold_line = np.full(audio_data_pre.get_number_of_samples(), threshold_db)
    input_signal_db = convert_linear_to_db(audio_data_pre.data)
    output_signal_db = convert_linear_to_db(audio_data_post.data)

    figure = make_subplots(rows=PLOT_ROWS, cols=PLOT_COLUMNS)
    figure.add_traces(
        [
            Scatter(x=t, y=input_signal_db, name="Input signal"),
            Scatter(x=t, y=threshold_line, name="Threshold"),
            Scatter(x=t, y=attenuation_db, name="Attenuation"),
            Scatter(x=t, y=output_signal_db, name="Output signal"),
        ],
        rows=[
            ROW_DYNAMICS_PROCESSING,
            ROW_DYNAMICS_PROCESSING,
            ROW_DYNAMICS_PROCESSING,
            ROW_DYNAMICS_RESULT,
        ],
        cols=[COLUMN for i in range(4)],
    )
    figure.update_xaxes(title_text="Time [s]", row=ROW_DYNAMICS_PROCESSING, col=COLUMN)
    figure.update_yaxes(
        title_text="Amplitude [dB]",
        range=PLOT_RANGE_Y_DYNAMICS,
        row=ROW_DYNAMICS_PROCESSING,
        col=COLUMN,
    )
    figure.update_xaxes(title_text="Time [s]", row=ROW_DYNAMICS_RESULT, col=COLUMN)
    figure.update_yaxes(
        title_text="Amplitude [dB]",
        range=PLOT_RANGE_Y_DYNAMICS,
        row=ROW_DYNAMICS_RESULT,
        col=COLUMN,
    )
    figure.update_layout(title=_get_plot_title(audio_data_post, PlotType.DYNAMICS))
    return figure
