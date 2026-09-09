let forecastChart = null;


const seriesSelect =
    document.getElementById("seriesSelect");

const forecastButton =
    document.getElementById("forecastButton");

const loading =
    document.getElementById("loading");

const forecastSection =
    document.getElementById("forecastSection");

const errorMessage =
    document.getElementById("errorMessage");


// -----------------------------------------------------
// Utility
// -----------------------------------------------------

function round(value, digits = 2) {

    return Number(value).toFixed(digits);
}


function formatDate(dateString) {

    const date = new Date(dateString);

    return date.toLocaleString(
        [],
        {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        }
    );
}


// -----------------------------------------------------
// Load model metrics
// -----------------------------------------------------

async function loadMetrics() {

    try {

        const response =
            await fetch("/api/metrics");

        const data =
            await response.json();

        const metrics =
            data.test_metrics;

        document.getElementById(
            "mae"
        ).textContent =
            round(metrics.MAE, 2);

        document.getElementById(
            "rmse"
        ).textContent =
            round(metrics.RMSE, 2);

        document.getElementById(
            "mase"
        ).textContent =
            round(metrics.MASE, 3);

        document.getElementById(
            "wape"
        ).textContent =
            round(
                metrics.WAPE_percent,
                2
            ) + "%";

    }

    catch (error) {

        console.error(
            "Could not load metrics:",
            error
        );
    }
}


// -----------------------------------------------------
// Load series
// -----------------------------------------------------

async function loadSeries() {

    try {

        const response =
            await fetch("/api/series");

        const data =
            await response.json();

        seriesSelect.innerHTML = "";

        data.series.forEach(
            seriesId => {

                const option =
                    document.createElement(
                        "option"
                    );

                option.value =
                    seriesId;

                option.textContent =
                    `Series ${seriesId}`;

                seriesSelect.appendChild(
                    option
                );

            }
        );

    }

    catch (error) {

        showError(
            "Unable to load series."
        );
    }
}


// -----------------------------------------------------
// Request forecast
// -----------------------------------------------------

async function generateForecast() {

    const seriesId =
        seriesSelect.value;

    if (!seriesId) {
        return;
    }


    forecastButton.disabled = true;

    forecastButton.textContent =
        "Forecasting...";

    loading.classList.remove(
        "hidden"
    );

    errorMessage.classList.add(
        "hidden"
    );


    try {

        const response =
            await fetch(
                `/api/forecast/${encodeURIComponent(seriesId)}`
            );


        if (!response.ok) {

            throw new Error(
                "Forecast request failed."
            );
        }


        const data =
            await response.json();


        updateSummary(data);

        drawChart(data);

        forecastSection.classList.remove(
            "hidden"
        );

    }

    catch (error) {

        showError(
            "Could not generate forecast."
        );

        console.error(error);
    }

    finally {

        loading.classList.add(
            "hidden"
        );

        forecastButton.disabled = false;

        forecastButton.textContent =
            "Generate Forecast";
    }
}


// -----------------------------------------------------
// Forecast summary
// -----------------------------------------------------

function updateSummary(data) {

    const summary =
        data.summary;


    document.getElementById(
        "forecastTitle"
    ).textContent =
        `Series ${data.series_id} Forecast`;


    document.getElementById(
        "currentActivity"
    ).textContent =
        round(
            summary.current_reviews_per_hour,
            1
        );


    document.getElementById(
        "totalForecast"
    ).textContent =
        round(
            summary.predicted_total_24h,
            0
        );


    document.getElementById(
        "averageForecast"
    ).textContent =
        round(
            summary.predicted_average_per_hour,
            1
        );


    document.getElementById(
        "peakForecast"
    ).textContent =
        round(
            summary.peak_prediction,
            1
        );


    document.getElementById(
        "peakTime"
    ).textContent =
        formatDate(
            summary.peak_time
        );
}


// -----------------------------------------------------
// Chart
// -----------------------------------------------------

function drawChart(data) {

    const history =
        data.history;

    const forecast =
        data.forecast;


    const labels = [

        ...history.map(
            item =>
                formatDate(item.ds)
        ),

        ...forecast.map(
            item =>
                formatDate(item.ds)
        )

    ];


    const historyValues = [

        ...history.map(
            item => item.y
        ),

        ...new Array(
            forecast.length
        ).fill(null)

    ];


    // Include final historical value
    // so forecast line connects cleanly.
    const forecastValues = [

        ...new Array(
            history.length - 1
        ).fill(null),

        history[
            history.length - 1
        ].y,

        ...forecast.map(
            item => item.yhat
        )

    ];


    const canvas =
        document.getElementById(
            "forecastChart"
        );


    if (forecastChart) {

        forecastChart.destroy();
    }


    forecastChart =
        new Chart(
            canvas,
            {

                type: "line",

                data: {

                    labels: labels,

                    datasets: [

                        {

                            label:
                                "Historical Review Activity",

                            data:
                                historyValues,

                            borderColor:
                                "#64748b",

                            backgroundColor:
                                "#64748b",

                            borderWidth:
                                2,

                            pointRadius:
                                0,

                            tension:
                                0.15

                        },

                        {

                            label:
                                "iTransformer Forecast",

                            data:
                                forecastValues,

                            borderColor:
                                "#2563eb",

                            backgroundColor:
                                "#2563eb",

                            borderWidth:
                                3,

                            pointRadius:
                                1,

                            tension:
                                0.15

                        }

                    ]

                },


                options: {

                    responsive:
                        true,

                    maintainAspectRatio:
                        false,

                    interaction: {

                        mode:
                            "index",

                        intersect:
                            false

                    },

                    plugins: {

                        legend: {

                            position:
                                "top"

                        }

                    },

                    scales: {

                        x: {

                            ticks: {

                                maxTicksLimit:
                                    12

                            }

                        },

                        y: {

                            beginAtZero:
                                true,

                            title: {

                                display:
                                    true,

                                text:
                                    "Reviews per hour"

                            }

                        }

                    }

                }

            }
        );
}


// -----------------------------------------------------
// Error
// -----------------------------------------------------

function showError(message) {

    errorMessage.textContent =
        message;

    errorMessage.classList.remove(
        "hidden"
    );
}


// -----------------------------------------------------
// Button
// -----------------------------------------------------

forecastButton.addEventListener(
    "click",
    generateForecast
);


// -----------------------------------------------------
// Initial load
// -----------------------------------------------------

async function initialize() {

    await Promise.all([
        loadMetrics(),
        loadSeries()
    ]);
}


initialize();